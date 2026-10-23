"""Light blockers: thick hidden walls in the map's void, against light leaks.

Roblox's sun shadows leak through thin walls and where brushes meet edge to edge (Source walls
are often 8-16 units, under 1.25 studs). Thickening the walls on their hidden side fixes it,
but only where that side can never be seen.

Source maps are sealed: VBSP refuses a map whose playable space reaches the outside. So the
space outside the hull (the void) is never meant to be seen, in outdoor maps too: the sky is a
sealed box of skybox brushes. On a grid of 16-unit cells (Hammer's usual wall grid):

* solid: cells overlapping a sealing brush (world / func_detail / func_brush; tool brushes
  like clips and triggers don't seal; displacement brushes don't either, as in VBSP)
* void: free cells connected to the grid's border. Pockets inside the map are not void, so
  space the grid closes off by mistake gets no blockers
* blockers: void cells within `depth` studs of an opaque, drawn brush, merged into boxes (which
  may also run through the rest of the hidden void and the inside of opaque walls)

Skybox brushes aren't built in Roblox, so through a sky opening the void IS visible, and the
sun shines into the map through it. Void cells seen from the inside through an opening (rays
in many directions) or on the sun's way into one only get a one-cell skin on drawn surfaces,
and none at all near the opening.
"""
from __future__ import annotations

import numpy as np

from .brushes import Brush
from .coords import SPU, to_roblox

CELL = 16.0                    # units
NON_SEALING = ("tools/toolsclip", "tools/toolsplayerclip", "tools/toolsnpcclip", "tools/toolsinvisible",
               "tools/toolstrigger", "tools/toolshint", "tools/toolsskip", "tools/toolsareaportal",
               "tools/toolsoccluder", "tools/toolsfog", "tools/toolsgrenadeclip")
SEALING_OWNERS = ("world", "func_detail", "func_brush")
SKY = "tools/toolsskybox"
VIEW_DIRS = 400                # rays per opening cell, for what's visible through sky openings
SEEN_PAD = 2                   # cells around what the rays saw (they spread apart with distance)
OPENING_PAD = 8                # no blockers this many cells around an opening at all


def _sphere_dirs(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(1 - z * z)
    phi = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


def _rasterize(b: Brush, lo: np.ndarray, n: np.ndarray, out: np.ndarray):
    """Mark cells the brush overlaps (conservative: a cell counts when its box isn't wholly
    outside any of the brush's planes)."""
    blo, bhi = b.bounds
    i0 = np.clip(np.floor((blo - lo) / CELL).astype(int), 0, n - 1)
    i1 = np.clip(np.ceil((bhi - lo) / CELL).astype(int), 1, n)
    axes = [lo[d] + CELL * (np.arange(i0[d], i1[d]) + 0.5) for d in range(3)]
    C = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    inside = np.ones(len(C), bool)
    for s in b.sides:
        reach = CELL / 2 * np.abs(s.normal).sum()
        inside &= C @ s.normal - s.dist < reach - 1e-6
    sub = out[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]]
    sub |= inside.reshape(sub.shape)


def _march(start: np.ndarray, dirs: np.ndarray, lo, n, stop: np.ndarray, mark: np.ndarray, max_cells: int,
           came_from: np.ndarray | None = None, back_cells: int = 4):
    """From every start point (units) along every direction, mark cells until a `stop` cell or
    the grid's edge. With came_from: only rays that, followed backwards, reach a came_from cell
    within back_cells before a stop cell (sight lines from the inside through an opening)."""
    def cells(p):
        idx = np.floor((p - lo) / CELL).astype(int)
        return idx, np.all((idx >= 0) & (idx < n), axis=1)

    for d in dirs:
        alive = np.ones(len(start), bool)
        if came_from is not None:
            ok = np.zeros(len(start), bool)
            blocked = np.zeros(len(start), bool)
            for k in range(1, 2 * back_cells + 1):
                idx, inb = cells(start - d * (CELL / 2) * k)
                idx = np.clip(idx, 0, n - 1)
                cf = inb & came_from[idx[:, 0], idx[:, 1], idx[:, 2]] & ~blocked
                ok |= cf
                blocked |= ~inb | stop[idx[:, 0], idx[:, 1], idx[:, 2]]
            alive &= ok
        p = start.copy()
        for _ in range(int(max_cells * 2)):
            if not alive.any():
                break
            p = p + d * (CELL / 2)
            idx, inb = cells(p)
            alive &= inb
            a = np.nonzero(alive)[0]
            ii = idx[a]
            hit = stop[ii[:, 0], ii[:, 1], ii[:, 2]]
            mark[ii[~hit, 0], ii[~hit, 1], ii[~hit, 2]] = True
            alive[a[hit]] = False


_AXIS_ORDERS = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))


def _grow(free: np.ndarray, seed, order) -> tuple[list[int], list[int]]:
    """The box from `seed` grown through free cells, one axis at a time (both ways), in `order`."""
    lo, hi = list(seed), [x + 1 for x in seed]
    n = free.shape
    for ax in order:
        while lo[ax] > 0:
            sl = [slice(lo[d], hi[d]) for d in range(3)]
            sl[ax] = slice(lo[ax] - 1, lo[ax])
            if not free[tuple(sl)].all():
                break
            lo[ax] -= 1
        while hi[ax] < n[ax]:
            sl = [slice(lo[d], hi[d]) for d in range(3)]
            sl[ax] = slice(hi[ax], hi[ax] + 1)
            if not free[tuple(sl)].all():
                break
            hi[ax] += 1
    return lo, hi


def _cover_boxes(need: np.ndarray, free: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Greedy boxes over `free` cells covering every `need` cell (need must be within free):
    boxes grow through any free cell, so few large boxes do the job. Each box is grown in all
    six axis orders and the one covering the most still-uncovered cells is kept (about a third
    fewer boxes than one fixed order on sp_a1_intro3)."""
    need = need.copy()
    out = []
    for seed in zip(*np.nonzero(need)):
        if not need[seed]:
            continue
        best = None
        for order in _AXIS_ORDERS:
            lo, hi = _grow(free, seed, order)
            gain = int(need[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].sum())
            if best is None or gain > best[0]:
                best = (gain, lo, hi)
        _, lo, hi = best
        need[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = False
        out.append((np.array(lo), np.array(hi)))
    return out


def _rasterize_inside(b: Brush, lo: np.ndarray, n: np.ndarray, out: np.ndarray):
    """Mark cells wholly inside the brush (never visible: inside an opaque wall)."""
    blo, bhi = b.bounds
    i0 = np.clip(np.ceil((blo - lo) / CELL).astype(int), 0, n)
    i1 = np.clip(np.floor((bhi - lo) / CELL).astype(int), 0, n)
    if np.any(i1 <= i0):
        return
    axes = [lo[d] + CELL * (np.arange(i0[d], i1[d]) + 0.5) for d in range(3)]
    C = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    inside = np.ones(len(C), bool)
    for s in b.sides:
        reach = CELL / 2 * np.abs(s.normal).sum()
        inside &= C @ s.normal - s.dist <= -reach + 1e-6
    sub = out[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]]
    sub |= inside.reshape(sub.shape)


def light_blockers(brushes: list[Brush], opaque, sun_dir=None, depth: float = 6.0,
                   report=None, debug: dict | None = None) -> list[dict]:
    """-> [{c, s}] axis-aligned boxes in Roblox studs.

    opaque(material) -> bool: drawn and casting shadows (not glass, not a texture light).
    sun_dir: Roblox-space unit vector toward the sun, or None (no sun)."""
    from scipy import ndimage
    from .shadow import _merge_boxes

    rep = report if report is not None else {}
    sealing, drawn, sky, disp = [], [], [], []
    for b in brushes:
        if b.owner not in SEALING_OWNERS or len(b.verts) == 0:
            continue
        if b.disp_sides:
            disp.append(b)
            continue
        mats = {s.material for s in b.sides}
        if SKY in mats:
            sky.append(b)
        elif mats - set(NON_SEALING):
            sealing.append(b)
            if any(opaque(s.material) for s in b.visible_sides):
                drawn.append(b)
    if not sealing:
        return []
    allb = sealing + sky + disp
    k = max(1, int(round(depth / (CELL * SPU))))                    # blocker depth, cells
    pad = (k + 2) * CELL                                             # room for them past the map
    lo = np.floor(np.min([b.bounds[0] for b in allb], 0) / CELL) * CELL - pad
    hi = np.ceil(np.max([b.bounds[1] for b in allb], 0) / CELL) * CELL + pad
    n = np.round((hi - lo) / CELL).astype(int)
    rep["light blocker grid cells"] = int(np.prod(n))

    solid = np.zeros(tuple(n), bool)
    for b in sealing:
        _rasterize(b, lo, n, solid)
    skyc = np.zeros(tuple(n), bool)
    for b in sky:
        _rasterize(b, lo, n, skyc)
    nearby_disp = np.zeros(tuple(n), bool)
    for b in disp:
        _rasterize(b, lo, n, nearby_disp)
    wall = solid | skyc

    # void: free cells connected to the border
    lab, _ = ndimage.label(~wall)
    border = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel(),
                                       lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
    void = np.isin(lab, border[border > 0])
    del lab
    cube = np.ones((3, 3, 3), bool)
    dsolid = np.zeros(tuple(n), bool)
    for b in drawn:
        _rasterize(b, lo, n, dsolid)
    # sealed? Most free cells against the drawn walls must be inside. If the void got in (a
    # leak, a --region cut, a decompile hole) it covers both sides of the walls: add nothing.
    against = ndimage.binary_dilation(dsolid, structure=cube, iterations=1) & ~wall
    if (against & ~void).sum() < 0.25 * max(1, against.sum()):
        rep["light blockers: map not sealed, none added"] = 1
        return []

    # sky openings: skybox cells next to the inside (free, not void)
    inside = ~wall & ~void
    near_inside = ndimage.binary_dilation(inside, iterations=1)
    opening = skyc & near_inside
    seen = np.zeros(tuple(n), bool)
    oc = np.argwhere(opening)
    rep["sky opening cells"] = len(oc)
    if len(oc):
        start = lo + (oc + 0.5) * CELL
        if len(start) > 4000:                                        # thin out large openings
            start = start[np.random.default_rng(0).choice(len(start), 4000, replace=False)]
        _march(start, _sphere_dirs(VIEW_DIRS), lo, n, solid, seen, max_cells=int(n.max()), came_from=inside)
        if sun_dir is not None:
            s = np.asarray(sun_dir, float)
            s_src = np.array([s[0], -s[2], s[1]])                       # Roblox -> Source axes
            _march(lo + (oc + 0.5) * CELL, [s_src / np.linalg.norm(s_src)], lo, n, solid, seen,
                   max_cells=int(n.max()), came_from=inside)

    near_drawn = ndimage.binary_dilation(dsolid, structure=cube, iterations=k)
    never = ndimage.binary_dilation(nearby_disp, structure=cube, iterations=1)
    if opening.any():
        never |= ndimage.binary_dilation(opening, structure=cube, iterations=OPENING_PAD)
    keep_out = never.copy()
    if seen.any():
        keep_out |= ndimage.binary_dilation(seen, structure=cube, iterations=SEEN_PAD)
    # where the void can be seen (a roof around its sky openings), only a one-cell skin on the
    # drawn surface: it reads as a slightly thicker roof, and still stops light at the seams
    skin = void & ndimage.binary_dilation(dsolid, structure=cube, iterations=1) & ~never
    block = (void & near_drawn & ~keep_out) | skin
    rep["light blocker skin cells (seen through sky openings)"] = int((skin & keep_out).sum())
    rep["light blocker cells"] = int(block.sum())
    # boxes may also run through the rest of the hidden void and through the inside of opaque
    # walls: far fewer, larger boxes
    deep = np.zeros(tuple(n), bool)
    for b in drawn:
        _rasterize_inside(b, lo, n, deep)
    # a box over a wall cell that touches the wall's drawn face would share that face (z-fighting,
    # seen in Studio): only cells with hidden space (void or other such cells) all around
    for _ in range(2):
        deep &= ~ndimage.binary_dilation(~(void | deep), structure=cube, iterations=1)
    free = (void & ~keep_out) | deep | skin

    out = []
    for a, b in _cover_boxes(block, free):
        c_src = lo + (a + b) / 2 * CELL
        size_src = (b - a) * CELL
        out.append(dict(c=[round(float(x), 4) for x in to_roblox(c_src) * SPU],
                        s=[round(float(x), 4) for x in np.abs(to_roblox(size_src)) * SPU]))
    rep["light blockers"] = len(out)
    if debug is not None:
        debug.update(solid=solid, sky=skyc, void=void, seen=seen, block=block, free=free, lo=lo)
    return out
