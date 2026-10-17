"""Shadow stand-ins for meshes.

Roblox doesn't draw shadows for MeshParts made from an EditableMesh (CastShadow is ignored;
measured in Studio, and reported on the devforum, "Editable meshes not casting shadows").
Ordinary Parts do cast them, and a Part needs Transparency below ~0.5 to keep its shadow, so
the stand-ins are opaque Parts hidden just inside each mesh: never seen, but in the shadow pass.

* brushes split into Parts that became meshes only for an angled texture: their exact pieces,
  shrunk by MARGIN
* other convex brushes: boxes filling the inside (voxels merged into boxes)
* props: volume boxes and thin plates inside the model's solid parts (see "prop models" below);
  open meshes, cards and foliage give no stand-ins rather than stray visible boxes

All sizes in studs.
"""
from __future__ import annotations

import numpy as np

from .geometry import PartSpec

MARGIN = 0.03          # stand-ins sit this far inside the surface
MAX_CELLS = 18         # voxels along the longest side
MIN_CELL = 0.12
MAX_BOXES = 12


def shrink(p: PartSpec, margin: float = MARGIN) -> PartSpec:
    return PartSpec(p.cls, p.center, np.maximum(p.size - 2 * margin, 0.01), p.rot, p.piece, p.kind)


def _merge_boxes(occ: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Greedy merge of an occupancy grid into boxes: (lo, hi) cell index ranges, hi exclusive."""
    occ = occ.copy()
    out = []
    nx, ny, nz = occ.shape
    for i, j, k in zip(*np.nonzero(occ)):
        if not occ[i, j, k]:
            continue
        i2 = i + 1
        while i2 < nx and occ[i2, j, k]:
            i2 += 1
        j2 = j + 1
        while j2 < ny and occ[i:i2, j2, k].all():
            j2 += 1
        k2 = k + 1
        while k2 < nz and occ[i:i2, j:j2, k2].all():
            k2 += 1
        occ[i:i2, j:j2, k:k2] = False
        out.append((np.array([i, j, k]), np.array([i2, j2, k2])))
    return out


def fill_boxes(inside, frame: np.ndarray, lo: np.ndarray, hi: np.ndarray, max_boxes: int = MAX_BOXES,
               margin: float = MARGIN) -> list[PartSpec]:
    """Boxes (in world space) filling the region where inside(points (N,3) world) is true, on a
    grid aligned with `frame` (rows = axes) spanning [lo, hi] in frame coordinates. A cell
    counts when all 8 of its corners are inside. Largest boxes first, shrunk by margin."""
    lo, hi = lo + margin, hi - margin            # stand-ins sit `margin` inside the surface
    ext = hi - lo
    if np.any(ext <= 0.02):
        return []
    cell = max(float(ext.max()) / MAX_CELLS, MIN_CELL)
    n = np.maximum(np.floor(ext / cell).astype(int), 1)
    step = ext / n
    axes = [lo[d] + step[d] * np.arange(n[d] + 1) for d in range(3)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1)                 # corner grid, frame coords
    ins = inside(G.reshape(-1, 3) @ frame).reshape(G.shape[:3])
    occ = (ins[:-1, :-1, :-1] & ins[1:, :-1, :-1] & ins[:-1, 1:, :-1] & ins[:-1, :-1, 1:]
           & ins[1:, 1:, :-1] & ins[1:, :-1, 1:] & ins[:-1, 1:, 1:] & ins[1:, 1:, 1:])
    boxes = sorted(_merge_boxes(occ), key=lambda b: -np.prod(b[1] - b[0]))[:max_boxes]
    out = []
    for a, b in boxes:
        c_frame = lo + step * (a + b) / 2
        size = step * (b - a)
        if np.all(size > 0.02):
            out.append(PartSpec("Part", c_frame @ frame, size, frame))
    return out


def brush_frame(R: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """An orthonormal frame for a convex brush (Roblox space), aligned with its faces: the
    normal of the side with the most vertices, then the side normal most perpendicular to it."""
    a = max(normals, key=lambda n: np.sum(np.isclose(R @ n, (R @ n).max(), atol=1e-3)))
    a = a / np.linalg.norm(a)
    proj = [n - (n @ a) * a for n in normals]
    b = max(proj, key=np.linalg.norm)
    if np.linalg.norm(b) < 1e-6:
        b = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
    b = b / np.linalg.norm(b)
    return np.array([b, np.cross(a, b), a])


def convex_brush_boxes(R: np.ndarray, planes: list[tuple[np.ndarray, float]], margin: float = MARGIN):
    """R: brush vertices in Roblox studs; planes: (outward normal, dist) in the same space."""
    N = np.array([n for n, _ in planes])
    D = np.array([d for _, d in planes])
    frame = brush_frame(R, N)
    F = R @ frame.T
    lo, hi = F.min(0), F.max(0)

    def inside(X):                               # within `margin` of every face
        return np.all(X @ N.T <= D - margin + 1e-6, axis=1)
    return fill_boxes(inside, frame, lo, hi, margin=margin)


def hull_planes(P: np.ndarray, T: np.ndarray) -> list[tuple[np.ndarray, float]]:
    """Outward planes (unit normal, dist) of a convex hull given as points + triangles."""
    c = P.mean(0)
    out = []
    for a, b, d in T:
        n = np.cross(P[b] - P[a], P[d] - P[a])
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n = n / ln
        if n @ (c - P[a]) > 0:
            n = -n
        out.append((n, float(n @ P[a])))
    return out


def hull_boxes(pieces: list[tuple[np.ndarray, np.ndarray]], margin: float = MARGIN, max_boxes: int = 16) -> list[PartSpec]:
    """Stand-ins from convex pieces (a .phy collision model, already in the mesh's space and
    units): boxes filling each piece, largest first."""
    out = []
    for P, T in pieces:
        planes = hull_planes(P, T)
        if len(planes) >= 4:
            out += convex_brush_boxes(P, planes, margin)
    return sorted(out, key=lambda p: -np.prod(p.size))[:max_boxes]


# ------------------------------------------------------------------ prop models (triangle meshes)
#
# Two kinds of stand-in, both kept strictly inside the model's solid parts:
#
# * volume boxes: a grid of points classified inside/outside by winding number (signed ray
#   crossings, counted along all three axes, all three must agree), cells with all 8 corners
#   inside, merged into boxes. Winding numbers handle models built from overlapping closed
#   pieces; an open model (a bin, a card) fails at least one axis and gets nothing there.
# * plates: Portal 2's broken walls, beams and panels are thin sheets (sp_a1_intro3's
#   wall_dest models average 0.07 studs thick), far thinner than any affordable grid. For each
#   large flat region of the surface, a ray from each grid point on it goes inward to the
#   first surface behind; where that surface faces back (so the ray stayed inside the solid),
#   a thin plate fills the gap, a little behind the face. Plates of the opposite face of the
#   same sheet are skipped.

def _grid_ray_hits(A, B, C, gu, gv):
    """Rays along +z through every grid point (gu[i], gv[j]) against triangles A, B, C (rows,
    already in a frame where z is the ray axis) -> (i, j, z, triangle) per hit."""
    a2, b2, c2 = A[:, :2], B[:, :2], C[:, :2]
    lo2 = np.minimum(np.minimum(a2, b2), c2)
    hi2 = np.maximum(np.maximum(a2, b2), c2)
    i0, i1 = np.searchsorted(gu, lo2[:, 0]), np.searchsorted(gu, hi2[:, 0], "right")
    j0, j1 = np.searchsorted(gv, lo2[:, 1]), np.searchsorted(gv, hi2[:, 1], "right")
    nj = np.maximum(j1 - j0, 0)
    cnt = np.maximum(i1 - i0, 0) * nj
    tri = np.repeat(np.arange(len(A)), cnt)
    if len(tri) == 0:
        e = np.zeros(0, int)
        return e, e, np.zeros(0), e
    loc = np.arange(len(tri)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    ii = i0[tri] + loc // nj[tri]
    jj = j0[tri] + loc % nj[tri]
    px, py = gu[ii], gv[jj]
    a, b, c = a2[tri], b2[tri], c2[tri]
    den = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    ok = np.abs(den) > 1e-14
    den = np.where(ok, den, 1.0)
    w1 = ((b[:, 1] - c[:, 1]) * (px - c[:, 0]) + (c[:, 0] - b[:, 0]) * (py - c[:, 1])) / den
    w2 = ((c[:, 1] - a[:, 1]) * (px - c[:, 0]) + (a[:, 0] - c[:, 0]) * (py - c[:, 1])) / den
    w3 = 1 - w1 - w2
    hit = ok & (w1 >= 0) & (w2 >= 0) & (w3 >= 0)
    z = w1 * A[tri, 2] + w2 * B[tri, 2] + w3 * C[tri, 2]
    return ii[hit], jj[hit], z[hit], tri[hit]


def _axis_winding(V, T, axis, g):
    """Winding number at the grid points g[0] x g[1] x g[2] from rays along +axis: the sum of
    +1 (leaving) / -1 (entering) over the crossings beyond each point."""
    o = [d for d in range(3) if d != axis] + [axis]
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    sgn = np.sign(np.cross(B - A, C - A)[:, axis]).astype(np.int32)   # before permuting (axis 1's is odd)
    A, B, C = A[:, o], B[:, o], C[:, o]
    ii, jj, z, tri = _grid_ray_hits(A, B, C, g[o[0]], g[o[1]])
    k = np.searchsorted(g[axis], z)                 # points before the crossing: index < k
    D = np.zeros((len(g[o[0]]), len(g[o[1]]), len(g[axis]) + 1), np.int32)
    np.add.at(D, (ii, jj, np.zeros_like(k)), sgn[tri])
    np.add.at(D, (ii, jj, k), -sgn[tri])
    W = np.cumsum(D, axis=2)[:, :, :-1]
    return np.transpose(W, np.argsort(o))


def winding_inside(V: np.ndarray, T: np.ndarray, g: list[np.ndarray]) -> np.ndarray:
    """Grid points inside the mesh (winding >= 1 along all three axes)."""
    ins = np.ones(tuple(len(x) for x in g), bool)
    for a in range(3):
        ins &= _axis_winding(V, T, a, g) >= 1
    return ins


def points_inside(V: np.ndarray, T: np.ndarray, P: np.ndarray) -> np.ndarray:
    """The same test for arbitrary points (N, 3)."""
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    ok = np.ones(len(P), bool)
    for axis in range(3):
        o = [d for d in range(3) if d != axis]
        a, b, c = A[:, o], B[:, o], C[:, o]
        sgn = np.sign(np.cross(B - A, C - A)[:, axis])
        den = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
        good = np.abs(den) > 1e-14
        a, b, c, den, sgn = a[good], b[good], c[good], den[good], sgn[good]
        Az, Bz, Cz = A[good, axis], B[good, axis], C[good, axis]
        step = max(1, 2_000_000 // max(len(a), 1))
        for s in range(0, len(P), step):
            q = P[s:s + step]
            px, py = q[:, o[0], None], q[:, o[1], None]
            w1 = ((b[:, 1] - c[:, 1]) * (px - c[:, 0]) + (c[:, 0] - b[:, 0]) * (py - c[:, 1])) / den
            w2 = ((c[:, 1] - a[:, 1]) * (px - c[:, 0]) + (a[:, 0] - c[:, 0]) * (py - c[:, 1])) / den
            w3 = 1 - w1 - w2
            z = w1 * Az + w2 * Bz + w3 * Cz
            hit = (w1 >= 0) & (w2 >= 0) & (w3 >= 0) & (z > q[:, axis, None])
            ok[s:s + step] &= (hit * sgn).sum(1) >= 1
    return ok


VOLUME_CELLS = 160                 # grid points along the longest side
VOLUME_CELL = (0.05, 0.25)         # studs, min and max


def volume_cell(V: np.ndarray) -> float:
    return float(np.clip((V.max(0) - V.min(0)).max() / VOLUME_CELLS, *VOLUME_CELL))


def mesh_boxes(V: np.ndarray, T: np.ndarray, margin: float = MARGIN, max_boxes: int = 24) -> list[PartSpec]:
    """Volume boxes (axis-aligned, in the mesh's own space), largest first."""
    if len(T) == 0:
        return []
    cell = volume_cell(V)
    lo, hi = V.min(0), V.max(0)
    n = np.maximum(np.ceil((hi - lo) / cell).astype(int), 1)
    step = (hi - lo) / n
    jitter = np.array([1.37e-4, 2.71e-4, 3.93e-4])     # keep rays off exact edges
    g = [lo[d] + jitter[d] + step[d] * np.arange(n[d] + 1) for d in range(3)]
    ins = winding_inside(V, T, g)
    occ = (ins[:-1, :-1, :-1] & ins[1:, :-1, :-1] & ins[:-1, 1:, :-1] & ins[:-1, :-1, 1:]
           & ins[1:, 1:, :-1] & ins[1:, :-1, 1:] & ins[:-1, 1:, 1:] & ins[1:, 1:, 1:])
    boxes = sorted(_merge_boxes(occ), key=lambda b: -np.prod(b[1] - b[0]))[:max_boxes]
    out = []
    for a, b in boxes:
        c = lo + jitter + step * (a + b) / 2
        size = step * (b - a) - 2 * margin
        if np.all(size > 0.02):
            out.append(PartSpec("Part", c, size, np.eye(3)))
    return out


PLATE_ANGLE = np.cos(np.radians(4))   # triangles within this of a region's normal...
PLATE_PLANE = 0.015                   # ...and this close to its plane (studs) join it
PLATE_MIN_AREA = 0.25                 # square studs
PLATE_REGIONS = 60                    # largest flat regions considered
PLATE_GAP = 0.01                      # plates stay this far from both faces
PLATE_MAX = 0.25                      # thickest plate
PLATE_MIN = 0.008
PLATE_MIN_FACE = 0.08                 # square studs: smaller plates add little shadow


def _plane_frame(n: np.ndarray) -> np.ndarray:
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(a, n)
    u /= np.linalg.norm(u)
    return np.array([u, np.cross(n, u), n])


def mesh_plates(V: np.ndarray, T: np.ndarray, max_plates: int = 150, skip_depth: float = np.inf) -> list[PartSpec]:
    """Plates (see above), in the mesh's own space, largest first. Regions where the solid is
    deeper than skip_depth are left to the volume boxes."""
    if len(T) == 0 or max_plates <= 0:
        return []
    A0, B0, C0 = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    N = np.cross(B0 - A0, C0 - A0)
    area = np.linalg.norm(N, axis=1) / 2
    good = area > 1e-8
    Nn = np.zeros_like(N)
    Nn[good] = N[good] / (2 * area[good])[:, None]
    cen = (A0 + B0 + C0) / 3
    region = -np.ones(len(T), int)
    regions = []
    for s in np.argsort(-area):
        if region[s] >= 0 or not good[s]:
            continue
        n, d = Nn[s], float(Nn[s] @ cen[s])
        m = (region < 0) & good & (Nn @ n > PLATE_ANGLE) & (np.abs(cen @ n - d) < PLATE_PLANE)
        region[m] = len(regions)
        regions.append((n, d, float(area[m].sum())))
    done = np.zeros(len(regions) + 1, bool)          # last slot: "no region"
    plates = []
    for ri in np.argsort([-r[2] for r in regions])[:PLATE_REGIONS]:
        n, d, ar = regions[ri]
        if ar < PLATE_MIN_AREA:
            break
        F = _plane_frame(n)
        A, B, C = A0 @ F.T, B0 @ F.T, C0 @ F.T
        sel = region == ri
        P2 = np.concatenate([A[sel], B[sel], C[sel]])[:, :2]
        lo, hi = P2.min(0), P2.max(0)
        h = max(float(np.clip(np.sqrt(ar) / 24, 0.05, 0.2)), float((hi - lo).max()) / 200)
        nu, nv = np.maximum(np.ceil((hi - lo) / h).astype(int), 1)
        su, sv = (hi - lo) / [nu, nv]
        gu = lo[0] + 1.7e-5 + su * np.arange(nu + 1)
        gv = lo[1] + 2.9e-5 + sv * np.arange(nv + 1)
        ii, jj, z, tri = _grid_ray_hits(A, B, C, gu, gv)
        on = sel[tri] & (np.abs(z - d) < 2 * PLATE_PLANE)
        cov = np.zeros((nu + 1, nv + 1), bool)
        cov[ii[on], jj[on]] = True
        # first surface behind the face, per grid point
        beh = z < d - 1e-4
        o = np.argsort(-z[beh])
        ib, jb, zb, tb = ii[beh][o], jj[beh][o], z[beh][o], tri[beh][o]
        first = np.unique(ib * (nv + 1) + jb, return_index=True)[1]
        ib, jb, zb, tb = ib[first], jb[first], zb[first], tb[first]
        backface = Nn[tb] @ n < -0.5
        depth = np.full((nu + 1, nv + 1), np.inf)
        depth[ib[backface], jb[backface]] = d - zb[backface]
        twin = np.zeros((nu + 1, nv + 1), bool)          # the sheet's other face already has plates
        twin[ib, jb] = done[region[tb]]
        ok = cov & np.isfinite(depth) & ~twin
        cell = ok[:-1, :-1] & ok[1:, :-1] & ok[:-1, 1:] & ok[1:, 1:]
        cdepth = np.minimum(np.minimum(depth[:-1, :-1], depth[1:, :-1]), np.minimum(depth[:-1, 1:], depth[1:, 1:]))
        th = np.minimum(cdepth - 2 * PLATE_GAP, PLATE_MAX)
        cell &= (th >= PLATE_MIN) & (cdepth <= skip_depth)
        if not cell.any():
            continue
        done[ri] = True
        bins = np.where(cell, np.floor(th / 0.02), -1).astype(int)
        for bval in np.unique(bins[bins >= 0]):
            thick = max(bval * 0.02, PLATE_MIN)
            for a, b in _merge_boxes((bins == bval)[:, :, None]):
                cu = gu[0] + su * (a[0] + b[0]) / 2
                cv = gv[0] + sv * (a[1] + b[1]) / 2
                size = np.array([su * (b[0] - a[0]), sv * (b[1] - a[1]), thick])
                if size[0] > 0.04 and size[1] > 0.04 and size[0] * size[1] >= PLATE_MIN_FACE:
                    plates.append(PartSpec("Part", np.array([cu, cv, d - PLATE_GAP - thick / 2]) @ F, size, F))
    plates.sort(key=lambda p: -p.size[0] * p.size[1])
    plates = plates[:max_plates * 2]
    # a plate whose corners aren't all inside the solid sits in an open shell (a bin, the back
    # of a one-sided panel): visible, so dropped
    if plates:
        corners = np.array([[x, y, z] for x in (-.45, .45) for y in (-.45, .45) for z in (0.0,)])
        P = np.concatenate([p.center + (corners * p.size) @ np.asarray(p.rot) for p in plates])
        ins = points_inside(V, T, P).reshape(len(plates), -1).all(1)
        plates = [p for p, k in zip(plates, ins) if k]
    return plates[:max_plates]


MODEL_PARTS = 700          # stand-ins per model, shared by all its placements...
MODEL_PARTS_RANGE = (6, 120)   # ...within these per placement (sp_a1_intro3: about 16k in all)


def model_budget(placements: int) -> int:
    return int(np.clip(MODEL_PARTS / max(placements, 1), *MODEL_PARTS_RANGE))


def model_stand_ins(V: np.ndarray, T: np.ndarray, budget: int = 150) -> list[PartSpec]:
    """Volume boxes for the chunky parts, plates for the thin ones, at most `budget` parts."""
    boxes = mesh_boxes(V, T, max_boxes=min(24, max(budget // 3, 4)))
    plates = mesh_plates(V, T, budget - len(boxes), skip_depth=4 * volume_cell(V))
    return boxes + plates


def part_record(p: PartSpec, color=None) -> dict:
    """color: the surface's average colour, so a stand-in showing through at a distance (depth
    precision) blends in."""
    rec = dict(cls=p.cls, c=[round(float(x), 5) for x in p.center], s=[round(float(x), 5) for x in p.size],
               r=[round(float(x), 6) for x in np.asarray(p.rot).flatten()])
    if color is not None:
        rec["col"] = [int(x) for x in color[:3]]
    return rec
