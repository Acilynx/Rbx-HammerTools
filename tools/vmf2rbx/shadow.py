"""Shadow stand-ins for meshes.

Roblox doesn't draw shadows for MeshParts made from an EditableMesh (CastShadow is ignored;
measured in Studio, and reported on the devforum, "Editable meshes not casting shadows").
Ordinary Parts do cast them, and a Part needs Transparency below ~0.5 to keep its shadow, so
the stand-ins are opaque Parts hidden just inside each mesh: never seen, but in the shadow pass.

* brushes split into Parts that became meshes only for an angled texture: their exact pieces,
  shrunk by MARGIN
* other convex brushes: boxes filling the inside (voxels merged into boxes)
* props: the same for the model's closed parts; inside = ray parity along two axes, both agreeing
  (open meshes, cards and foliage give no stand-ins rather than stray visible boxes)

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


# ------------------------------------------------------------------ closed triangle meshes

def _crossings(V: np.ndarray, T: np.ndarray, axis: int, u: np.ndarray, v: np.ndarray) -> list[np.ndarray]:
    """For each line parallel to `axis` through (u[k], v[k]) in the other two axes: the
    coordinates along `axis` where it crosses the mesh."""
    o = [d for d in range(3) if d != axis]
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    a, b, c = A[:, o], B[:, o], C[:, o]
    den = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    ok = np.abs(den) > 1e-12
    a, b, c, den = a[ok], b[ok], c[ok], den[ok]
    A, B, C = A[ok], B[ok], C[ok]
    out = []
    for k in range(len(u)):
        px, py = u[k], v[k]
        w1 = ((b[:, 1] - c[:, 1]) * (px - c[:, 0]) + (c[:, 0] - b[:, 0]) * (py - c[:, 1])) / den
        w2 = ((c[:, 1] - a[:, 1]) * (px - c[:, 0]) + (a[:, 0] - c[:, 0]) * (py - c[:, 1])) / den
        w3 = 1 - w1 - w2
        hit = (w1 >= 0) & (w2 >= 0) & (w3 >= 0)
        out.append(np.sort(w1[hit] * A[hit, axis] + w2[hit] * B[hit, axis] + w3[hit] * C[hit, axis]))
    return out


def mesh_inside_grid(V: np.ndarray, T: np.ndarray, axes: list[np.ndarray]) -> np.ndarray:
    """Inside/outside of the grid points axes[0] x axes[1] x axes[2] (axis-aligned), by ray
    parity along x and along y; a point is inside only when both agree."""
    gx, gy, gz = axes
    res = np.ones((len(gx), len(gy), len(gz)), bool)
    # lines along x through every (y, z)
    yy, zz = np.meshgrid(gy, gz, indexing="ij")
    cr = _crossings(V, T, 0, yy.ravel(), zz.ravel())
    for k, (j, l) in enumerate(np.ndindex(len(gy), len(gz))):
        res[:, j, l] &= (np.searchsorted(cr[k], gx) % 2) == 1
    xx, zz = np.meshgrid(gx, gz, indexing="ij")
    cr = _crossings(V, T, 1, xx.ravel(), zz.ravel())
    for k, (i, l) in enumerate(np.ndindex(len(gx), len(gz))):
        res[i, :, l] &= (np.searchsorted(cr[k], gy) % 2) == 1
    return res


MESH_MIN_CELL = 0.06     # studs
MESH_MAX_CELLS = 64      # per axis
CELLS_ACROSS = 8         # cells across the thinnest side (the wall layer costs about one)


def _surface_cells(V: np.ndarray, T: np.ndarray, lo: np.ndarray, step: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Cells any triangle passes through (sampled densely on each triangle)."""
    occ = np.zeros(tuple(n), bool)
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    edge = np.maximum(np.linalg.norm(B - A, axis=1), np.maximum(np.linalg.norm(C - A, axis=1), np.linalg.norm(C - B, axis=1)))
    k_all = np.clip(np.ceil(edge / (0.4 * step.min())).astype(int), 1, 200)
    for k in np.unique(k_all):
        sel = k_all == k
        i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
        keep = i + j <= k
        u, v = (i[keep] / k)[None, :, None], (j[keep] / k)[None, :, None]
        P = A[sel][:, None] + u * (B[sel] - A[sel])[:, None] + v * (C[sel] - A[sel])[:, None]
        f = (P.reshape(-1, 3) - lo) / step
        for e in (-1e-6, 1e-6):              # a surface exactly on a cell boundary walls both cells
            idx = np.clip(np.floor(f + e).astype(int), 0, n - 1)
            occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return occ


def _flood_outside(blocked: np.ndarray) -> np.ndarray:
    """Cells reachable from the grid's border without crossing `blocked` (6-connected)."""
    pad = np.pad(blocked, 1, constant_values=False)
    out = np.zeros_like(pad)
    out[0, :, :] = out[-1, :, :] = out[:, 0, :] = out[:, -1, :] = out[:, :, 0] = out[:, :, -1] = True
    out &= ~pad
    while True:
        grown = out.copy()
        grown[1:] |= out[:-1]; grown[:-1] |= out[1:]
        grown[:, 1:] |= out[:, :-1]; grown[:, :-1] |= out[:, 1:]
        grown[:, :, 1:] |= out[:, :, :-1]; grown[:, :, :-1] |= out[:, :, 1:]
        grown &= ~pad
        if (grown == out).all():
            return out[1:-1, 1:-1, 1:-1]
        out = grown


def mesh_boxes(V: np.ndarray, T: np.ndarray, margin: float = 0.01, max_boxes: int = 16) -> list[PartSpec]:
    """Stand-in boxes (axis-aligned, in the mesh's own space) filling a mesh's enclosed volume.

    Cells the surface passes through are walls; the outside is flood-filled in from the grid's
    border; what it can't reach is inside. Works for models made of overlapping closed pieces
    (where ray parity fails); an open model leaks and gets nothing, rather than a stray box.
    Inside cells never touch a surface cell's interior, so the boxes stay hidden."""
    if len(T) == 0:
        return []
    lo, hi = V.min(0), V.max(0)
    ext = hi - lo
    if ext.min() < 3 * MESH_MIN_CELL:
        return []
    cell = max(MESH_MIN_CELL, float(ext.min()) / CELLS_ACROSS, float(ext.max()) / MESH_MAX_CELLS)
    lo, hi = lo - cell, hi + cell                    # a free border for the flood to start in
    n = np.minimum(np.maximum(np.ceil((hi - lo) / cell).astype(int), 1), MESH_MAX_CELLS + 2)
    step = (hi - lo) / n
    wall = _surface_cells(V, T, lo, step, n)
    inside = ~wall & ~_flood_outside(wall)
    boxes = sorted(_merge_boxes(inside), key=lambda b: -np.prod(b[1] - b[0]))[:max_boxes]
    out = []
    for a, b in boxes:
        c = lo + step * (a + b) / 2
        size = step * (b - a) - 2 * margin
        if np.all(size > 0.02):
            out.append(PartSpec("Part", c, size, np.eye(3)))
    return out


def part_record(p: PartSpec) -> dict:
    return dict(cls=p.cls, c=[round(float(x), 5) for x in p.center], s=[round(float(x), 5) for x in p.size],
                r=[round(float(x), 6) for x in np.asarray(p.rot).flatten()])
