"""Shadow stand-ins (shadow.py): hidden inside their mesh, and filling most of it."""
import numpy as np

from helpers import brush_from_halfspaces
from vmf2rbx.coords import SPU, to_roblox
from vmf2rbx.geometry import PartSpec, brush_volume, inside_piece
from vmf2rbx.shadow import MARGIN, convex_brush_boxes, mesh_boxes, shrink


def _corners(p: PartSpec):
    h = p.size / 2
    sgn = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    return p.center + (sgn * h) @ p.rot


def bevel_strip():
    """A 128-unit strip with a 45-degree mitred end: not a prism."""
    hs = [(np.array([1.0, 0, 0]), 128), (np.array([-1.0, 0, 0]), 0), (np.array([0, 1.0, 0]), 8),
          (np.array([0, -1.0, 0]), 0), (np.array([0, 0, 1.0]), 16), (np.array([0, 0, -1.0]), 0),
          (np.array([1.0, 0, 1.0]) / np.sqrt(2), 136 / np.sqrt(2))]
    return brush_from_halfspaces(1, hs)


def _planes(b):
    return [(to_roblox(s.normal), s.dist * SPU) for s in b.sides]


def test_convex_brush_boxes_are_inside_and_fill_it():
    b = bevel_strip()
    R = to_roblox(b.verts) * SPU
    boxes = convex_brush_boxes(R, _planes(b))
    assert boxes
    N = np.array([n for n, _ in _planes(b)]); D = np.array([d for _, d in _planes(b)])
    for p in boxes:
        assert np.all(_corners(p) @ N.T <= D - MARGIN * 0.99)          # strictly inside, by the margin
    filled = sum(np.prod(p.size) for p in boxes)
    assert filled > 0.6 * brush_volume(b)


def _cube_mesh(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    V = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    T = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return V, T


def test_closed_mesh_gets_boxes_inside_it():
    V, T = _cube_mesh((0, 0, 0), (2, 1, 3))
    boxes = mesh_boxes(V, T)
    assert boxes
    for p in boxes:
        c = _corners(p)
        assert np.all(c > -1e-9 + MARGIN * 0.99) and np.all(c < np.array([2, 1, 3]) - MARGIN * 0.99)
    assert sum(np.prod(p.size) for p in boxes) > 0.5 * 6          # inset by about one cell (the wall layer)


def test_open_mesh_gets_no_boxes():
    V = np.array([[0, 0, 0], [2, 0, 0], [2, 0, 2], [0, 0, 2.0]])          # a single card
    T = np.array([[0, 1, 2], [0, 2, 3]])
    assert mesh_boxes(V, T) == []
    V, T = _cube_mesh((0, 0, 0), (2, 2, 2))
    assert mesh_boxes(V, T[:-2]) == [] or all(np.prod(p.size) < 8 for p in mesh_boxes(V, T[:-2]))


def test_shrink_keeps_centre():
    p = PartSpec("Part", np.array([1.0, 2, 3]), np.array([2.0, 2, 2]), np.eye(3))
    q = shrink(p)
    assert np.allclose(q.center, p.center) and np.allclose(q.size, 2 - 2 * MARGIN)
