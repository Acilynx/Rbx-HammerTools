"""Displacements -> meshes (brushmesh.disp_grid / displacement_record)."""
import numpy as np

from helpers import box
from vmf2rbx.brushes import Displacement
from vmf2rbx.brushmesh import displacement_record, disp_grid, face_polygon
from vmf2rbx.coords import SPU, to_roblox
from vmf2rbx.geometry import brush_to_parts


def disp_box(power=2, dist=None, start=(0, 0, 64)):
    """A 128x128x64 box whose top face (z = 64) is a displacement."""
    b = box(1, (0, 0, 0), (128, 128, 64))
    top = next(s for s in b.sides if s.normal[2] > 0.5)
    n = (1 << power) + 1
    N = n * n
    top.disp = Displacement(power, np.array(start, float), 0.0, np.tile([0, 0, 1.0], (N, 1)),
                            np.zeros(N) if dist is None else np.asarray(dist, float),
                            np.zeros((N, 3)), np.zeros(N))
    return b, top


def _unpack(blob, n, m):
    a = np.frombuffer(blob[: 32 * n], "<f4")
    P = a[: 3 * n].reshape(n, 3); N = a[3 * n: 6 * n].reshape(n, 3); U = a[6 * n:].reshape(n, 2)
    T = np.frombuffer(blob[32 * n:], "<u4").reshape(m, 3)
    return P.astype(float), N.astype(float), U.astype(float), T


def test_brush_with_displacement_becomes_displacement_mesh():
    b, _ = disp_box()
    assert brush_to_parts(b)[0].kind == "displacement"


def test_flat_displacement_reproduces_the_face_outward():
    b, top = disp_box()
    rec, blobs = displacement_record(b, {})
    assert len(rec["meshes"]) == 1                       # only the displaced side is kept
    m = rec["meshes"][0]
    P, N, U, T = _unpack(blobs[m["blob"]], m["n"], m["m"])
    assert m["n"] == 25 and m["m"] == 32
    g = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    area = np.linalg.norm(g, axis=1).sum() / 2
    assert np.isclose(area, (128 * SPU) ** 2)
    up = to_roblox(top.normal)
    assert np.all(g @ up > 0)                            # counter-clockwise seen from outside
    assert np.allclose(N, up)


def test_grid_starts_at_startposition_rows_clockwise():
    b, top = disp_box(start=(128, 0, 64))
    base, pos = disp_grid(b, top)
    assert np.allclose(base[0], (128, 0, 64))
    # clockwise seen from above (+z): from (128, 0) the next corner is (0, 0)
    assert np.allclose(base[5 * 1], (96, 0, 64))         # row 1, column 0: a quarter toward corner 1
    assert np.allclose(base[1], (128, 32, 64))           # row 0, column 1: toward corner 3


def test_distances_move_vertices_and_uvs_stay_on_the_flat_face():
    dist = np.zeros(25); dist[12] = 40.0                 # centre vertex up 40 units
    b, top = disp_box(dist=dist)
    base, pos = disp_grid(b, top)
    assert np.allclose(pos[12], base[12] + [0, 0, 40])
    rec, blobs = displacement_record(b, {"test/tex": (256, 256)})
    m = rec["meshes"][0]
    P, N, U, T = _unpack(blobs[m["blob"]], m["n"], m["m"])
    want = ((base[12] @ top.u.vec) / top.u.scale + top.u.offset) / 256
    assert np.isclose(U[12, 0], want, atol=1e-5)
    assert rec["fidelity"] == "precise" and rec["collider"] is None
