"""Brushes the part decomposer can't express become meshes (brushmesh.py)."""
import numpy as np

from helpers import brush_from_halfspaces, box
from vmf2rbx.brushes import TexAxis, make_brush
from vmf2rbx.brushmesh import brush_mesh_record, face_polygon
from vmf2rbx.geometry import brush_to_parts, brush_volume, classify
from vmf2rbx.coords import SPU, to_roblox


def _unpack(blob, n, m, pad=True):
    a = np.frombuffer(blob[: 4 * 8 * n], "<f4")
    P = a[: 3 * n].reshape(n, 3); N = a[3 * n: 6 * n].reshape(n, 3); U = a[6 * n:].reshape(n, 2)
    T = np.frombuffer(blob[4 * 8 * n:], "<u4").reshape(m, 3)
    # the last 2 triangles / 6 vertices are the zero-area bounds padding (brushmesh.with_bounds)
    if not pad:                                          # collision hulls have none
        return P.astype(float), N.astype(float), U.astype(float), T
    return P[:-6].astype(float), N[:-6].astype(float), U[:-6].astype(float), T[:-2]


def _volume(P, T):
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)


def chamfered_box(id=1, tool_sides=()):
    """A 64-unit cube with one corner cut off by a (1,1,1) plane: 7 sides, not a prism."""
    hs = []
    for i in range(3):
        e = np.eye(3)[i]
        hs += [(e, 64.0), (-e, 0.0)]
    hs.append((np.ones(3), 140.0))
    b = brush_from_halfspaces(id, hs)
    for i in tool_sides:
        b.sides[i].material = "tools/toolsnodraw"
    return b


def octagon_cylinder(id=2, sides=12):
    hs = [(np.array([0, 0, 1.0]), 80.0), (np.array([0, 0, -1.0]), 0.0)]
    for k in range(sides):
        t = 2 * np.pi * (k + 0.5) / sides
        hs.append((np.array([np.cos(t), np.sin(t), 0]), 20.0))
    return brush_from_halfspaces(id, hs)


def test_non_prism_becomes_a_mesh():
    b = chamfered_box()
    assert classify(b) == "other_7"
    assert brush_to_parts(b)[0].cls == "Mesh"


def test_many_sided_cylinder_becomes_a_mesh():
    b = octagon_cylinder(sides=24)      # slices into more than MAX_PIECES parts
    assert brush_to_parts(b)[0].cls == "Mesh"


def test_mesh_encloses_the_brush_outward():
    for b in (chamfered_box(), octagon_cylinder()):
        rec, blobs = brush_mesh_record(b, {})
        assert rec["collider"] is None and len(rec["meshes"]) == 1      # one material, all visible
        m = rec["meshes"][0]
        P, N, U, T = _unpack(blobs[m["blob"]], m["n"], m["m"])
        # positive volume = counter-clockwise from outside; equal to the brush = closed and exact
        assert np.isclose(_volume(P, T), brush_volume(b), rtol=1e-6)
        g = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        assert np.allclose(g, N[T[:, 0]], atol=1e-5)
        # placed at the brush's bounding-box centre
        R = to_roblox(b.verts) * SPU
        assert np.allclose(np.array(rec["c"]) + P.min(0), R.min(0), atol=1e-5)


def test_uvs_follow_hammer_mapping():
    b = chamfered_box()
    W, H = 256, 128
    rec, blobs = brush_mesh_record(b, {"test/tex": (W, H)})
    m = rec["meshes"][0]
    P, N, U, T = _unpack(blobs[m["blob"]], m["n"], m["m"])
    src = (P + rec["c"]) / SPU
    S = np.stack([src[:, 0], -src[:, 2], src[:, 1]], 1)           # back to Source axes
    k = 0
    for s in b.visible_sides:
        poly = face_polygon(b, s)
        shift = None
        for v in poly:
            assert np.allclose(S[k], v, atol=1e-3)
            want = np.array([((v @ s.u.vec) / s.u.scale + s.u.offset) / W, ((v @ s.v.vec) / s.v.scale + s.v.offset) / H])
            d = U[k] - want                    # whole tiles, the same across the face (near_zero)
            assert np.allclose(d, np.round(d), atol=1e-5)
            shift = d if shift is None else shift
            assert np.allclose(d, shift, atol=1e-5)
            k += 1
    assert U.min() >= -1e-6 and U.min(0).max() < 1          # each face starts within the first tile


def test_hidden_side_whose_vertices_stay_covered_needs_no_collider():
    # the other faces still touch every vertex, so the visible mesh's hull is the whole brush
    rec, _ = brush_mesh_record(chamfered_box(tool_sides=[1]), {})
    assert rec["collider"] is None


def test_hidden_sides_get_a_collision_hull():
    # only the chamfer is visible: its hull would be a flat triangle
    b = chamfered_box(tool_sides=range(6))
    rec, blobs = brush_mesh_record(b, {})
    assert rec["collider"] is not None
    c = rec["collider"]
    P, _, _, T = _unpack(blobs[c["blob"]], c["n"], c["m"], pad=False)
    assert np.isclose(_volume(P, T), brush_volume(b), rtol=1e-6)
    assert rec["meshes"][0]["m"] == 1 + 2             # the chamfer, plus the bounds padding


def test_identical_shapes_share_a_blob():
    a, _ = brush_mesh_record(chamfered_box(1), {})
    b, _ = brush_mesh_record(chamfered_box(2), {})
    assert a["meshes"][0]["blob"] == b["meshes"][0]["blob"]


def test_flat_mesh_gets_the_brushes_full_box():
    # only the chamfer is visible; its MeshPart must still span the brush (occlusion culling)
    b = chamfered_box(tool_sides=range(6))
    rec, blobs = brush_mesh_record(b, {})
    m = rec["meshes"][0]
    a = np.frombuffer(blobs[m["blob"]][: 12 * m["n"]], "<f4").reshape(-1, 3)
    R = to_roblox(b.verts) * SPU
    assert np.allclose(a.min(0) + rec["c"], R.min(0), atol=1e-3)
    assert np.allclose(a.max(0) + rec["c"], R.max(0), atol=1e-3)
