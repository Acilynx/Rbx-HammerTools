"""Overlays: the texel Roblox draws on the overlay part matches the overlay's u/v ramp."""
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest

from test_textures import oracle_frame, roblox_texel
from vmf2rbx.coords import point_to_source
from vmf2rbx.overlays import overlay_parts


def overlay(**kv):
    e = defaultdict(str, classname="info_overlay", material="signage/test")
    e.update(kv)
    return e


CASES = [
    # floor strip like Portal's indicator lights: U repeats 6x, V mirrored
    dict(BasisOrigin="-800 -956 128", BasisU="1 0 0", BasisV="0 1 0", BasisNormal="0 0 1",
         uv0="-96 -4 0", uv1="-96 4 0", uv2="96 4 0", uv3="96 -4 0", StartU="0", EndU="6", StartV="1", EndV="0"),
    # wall sign facing -y
    dict(BasisOrigin="10 20 64", BasisU="-1 0 0", BasisV="0 0 -1", BasisNormal="0 -1 0",
         uv0="-16 -32 0", uv1="-16 32 0", uv2="16 32 0", uv3="16 -32 0", StartU="0", EndU="1", StartV="0", EndV="1"),
    # ceiling, off-centre corners, partial texture range
    dict(BasisOrigin="0 0 256", BasisU="0 1 0", BasisV="1 0 0", BasisNormal="0 0 -1",
         uv0="0 -8 0", uv1="0 24 0", uv2="40 24 0", uv3="40 -8 0", StartU="0.25", EndU="0.75", StartV="0", EndV="0.5"),
]


@pytest.mark.parametrize("kv", CASES)
def test_overlay_texels(kv):
    W, H = 128, 64
    recs, mats, rep = overlay_parts(SimpleNamespace(entities=[overlay(**kv)]), {"signage/test": (W, H)})
    assert len(recs) == 1 and rep["overlays"] == 1
    part, ft, info = recs[0]
    assert ft is not None
    O = np.array([float(x) for x in kv["BasisOrigin"].split()])
    U = np.array([float(x) for x in kv["BasisU"].split()])
    V = np.array([float(x) for x in kv["BasisV"].split()])
    uv = np.array([[float(x) for x in kv[f"uv{i}"].split()][:2] for i in range(4)])
    x0, x1, y0, y1 = uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].min(), uv[:, 1].max()
    su, eu, sv, ev = (float(kv[k]) for k in ("StartU", "EndU", "StartV", "EndV"))
    Of, Ut, Vt = oracle_frame(part.cls, part.size, "Front")
    hs = part.size / 2
    lu, lv = np.abs(Ut) @ hs * 2, np.abs(Vt) @ hs * 2
    rng = np.random.default_rng(0)
    for _ in range(100):
        u, v = rng.uniform(0, lu), rng.uniform(0, lv)
        pw = part.center + part.rot.T @ (Of + u * Ut + v * Vt)
        ps = point_to_source(pw)
        a = ((ps - O) @ U - x0) / (x1 - x0)
        b = ((ps - O) @ V - y0) / (y1 - y0)
        want = (W * (su + (eu - su) * a), H * (sv + (ev - sv) * b))
        got = roblox_texel(part, ft, u, v, W, H)
        ds, dt = (got[0] - want[0]) % W, (got[1] - want[1]) % H
        assert min(ds, W - ds) < 1e-3 and min(dt, H - dt) < 1e-3
    # the part sits just off the surface, on the normal side
    N = np.array([float(x) for x in kv["BasisNormal"].split()])
    assert 0 < (point_to_source(part.center) - O) @ N < 1.0


# ------------------------------------------------------------------ projected overlay meshes

from vmf2rbx.coords import SPU

W, H = 128, 64

def _box_with_ids():
    from helpers import box
    b = box(1, (0, 0, 0), (128, 128, 64))
    for k, s in enumerate(b.sides, 1):
        s.sid = 100 + k
    return b


def _side(b, n):
    return next(s for s in b.sides if np.allclose(s.normal, n))


def _mesh_points(rec, blobs):
    m = rec["meshes"][0]
    a = np.frombuffer(blobs[m["blob"]][: 12 * m["n"]], "<f4").reshape(-1, 3)[:-6] + rec["c"]
    uv = np.frombuffer(blobs[m["blob"]][24 * m["n"]: 32 * m["n"]], "<f4").reshape(-1, 2)[:-6]
    S = np.stack([a[:, 0], -a[:, 2], a[:, 1]], 1) / SPU           # back to Source units
    return S, uv


def test_overlay_mesh_on_one_face():
    from vmf2rbx.overlays import LIFT, overlay_meshes
    b = _box_with_ids()
    top = _side(b, [0, 0, 1])
    e = overlay(BasisOrigin="64 64 64", BasisU="1 0 0", BasisV="0 1 0", BasisNormal="0 0 1",
                sides=str(top.sid), uv0="-16 -16 0", uv1="-16 16 0", uv2="16 16 0", uv3="16 -16 0")
    out, rep = overlay_meshes(SimpleNamespace(entities=[e]), [b], {"signage/test": (W, H)})
    assert rep["overlays"] == 1
    rec, blobs = out[0]
    S, uv = _mesh_points(rec, blobs)
    assert np.allclose(S[:, 2], 64 + LIFT / SPU, atol=1e-3)            # on the face, lifted
    assert np.allclose(np.sort(np.unique(S[:, 0].round(3))), [48, 80])  # clipped to the overlay
    assert rec["nocollide"]


def test_overlay_mesh_wraps_an_edge():
    from vmf2rbx.overlays import overlay_meshes
    b = _box_with_ids()
    top, side = _side(b, [0, 0, 1]), _side(b, [1, 0, 0])
    # centred on the top face's +x edge, half on the top, half hanging over the side
    e = overlay(BasisOrigin="128 64 64", BasisU="1 0 0", BasisV="0 1 0", BasisNormal="0.7071 0 0.7071",
                sides=f"{top.sid} {side.sid}", uv0="-16 -16 0", uv1="-16 16 0", uv2="16 16 0", uv3="16 -16 0")
    out, rep = overlay_meshes(SimpleNamespace(entities=[e]), [b], {"signage/test": (W, H)})
    assert rep["overlays wrapping an edge"] == 1
    S, _ = _mesh_points(*out[0])
    on_top = S[:, 2] > 64 + 0.05            # lifted off the top face
    on_side = S[:, 0] > 128 + 0.05          # lifted off the side face
    assert on_top.any() and on_side.any()
    assert np.all(on_top | on_side)                                     # every piece hugs a face
    assert S[on_top][:, 0].max() <= 128 + 1e-3 and S[on_side][:, 2].max() <= 64 + 1e-3


def test_overlay_without_faces_is_dropped():
    from vmf2rbx.overlays import overlay_meshes
    b = _box_with_ids()
    e = overlay(BasisOrigin="64 64 64", BasisU="1 0 0", BasisV="0 1 0", BasisNormal="0 0 1", sides="")
    out, rep = overlay_meshes(SimpleNamespace(entities=[e]), [b], {"signage/test": (W, H)})
    assert out == [] and rep["overlay on no faces (not drawn)"] == 1
