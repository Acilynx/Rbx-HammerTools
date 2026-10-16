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
