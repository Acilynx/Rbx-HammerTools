import numpy as np

from helpers import box
from vmf2rbx.coords import SPU, to_roblox
from vmf2rbx.sealing import light_blockers

OPAQUE = lambda m: not m.startswith("tools/")


def room(lo=(0, 0, 0), hi=(512, 512, 256), t=16, ceiling="test/tex", missing=None):
    """A hollow box room, walls t units thick, inside [lo, hi] (Source units)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    walls = []
    for axis in range(3):
        for side, name in ((0, "lo"), (1, "hi")):
            a, b = lo - t, hi + t
            if side == 0:
                b = b.copy(); b[axis] = lo[axis]
            else:
                a = a.copy(); a[axis] = hi[axis]
            if missing == (axis, side):
                continue
            mat = ceiling if (axis == 2 and side == 1) else "test/tex"
            walls.append(box(len(walls) + 1, a, b, material=mat))
    return walls


def _corners_src(r):
    c, s = np.asarray(r["c"]), np.asarray(r["s"])
    lo_r, hi_r = c - s / 2, c + s / 2
    # Roblox (x, z, -y) -> Source: back-convert the two corners
    pts = np.array([[x, y, z] for x in (lo_r[0], hi_r[0]) for y in (lo_r[1], hi_r[1]) for z in (lo_r[2], hi_r[2])])
    return np.stack([pts[:, 0], -pts[:, 2], pts[:, 1]], 1) / SPU


def test_sealed_room_gets_blockers_outside_only():
    rep = {}
    bl = light_blockers(room(), OPAQUE, report=rep)
    assert bl
    for r in bl:
        P = _corners_src(r)
        lo, hi = P.min(0), P.max(0)
        # never overlaps the room's inside
        assert np.any(hi <= np.array([0, 0, 0]) + 1e-6) or np.any(lo >= np.array([512, 512, 256]) - 1e-6)
    # they wrap the room: something above, below and on each side
    allp = np.concatenate([_corners_src(r) for r in bl])
    assert allp[:, 2].max() > 256 + 16 + 32 and allp[:, 2].min() < -16 - 32
    assert allp[:, 0].max() > 512 + 16 + 32 and allp[:, 0].min() < -16 - 32


def test_unsealed_room_gets_none():
    rep = {}
    assert light_blockers(room(missing=(0, 1)), OPAQUE, report=rep) == []
    assert rep.get("light blockers: map not sealed, none added")


def test_sky_ceiling_keeps_the_view_up_clear():
    sun = to_roblox([0.0, 0.0, 1.0])             # straight up
    bl = light_blockers(room(ceiling="tools/toolsskybox"), OPAQUE, sun_dir=sun)
    # nothing in the column above the open ceiling (only the side walls' skins may rise past it)
    for r in bl:
        P = _corners_src(r)
        lo, hi = P.min(0), P.max(0)
        above = hi[2] > 256 + 16 + 1
        over_room = np.all(lo[:2] < np.array([512, 512]) - 1) and np.all(hi[:2] > np.array([0, 0]) + 1)
        assert not (above and over_room)
