"""Texture mapping tests against an independent oracle.

ROBLOX_FACES is Roblox's per-face texture layout written out *literally* from Studio
measurements (a test image with marked u=0/v=0 edges on a 3x5x7 part, and on an 8x8x8 cube),
not computed. The code under test derives the same layout with cross products; if either is
wrong they disagree. (An earlier version shared one function between code and check, and a
precedence bug - n*h@|n| - slipped through because the check had the same bug. Hence this.)

Per face: U axis, V axis (part-local), and the origin corner as signs of the half-size.
"Back" was not imaged directly; it follows the same pattern as the five measured faces.
"""
import numpy as np
import pytest

from helpers import box, chamfered_box, diagonal_strip, ngon_prism, right_wedge, rotated_box
from vmf2rbx.brushes import TexAxis
from vmf2rbx.coords import point_to_source
from vmf2rbx.geometry import brush_to_parts
from vmf2rbx.textures import FACE_NORMALS, face_frame, find_side, map_face

ROBLOX_FACES = {
    #          U            V            origin corner (sign of hx, hy, hz)
    "Right":  ((0, 0, -1), (0, -1, 0), (+1, +1, +1)),
    "Left":   ((0, 0, +1), (0, -1, 0), (-1, +1, -1)),
    "Front":  ((-1, 0, 0), (0, -1, 0), (+1, +1, -1)),
    "Back":   ((+1, 0, 0), (0, -1, 0), (-1, +1, +1)),
    "Top":    ((-1, 0, 0), (0, 0, -1), (+1, +1, +1)),
    "Bottom": ((+1, 0, 0), (0, 0, -1), (-1, -1, +1)),
}


def oracle_frame(cls, size, face):
    h = np.asarray(size, float) / 2
    if cls == "WedgePart" and face == "Front":
        # measured: U = -X, V down the slope measured as height, origin at the top-back edge
        hx, hy, hz = h
        return np.array([hx, hy, hz]), np.array([-1.0, 0, 0]), np.array([0, -1.0, -hz / hy])
    U, V, corner = ROBLOX_FACES[face]
    return np.array(corner) * h, np.array(U, float), np.array(V, float)


@pytest.mark.parametrize("size", [(8, 8, 8), (3, 5, 7), (13.75, 1.25, 10), (0.3125, 17.5, 2.5)])
@pytest.mark.parametrize("face", list(FACE_NORMALS))
def test_face_frame_matches_measured_layout(size, face):
    O, U, V, _ = face_frame("Part", size, face)
    Oo, Uo, Vo = oracle_frame("Part", size, face)
    assert np.allclose(U, Uo) and np.allclose(V, Vo)
    assert np.allclose(O, Oo), f"origin {O} vs measured {Oo}"


def test_wedge_slope_frame():
    O, U, V, _ = face_frame("WedgePart", (3, 4, 16), "Front")
    Oo, Uo, Vo = oracle_frame("WedgePart", (3, 4, 16), "Front")
    assert np.allclose(O, Oo) and np.allclose(U, Uo) and np.allclose(V, Vo)


def roblox_texel(part, ft, u, v, W, H):
    """Which source texel Roblox draws at face coords (u, v), using only the oracle frame and
    the FaceTex parameters (image variant + tiling + offset)."""
    Nx, Ny = (H, W) if ft.swap else (W, H)
    xp = ((u + ft.offset_u) / ft.studs_per_tile_u * Nx) % Nx
    yp = ((v + ft.offset_v) / ft.studs_per_tile_v * Ny) % Ny
    if ft.flip_x:
        xp = Nx - xp
    if ft.flip_y:
        yp = Ny - yp
    return (yp, xp) if ft.swap else (xp, yp)


def hammer_texel(side, P):
    return (P @ side.u.vec / side.u.scale + side.u.offset, P @ side.v.vec / side.v.scale + side.v.offset)


def random_axes(rng, yaw_deg=0.0):
    """Random 90-degree texture alignment on any face: u and v along ± the face's edge
    directions (for a yawed brush, its own edges), swapped or not."""
    yaw = np.radians(yaw_deg)

    def axes(n):
        n = np.asarray(n, float)
        t1 = np.cross([0, 0, 1.0], n) if abs(n[2]) < 0.99 else np.array([np.cos(yaw), np.sin(yaw), 0])
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        u, v = (t1, t2) if rng.random() < 0.5 else (t2, t1)
        return u * rng.choice([-1, 1]), v * rng.choice([-1, 1])
    return axes


def all_test_brushes(rng):
    kw = lambda yaw=0.0: dict(axes=random_axes(rng, yaw), scale=float(rng.choice([0.125, 0.25, 0.5, 1.0])),
                              offsets=(float(rng.integers(0, 512)), float(rng.integers(0, 512))))
    return [box(1, [-40, 8, 0], [88, 72, 96], **kw()),
            box(2, [0, 0, 0], [16, 224, 4], **kw()),
            rotated_box(3, [10, 20, 30], [96, 32, 64], 45, **kw(45.0)),
            chamfered_box(4, [0, 0, 0], [192, 16, 128], 16, **kw()),
            diagonal_strip(5, **kw(45.0)),      # caps textured along the wall, as a mapper would
            right_wedge(6, **kw()),
            ngon_prism(7, **kw())]


@pytest.mark.parametrize("seed", range(6))
def test_roblox_draws_the_same_texel_as_hammer(seed):
    rng = np.random.default_rng(seed)
    W, H = 512, 256
    checked = 0
    for brush in all_test_brushes(rng):
        for part in brush_to_parts(brush):
            for face in FACE_NORMALS:
                side = find_side(brush, part, face)
                if side is None:
                    continue
                ft = map_face(part, face, side, W, H)
                assert ft is not None, f"brush {brush.id} {face}: mapping should be 90-degree"
                O, Ut, Vt = oracle_frame(part.cls, part.size, face)
                hs = part.size / 2
                lu = np.abs(Ut) @ hs * 2
                lv = hs[1] * 2 if (part.cls == "WedgePart" and face == "Front") else np.abs(Vt) @ hs * 2
                for _ in range(40):
                    u, v = rng.uniform(0, lu), rng.uniform(0, lv)
                    pw = part.center + part.rot.T @ (O + u * Ut + v * Vt)
                    s, t = hammer_texel(side, point_to_source(pw))
                    rs, rt = roblox_texel(part, ft, u, v, W, H)
                    ds, dt = (rs - s) % W, (rt - t) % H
                    assert min(ds, W - ds) < 1e-3 and min(dt, H - dt) < 1e-3, \
                        f"brush {brush.id} piece {part.piece} {face}: texel ({rs:.2f},{rt:.2f}) vs Hammer ({s % W:.2f},{t % H:.2f})"
                    checked += 1
    assert checked > 500
