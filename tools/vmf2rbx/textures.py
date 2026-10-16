"""Hammer texture alignment -> Roblox Texture parameters.

Roblox's per-face texture layout, measured in Studio (see docs/texture-rules.md):

* side faces (local +-X, +-Z): reference R = local Y;  top/bottom (+-Y): R = local Z
* U = R x n,  V = -R,  tiling starts at the face corner where u and v are smallest
* image x = (u + OffsetStudsU) / StudsPerTileU,  image y likewise with V
* WedgePart slope ("Front"): U = -X, V runs down the slope but is measured as *height*
  (the slope is stretched), starting at the top-back edge
* Texture.Rotation and a negative UVScale can't express Hammer's rotations/mirroring, so
  each face instead uses one of 8 orientations of the source image (transpose, flip x, flip y)

Hammer: texel s = (P . uaxis) / uscale + uoffset, P in units; t likewise (t runs down the image).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .brushes import Brush, Side
from .coords import SPU, to_roblox
from .geometry import PartSpec

X, Y, Z = np.eye(3)
FACE_NORMALS = {"Right": X, "Left": -X, "Top": Y, "Bottom": -Y, "Back": Z, "Front": -Z}


@dataclass
class FaceTex:
    face: str
    material: str
    swap: bool        # transpose the source image
    flip_x: bool      # then mirror horizontally
    flip_y: bool      # then mirror vertically
    studs_per_tile_u: float
    studs_per_tile_v: float
    offset_u: float
    offset_v: float
    image: str | None = None   # image key when it isn't the material's own image (baked faces)


def face_frame(cls: str, size, face: str):
    """Roblox texture frame of one face in part-local space.

    Returns (origin, u_tangent, v_tangent, normal) where a point on the face is
    origin + u * u_tangent + v * v_tangent for texture coordinates (u, v) in studs,
    or None if the face doesn't exist (WedgePart has no top)."""
    h = np.asarray(size, float) / 2
    if cls == "WedgePart" and face == "Front":
        hx, hy, hz = h
        n = np.array([0.0, 2 * hz, -2 * hy]) / np.hypot(2 * hy, 2 * hz)
        return np.array([hx, hy, hz]), -X, np.array([0.0, -1.0, -hz / hy]), n
    if cls == "WedgePart" and face == "Top":
        return None
    n = FACE_NORMALS[face]
    R = Y if abs(n[1]) < 0.5 else Z
    U = np.cross(R, n)
    V = -R
    origin = n * (h @ np.abs(n)) - U * (h @ np.abs(U)) - V * (h @ np.abs(V))
    return origin, U, V, n


def find_side(brush: Brush, part: PartSpec, face: str, plane_tol=2e-3) -> Side | None:
    """The brush side lying on this part face: same outward normal, same plane."""
    fr = face_frame(part.cls, part.size, face)
    if fr is None:
        return None
    O, _, _, n = fr
    Rm = part.rot.T
    Ow, nw = part.center + Rm @ O, Rm @ n
    for s in brush.sides:
        if to_roblox(s.normal) @ nw > 0.999 and abs(to_roblox(s.point) * SPU @ nw - Ow @ nw) < plane_tol:
            return s
    return None


def map_face(part: PartSpec, face: str, side: Side, tex_w: int, tex_h: int):
    """Roblox Texture parameters reproducing Hammer's alignment on this face.

    Returns a FaceTex, or None if the mapping isn't a multiple of 90 degrees."""
    O, Ut, Vt, _ = face_frame(part.cls, part.size, face)
    Rm = part.rot.T
    Ow, Uw, Vw = part.center + Rm @ O, Rm @ Ut, Rm @ Vt

    def linear(ax):
        g = to_roblox(ax.vec) / (SPU * ax.scale)     # texels per stud (as a Roblox-space gradient)
        return Ow @ g + ax.offset, Uw @ g, Vw @ g    # texel = c0 + kU*u + kV*v
    s0, sU, sV = linear(side.u)
    t0, tU, tV = linear(side.v)
    eps = 1e-6 * max(abs(sU), abs(sV), abs(tU), abs(tV))
    if abs(sV) < eps and abs(tU) < eps:
        swap, ax_u, ax_v = False, (sU, s0, tex_w), (tV, t0, tex_h)
    elif abs(sU) < eps and abs(tV) < eps:
        swap, ax_u, ax_v = True, (tU, t0, tex_h), (sV, s0, tex_w)
    else:
        return None

    def axis(k, c0, N):
        flip = k < 0
        k = abs(k)
        c0 = (N - c0) if flip else c0
        spt = N / k
        return flip, spt, (c0 / k) % spt
    fx, spt_u, off_u = axis(*ax_u)
    fy, spt_v, off_v = axis(*ax_v)
    return FaceTex(face, side.material, bool(swap), bool(fx), bool(fy), spt_u, spt_v, off_u, off_v)


BAKE_MAX_PX = 512      # longest side of a baked face image
BAKE_MIN_PX = 8


def bake_face(part: PartSpec, face: str, side: Side, image, key: str):
    """Fallback for faces whose texture runs at an angle Roblox's tiling can't express.

    Renders Hammer's mapping of `image` onto exactly this face into a new image that covers the
    face once (StudsPerTile = face size, offset 0). Works for any angle, at the cost of one
    image per face. Returns (FaceTex, PIL image)."""
    from PIL import Image
    O, Ut, Vt, _ = face_frame(part.cls, part.size, face)
    h = np.asarray(part.size, float) / 2
    if part.cls == "WedgePart" and face == "Front":
        W, H = 2 * h[0], 2 * h[1]                      # V is measured as height on the slope
    else:
        W, H = 2 * (h @ np.abs(Ut)), 2 * (h @ np.abs(Vt))
    Rm = part.rot.T
    Ow, Uw, Vw = part.center + Rm @ O, Rm @ Ut, Rm @ Vt
    gs = to_roblox(side.u.vec) / (SPU * side.u.scale)
    gt = to_roblox(side.v.vec) / (SPU * side.v.scale)
    s0, sU, sV = Ow @ gs + side.u.offset, Uw @ gs, Vw @ gs
    t0, tU, tV = Ow @ gt + side.v.offset, Uw @ gt, Vw @ gt
    # keep the source's texel density along each face axis, within limits
    dens_u, dens_v = np.hypot(sU, tU), np.hypot(sV, tV)
    nx, ny = max(BAKE_MIN_PX, int(np.ceil(W * dens_u))), max(BAKE_MIN_PX, int(np.ceil(H * dens_v)))
    scale = min(1.0, BAKE_MAX_PX / max(nx, ny))
    nx, ny = max(1, int(round(nx * scale))), max(1, int(round(ny * scale)))
    src = np.asarray(image.convert("RGBA"))
    th, tw = src.shape[:2]
    u = (np.arange(nx) + 0.5) / nx * W
    v = (np.arange(ny) + 0.5) / ny * H
    uu, vv = np.meshgrid(u, v)
    sx = np.floor(s0 + sU * uu + sV * vv).astype(np.int64) % tw
    ty = np.floor(t0 + tU * uu + tV * vv).astype(np.int64) % th
    out = Image.fromarray(src[ty, sx], "RGBA")
    return FaceTex(face, side.material, False, False, False, float(W), float(H), 0.0, 0.0, image=key), out


def texture_part(brush: Brush, part: PartSpec, sizes: dict[str, tuple[int, int]], images=None, baked=None):
    """FaceTex for every visible face of a part. Returns (faces, skipped_reasons).

    With `images` (material -> PIL image) and `baked` (a dict to collect new images in), faces
    whose mapping isn't a multiple of 90 degrees get a baked image instead of being skipped."""
    faces, skipped = [], []
    if part.cls == "Placeholder":
        return faces, skipped
    for face in FACE_NORMALS:
        side = find_side(brush, part, face)
        if side is None:
            continue
        if side.is_tool:
            continue
        if side.material not in sizes:
            skipped.append(("no texture", side.material))
            continue
        ft = map_face(part, face, side, *sizes[side.material])
        if ft is None and images is not None and baked is not None and side.material in images:
            # On a decomposed brush this is almost always a cap whose texture runs at an angle
            # to every geometrically valid slicing direction (thin diagonal strips, slanted
            # triangles, rotated slabs in a doorway).
            key = f"bake:{brush.id}:{part.piece}:{face}"
            ft, baked[key] = bake_face(part, face, side, images[side.material], key)
            faces.append(ft)
        elif ft is None:
            skipped.append(("texture not at 90 degrees to part" + (" (decomposed piece)" if part.piece else ""), side.material))
        else:
            faces.append(ft)
    return faces, skipped
