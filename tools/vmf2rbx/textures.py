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


def map_linear(part: PartSpec, face: str, gs, s_off: float, gt, t_off: float, material: str,
               tex_w: int, tex_h: int):
    """Roblox Texture parameters for a face whose texel coordinates are linear in world
    (Roblox, studs) position: s = gs . P + s_off, t = gt . P + t_off (t runs down the image).

    Returns a FaceTex, or None if the mapping isn't a multiple of 90 degrees."""
    O, Ut, Vt, _ = face_frame(part.cls, part.size, face)
    Rm = part.rot.T
    Ow, Uw, Vw = part.center + Rm @ O, Rm @ Ut, Rm @ Vt
    s0, sU, sV = Ow @ gs + s_off, Uw @ gs, Vw @ gs     # texel = c0 + kU*u + kV*v
    t0, tU, tV = Ow @ gt + t_off, Uw @ gt, Vw @ gt
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
    return FaceTex(face, material, bool(swap), bool(fx), bool(fy), spt_u, spt_v, off_u, off_v)


def map_face(part: PartSpec, face: str, side: Side, tex_w: int, tex_h: int):
    """Roblox Texture parameters reproducing Hammer's alignment on this face.

    Returns a FaceTex, or None if the mapping isn't a multiple of 90 degrees."""
    gs = to_roblox(side.u.vec) / (SPU * side.u.scale)     # texels per stud (as a Roblox-space gradient)
    gt = to_roblox(side.v.vec) / (SPU * side.v.scale)
    return map_linear(part, face, gs, side.u.offset, gt, side.v.offset, side.material, tex_w, tex_h)


BAKE_MAX_PX = 512      # longest side of a baked face image
BAKE_MIN_PX = 8


class FaceBake:
    """Hammer's mapping of one face, sampled on a grid that covers the face once.

    `sample(arr)` resamples any map of the material (colour, roughness, ...) onto the face.
    Texel coordinates are in the material's original texture size, so a replacement image of a
    different resolution (Portal RTX) lands in the same place. `normal_matrix` takes a normal
    map's tangential components (image right, image down) from the source image's frame to the
    baked image's frame: the transpose of the rotation/mirror part of d(s,t)/d(u,v) (its polar
    factor). Stretch is left out, as the importer's 90-degree path leaves it out: on a wedge
    slope the V coordinate is measured as height, so its stretch isn't physical anyway."""

    def __init__(self, part: PartSpec, face: str, side: Side, size: tuple[int, int]):
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
        self.nx, self.ny = max(1, int(round(nx * scale))), max(1, int(round(ny * scale)))
        tw, th = size
        u = (np.arange(self.nx) + 0.5) / self.nx * W
        v = (np.arange(self.ny) + 0.5) / self.ny * H
        uu, vv = np.meshgrid(u, v)
        # position inside the tile as a 0..1 fraction of the original texture
        self.fx = np.mod(s0 + sU * uu + sV * vv, tw) / tw
        self.fy = np.mod(t0 + tU * uu + tV * vv, th) / th
        J = np.array([[sU, sV], [tU, tV]])
        Us, _, Vts = np.linalg.svd(J)
        self.normal_matrix = (Us @ Vts).T
        self.tex = FaceTex(face, side.material, False, False, False, float(W), float(H), 0.0, 0.0)

    def sample(self, arr: np.ndarray) -> np.ndarray:
        h, w = arr.shape[:2]
        x = np.minimum((self.fx * w).astype(np.int64), w - 1)
        y = np.minimum((self.fy * h).astype(np.int64), h - 1)
        return arr[y, x]

    def sample_normal(self, arr: np.ndarray) -> np.ndarray:
        """`arr`: uint8 (H, W, 2) of (right, down) components encoded as (v + 1) / 2 * 255."""
        n = self.sample(arr).astype(float) / 255 * 2 - 1
        out = n @ self.normal_matrix.T
        ln = np.linalg.norm(out, axis=-1, keepdims=True)
        out = np.where(ln > 1, out / np.maximum(ln, 1e-9), out)
        return np.round((out + 1) / 2 * 255).astype(np.uint8)


def bake_face(part: PartSpec, face: str, side: Side, image, key: str, size=None):
    """Fallback for faces whose texture runs at an angle Roblox's tiling can't express.

    Renders Hammer's mapping of `image` onto exactly this face into a new image that covers the
    face once (StudsPerTile = face size, offset 0). Works for any angle, at the cost of one
    image per face. Returns (FaceTex, PIL image)."""
    from PIL import Image
    fb = FaceBake(part, face, side, size or image.size)
    fb.tex.image = key
    return fb.tex, Image.fromarray(fb.sample(np.asarray(image.convert("RGBA"))), "RGBA")


def texture_part(brush: Brush, part: PartSpec, sizes: dict[str, tuple[int, int]], images=None, baked=None,
                 pbr=None):
    """FaceTex for every visible face of a part. Returns (faces, skipped_reasons).

    With `images` (material -> PIL image) and `baked` (a dict to collect new images in), faces
    whose mapping isn't a multiple of 90 degrees get a baked image instead of being skipped.
    `pbr` (material -> {"n"|"r"|"m": uint8 array}) bakes those maps too, as key + "#n" etc.
    `sizes` holds each material's original texture size, which defines its texel space."""
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
            fb = FaceBake(part, face, side, sizes[side.material])
            fb.tex.image = key
            from PIL import Image
            baked[key] = Image.fromarray(fb.sample(np.asarray(images[side.material].convert("RGBA"))), "RGBA")
            for kind, arr in ((pbr or {}).get(side.material) or {}).items():
                if isinstance(arr, np.ndarray):
                    baked[f"{key}#{kind}"] = fb.sample_normal(arr) if kind == "n" else fb.sample(arr)
            faces.append(fb.tex)
        elif ft is None:
            skipped.append(("texture not at 90 degrees to part" + (" (decomposed piece)" if part.piece else ""), side.material))
        else:
            faces.append(ft)
    return faces, skipped
