"""Lights from the compiled BSP.

VRAD bakes texture lights (materials listed in lights.rad) into the BSP's world-light lump as
`emit_surface` lights, one per lighting patch, with the intensity it computed. So the BSP
carries what lights.rad would have said, even when lights.rad isn't shipped. Point and spot
lights are read from the same lump, using VRAD's own intensity and attenuation instead of the
(sometimes odd) values in a decompiled VMF.

The lump's texinfo field is not reliable for surfaces (it's 0 in Portal's maps), so each patch
is matched back to a brush face by position and normal.

Brightness/range conversion uses tunable constants. They are not derived: they were set by eye
in Studio on testchmb_a_00 with the importer's indoor lighting (no sun or sky light, dim
ambient), aiming for Portal's overall level of light rather than its exact numbers.
"""
from __future__ import annotations

import math
import struct
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .brushes import Brush
from .coords import SPU, to_roblox

LUMP_WORLDLIGHTS = 15
LUMP_WORLDLIGHTS_HDR = 54
EMIT = ["surface", "point", "spot", "skylight", "quakelight", "skyambient"]


@dataclass
class LightTuning:
    cutoff: float = 0.05                 # illuminance treated as "faded out" (lightmap units)
    surface_brightness_per_radiance: float = 0.03
    point_brightness_divisor: float = 25000.0
    max_range: float = 120.0             # Roblox's Range limit (was 60 until 2026)
    reach_margin: float = 1.25           # a light may reach this far past the first wall it faces
    saturation: float = 0.5              # 1 = VRAD's colour as is; Source's tonemapping washes it out


def _tint(rgb, saturation: float) -> list[int]:
    """0..1 colour -> 0..255, pulled toward white by (1 - saturation)."""
    return [round(255 * (1 - saturation + saturation * float(c))) for c in rgb]


@dataclass
class WorldLight:
    origin: np.ndarray
    intensity: np.ndarray
    normal: np.ndarray
    type: str
    style: int
    stopdot: float
    stopdot2: float
    exponent: float
    radius: float
    attn: tuple
    flags: int = 0


def bsp_lump(bsp_bytes: bytes, i: int) -> tuple[bytes, int]:
    """(data, lump version) of lump i. Later BSPs (Portal 2, CS:GO) may LZMA-compress a lump:
    it then starts with "LZMA", u32 uncompressed size, u32 compressed size, 5 property bytes."""
    o, ln, ver, _ = struct.unpack_from("<iii4s", bsp_bytes, 8 + 16 * i)
    data = bsp_bytes[o:o + ln]
    if data[:4] == b"LZMA":
        import lzma
        size, csize = struct.unpack_from("<II", data, 4)
        props = data[12:17]
        d = props[0]
        lc, lp, pb = d % 9, (d // 9) % 5, d // 45
        dict_size = struct.unpack_from("<I", props, 1)[0]
        filt = [{"id": lzma.FILTER_LZMA1, "lc": lc, "lp": lp, "pb": pb, "dict_size": dict_size}]
        data = lzma.LZMADecompressor(lzma.FORMAT_RAW, filters=filt).decompress(data[17:17 + csize], size)
    return data, ver


def read_world_lights(bsp_bytes: bytes) -> list[WorldLight]:
    """dworldlight_t records. Lump version 0 (Portal, HL2): 88 bytes (84 in old BSPs without
    flags/texinfo/owner... the last int). Version 1 (Portal 2, CS:GO): 100 bytes, with a
    shadow_cast_offset vector after the normal."""
    data, ver = bsp_lump(bsp_bytes, LUMP_WORLDLIGHTS_HDR)
    if not data:
        data, ver = bsp_lump(bsp_bytes, LUMP_WORLDLIGHTS)
    if ver >= 1:
        size, fmt = 100, "<3f3f3f3fiiifffffffiii"
    else:
        size = 88 if len(data) % 88 == 0 else 84
        fmt = "<3f3f3fiiifffffff" + ("iii" if size == 88 else "ii")
    out = []
    for i in range(len(data) // size):
        f = struct.unpack_from(fmt, data, size * i)
        if ver >= 1:
            f = f[:9] + f[12:]                     # drop shadow_cast_offset
        typ = EMIT[f[10]] if f[10] < len(EMIT) else str(f[10])
        out.append(WorldLight(np.array(f[0:3]), np.array(f[3:6]), np.array(f[6:9]), typ, f[11],
                              f[12], f[13], f[14], f[15], tuple(f[16:19]), f[19] if size >= 88 else 0))
    return out


def _face_polys(brushes: list[Brush]):
    polys = []
    for b in brushes:
        for s in b.visible_sides:
            fv = np.array([v for v in b.verts if abs(s.normal @ v - s.dist) < 1e-2])
            if len(fv) >= 3:
                polys.append((b, s, fv))
    return polys


def _in_poly(p, n, fv, tol=0.5):
    c = fv.mean(0)
    a = fv[0] - c; a /= np.linalg.norm(a)
    b = np.cross(n, a)
    P = np.array([((v - c) @ a, (v - c) @ b) for v in fv])
    P = P[np.argsort(np.arctan2(P[:, 1], P[:, 0]))]
    q = np.array([(p - c) @ a, (p - c) @ b])
    for i in range(len(P)):
        e = P[(i + 1) % len(P)] - P[i]
        w = q - P[i]
        if e[0] * w[1] - e[1] * w[0] < -tol:
            return False
    return True


def _area(fv, n):
    c = fv.mean(0)
    a = fv[0] - c; a /= np.linalg.norm(a)
    b = np.cross(n, a)
    P = np.array([((v - c) @ a, (v - c) @ b) for v in fv])
    P = P[np.argsort(np.arctan2(P[:, 1], P[:, 0]))]
    return 0.5 * abs(sum(P[i, 0] * P[(i + 1) % len(P), 1] - P[(i + 1) % len(P), 0] * P[i, 1] for i in range(len(P))))


def solid_planes(brushes: list[Brush]) -> list[tuple[np.ndarray, np.ndarray]]:
    """(normals, dists) per brush that blocks light: world and func_detail, not tool-only."""
    out = []
    for b in brushes:
        if b.owner in ("world", "func_detail") and not b.is_tool_only:
            out.append((np.array([sd.normal for sd in b.sides]), np.array([sd.dist for sd in b.sides])))
    return out


def ray_reach(origin, direction, solids, start=0.5, far=1e5) -> float:
    """Distance (Source units) from origin along direction to the first solid brush, ignoring
    the first `start` units (so a light on a brush face doesn't hit its own brush)."""
    o, d = np.asarray(origin, float), np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    best = far
    for N, D in solids:
        nd, no = N @ d, N @ o
        par = np.abs(nd) < 1e-9
        if np.any(par & (no > D + 1e-6)):
            continue                           # parallel to a plane and outside it
        t = (D[~par] - no[~par]) / nd[~par]
        ent, ext = t[nd[~par] < 0], t[nd[~par] > 0]
        t0 = ent.max() if len(ent) else -np.inf
        t1 = ext.min() if len(ext) else np.inf
        if t0 <= t1 and t1 > start:
            hit = max(t0, start)
            if hit < best:
                best = hit
    return best


AXES = np.vstack([np.eye(3), -np.eye(3)])


def cone_dirs(axis, half_angle, ring=8) -> list[np.ndarray]:
    """The axis plus a ring of directions at 70% of the cone's half-angle: one ray down the
    middle of a spot is easily stopped by a window sill or rail the cone mostly clears."""
    a = np.asarray(axis, float); a /= np.linalg.norm(a)
    t1 = np.cross(a, [0.0, 0.0, 1.0] if abs(a[2]) < 0.9 else [1.0, 0.0, 0.0]); t1 /= np.linalg.norm(t1)
    t2 = np.cross(a, t1)
    th = 0.7 * half_angle
    return [a] + [math.cos(th) * a + math.sin(th) * (math.cos(k) * t1 + math.sin(k) * t2)
                  for k in np.linspace(0, 2 * math.pi, ring, endpoint=False)]


@dataclass
class LightReport:
    texlights: dict = field(default_factory=dict)       # material -> {color, radiance}
    surface: list = field(default_factory=list)         # per emitting face
    points: list = field(default_factory=list)          # point and spot lights
    unmatched_patches: int = 0
    dark_lights: int = 0                                # point/spot lights too dim (or negative) to use
    sun: dict | None = None                             # light_environment: direction toward the sun etc.


def fade_distance(attn, lum: float, cutoff: float) -> float | None:
    """Distance (Source units) where lum / (c + l d + q d^2) drops to `cutoff`, or None when
    the light never reaches the cutoff at all (zero or negative brightness: Source allows
    negative "darkness" lights; or a large constant term)."""
    c, l, q = attn
    k = c - lum / cutoff                 # q d^2 + l d + k = 0
    if lum <= 0 or k >= 0:
        return None
    if q > 0:
        return (-l + math.sqrt(l * l - 4 * q * k)) / (2 * q)    # k < 0: discriminant > l^2
    if l > 0:
        return -k / l
    return 1e9                           # no falloff


def build_lights(world_lights: list[WorldLight], brushes: list[Brush], tuning=LightTuning(),
                 keep_brush=lambda b: True, keep_point=lambda p: True) -> LightReport:
    rep = LightReport()
    polys = _face_polys(brushes)
    solids = solid_planes(brushes)

    def reach_cap(origin, dirs) -> float:
        """Range limit: just past the farthest of the first walls hit along `dirs`. The cutoff
        formula alone lets strong lights run far past the room they're in (or, capped at the
        old 60-stud limit, stop short of the far wall)."""
        r = max(ray_reach(origin, d, solids) for d in dirs)
        return min(tuning.max_range, r * SPU * tuning.reach_margin)
    hits = defaultdict(list)
    for wl in world_lights:
        if wl.type != "surface":
            continue
        best = None
        for idx, (b, s, fv) in enumerate(polys):
            if s.normal @ wl.normal < 0.99:
                continue
            d = abs(s.normal @ wl.origin - s.dist)
            if d > 4 or not _in_poly(wl.origin, s.normal, fv):
                continue
            if best is None or d < best[0]:
                best = (d, idx)
        if best:
            hits[best[1]].append(wl)
        else:
            rep.unmatched_patches += 1
    by_mat = defaultdict(lambda: [np.zeros(3), 0.0])
    for idx, ws in hits.items():
        b, s, fv = polys[idx]
        by_mat[s.material][0] += np.sum([w.intensity for w in ws], 0)
        by_mat[s.material][1] += _area(fv, s.normal)
    for m, (I, A) in by_mat.items():
        rad = I / A if A > 0 else I
        if rad.max() <= 0:
            rad = np.full(3, 1e-9)
        rep.texlights[m] = dict(radiance=rad.tolist(),
                                color=[c / 255 for c in _tint(rad / rad.max(), tuning.saturation)])
    for idx, ws in hits.items():
        b, s, fv = polys[idx]
        if not keep_brush(b):
            continue
        I = np.sum([w.intensity for w in ws], 0)
        tl = rep.texlights[s.material]
        rng = min(reach_cap(fv.mean(0), [s.normal]), math.sqrt(max(0.0, float(I.mean())) / tuning.cutoff) * SPU)
        rep.surface.append(dict(brush=b.id, material=s.material, normal=to_roblox(s.normal).tolist(),
                                color=[round(255 * c) for c in tl["color"]],
                                brightness=tuning.surface_brightness_per_radiance * float(np.mean(tl["radiance"])),
                                range=rng))
    sky = {wl.type: wl for wl in world_lights if wl.type in ("skylight", "skyambient")}
    if "skylight" in sky:
        wl = sky["skylight"]
        I = wl.intensity
        amb = sky["skyambient"].intensity if "skyambient" in sky else np.zeros(3)
        rep.sun = dict(direction=to_roblox(-wl.normal).tolist(),          # toward the sun
                       color=_tint(I / max(I.max(), 1e-9), tuning.saturation),
                       intensity=float(I.mean()),
                       ambient_color=_tint(amb / max(amb.max(), 1e-9), tuning.saturation),
                       ambient=float(amb.mean()))
    for wl in world_lights:
        if wl.type not in ("point", "spot"):
            continue
        lum = float(wl.intensity.mean())
        d = fade_distance(wl.attn, lum, tuning.cutoff)
        if d is None or wl.intensity.max() <= 0:
            rep.dark_lights += 1
            continue
        rec = dict(kind=wl.type, position=(to_roblox(wl.origin) * SPU).tolist(), direction=to_roblox(wl.normal).tolist(),
                   color=_tint(wl.intensity / wl.intensity.max(), tuning.saturation),
                   brightness=lum / tuning.point_brightness_divisor,
                   range=min(reach_cap(wl.origin, cone_dirs(wl.normal, math.acos(max(-1.0, min(1.0, wl.stopdot2))))
                                       if wl.type == "spot" else AXES), max(0.0, d) * SPU),
                   style=wl.style)
        if wl.type == "spot":
            rec["angle"] = 2 * math.degrees(math.acos(max(-1.0, min(1.0, wl.stopdot2))))
        if keep_point(rec):
            rep.points.append(rec)
    return rep
