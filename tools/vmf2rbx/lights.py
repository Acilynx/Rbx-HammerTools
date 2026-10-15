"""Lights from the compiled BSP.

VRAD bakes texture lights (materials listed in lights.rad) into the BSP's world-light lump as
`emit_surface` lights, one per lighting patch, with the intensity it computed. So the BSP
carries what lights.rad would have said, even when lights.rad isn't shipped. Point and spot
lights are read from the same lump, using VRAD's own intensity and attenuation instead of the
(sometimes odd) values in a decompiled VMF.

The lump's texinfo field is not reliable for surfaces (it's 0 in Portal's maps), so each patch
is matched back to a brush face by position and normal.

Brightness/range conversion uses tunable constants; they are a starting point, not derived.
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
    surface_brightness_per_radiance: float = 0.3
    point_brightness_divisor: float = 2500.0
    max_range: float = 60.0              # Roblox light range cap (studs)


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


def read_world_lights(bsp_bytes: bytes) -> list[WorldLight]:
    def lump(i):
        o, ln, _, _ = struct.unpack_from("<iii4s", bsp_bytes, 8 + 16 * i)
        return bsp_bytes[o:o + ln]
    data = lump(LUMP_WORLDLIGHTS_HDR) or lump(LUMP_WORLDLIGHTS)
    size = 88 if len(data) % 88 == 0 else 84
    fmt = "<3f3f3fiiifffffff" + ("iii" if size == 88 else "ii")
    out = []
    for i in range(len(data) // size):
        f = struct.unpack_from(fmt, data, size * i)
        typ = EMIT[f[10]] if f[10] < len(EMIT) else str(f[10])
        out.append(WorldLight(np.array(f[0:3]), np.array(f[3:6]), np.array(f[6:9]), typ, f[11],
                              f[12], f[13], f[14], f[15], tuple(f[16:19]), f[19] if size == 88 else 0))
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


@dataclass
class LightReport:
    texlights: dict = field(default_factory=dict)       # material -> {color, radiance}
    surface: list = field(default_factory=list)         # per emitting face
    points: list = field(default_factory=list)          # point and spot lights
    unmatched_patches: int = 0


def build_lights(world_lights: list[WorldLight], brushes: list[Brush], tuning=LightTuning(),
                 keep_brush=lambda b: True, keep_point=lambda p: True) -> LightReport:
    rep = LightReport()
    polys = _face_polys(brushes)
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
        rad = I / A
        rep.texlights[m] = dict(radiance=rad.tolist(), color=(rad / rad.max()).tolist())
    for idx, ws in hits.items():
        b, s, fv = polys[idx]
        if not keep_brush(b):
            continue
        I = np.sum([w.intensity for w in ws], 0)
        tl = rep.texlights[s.material]
        rng = min(tuning.max_range, math.sqrt(I.mean() / tuning.cutoff) * SPU)
        rep.surface.append(dict(brush=b.id, material=s.material, normal=to_roblox(s.normal).tolist(),
                                color=[round(255 * c) for c in tl["color"]],
                                brightness=tuning.surface_brightness_per_radiance * float(np.mean(tl["radiance"])),
                                range=rng))
    for wl in world_lights:
        if wl.type not in ("point", "spot"):
            continue
        lum = float(wl.intensity.mean())
        c, l, q = wl.attn
        if q > 0:
            d = (-l + math.sqrt(l * l - 4 * q * (c - lum / tuning.cutoff))) / (2 * q)
        elif l > 0:
            d = (lum / tuning.cutoff - c) / l
        else:
            d = 1e9
        rec = dict(kind=wl.type, position=(to_roblox(wl.origin) * SPU).tolist(), direction=to_roblox(wl.normal).tolist(),
                   color=[round(255 * x / wl.intensity.max()) for x in wl.intensity],
                   brightness=lum / tuning.point_brightness_divisor, range=min(tuning.max_range, max(0.0, d) * SPU),
                   style=wl.style)
        if wl.type == "spot":
            rec["angle"] = 2 * math.degrees(math.acos(max(-1.0, min(1.0, wl.stopdot2))))
        if keep_point(rec):
            rep.points.append(rec)
    return rep
