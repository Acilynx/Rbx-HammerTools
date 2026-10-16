"""info_overlay -> thin see-through parts carrying the overlay texture.

An overlay is a textured quad that VBSP projects onto the brush faces listed in "sides". Roblox
can't put a partial texture on part of a face, so each overlay becomes its own paper-thin,
non-colliding part just in front of the surface, with a Texture on the side facing out.

Overlay keyvalues: BasisOrigin / BasisU / BasisV / BasisNormal (the quad's frame, Source
units), uv0..uv3 (its corners in that frame), StartU/EndU/StartV/EndV (texture coordinates at
the corners: u runs along +BasisU, v along +BasisV; End > 1 repeats the texture, End < Start
mirrors it), RenderOrder (stacking order of overlapping overlays).

Overlays that wrap around a corner (sides on different planes) come out flat, on the basis
plane.
"""
from __future__ import annotations

import numpy as np

from .coords import SPU, point_to_roblox, to_roblox
from .geometry import PartSpec
from .textures import map_linear

THICKNESS = 0.02          # studs
LIFT = 0.012              # studs off the surface, plus a little per RenderOrder step
LIFT_PER_ORDER = 0.006


def _vec(s, n=3):
    try:
        v = [float(x) for x in (s or "").split()]
    except ValueError:
        v = []
    return np.array((v + [0.0] * n)[:n])


def overlay_parts(vmf, sizes: dict[str, tuple[int, int]], region=None):
    """Returns (records, materials, report) where records are (PartSpec, FaceTex | None, info)."""
    from collections import Counter
    rep = Counter()
    out, mats = [], set()
    for e in vmf.entities:
        if e["classname"] != "info_overlay":
            continue
        m = (e["material"] or "").replace("\\", "/").lower()
        O = _vec(e["BasisOrigin"] or e["origin"])
        if region is not None:
            lo, hi = map(np.asarray, region)
            if not (np.all(O > lo) and np.all(O < hi)):
                continue
        U, V, N = _vec(e["BasisU"]), _vec(e["BasisV"]), _vec(e["BasisNormal"])
        if np.linalg.norm(U) < 1e-6 or np.linalg.norm(V) < 1e-6:
            rep["overlay without a basis"] += 1
            continue
        U, V = U / np.linalg.norm(U), V / np.linalg.norm(V)
        N = N / np.linalg.norm(N) if np.linalg.norm(N) > 1e-6 else np.cross(U, V)
        uv = np.array([_vec(e[f"uv{i}"])[:2] for i in range(4)])
        x0, x1 = uv[:, 0].min(), uv[:, 0].max()
        y0, y1 = uv[:, 1].min(), uv[:, 1].max()
        if x1 - x0 < 1e-3 or y1 - y0 < 1e-3:
            rep["degenerate overlay"] += 1
            continue
        su, eu = float(e["StartU"] or 0), float(e["EndU"] or 1)
        sv, ev = float(e["StartV"] or 0), float(e["EndV"] or 1)
        order = int(float(e["RenderOrder"] or 0))

        # frame in Roblox space (studs)
        Uh, Vh, Nh = to_roblox(U), to_roblox(V), to_roblox(N)
        len_u, len_v = (x1 - x0) * SPU, (y1 - y0) * SPU
        c_src = O + U * (x0 + x1) / 2 + V * (y0 + y1) / 2
        lift = LIFT + LIFT_PER_ORDER * order + THICKNESS / 2
        centre = point_to_roblox(c_src) + Nh * lift
        corner0 = point_to_roblox(O + U * x0 + V * y0)        # where (StartU, StartV) sits
        # part axes: the outward face is "Front" (-Z), so Z = -N; X along U, Y along +-V
        X, Z = Uh, -Nh
        Y = np.cross(Z, X)
        part = PartSpec("Part", centre, np.array([len_u, len_v, THICKNESS]), np.array([X, Y, Z]), 0, "overlay")
        ft = None
        if m in sizes:
            W, H = sizes[m]
            gs = W * (eu - su) / len_u * Uh
            gt = H * (ev - sv) / len_v * Vh
            ft = map_linear(part, "Front", gs, W * su - gs @ corner0, gt, H * sv - gt @ corner0, m, W, H)
            rep["overlays"] += 1
        else:
            rep["overlay without texture"] += 1
        mats.add(m)
        out.append((part, ft, dict(name=e["targetname"] or "", material=m, order=order)))
    return out, sorted(mats), rep
