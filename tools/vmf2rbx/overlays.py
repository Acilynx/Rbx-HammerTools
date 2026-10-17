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


# ------------------------------------------------------------------ overlays as meshes on their faces

def _clip(poly2: np.ndarray, poly3: np.ndarray, axis: int, bound: float, keep_below: bool):
    """Sutherland-Hodgman against one axis-aligned line in the overlay's (u, v) plane, carrying
    the 3D positions along."""
    out2, out3 = [], []
    n = len(poly2)
    for i in range(n):
        a2, b2, a3, b3 = poly2[i], poly2[(i + 1) % n], poly3[i], poly3[(i + 1) % n]
        ina = a2[axis] <= bound if keep_below else a2[axis] >= bound
        inb = b2[axis] <= bound if keep_below else b2[axis] >= bound
        if ina:
            out2.append(a2); out3.append(a3)
        if ina != inb:
            t = (bound - a2[axis]) / (b2[axis] - a2[axis])
            out2.append(a2 + t * (b2 - a2)); out3.append(a3 + t * (b3 - a3))
    return np.array(out2), np.array(out3)


def overlay_meshes(vmf, brushes, sizes: dict[str, tuple[int, int]], region=None):
    """info_overlay -> meshes on exactly the faces it lists, as VBSP does: each face is projected
    along the overlay's normal into its (u, v) frame and clipped to the overlay's rectangle; the
    pieces keep the face's own 3D positions (lifted a little off it) and get the overlay's
    texture coordinates. Overlays that wrap around an edge follow both faces; overlays listing
    no faces aren't drawn (neither does Source).
    Returns ([(record fields, blobs)], report)."""
    from collections import Counter
    from .brushmesh import _blob_name, _fan, face_polygon, near_zero, with_bounds
    from .props import mesh_blob
    rep = Counter()
    by_sid = {s.sid: (b, s) for b in brushes for s in b.sides if s.sid}
    out = []
    for i, e in enumerate(e for e in vmf.entities if e["classname"] == "info_overlay"):
        m = (e["material"] or "").replace("\\", "/").lower()
        O = _vec(e["BasisOrigin"] or e["origin"])
        if region is not None:
            lo, hi = map(np.asarray, region)
            if not (np.all(O > lo) and np.all(O < hi)):
                continue
        sides = [int(x) for x in (e["sides"] or "").split() if x.lstrip("-").isdigit()]
        faces = [by_sid[s] for s in sides if s in by_sid]
        if not faces:
            rep["overlay on no faces (not drawn)"] += 1
            continue
        if m not in sizes:
            rep["overlay without texture"] += 1
            continue
        U, V, N = _vec(e["BasisU"]), _vec(e["BasisV"]), _vec(e["BasisNormal"])
        if np.linalg.norm(U) < 1e-6 or np.linalg.norm(V) < 1e-6:
            rep["overlay without a basis"] += 1
            continue
        U, V = U / np.linalg.norm(U), V / np.linalg.norm(V)
        N = N / np.linalg.norm(N) if np.linalg.norm(N) > 1e-6 else np.cross(U, V)
        uv = np.array([_vec(e[f"uv{k}"])[:2] for k in range(4)])
        x0, x1, y0, y1 = uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].min(), uv[:, 1].max()
        if x1 - x0 < 1e-3 or y1 - y0 < 1e-3:
            rep["degenerate overlay"] += 1
            continue
        su, eu = float(e["StartU"] or 0), float(e["EndU"] or 1)
        sv, ev = float(e["StartV"] or 0), float(e["EndV"] or 1)
        order = int(float(e["RenderOrder"] or 0))
        lift = (LIFT + LIFT_PER_ORDER * order) / SPU             # units
        P, Nn, UV, T = [], [], [], []
        for b, s in faces:
            if abs(s.normal @ N) < 0.1:                           # face edge-on to the projection
                continue
            p3 = face_polygon(b, s)
            p2 = np.stack([(p3 - O) @ U, (p3 - O) @ V], 1)
            for axis, bound, below in ((0, x1, True), (0, x0, False), (1, y1, True), (1, y0, False)):
                if len(p2) < 3:
                    break
                p2, p3 = _clip(p2, p3, axis, bound, below)
            if len(p2) < 3:
                continue
            q = p3 + s.normal * lift
            tu = su + (p2[:, 0] - x0) / (x1 - x0) * (eu - su)
            tv = sv + (p2[:, 1] - y0) / (y1 - y0) * (ev - sv)
            T += _fan(sum(len(x) for x in P), len(q))
            P.append(to_roblox(q) * SPU)
            Nn.append(np.repeat(to_roblox(s.normal)[None], len(q), 0))
            UV.append(near_zero(np.stack([tu, tv], 1)))
        if not P:
            rep["overlay clipped away"] += 1
            continue
        P, Nn, UV = np.concatenate(P), np.concatenate(Nn), np.concatenate(UV)
        lo, hi = P.min(0) - 0.1, P.max(0) + 0.1
        centre = (lo + hi) / 2
        P, Nn, UV, T = with_bounds(P - centre, Nn, UV, T, lo - centre, hi - centre)
        blob = mesh_blob(P, Nn, UV, T)
        name = _blob_name("overlaymesh", blob)
        rec = dict(id=-(i + 1), piece=0, cls="Mesh", owner="info_overlay", entity=e["targetname"] or "",
                   mat=m, kind="overlay", faces=[], c=[round(float(x), 6) for x in centre],
                   s=[round(float(x), 6) for x in hi - lo],
                   meshes=[dict(blob=name, material=m, n=len(P), m=len(T))], collider=None,
                   nocollide=True, shadow=[])
        out.append((rec, {name: blob}))
        rep["overlays"] += 1
        if len({s.sid for _, s in faces}) > 1 and len({tuple(np.round(s.normal, 3)) for _, s in faces}) > 1:
            rep["overlays wrapping an edge"] += 1
    return out, rep
