"""Brush -> Roblox parts.

* Axis-aligned box  -> one Part (identity rotation)
* Rotated box       -> one Part
* Convex prism      -> exact split into Parts and WedgeParts (slab decomposition)
* Anything else     -> a placeholder (bounding box), flagged for a CSG/mesh fallback

All output is in Roblox space and studs.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .brushes import Brush
from .coords import SPU, to_roblox

EPS = 1e-4
MAX_PIECES = 16


@dataclass
class PartSpec:
    cls: str                 # "Part", "WedgePart" or "Placeholder"
    center: np.ndarray       # studs, Roblox space
    size: np.ndarray         # studs, local X/Y/Z
    rot: np.ndarray          # 3x3, rows = local X, Y, Z axes in world space
    piece: int = 0           # 0 = whole brush; 1.. = pieces of a decomposed brush
    kind: str = ""           # classification of the source brush


def _par(a, b, t=1e-4):
    return abs(abs(a @ b) - 1) < t


def _perp(a, b, t=1e-4):
    return abs(a @ b) < t


AXES = np.eye(3)


def classify(brush: Brush) -> str:
    ns = [s.normal for s in brush.sides]
    k = len(ns)
    if k == 6:
        used, pairs = set(), []
        for i in range(6):
            if i in used:
                continue
            for j in range(i + 1, 6):
                if j not in used and ns[i] @ ns[j] < -1 + 1e-4:
                    pairs.append(ns[i]); used |= {i, j}
                    break
        if len(pairs) == 3 and all(_perp(pairs[a], pairs[b]) for a, b in ((0, 1), (0, 2), (1, 2))):
            return "box_axis" if all(any(_par(p, a) for a in AXES) for p in pairs) else "box_rotated"
    if k == 5:
        for i, j in itertools.combinations(range(5), 2):
            if ns[i] @ ns[j] < -1 + 1e-4:
                rest = [ns[x] for x in range(5) if x not in (i, j)]
                if all(_perp(r, ns[i]) for r in rest):
                    return "wedge" if any(_perp(a, b) for a, b in itertools.combinations(rest, 2)) else "tri_prism"
        return "five_sided"
    return f"other_{k}"


# ---------------------------------------------------------------- prism slab decomposition

def _hull2d(points):
    P = sorted(set((round(p[0], 6), round(p[1], 6)) for p in points))
    if len(P) < 3:
        return [np.array(p) for p in P]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 1e-9:
            lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 1e-9:
            up.pop()
        up.append(p)
    return [np.array(p) for p in lo[:-1] + up[:-1]]


def _vrange(poly, u):
    vs = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        if abs(b[0] - a[0]) < 1e-9:
            if abs(u - a[0]) < 1e-9:
                vs += [a[1], b[1]]
            continue
        t = (u - a[0]) / (b[0] - a[0])
        if -1e-9 <= t <= 1 + 1e-9:
            vs.append(a[1] + t * (b[1] - a[1]))
    return min(vs), max(vs)


def _slab_decompose(poly):
    """Convex polygon in (u, v) -> rectangles and right triangles, or None if this slicing
    direction can't be tiled by rect + 2 triangles per slab (top chain dips below bottom)."""
    us = sorted(set(round(p[0], 6) for p in poly))
    out = []
    for u0, u1 in zip(us, us[1:]):
        if u1 - u0 < EPS:
            continue
        ua, ub = u0 + (u1 - u0) * 0.25, u0 + (u1 - u0) * 0.75
        (la, ha), (lb, hb) = _vrange(poly, ua), _vrange(poly, ub)

        def ext(fa, fb, u):
            return fa + (fb - fa) * (u - ua) / (ub - ua)
        lo0, lo1, hi0, hi1 = ext(la, lb, u0), ext(la, lb, u1), ext(ha, hb, u0), ext(ha, hb, u1)
        r0, r1 = max(lo0, lo1), min(hi0, hi1)
        if r1 - r0 < -EPS:
            return None
        if r1 - r0 > EPS:
            out.append(["rect", u0, u1, r0, r1])
        if abs(lo0 - lo1) > EPS:     # bottom triangle; right angle at the low end, top of the chain
            uh, ul = (u1, u0) if lo1 > lo0 else (u0, u1)
            out.append(["tri", np.array([ul, max(lo0, lo1)]), np.array([ul, min(lo0, lo1)]), np.array([uh, max(lo0, lo1)])])
        if abs(hi0 - hi1) > EPS:     # top triangle; right angle at the high end, bottom of the chain
            uh, ul = (u0, u1) if hi0 > hi1 else (u1, u0)
            out.append(["tri", np.array([uh, min(hi0, hi1)]), np.array([uh, max(hi0, hi1)]), np.array([ul, min(hi0, hi1)])])
    rects = sorted([r for r in out if r[0] == "rect"], key=lambda r: r[1])
    merged = []
    for r in rects:
        if merged and abs(merged[-1][2] - r[1]) < EPS and abs(merged[-1][3] - r[3]) < EPS and abs(merged[-1][4] - r[4]) < EPS:
            merged[-1][2] = r[2]
        else:
            merged.append(r)
    return merged + [t for t in out if t[0] == "tri"]


def prism_axis(brush: Brush):
    ns = [s.normal for s in brush.sides]
    for i, j in itertools.combinations(range(len(ns)), 2):
        if ns[i] @ ns[j] < -1 + 1e-4 and all(abs(ns[k] @ ns[i]) < 1e-4 for k in range(len(ns)) if k not in (i, j)):
            return ns[i]
    return None


def _unit(v):
    return v / np.linalg.norm(v)


def decompose_prism(brush: Brush):
    """Exact Part/WedgePart split of a convex prism brush; None if it isn't a prism,
    [] if no valid slicing direction exists."""
    ax = prism_axis(brush)
    if ax is None:
        return None
    R = to_roblox(brush.verts) * SPU
    e = _unit(to_roblox(ax))
    t = R @ e
    L, tm = t.max() - t.min(), (t.max() + t.min()) / 2
    a0 = _unit(np.cross(e, [1, 0, 0] if abs(e[0]) < 0.9 else [0, 1, 0]))
    b0 = np.cross(e, a0)
    poly = _hull2d([(p @ a0, p @ b0) for p in R])
    # Texture direction of each visible cap face (sides perpendicular to the prism axis). A
    # slicing direction that isn't parallel/perpendicular to it leaves that cap unmappable
    # (Roblox textures only rotate in 90-degree steps relative to the part).
    cap_dirs = []
    for s in brush.sides:
        if not s.is_tool and abs(abs(s.normal @ ax) - 1) < 1e-4:
            u = to_roblox(s.u.vec)
            u = u - (u @ e) * e
            if np.linalg.norm(u) > 1e-6:
                cap_dirs.append(_unit(u))
    best = None
    for i in range(len(poly)):
        d = _unit(poly[(i + 1) % len(poly)] - poly[i])
        a = d[0] * a0 + d[1] * b0
        b = np.cross(e, a)
        pieces = _slab_decompose(_hull2d([(p @ a, p @ b) for p in R]))
        if pieces is None:
            continue
        bad_caps = sum(1 for c in cap_dirs if not (_par(c, a, 1e-3) or _perp(c, a, 1e-3)))
        score = (bad_caps, len(pieces))
        if best is None or score < best[3]:
            best = (pieces, a, b, score)
    if best is None:
        return []
    pieces, a, b, _ = best

    def W(uv):
        return uv[0] * a + uv[1] * b + tm * e
    out = []
    for pc in pieces:
        if pc[0] == "rect":
            _, u0, u1, v0, v1 = pc
            out.append(PartSpec("Part", W(((u0 + u1) / 2, (v0 + v1) / 2)), np.array([u1 - u0, v1 - v0, L]), np.array([a, b, e])))
        else:
            # Roblox WedgePart: right angle at local (y=-h/2, z=+d/2); legs along +Y (height) and -Z (depth)
            _, P, A, B = pc
            P3, A3, B3 = W(P), W(A), W(B)
            Y = _unit(A3 - P3); h = np.linalg.norm(A3 - P3)
            Z = -_unit(B3 - P3); dd = np.linalg.norm(B3 - P3)
            X = np.cross(Y, Z)
            out.append(PartSpec("WedgePart", P3 + 0.5 * (A3 - P3) + 0.5 * (B3 - P3), np.array([L, h, dd]), np.array([X, Y, Z])))
    return out


def brush_to_parts(brush: Brush) -> list[PartSpec]:
    kind = classify(brush)
    R = to_roblox(brush.verts) * SPU
    if kind == "box_axis":
        return [PartSpec("Part", (R.min(0) + R.max(0)) / 2, R.max(0) - R.min(0), np.eye(3), 0, kind)]
    if kind == "box_rotated":
        ns = []
        for s in brush.sides:
            if not any(_par(s.normal, m) for m in ns):
                ns.append(s.normal)
        n1, n2 = ns[0], ns[1]
        n3 = np.cross(n1, n2)
        ext = [(brush.verts @ n).max() - (brush.verts @ n).min() for n in (n1, n2, n3)]
        r0, r1 = to_roblox(n1), to_roblox(n2)
        return [PartSpec("Part", to_roblox(brush.verts.mean(0)) * SPU, np.array(ext) * SPU, np.array([r0, r1, np.cross(r0, r1)]), 0, kind)]
    pieces = decompose_prism(brush)
    if pieces and len(pieces) <= MAX_PIECES:
        for i, p in enumerate(pieces, 1):
            p.piece, p.kind = i, kind
        return pieces
    return [PartSpec("Placeholder", (R.min(0) + R.max(0)) / 2, R.max(0) - R.min(0), np.eye(3), 0, kind)]


# ---------------------------------------------------------------- verification helpers

def brush_volume(brush: Brush) -> float:
    """Exact volume (studs^3) of the convex brush, independent of the decomposition."""
    V = brush.verts * SPU
    c = V.mean(0)
    vol = 0.0
    for s in brush.sides:
        fv = [v * SPU for v in brush.verts if abs(s.normal @ v - s.dist) < 1e-2]
        if len(fv) < 3:
            continue
        fc = np.mean(fv, 0)
        a = _unit(fv[0] - fc)
        b = np.cross(s.normal, a)
        fv.sort(key=lambda p: np.arctan2((p - fc) @ b, (p - fc) @ a))
        for i in range(len(fv)):
            vol += abs(np.dot(fv[i] - c, np.cross(fv[(i + 1) % len(fv)] - c, fc - c))) / 6
    return vol


def piece_volume(p: PartSpec) -> float:
    return float(np.prod(p.size)) * (0.5 if p.cls == "WedgePart" else 1.0)


def inside_piece(p: PartSpec, X: np.ndarray) -> np.ndarray:
    """Which of the Roblox-space points X (N,3) lie inside the piece."""
    loc = (X - p.center) @ p.rot.T
    h = p.size / 2
    ok = np.all(np.abs(loc) <= h + 1e-6, axis=1)
    if p.cls == "WedgePart":
        H, D = p.size[1], p.size[2]
        ok &= (loc[:, 1] + H / 2) / H <= (loc[:, 2] + D / 2) / D + 1e-6
    return ok


def inside_brush(brush: Brush, X: np.ndarray, margin: float = 0.0) -> np.ndarray:
    S = np.stack([X[:, 0], -X[:, 2], X[:, 1]], 1) / SPU
    ok = np.ones(len(X), bool)
    for s in brush.sides:
        ok &= S @ s.normal <= s.dist + 1e-3 - margin
    return ok


def near_surface(brush: Brush, X: np.ndarray, tol_units: float = 0.05) -> np.ndarray:
    S = np.stack([X[:, 0], -X[:, 2], X[:, 1]], 1) / SPU
    near = np.zeros(len(X), bool)
    for s in brush.sides:
        near |= np.abs(S @ s.normal - s.dist) <= tol_units
    return near
