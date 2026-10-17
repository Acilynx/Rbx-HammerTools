"""Brushes as convex solids: planes with outward normals, plus the vertices they enclose.

Everything here is in Source units and Source axes. Conversion to Roblox happens later.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

EPS_IN = 1e-3      # tolerance for "inside all half-spaces" (units)


@dataclass
class TexAxis:
    """One Hammer texture axis: texel = (P . vec) / scale + offset."""
    vec: np.ndarray
    offset: float
    scale: float


@dataclass
class Displacement:
    """A side's dispinfo: a (2^power + 1)^2 grid over the (4-sided) face. Arrays are indexed
    [row * size + column], rows as in the VMF ("row0", "row1", ...)."""
    power: int
    start: np.ndarray         # startposition: the face corner the grid starts at
    elevation: float
    normals: np.ndarray       # (N, 3) displacement direction per vertex
    distances: np.ndarray     # (N,)
    offsets: np.ndarray       # (N, 3)
    alphas: np.ndarray        # (N,) blend between $basetexture (0) and $basetexture2 (255)

    @property
    def size(self) -> int:
        return (1 << self.power) + 1


@dataclass
class Side:
    normal: np.ndarray        # outward unit normal
    dist: float               # plane: normal . x = dist
    material: str             # lower-case material path
    u: TexAxis
    v: TexAxis
    point: np.ndarray         # a point on the plane
    disp: "Displacement | None" = None
    sid: int = 0              # Hammer side id (info_overlay "sides" refer to it)

    @property
    def is_tool(self) -> bool:
        return self.material.startswith("tools/")


@dataclass
class Brush:
    id: int
    sides: list[Side]
    owner: str = "world"      # entity classname, or "world"
    entity_name: str = ""     # entity targetname
    verts: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    @property
    def visible_sides(self) -> list[Side]:
        return [s for s in self.sides if not s.is_tool]

    @property
    def is_tool_only(self) -> bool:
        return not self.visible_sides

    @property
    def disp_sides(self) -> list[Side]:
        return [s for s in self.sides if s.disp is not None]

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.verts.min(0), self.verts.max(0)

    def dominant_material(self) -> str:
        mats = [s.material for s in self.visible_sides] or [s.material for s in self.sides]
        return max(dict.fromkeys(mats), key=mats.count)      # ties: first side wins (set order varies per run)


def _intersections(normals: np.ndarray, dists: np.ndarray) -> np.ndarray:
    pts: list[np.ndarray] = []
    for a, b, c in itertools.combinations(range(len(normals)), 3):
        A = normals[[a, b, c]]
        if abs(np.linalg.det(A)) < 1e-9:
            continue
        x = np.linalg.solve(A, dists[[a, b, c]])
        if np.all(normals @ x <= dists + EPS_IN):
            if not any(np.linalg.norm(x - p) < EPS_IN for p in pts):
                pts.append(x)
    return np.array(pts) if pts else np.zeros((0, 3))


def solve_vertices(sides: list[Side]) -> np.ndarray:
    n = np.array([s.normal for s in sides])
    d = np.array([s.dist for s in sides])
    return _intersections(n, d)


def make_brush(id: int, planes: list[tuple], owner="world", entity_name="") -> Brush:
    """Build a brush from (normal, dist, material, u, v) tuples whose normals are already outward."""
    sides = []
    for normal, dist, mat, u, v in planes:
        normal = np.asarray(normal, float)
        normal = normal / np.linalg.norm(normal)
        sides.append(Side(normal, float(dist), mat.lower(), u, v, normal * dist))
    b = Brush(id, sides, owner, entity_name)
    b.verts = solve_vertices(sides)
    if len(b.verts) < 4:
        raise ValueError(f"brush {id} is degenerate")
    return b


def _displacement(side) -> Displacement | None:
    if not getattr(side, "is_disp", False) or not side._disp_verts:
        return None
    vs = side._disp_verts
    vec = lambda v: [v.x, v.y, v.z]
    return Displacement(
        int(side.disp_power), np.array(vec(side.disp_pos), float), float(side.disp_elevation),
        np.array([vec(d.normal) for d in vs], float), np.array([d.distance for d in vs], float),
        np.array([vec(d.offset) for d in vs], float), np.array([d.alpha for d in vs], float))


def from_srctools(solid, owner="world", entity_name="") -> Brush | None:
    """Convert a srctools Solid. Hammer stores each plane as three points; the winding gives
    the normal. Rather than trusting one winding convention, try it and flip if the brush
    comes out empty."""
    raw = []
    for s in solid.sides:
        p = [np.array([v.x, v.y, v.z], float) for v in s.planes]
        n = np.cross(p[2] - p[0], p[1] - p[0])
        ln = np.linalg.norm(n)
        if ln < 1e-12:
            continue
        n /= ln
        u = TexAxis(np.array([s.uaxis.x, s.uaxis.y, s.uaxis.z], float), float(s.uaxis.offset), float(s.uaxis.scale))
        v = TexAxis(np.array([s.vaxis.x, s.vaxis.y, s.vaxis.z], float), float(s.vaxis.offset), float(s.vaxis.scale))
        raw.append([n, float(n @ p[0]), s.mat.lower(), u, v, p[0], _displacement(s), int(s.id)])
    for flip in (1.0, -1.0):
        sides = [Side(flip * n, flip * d, m, u, v, p0, dp, sid) for n, d, m, u, v, p0, dp, sid in raw]
        verts = solve_vertices(sides)
        if len(verts) >= 4:
            c = verts.mean(0)
            if all(s.normal @ c < s.dist for s in sides):
                return Brush(int(solid.id), sides, owner, entity_name, verts)
    return None


def load_vmf_brushes(vmf) -> list[Brush]:
    """All brushes of a parsed srctools VMF: world brushes plus brush entities."""
    out: list[Brush] = []
    for s in vmf.brushes:
        b = from_srctools(s)
        if b is not None:
            out.append(b)
    for e in vmf.entities:
        for s in e.solids:
            b = from_srctools(s, e["classname"], e["targetname"] or "")
            if b is not None:
                out.append(b)
    return out
