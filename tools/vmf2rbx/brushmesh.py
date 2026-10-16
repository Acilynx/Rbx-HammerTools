"""Brushes the part decomposer can't express -> triangle meshes (Roblox MeshParts).

Parts and wedges cover boxes and prisms that slice into at most MAX_PIECES pieces. Everything
else (bevelled trims, domes, frustums, many-sided cylinders) is still a convex solid, so it
converts exactly as a mesh:

* one mesh per visible material, made of that material's faces. Each face keeps its own
  vertices (flat shading) and gets Hammer's texture coordinates directly,
  u = ((P . uaxis) / uscale + uoffset) / texture width, v likewise (v runs down the image,
  as in EditableMesh), so any texture angle maps exactly and nothing needs baking
* one collision hull from all of the brush's vertices (nodraw/clip sides are solid too),
  unless the brush is a single visible mesh covering every vertex, which then collides itself

Positions are Roblox studs relative to the brush's bounding-box centre; the importer places
every mesh of a brush at that centre with identity rotation. Triangles are counter-clockwise
seen from outside. Blobs use the props layout (see props.mesh_blob) and are named by content
hash, so identical shapes share one mesh in the importer.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .brushes import Brush, Side
from .coords import SPU, to_roblox
from .props import mesh_blob

DEFAULT_TEX = (512, 512)        # texel size for materials without a loaded texture
ON_PLANE = 1e-2                 # units


def face_polygon(b: Brush, s: Side) -> np.ndarray:
    """The brush's vertices on side `s`, ordered counter-clockwise seen from outside."""
    fv = b.verts[np.abs(b.verts @ s.normal - s.dist) < ON_PLANE]
    if len(fv) < 3:
        return fv
    c = fv.mean(0)
    a = fv[0] - c
    a /= np.linalg.norm(a)
    bb = np.cross(s.normal, a)
    ang = np.arctan2((fv - c) @ bb, (fv - c) @ a)
    return fv[np.argsort(ang)]


def face_uv(s: Side, P: np.ndarray, size) -> np.ndarray:
    W, H = size
    su = (P @ s.u.vec) / s.u.scale + s.u.offset
    tv = (P @ s.v.vec) / s.v.scale + s.v.offset
    return np.stack([su / W, tv / H], 1)


def _fan(n0: int, k: int) -> list[tuple[int, int, int]]:
    return [(n0, n0 + i, n0 + i + 1) for i in range(1, k - 1)]


def _blob_name(prefix: str, blob: bytes) -> str:
    return f"{prefix}:{hashlib.sha1(blob).hexdigest()[:20]}"


def brush_mesh_record(b: Brush, sizes: dict):
    """-> (record fields, {blob name: bytes}). Record fields: c (centre, studs), meshes
    [{blob, material, n, m}], collider {blob, n, m} or None (a visual mesh collides)."""
    R = to_roblox(b.verts) * SPU
    centre = (R.min(0) + R.max(0)) / 2
    groups: dict[str, list] = {}
    for s in b.visible_sides:
        poly = face_polygon(b, s)
        if len(poly) >= 3:
            groups.setdefault(s.material, []).append((s, poly))
    blobs, meshes = {}, []
    for mat, faces in groups.items():
        P, N, U, T = [], [], [], []
        for s, poly in faces:
            T += _fan(sum(len(p) for p in P), len(poly))
            P.append(to_roblox(poly) * SPU - centre)
            N.append(np.repeat(to_roblox(s.normal)[None], len(poly), 0))
            U.append(face_uv(s, poly, sizes.get(mat, DEFAULT_TEX)))
        P, N, U = np.concatenate(P), np.concatenate(N), np.concatenate(U)
        blob = mesh_blob(P, N, U, np.array(T))
        name = _blob_name("brushmesh", blob)
        blobs[name] = blob
        meshes.append(dict(blob=name, material=mat, n=len(P), m=len(T)))
    covered = {tuple(np.round(v, 3)) for s, poly in (f for fs in groups.values() for f in fs) for v in poly}
    collider = None
    if len(meshes) != 1 or len(covered) < len(b.verts):
        # closed hull of every side, for collision only
        P, T = [], []
        for s in b.sides:
            poly = face_polygon(b, s)
            if len(poly) >= 3:
                T += _fan(sum(len(p) for p in P), len(poly))
                P.append(to_roblox(poly) * SPU - centre)
        P = np.concatenate(P)
        blob = mesh_blob(P, np.zeros_like(P), np.zeros((len(P), 2)), np.array(T))
        name = _blob_name("brushhull", blob)
        blobs[name] = blob
        collider = dict(blob=name, n=len(P), m=len(T))
    return dict(c=[round(float(x), 6) for x in centre], meshes=meshes, collider=collider), blobs


# ------------------------------------------------------------------ displacements

def disp_grid(b: Brush, s: Side):
    """(base, displaced) vertex positions (N, 3), Source units, indexed [row * size + column].

    As builddisp.cpp: the face's corners clockwise seen from outside, starting at the one at
    startposition; row r runs from corner 0 toward corner 1, column c across to the edge
    corner 3 -> corner 2. position = base + normal * distance + offset + face normal * elevation.
    (Clockwise is checked on sp_a1_intro3: with it, sewn neighbours' edges coincide exactly.)"""
    poly = face_polygon(b, s)[::-1]
    if len(poly) != 4:
        return None
    k = int(np.argmin(np.linalg.norm(poly - s.disp.start, axis=1)))
    p = np.roll(poly, -k, axis=0)
    d = s.disp
    n = d.size
    t = np.linspace(0, 1, n)
    e0 = p[0] + np.outer(t, p[1] - p[0])
    e1 = p[3] + np.outer(t, p[2] - p[3])
    base = (e0[:, None, :] + (e1 - e0)[:, None, :] * t[None, :, None]).reshape(-1, 3)
    pos = base + d.normals * d.distances[:, None] + d.offsets + s.normal * d.elevation
    return base, pos


def disp_triangles(size: int) -> np.ndarray:
    """Two triangles per grid quad, diagonals alternating as in builddisp.cpp."""
    T = []
    for y in range(size - 1):
        for x in range(size - 1):
            i = y * size + x
            tl, tr, bl, br = i, i + 1, i + size, i + size + 1
            if i % 2 == 1:
                T += [(tl, bl, tr), (tr, bl, br)]
            else:
                T += [(tl, bl, br), (tl, br, tr)]
    return np.array(T)


def displacement_record(b: Brush, sizes: dict):
    """A brush with displacements: only the displaced surfaces exist (VBSP drops the brush's
    other faces). One mesh per displacement, smooth normals, Hammer UVs from the undisplaced
    surface (as Source does); it collides itself (precise, since it isn't convex)."""
    R = to_roblox(b.verts) * SPU
    meshes, blobs, pts = [], {}, []
    for s in b.disp_sides:
        g = disp_grid(b, s)
        if g is None:
            continue
        base, pos = g
        T = disp_triangles(s.disp.size)
        # counter-clockwise from outside: judge on the flat base surface
        a, c, e = base[T[:, 0]], base[T[:, 1]], base[T[:, 2]]
        flip = np.cross(c - a, e - a) @ s.normal < 0
        T[flip] = T[flip][:, ::-1]
        P = to_roblox(pos) * SPU
        fn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
        N = np.zeros_like(P)
        for k in range(3):
            np.add.at(N, T[:, k], fn)
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
        U = face_uv(s, base, sizes.get(s.material, DEFAULT_TEX))
        meshes.append((s.material, P, N, U, T))
        pts.append(P)
    if not meshes:
        return None, {}
    allp = np.concatenate(pts)
    centre = (allp.min(0) + allp.max(0)) / 2
    out = []
    for mat, P, N, U, T in meshes:
        blob = mesh_blob(P - centre, N, U, T)
        name = _blob_name("dispmesh", blob)
        blobs[name] = blob
        out.append(dict(blob=name, material=mat, n=len(P), m=len(T)))
    size = allp.max(0) - allp.min(0)
    return dict(c=[round(float(x), 6) for x in centre], s=[round(float(x), 6) for x in size],
                meshes=out, collider=None, fidelity="precise"), blobs
