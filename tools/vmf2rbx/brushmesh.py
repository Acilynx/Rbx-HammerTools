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
