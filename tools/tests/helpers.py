"""Synthetic brushes for tests (no game content)."""
from __future__ import annotations

import math

import numpy as np

from vmf2rbx.brushes import TexAxis, make_brush

TEX = "test/tex"


def world_aligned_axes(normal):
    """Hammer's default world-aligned texture axes for a face normal (Source space)."""
    n = np.asarray(normal, float)
    ax = int(np.argmax(np.abs(n)))
    if ax == 2:
        return np.array([1.0, 0, 0]), np.array([0, -1.0, 0])
    if ax == 0:
        return np.array([0, 1.0, 0]), np.array([0, 0, -1.0])
    return np.array([1.0, 0, 0]), np.array([0, 0, -1.0])


def face_aligned_axes(normal):
    """u along the horizontal edge of the face, v down (works for vertical faces), else world-aligned."""
    n = np.asarray(normal, float)
    if abs(n[2]) > 0.99:
        return world_aligned_axes(n)
    u = np.cross([0, 0, 1.0], n); u /= np.linalg.norm(u)
    return u, np.array([0, 0, -1.0])


def brush_from_halfspaces(id, halfspaces, axes=face_aligned_axes, scale=0.25, offsets=(0.0, 0.0), material=TEX, owner="world"):
    planes = []
    for n, d in halfspaces:
        n = np.asarray(n, float)
        ln = np.linalg.norm(n)
        n, d = n / ln, d / ln
        u, v = axes(n)
        planes.append((n, d, material, TexAxis(u, offsets[0], scale), TexAxis(v, offsets[1], scale)))
    return make_brush(id, planes, owner)


def box(id, lo, hi, **kw):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    hs = []
    for i in range(3):
        e = np.eye(3)[i]
        hs += [(e, hi[i]), (-e, -lo[i])]
    return brush_from_halfspaces(id, hs, **kw)


def rotated_box(id, center, size, yaw_deg, **kw):
    c = np.asarray(center, float); h = np.asarray(size, float) / 2
    t = math.radians(yaw_deg)
    ex = np.array([math.cos(t), math.sin(t), 0]); ey = np.array([-math.sin(t), math.cos(t), 0]); ez = np.array([0, 0, 1.0])
    hs = []
    for e, hh in ((ex, h[0]), (ey, h[1]), (ez, h[2])):
        hs += [(e, e @ c + hh), (-e, -(e @ c) + hh)]
    return brush_from_halfspaces(id, hs, **kw)


def chamfered_box(id, lo, hi, cut, **kw):
    """Axis box with its (-x, -y) vertical edge cut at 45 degrees by `cut` units."""
    b = box(id, lo, hi, **kw)
    lo = np.asarray(lo, float)
    n = np.array([-1.0, -1.0, 0]) / math.sqrt(2)
    corner = np.array([lo[0] + cut, lo[1], 0])
    hs = [(s.normal, s.dist) for s in b.sides] + [(n, n @ corner)]
    return brush_from_halfspaces(id, hs, **kw)


def diagonal_strip(id, thickness=16, length=128, z=(0, 128), **kw):
    """Thin wall running at 45 degrees, clipped by x and y caps (the shape that broke slicing once)."""
    n = np.array([1.0, -1.0, 0]) / math.sqrt(2)
    hs = [(n, thickness / 2), (-n, thickness / 2),
          (np.array([1.0, 0, 0]), length / 2), (np.array([0, 1.0, 0]), length / 2),
          (np.array([-1.0, 0, 0]), length / 2), (np.array([0, -1.0, 0]), length / 2),
          (np.array([0, 0, 1.0]), z[1]), (np.array([0, 0, -1.0]), -z[0])]
    return brush_from_halfspaces(id, hs, **kw)


def right_wedge(id, w=64, d=128, h=64, **kw):
    """Ramp rising along +x: bottom, back wall at x=d, two sides and a slope."""
    slope = np.array([-h, 0, d], float)
    hs = [(np.array([0, 0, -1.0]), 0), (np.array([1.0, 0, 0]), d),
          (np.array([0, 1.0, 0]), w / 2), (np.array([0, -1.0, 0]), w / 2), (slope, 0.0)]
    return brush_from_halfspaces(id, hs, **kw)


def ngon_prism(id, n=12, r=16, z=(0, 32), **kw):
    hs = [(np.array([math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n), 0]), r) for k in range(n)]
    hs += [(np.array([0, 0, 1.0]), z[1]), (np.array([0, 0, -1.0]), -z[0])]
    return brush_from_halfspaces(id, hs, **kw)


def mitred_bevel_strip(id, **kw):
    """Not a prism in any direction: mitred 45-degree ends plus a bevelled underside."""
    hs = [(np.array([0, 0, 1.0]), 4), (np.array([0, 0, -1.0]), 0),
          (np.array([0, -1.0, 0]), 0), (np.array([0, 1.0, 0]), 8),
          (np.array([1.0, 1.0, 0]) / math.sqrt(2), 96 / math.sqrt(2)),
          (np.array([-1.0, 1.0, 0]) / math.sqrt(2), 96 / math.sqrt(2)),
          (np.array([0, 1.0, -1.0]) / math.sqrt(2), 6 / math.sqrt(2))]
    return brush_from_halfspaces(id, hs, **kw)
