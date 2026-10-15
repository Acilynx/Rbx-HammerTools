"""Coordinate conventions shared by every stage.

Source is right-handed and Z-up; Roblox is right-handed and Y-up. Swapping Y and Z alone is
a reflection (it flips handedness and mirrors the map), so one axis is negated:

    Source (x, y, z)  ->  Roblox (x, z, -y)

Source north (+Y) therefore lands on Roblox -Z, which is also Roblox's default "forward".
"""
from __future__ import annotations

import numpy as np

UNITS_PER_STUD = 12.8          # 128 Hammer units = 10 studs
SPU = 1.0 / UNITS_PER_STUD     # studs per unit (0.078125)


def to_roblox(v) -> np.ndarray:
    """Source vector -> Roblox vector (no scaling). Works on (3,) or (N, 3)."""
    v = np.asarray(v, dtype=float)
    return np.stack([v[..., 0], v[..., 2], -v[..., 1]], axis=-1)


def to_source(v) -> np.ndarray:
    """Inverse of :func:`to_roblox`."""
    v = np.asarray(v, dtype=float)
    return np.stack([v[..., 0], -v[..., 2], v[..., 1]], axis=-1)


def point_to_roblox(p) -> np.ndarray:
    """Source position (units) -> Roblox position (studs)."""
    return to_roblox(p) * SPU


def point_to_source(p) -> np.ndarray:
    """Roblox position (studs) -> Source position (units)."""
    return to_source(p) / SPU
