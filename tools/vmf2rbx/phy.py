"""Source collision models (.phy): the convex pieces of a prop's physics shape.

Layout (VPhysics compact surfaces, as written by studiomdl):
  header: int size (16), int id, int solidCount, int checksum
  per solid: int byteSize, then
    compactsurfaceheader_t (28 bytes): "VPHY", short version, short modelType, int surfaceSize,
      Vector dragAxisAreas, int axisMapSize
    IVP_Compact_Surface (48 bytes): mass_center[3], rotation_inertia[3], upper_limit_radius,
      int (max_deviation:8 | byte_size:24), int offset_ledgetree_root, int dummy[3] ("IVPS")
    ledges (one per convex piece), back to back: IVP_Compact_Ledge (16 bytes): int c_point_offset
      (from the ledge, to the shared point pool), int, int flags, short n_triangles, short; then
      n_triangles x 16 bytes: int (tri_index etc.), then 3 edges of u16 start_point_index +
      u16 (opposite:15 | virtual:1). After the last ledge: the point pool, float x, y, z, w each
      (measured on Portal 2's models: the ledges end exactly where every c_point_offset points)
IVP works in metres with its own axes; Source = (x, z, -y) * 39.3701 (ConvertPositionToHL).
"""
from __future__ import annotations

import struct

import numpy as np

M2I = 1.0 / 0.0254


def parse_phy(data: bytes) -> list[np.ndarray]:
    """Convex pieces as point arrays (Source units, in the space of the bone they're attached to;
    for one-bone props apply that bone's rest transform, models.rest_pose(bone_to_model=True)).
    Empty on anything unexpected."""
    return [p for p, _ in parse_phy_faces(data)]


def parse_phy_faces(data: bytes) -> list[tuple[np.ndarray, np.ndarray]]:
    """Convex pieces as (points (n, 3), triangles (m, 3) indexing them)."""
    out = []
    try:
        hsize, _, nsolids, _ = struct.unpack_from("<4i", data, 0)
        off = hsize
        for _ in range(nsolids):
            size = struct.unpack_from("<i", data, off)[0]
            start = off + 4
            if data[start:start + 4] != b"VPHY":
                off = start + size
                continue                       # old-style (pre-VPHY) solid: skip
            surf = start + 28
            led = surf + 48
            # ledges sit back to back (16-byte header + 16 bytes per triangle) and all index one
            # shared point pool, which starts right after the last ledge
            pool = led + struct.unpack_from("<i", data, led)[0]
            while led < pool:
                cpo, _, _, ntri = struct.unpack_from("<iiIh", data, led)
                if ntri <= 0 or led + cpo != pool:
                    break
                tris = [[struct.unpack_from("<H", data, led + 16 + 16 * t + 4 + 4 * e)[0] for e in range(3)]
                        for t in range(ntri)]
                idx = sorted({i for tr in tris for i in tr})
                if pool + 16 * (idx[-1] + 1) <= len(data):
                    pts = np.array([struct.unpack_from("<3f", data, pool + 16 * i) for i in idx])
                    if len(pts) >= 4 and np.all(np.isfinite(pts)) and np.abs(pts).max() < 1000:
                        local = {g: k for k, g in enumerate(idx)}
                        out.append((np.stack([pts[:, 0], pts[:, 2], -pts[:, 1]], 1) * M2I,
                                    np.array([[local[i] for i in tr] for tr in tris])))
                led += 16 + 16 * ntri
            off = start + size
    except struct.error:
        return []
    return out
