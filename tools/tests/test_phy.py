"""The .phy reader on a hand-built collision model: one convex piece, a box."""
import struct

import numpy as np

from vmf2rbx.phy import M2I, parse_phy_faces

# a 2 x 4 x 6 inch box in Source space, stored in IVP's metres and axes (ivp = (x, -z, y) / M2I)
SRC = np.array([[x, y, z] for x in (0, 2) for y in (0, 4) for z in (0, 6)], float)
TRIS = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
        (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]


def build_phy() -> bytes:
    ivp = np.stack([SRC[:, 0], -SRC[:, 2], SRC[:, 1]], 1) / M2I
    ntri = len(TRIS)
    ledge_size = 16 + 16 * ntri
    tris = b"".join(struct.pack("<I", 0) + b"".join(struct.pack("<HH", i, 0) for i in t) for t in TRIS)
    ledge = struct.pack("<iiIhh", ledge_size, 0, 0, ntri, 0) + tris      # c_point_offset -> right after
    pool = b"".join(struct.pack("<4f", *p, 0.0) for p in ivp)
    surface = struct.pack("<3f3ffIi3i", 0, 0, 0, 0, 0, 0, 0, 0, 48 + ledge_size + len(pool), 0, 0, 0)
    vphy = b"VPHY" + struct.pack("<hhi3fi", 256, 0, 0, 0, 0, 0, 0)
    solid = vphy + surface + ledge + pool
    return struct.pack("<4i", 16, 0, 1, 0) + struct.pack("<i", len(solid)) + solid


def test_parse_box_piece():
    pieces = parse_phy_faces(build_phy())
    assert len(pieces) == 1
    P, T = pieces[0]
    assert np.allclose(np.sort(P, axis=0), np.sort(SRC, axis=0), atol=1e-4)
    assert T.shape == (12, 3) and T.max() == 7
