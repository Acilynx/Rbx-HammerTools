import numpy as np

from vmf2rbx.coords import SPU, point_to_roblox, point_to_source, to_roblox, to_source


def test_scale():
    assert SPU * 128 == 10


def test_round_trip():
    p = np.array([[123.0, -45.5, 7.25], [0, 0, 0], [-1, 2, -3]])
    assert np.allclose(to_source(to_roblox(p)), p)
    assert np.allclose(point_to_source(point_to_roblox(p)), p)


def test_up_is_up_and_north_is_forward():
    assert np.allclose(to_roblox([0, 0, 1]), [0, 1, 0])     # Source up -> Roblox up
    assert np.allclose(to_roblox([0, 1, 0]), [0, 0, -1])    # Source north -> Roblox -Z (forward)


def test_mapping_preserves_handedness():
    """A pure Y<->Z swap has determinant -1 (a mirror). Ours must be a rotation."""
    M = np.array([to_roblox(e) for e in np.eye(3)]).T
    assert np.isclose(np.linalg.det(M), 1.0)
