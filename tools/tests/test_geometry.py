"""Decomposition must be exact: same volume, and every sampled point inside the brush is inside
exactly one piece (and points outside are in none). The checks only use the brush's planes and
the WedgePart shape measured in Studio, not the decomposition code."""
import numpy as np
import pytest

from helpers import (box, chamfered_box, diagonal_strip, mitred_bevel_strip, ngon_prism,
                     right_wedge, rotated_box)
from vmf2rbx.geometry import brush_to_parts, classify
from vmf2rbx.pipeline import verify_decomposition


def check_exact(brush, samples=6000):
    parts = brush_to_parts(brush)
    err = verify_decomposition(brush, parts, np.random.default_rng(1), samples)
    assert err is None, err
    return parts


def test_axis_box_is_one_part():
    parts = brush_to_parts(box(1, [0, 0, 0], [128, 64, 32]))
    assert len(parts) == 1 and parts[0].cls == "Part"
    assert np.allclose(parts[0].size, [10, 2.5, 5])          # x, z, y extents in studs
    assert np.allclose(parts[0].center, [5, 1.25, -2.5])


@pytest.mark.parametrize("yaw", [15, 45, 70])
def test_rotated_box(yaw):
    b = rotated_box(2, [10, 20, 30], [96, 32, 64], yaw)
    assert classify(b) == "box_rotated"
    parts = check_exact(b)
    assert len(parts) == 1


def test_chamfered_box_is_box_plus_wedge():
    parts = check_exact(chamfered_box(3, [0, 0, 0], [192, 16, 128], 16))
    assert sorted(p.cls for p in parts) == ["Part", "WedgePart"]


def test_thin_diagonal_strip():
    """The shape where slicing the wrong way once overfilled the brush 30x."""
    check_exact(diagonal_strip(4))


def test_right_wedge():
    b = right_wedge(5)
    assert classify(b) == "wedge"
    parts = check_exact(b)
    assert [p.cls for p in parts] == ["WedgePart"]


def test_round_pillar():
    parts = check_exact(ngon_prism(6, 12))
    assert 1 < len(parts) <= 16


def test_non_prism_becomes_mesh():
    parts = brush_to_parts(mitred_bevel_strip(7))
    assert [p.cls for p in parts] == ["Mesh"]
