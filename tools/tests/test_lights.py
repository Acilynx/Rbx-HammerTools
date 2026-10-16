"""Light reach: rays against convex brushes."""
import math

import numpy as np

from helpers import box
from vmf2rbx.lights import cone_dirs, ray_reach, solid_planes


def room():
    """A closed 512 x 256 x 128 room made of six wall brushes, 16 units thick."""
    return [box(1, [-16, -16, -16], [528, 272, 0]),      # floor
            box(2, [-16, -16, 128], [528, 272, 144]),    # ceiling
            box(3, [-16, -16, 0], [0, 272, 128]),        # -x
            box(4, [512, -16, 0], [528, 272, 128]),      # +x
            box(5, [0, -16, 0], [512, 0, 128]),          # -y
            box(6, [0, 256, 0], [512, 272, 128])]        # +y


def test_ray_hits_first_wall():
    s = solid_planes(room())
    assert abs(ray_reach([100, 100, 64], [1, 0, 0], s) - 412) < 1e-9
    assert abs(ray_reach([100, 100, 64], [-1, 0, 0], s) - 100) < 1e-9
    assert abs(ray_reach([100, 100, 64], [0, 0, -1], s) - 64) < 1e-9
    d = np.array([1.0, 1.0, 0.0])
    assert abs(ray_reach([100, 100, 64], d, s) - 156 * math.sqrt(2)) < 1e-9   # +y wall first


def test_light_on_a_face_ignores_its_own_brush():
    s = solid_planes(room())
    # a ceiling light: origin on the ceiling's underside, shining down
    assert abs(ray_reach([256, 128, 128], [0, 0, -1], s) - 128) < 1e-9


def test_tool_brushes_and_entities_dont_block():
    walls = room()
    walls.append(box(7, [200, 0, 0], [216, 256, 128], material="tools/toolsclip"))
    walls.append(box(8, [300, 0, 0], [316, 256, 128], owner="func_door"))
    assert abs(ray_reach([100, 100, 64], [1, 0, 0], solid_planes(walls)) - 412) < 1e-9


def test_cone_dirs():
    dirs = cone_dirs([0, 0, -1], math.radians(40))
    assert len(dirs) == 9
    for d in dirs:
        assert abs(np.linalg.norm(d) - 1) < 1e-9
    angles = sorted(math.degrees(math.acos(-d[2])) for d in dirs)
    assert angles[0] < 1e-6 and all(abs(a - 28) < 1e-6 for a in angles[1:])
