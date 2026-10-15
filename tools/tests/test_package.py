import numpy as np
from PIL import Image

from vmf2rbx.package import read_package, write_package


def test_round_trip(tmp_path):
    img = Image.fromarray((np.random.default_rng(0).random((16, 32, 4)) * 255).astype("uint8"), "RGBA")
    path = tmp_path / "t.htmp"
    write_package(str(path), {"parts": [{"id": 1}]}, {"a/b": img})
    meta, blob = read_package(str(path))
    assert meta["parts"] == [{"id": 1}] and meta["version"] == 1
    m = meta["images"][0]
    assert (m["name"], m["w"], m["h"]) == ("a/b", 32, 16)
    pal = blob[m["off"]: m["off"] + 4 * m["ncol"]]
    idx = blob[m["off"] + 4 * m["ncol"]: m["off"] + 4 * m["ncol"] + 32 * 16]
    assert len(idx) == 512 and max(idx) < m["ncol"] and len(pal) == 4 * m["ncol"]
