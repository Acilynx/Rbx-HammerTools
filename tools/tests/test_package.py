import numpy as np
from PIL import Image

from vmf2rbx.package import read_package, write_package


def test_round_trip(tmp_path):
    img = Image.fromarray((np.random.default_rng(0).random((16, 32, 4)) * 255).astype("uint8"), "RGBA")
    path = tmp_path / "t.htmp"
    write_package(str(path), {"parts": [{"id": 1}]}, {"a/b": img})
    meta, blob = read_package(str(path))
    assert meta["parts"] == [{"id": 1}] and meta["version"] == 2
    m = meta["images"][0]
    assert (m["name"], m["w"], m["h"], m["enc"]) == ("a/b", 32, 16, "palz")
    import zstandard
    raw = zstandard.ZstdDecompressor().decompress(blob[m["off"]: m["off"] + m["len"]])
    pal = raw[: 4 * m["ncol"]]
    idx = raw[4 * m["ncol"]: 4 * m["ncol"] + 32 * 16]
    assert len(idx) == 512 and max(idx) < m["ncol"] and len(pal) == 4 * m["ncol"]


def test_data_maps_are_zstd(tmp_path):
    import zstandard
    n = (np.random.default_rng(1).random((8, 12, 2)) * 255).astype("uint8")
    g = (np.random.default_rng(2).random((4, 4)) * 255).astype("uint8")
    path = tmp_path / "t.htmp"
    write_package(str(path), {}, {"m#n": n, "m#r": g})
    meta, blob = read_package(str(path))
    by = {i["name"]: i for i in meta["images"]}
    assert by["m#n"]["enc"] == "zstd" and by["m#n"]["ch"] == 2 and (by["m#n"]["w"], by["m#n"]["h"]) == (12, 8)
    assert by["m#r"]["ch"] == 1
    for name, arr in (("m#n", n), ("m#r", g)):
        i = by[name]
        raw = zstandard.ZstdDecompressor().decompress(blob[i["off"]: i["off"] + i["len"]])
        assert raw == arr.tobytes()
