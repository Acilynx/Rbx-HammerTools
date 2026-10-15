"""Map package (.htmp): everything the Studio importer needs, in one file.

Layout:  b"HTMP" | u32 json_length | json (utf-8) | blob
Images are stored palettized (<=256 colours): palette (ncol * RGBA) followed by one index byte
per pixel, at blob offset `off`. The importer builds the per-face orientation variants itself.
"""
from __future__ import annotations

import json
import struct

from PIL import Image

MAGIC = b"HTMP"
VERSION = 1


def palettize(img: Image.Image) -> tuple[bytes, int]:
    q = img.quantize(colors=256, method=Image.Quantize.FASTOCTREE)
    idx = bytes(q.get_flattened_data()) if hasattr(q, "get_flattened_data") else bytes(q.getdata())
    pal = q.getpalette(rawmode="RGBA")[: 4 * (max(idx) + 1)]
    return bytes(pal) + idx, len(pal) // 4


def write_package(path: str, meta: dict, images: dict[str, Image.Image]) -> int:
    blob = bytearray()
    mats = []
    for name, img in images.items():
        data, ncol = palettize(img)
        mats.append(dict(name=name, w=img.width, h=img.height, ncol=ncol, off=len(blob)))
        blob += data
    meta = dict(meta, version=VERSION, images=mats)
    js = json.dumps(meta, separators=(",", ":")).encode()
    out = MAGIC + struct.pack("<I", len(js)) + js + bytes(blob)
    with open(path, "wb") as fh:
        fh.write(out)
    return len(out)


def read_package(path: str) -> tuple[dict, bytes]:
    with open(path, "rb") as fh:
        data = fh.read()
    assert data[:4] == MAGIC, "not a map package"
    (n,) = struct.unpack_from("<I", data, 4)
    return json.loads(data[8:8 + n]), data[8 + n:]
