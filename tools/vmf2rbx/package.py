"""Map package (.htmp): everything the Studio importer needs, in one file.

Layout:  b"HTMP" | u32 json_length | json (utf-8) | blob

Images (meta["images"], each at blob offset `off`, `len` bytes):
* colour (a PIL image): enc "palz", palettized to <=256 colours: palette (ncol * RGBA), then one
  index byte per pixel, the whole thing zstd-compressed (enc "pal" = the same uncompressed)
* data maps (a uint8 numpy array, H x W x ch): enc "zstd", the raw bytes zstd-compressed
  (Roblox's EncodingService decompresses them). ch 1 = roughness/metalness, ch 2 = normal map
  (right, down) components. Palettizing would band normal maps badly.
The importer builds the per-face orientation variants itself.

Blobs (meta["blobs"]: name, off, len, size): other binary data, zstd-compressed, e.g. prop
meshes (layout in props.py).
"""
from __future__ import annotations

import json
import struct

from PIL import Image

MAGIC = b"HTMP"
VERSION = 2


def palettize(img: Image.Image) -> tuple[bytes, int]:
    q = img.quantize(colors=256, method=Image.Quantize.FASTOCTREE)
    idx = bytes(q.get_flattened_data()) if hasattr(q, "get_flattened_data") else bytes(q.getdata())
    pal = q.getpalette(rawmode="RGBA")[: 4 * (max(idx) + 1)]
    return bytes(pal) + idx, len(pal) // 4


def write_package(path: str, meta: dict, images: dict[str, Image.Image], blobs: dict[str, bytes] | None = None) -> int:
    blob = bytearray()
    mats = []
    zc = None
    for name, img in images.items():
        if isinstance(img, Image.Image):
            import zstandard
            zc = zc or zstandard.ZstdCompressor(level=19)
            data, ncol = palettize(img)
            data = zc.compress(data)
            mats.append(dict(name=name, enc="palz", w=img.width, h=img.height, ncol=ncol, off=len(blob), len=len(data)))
        else:
            import numpy as np
            import zstandard
            zc = zc or zstandard.ZstdCompressor(level=19)
            a = np.ascontiguousarray(img, dtype=np.uint8)
            if a.ndim == 2:
                a = a[..., None]
            data = zc.compress(a.tobytes())
            mats.append(dict(name=name, enc="zstd", w=a.shape[1], h=a.shape[0], ch=a.shape[2], off=len(blob), len=len(data)))
        blob += data
    blob_meta = []
    for name, raw in (blobs or {}).items():
        import zstandard
        zc = zc or zstandard.ZstdCompressor(level=19)
        data = zc.compress(raw)
        blob_meta.append(dict(name=name, off=len(blob), len=len(data), size=len(raw)))
        blob += data
    meta = dict(meta, version=VERSION, images=mats, blobs=blob_meta)
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
