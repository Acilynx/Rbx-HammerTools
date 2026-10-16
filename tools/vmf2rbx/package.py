"""Map package (.htmp): everything the Studio importer needs, in one file.

Layout:  b"HTMP" | u32 json_length | json (utf-8) | blob

Images (meta["images"], each at blob offset `off`, `len` bytes):
* colour (a PIL image): enc "palz", palettized to <=256 colours: palette (ncol * RGBA), then one
  index byte per pixel, the whole thing zstd-compressed (enc "pal" = the same uncompressed);
  or enc "rgbaz", raw RGBA zstd-compressed, for see-through images and ones a palette would band
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


PAL_MIN_PSNR = 42.0     # below this (dB) a palette shows banding (e.g. Portal's door gradients)


def palettize(img: Image.Image) -> tuple[bytes, int]:
    """<=256-colour palette (RGBA entries) + one index byte per pixel. Median cut on RGB: on
    Portal's textures it's 4-10 dB better than the octree method."""
    rgba = img.convert("RGBA")
    q = rgba.convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    idx = bytes(q.get_flattened_data()) if hasattr(q, "get_flattened_data") else bytes(q.getdata())
    n = max(idx) + 1
    pal = bytearray()
    rgb = q.getpalette()[: 3 * n]
    for i in range(n):
        pal += bytes(rgb[3 * i: 3 * i + 3]) + b"\xff"
    return bytes(pal) + idx, n


def palette_psnr(img: Image.Image, data: bytes, n: int) -> float:
    import numpy as np
    pal = np.frombuffer(data[: 4 * n], np.uint8).reshape(-1, 4)
    rec = pal[np.frombuffer(data[4 * n:], np.uint8)].reshape(img.height, img.width, 4)[..., :3].astype(float)
    mse = ((rec - np.asarray(img.convert("RGB"), float)) ** 2).mean()
    return 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def write_package(path: str, meta: dict, images: dict[str, Image.Image], blobs: dict[str, bytes] | None = None) -> int:
    blob = bytearray()
    mats = []
    zc = None
    for name, img in images.items():
        if isinstance(img, Image.Image):
            import numpy as np
            import zstandard
            zc = zc or zstandard.ZstdCompressor(level=19)
            rgba = np.asarray(img.convert("RGBA"))
            data, ncol = (None, 0) if rgba[..., 3].min() < 255 else palettize(img)
            if data is not None and palette_psnr(img, data, ncol) >= PAL_MIN_PSNR:
                data = zc.compress(data)
                mats.append(dict(name=name, enc="palz", w=img.width, h=img.height, ncol=ncol, off=len(blob), len=len(data)))
            else:
                # see-through, or a palette would band: full RGBA
                data = zc.compress(np.ascontiguousarray(rgba).tobytes())
                mats.append(dict(name=name, enc="rgbaz", w=img.width, h=img.height, off=len(blob), len=len(data)))
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
