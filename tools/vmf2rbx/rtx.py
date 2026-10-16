"""Portal with RTX (RTX Remix) replacement materials as a PBR source. See docs/portal-rtx.md.

* Remix keys each replaced material by XXH64 (seed 0) of mip 0 of the original texture's
  pixel data, exactly as stored in the VTF. `match_materials` uses that to find the RTX
  replacement for a map's materials, exactly (no perceptual matching).
* The replacement textures live in Remix `.pkg` packages (GBs, on the PC with the game).
  `tools/rtx_extract.py` copies the needed mip of each into a small `.rtxb` file without
  decoding; `read_rtxb` / `decode` here inflate (GDeflate) and decode (BCn) them.
"""
from __future__ import annotations

import ctypes
import io
import json
import os
import re
import struct
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

CHANNELS = ("diffuse", "normalmap", "reflectionroughness", "metallic", "emissive_mask")

# ------------------------------------------------------------------ hashing / matching

# VTF image formats -> (kind, bytes per pixel or per 4x4 block)
_VTF_FMT = {0: ("px", 4), 1: ("px", 4), 2: ("px", 3), 3: ("px", 3), 4: ("px", 2), 5: ("px", 1),
            8: ("px", 1), 11: ("px", 4), 12: ("px", 4), 13: ("blk", 8), 14: ("blk", 16),
            15: ("blk", 16), 16: ("px", 4), 20: ("blk", 8), 24: ("px", 8), 25: ("px", 8)}


def vtf_mip0(data: bytes) -> bytes | None:
    """Raw pixel data of the largest mip, first frame. VTF stores mips smallest first, so the
    largest is at the end of the file."""
    w, h = struct.unpack_from("<HH", data, 16)
    frames = struct.unpack_from("<H", data, 24)[0]
    fmt = struct.unpack_from("<i", data, 52)[0]
    if fmt not in _VTF_FMT:
        return None
    kind, b = _VTF_FMT[fmt]
    size = w * h * b if kind == "px" else max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * b
    start = len(data) - size * frames
    return data[start:start + size]


def remix_hash(vtf_bytes: bytes) -> str | None:
    import xxhash
    m = vtf_mip0(vtf_bytes)
    return None if m is None else xxhash.xxh64_hexdigest(m).upper()


def parse_replacements(usda_text: str) -> dict[str, dict[str, str]]:
    """mat_<HASH> blocks in a Remix replacement layer -> {HASH: {channel: texture path}}."""
    starts = [(m.start(), m.group(1)) for m in re.finditer(r'over "mat_([0-9A-F]{16})"', usda_text)]
    out = {}
    for i, (pos, h) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(usda_text)
        block = usda_text[pos:end]
        tex = {ch: path for ch, path in re.findall(r"asset inputs:(\w+)_texture = @\./([^@]+)@", block)
               if ch in CHANNELS}
        enc = re.search(r"int inputs:encoding = (\d)", block)
        if tex and enc:
            tex["encoding"] = int(enc.group(1))
        out[h] = tex
    return out


def match_materials(fsys, materials, replacements) -> dict[str, dict]:
    """{material: {"hash": H, channel: path, ...}} for the materials Portal RTX replaces."""
    from .assets import _read, resolve_material
    out = {}
    for m in materials:
        info = resolve_material(fsys, m)
        bt = info.base_texture
        if not bt:
            continue
        data = _read(fsys, "materials/" + bt.lower().replace("\\", "/") + ".vtf")
        h = remix_hash(data) if data else None
        if h in replacements and replacements[h]:
            out[m] = {"hash": h, **replacements[h]}
    return out


# ------------------------------------------------------------------ .rtxb reading / decoding

def _load_gdeflate():
    here = os.path.join(os.path.dirname(__file__), "..", "native", "gdeflate")
    for name in (os.environ.get("VMF2RBX_GDEFLATE", ""), "libgdeflate.so", "gdeflate.dll", "libgdeflate.dylib"):
        if not name:
            continue
        p = name if os.path.isabs(name) else os.path.join(here, name)
        if os.path.exists(p):
            lib = ctypes.CDLL(p)
            lib.gdeflate_tiles.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p,
                                           ctypes.c_size_t, ctypes.c_uint32]
            return lib
    raise RuntimeError("GDeflate library not built: run tools/native/gdeflate/build.sh")


_gd = None


def gdeflate(buf: bytes) -> bytes:
    """Inflate an RTX IO GDeflate tile stream. Header: 04 FB 00 00, then a u32 with the tile
    count in bits 4-13 and the last tile's size in bits 14-31; then the tile offsets."""
    global _gd
    if buf[0] != 4 or buf[1] != 0xFB:
        raise ValueError("not a GDeflate stream")
    _gd = _gd or _load_gdeflate()
    w = int.from_bytes(buf[4:8], "little")
    n, last = (w >> 4) & 0x3FF, w >> 14
    size = n * 65536 - (65536 - last if last else 0)
    out = ctypes.create_string_buffer(size)
    if not _gd.gdeflate_tiles(out, size, buf, len(buf), n):
        raise ValueError("GDeflate decompression failed")
    return out.raw


# Vulkan BCn format -> (DXGI format, bytes per 4x4 block)
_VK = {131: (71, 8), 132: (72, 8), 133: (71, 8), 134: (72, 8), 135: (74, 16), 136: (75, 16),
       137: (77, 16), 138: (78, 16), 139: (80, 8), 140: (81, 8), 141: (83, 16), 142: (84, 16),
       143: (95, 16), 144: (96, 16), 145: (98, 16), 146: (99, 16)}


def _bc_size(w, h, bs):
    return max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * bs


def _dds(dxgi: int, w: int, h: int, data: bytes) -> Image.Image:
    hdr = struct.pack("<4s7I44s8I5I5I", b"DDS ", 124, 0x1007 | 0x80000, h, w, len(data), 0, 1, b"\0" * 44,
                      32, 4, int.from_bytes(b"DX10", "little"), 0, 0, 0, 0, 0,
                      0x1000, 0, 0, 0, 0,
                      dxgi, 3, 0, 1, 0)
    return Image.open(io.BytesIO(hdr + data))


@dataclass
class RtxTexture:
    name: str
    image: Image.Image           # RGBA, the extracted mip
    source_size: tuple           # full-resolution size


def decode(entry: dict, data: bytes) -> RtxTexture:
    raw = gdeflate(data) if entry["compression"] else data
    dxgi, bs = _VK[entry["vk_format"]]
    if entry["in_tail"]:
        # the mip tail packs its levels largest first
        lv0 = entry["tail_first_level"]
        off = sum(_bc_size(max(1, entry["width"] >> l), max(1, entry["height"] >> l), bs)
                  for l in range(lv0, lv0 + entry["level"]))
        level = lv0 + entry["level"]
    else:
        off, level = 0, entry["level"]
    w, h = max(1, entry["width"] >> level), max(1, entry["height"] >> level)
    img = _dds(dxgi, w, h, raw[off:off + _bc_size(w, h, bs)])
    img.load()
    return RtxTexture(entry["name"], img.convert("RGBA"), (entry["width"], entry["height"]))


def read_rtxb(path: str) -> dict[str, RtxTexture]:
    with open(path, "rb") as fh:
        b = fh.read()
    if b[:4] != b"RTXB":
        raise ValueError(f"{path}: not an .rtxb file")
    n = struct.unpack_from("<I", b, 4)[0]
    entries = json.loads(b[8:8 + n])
    blob0 = 8 + n
    return {e["name"]: decode(e, b[blob0 + e["offset"]: blob0 + e["offset"] + e["size"]]) for e in entries}


# Remix normal map encodings (AperturePBR_Normal.mdl "encoding")
OCTAHEDRAL, TANGENT_OGL, TANGENT_DX = 0, 1, 2


def normal_components(img: Image.Image, encoding: int = OCTAHEDRAL) -> np.ndarray:
    """Remix normal map -> (H, W, 2) float tangential components (image right, image down).

    Remix's tangent frame is x = +u, y = +v, and v runs down the image. Octahedral maps use
    Remix's hemisphere encoding (packing.slangh signedOctahedralToHemisphereDirection)."""
    a = np.asarray(img.convert("RGBA"), float)[..., :3] / 255 * 2 - 1
    if encoding == OCTAHEDRAL:
        tx, ty = (a[..., 0] + a[..., 1]) / 2, (a[..., 0] - a[..., 1]) / 2
        v = np.stack([tx, ty, 1 - np.abs(tx) - np.abs(ty)], -1)
    elif encoding == TANGENT_OGL:
        v = a * [1, -1, 1]
    else:
        v = a
    v /= np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)
    return v[..., :2]


def encode_components(n: np.ndarray) -> np.ndarray:
    return np.round((np.clip(n, -1, 1) + 1) / 2 * 255).astype(np.uint8)


def gray_map(img: Image.Image, max_size: int = 256, flat_range: int = 6):
    """Single-channel map (red) as uint8, downsized to max_size; an int when it's flat."""
    a = np.asarray(img.convert("RGBA"))[..., 0]
    if int(a.max()) - int(a.min()) < flat_range:
        return int(round(float(a.mean())))
    im = Image.fromarray(a, "L")
    if max(im.size) > max_size:
        k = max_size / max(im.size)
        im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.Resampling.BOX)
    return np.asarray(im)


def norm(name: str) -> str:
    return name.replace("\\", "/").lstrip("./").lower()
