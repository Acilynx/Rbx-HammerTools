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


def vtf_mip(data: bytes, level: int = 0) -> bytes | None:
    """Raw pixel data of one mip level (0 = largest), first frame. VTF stores mips smallest
    first, so the largest is at the end of the file."""
    w, h = struct.unpack_from("<HH", data, 16)
    frames = struct.unpack_from("<H", data, 24)[0]
    fmt = struct.unpack_from("<i", data, 52)[0]
    mips = data[56]
    if fmt not in _VTF_FMT or level >= mips:
        return None
    kind, b = _VTF_FMT[fmt]

    def size(l):
        mw, mh = max(1, w >> l), max(1, h >> l)
        return mw * mh * b if kind == "px" else max(1, (mw + 3) // 4) * max(1, (mh + 3) // 4) * b
    end = len(data)
    for l in range(level):
        end -= size(l) * frames
    start = end - size(level) * frames
    return data[start:start + size(level)]


def vtf_mip0(data: bytes) -> bytes | None:
    return vtf_mip(data, 0)


def remix_hashes(vtf_bytes: bytes) -> list[str]:
    """Candidate Remix texture hashes, most likely first. Portal RTX (2022) keys textures by
    XXH64 (seed 0) of mip 0; newer Remix builds (Portal 2 RTX) by XXH3-64 of mip 0, or of mip 1
    for textures the game uploads a level down (measured: 79 + 7 of sp_a1_intro3's textures)."""
    import xxhash
    out = []
    for level in (0, 1):
        m = vtf_mip(vtf_bytes, level)
        if m is None:
            continue
        if level == 0:
            out.append(xxhash.xxh64_hexdigest(m).upper())
        out.append(xxhash.xxh3_64_hexdigest(m).upper())
    return out


def remix_hash(vtf_bytes: bytes) -> str | None:
    h = remix_hashes(vtf_bytes)
    return h[0] if h else None


def _block_end(text: str, open_brace: int) -> int:
    """Index of the brace closing the one at `open_brace` (skipping quoted strings)."""
    depth, k, n = 0, open_brace, len(text)
    while k < n:
        c = text[k]
        if c == '"':
            k = text.index('"', k + 1)
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return k
        k += 1
    return n


def parse_replacements(usda_text: str) -> dict[str, dict]:
    """mat_<HASH> blocks in a Remix replacement layer -> {HASH: {channel: texture path,
    "encoding": normal map encoding, "emissive_intensity": float}}.

    Each block is cut at its own closing brace: mesh and light overrides sit between material
    blocks and carry their own textures (cutting at the next "mat_" instead gave the office
    floor the observation glass's textures)."""
    out = {}
    for m in re.finditer(r'over "mat_([0-9A-F]{16})"', usda_text):
        h = m.group(1)
        brace = usda_text.index("{", m.end())
        block = usda_text[m.start(): _block_end(usda_text, brace) + 1]
        tex = out.setdefault(h, {})
        for ch, path in re.findall(r"asset inputs:(\w+)_texture = @\./([^@]+)@", block):
            if ch in CHANNELS:
                tex[ch] = path
        enc = re.search(r"int inputs:encoding = (\d)", block)
        if enc:
            tex["encoding"] = int(enc.group(1))
        ei = re.search(r"float inputs:emissive_intensity = ([-\d.e]+)", block)
        if ei:
            tex["emissive_intensity"] = float(ei.group(1))
    return {h: t for h, t in out.items()}


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
        for h in (remix_hashes(data) if data else []):
            if h in replacements and replacements[h]:
                out[m] = {"hash": h, **replacements[h]}
                break
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


# ------------------------------------------------------------------ loose DDS files (Remix mods)

# DDS FourCC (pre-DX10 header) -> DXGI format
_FOURCC_DXGI = {b"DXT1": 71, b"DXT3": 74, b"DXT5": 77, b"ATI1": 80, b"BC4U": 80, b"ATI2": 83, b"BC5U": 83}
_DXGI_BLOCK = {71: 8, 72: 8, 74: 16, 75: 16, 77: 16, 78: 16, 80: 8, 81: 8, 83: 16, 84: 16,
               95: 16, 96: 16, 98: 16, 99: 16}


def read_dds(path: str, max_size: int = 512) -> RtxTexture:
    """One mip of a DDS file (the largest no bigger than max_size), as RGBA. Remix mods that
    ship their textures as loose files (Portal 2 RTX mods) use BC7 / BC5 / BC4 with DX10
    headers; old-style FourCC and uncompressed files work too."""
    with open(path, "rb") as fh:
        b = fh.read()
    if b[:4] != b"DDS ":
        raise ValueError(f"{path}: not a DDS file")
    h, w = struct.unpack_from("<II", b, 12)
    mips = max(1, struct.unpack_from("<I", b, 28)[0])
    fourcc = b[84:88]
    if fourcc == b"DX10":
        dxgi, off = struct.unpack_from("<I", b, 128)[0], 148
    else:
        dxgi, off = _FOURCC_DXGI.get(fourcc), 128
    name = os.path.basename(path)
    if dxgi not in _DXGI_BLOCK:                    # uncompressed or unusual: let Pillow read it
        img = Image.open(io.BytesIO(b)).convert("RGBA")
        if max(img.size) > max_size:
            k = max_size / max(img.size)
            img = img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))), Image.Resampling.BOX)
        return RtxTexture(name, img, (w, h))
    bs = _DXGI_BLOCK[dxgi]
    level = 0
    while level < mips - 1 and max(w >> level, h >> level) > max_size:
        off += _bc_size(max(1, w >> level), max(1, h >> level), bs)
        level += 1
    lw, lh = max(1, w >> level), max(1, h >> level)
    img = _dds(dxgi, lw, lh, b[off:off + _bc_size(lw, lh, bs)])
    img.load()
    return RtxTexture(name, img.convert("RGBA"), (w, h))


def read_remix_dir(mod_dir: str, names, max_size: int = 512) -> dict[str, RtxTexture]:
    """{norm(name): texture} for the given usda texture paths, read from a Remix mod folder.
    Matching ignores case (the usda was written on Windows)."""
    index = {}
    for root, _, files in os.walk(mod_dir):
        for f in files:
            full = os.path.join(root, f)
            index[norm(os.path.relpath(full, mod_dir))] = full
    out = {}
    for n in names:
        p = index.get(norm(n))
        if p is not None:
            try:
                out[norm(n)] = read_dds(p, max_size)
            except Exception:                        # pragma: no cover - a broken file loses one map
                pass
    return out
