"""Pull selected textures out of Portal RTX's Remix packages (mod_*.pkg).

Runs on the PC where Portal with RTX is installed. Plain Python 3.8+, nothing to install.

    python rtx_extract.py LIST [-o OUT] [--rtx PORTALRTX_DIR] [--max-size 512]

LIST is a text file of texture paths, one per line, as the Remix USD names them
(e.g. SubUSDs/textures/T_DimpledMoonPanels_Clean_basecolor.dds); `vmf2rbx rtx-list` writes it.
For each texture this copies ONE mip level (the largest one no bigger than --max-size) into
OUT, still compressed exactly as stored. Nothing is decoded here: decompression (GDeflate) and
BCn decoding happen in vmf2rbx, so this script needs no native libraries.

Package format (dxvk-remix src/dxvk/rtx_render/rtx_asset_package.h): header u32 magic
0xBAADD00D, u32 version 1, u64 dictOffset; at dictOffset u16 assetCount, u16 blobCount,
20-byte asset descriptors, 16-byte blob descriptors, then NUL-separated asset names.

OUT layout: b"RTXB" | u32 json_len | json (list of entries) | blob bytes.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import sys

DEFAULT_RTX = r"C:\Program Files (x86)\Steam\steamapps\common\PortalRTX"


def norm(name: str) -> str:
    return name.replace("\\", "/").lstrip("./").lower()


def read_index(path: str):
    with open(path, "rb") as f:
        magic, version, dict_off = struct.unpack("<IIQ", f.read(16))
        if magic != 0xBAADD00D or version != 1:
            raise ValueError(f"{path}: not a v1 Remix asset package")
        f.seek(dict_off)
        d = f.read()
    na, nb = struct.unpack_from("<HH", d, 0)
    p = 4
    assets = [struct.unpack_from("<HBBHHHHHHHH", d, p + 20 * i) for i in range(na)]
    p += 20 * na
    blobs = []
    for i in range(nb):
        q, size, crc = struct.unpack_from("<QII", d, p + 16 * i)
        blobs.append((q & ((1 << 40) - 1), (q >> 40) & 255, size))
    p += 16 * nb
    names = d[p:].split(b"\0")
    index = {}
    for a in assets:
        index.setdefault(norm(names[a[0]].decode("utf-8", "replace")), a)
    return index, blobs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("list")
    ap.add_argument("-o", "--out", default=None, help="default: LIST with .rtxb extension")
    ap.add_argument("--rtx", default=DEFAULT_RTX, help="Portal with RTX install folder")
    ap.add_argument("--max-size", type=int, default=512)
    a = ap.parse_args(argv)

    want = []
    with open(a.list, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                want.append(norm(line))
    pkgs = sorted(glob.glob(os.path.join(a.rtx, "rtx-remix", "mods", "*", "*.pkg")))
    if not pkgs:
        print(f"no .pkg files under {a.rtx}\\rtx-remix\\mods", file=sys.stderr)
        return 1

    found = {}
    for pkg in pkgs:
        index, blobs = read_index(pkg)
        hits = [n for n in want if n in index and n not in found]
        if not hits:
            continue
        with open(pkg, "rb") as f:
            for n in hits:
                _, typ, fmt, w, h, depth, mips, tail, layers, base, tail_blob = index[n]
                if typ != 2:                       # 2D images only
                    continue
                loose = mips - tail
                level = 0
                while level + 1 < mips and max(w >> level, h >> level) > a.max_size:
                    level += 1
                in_tail = level >= loose
                bi = tail_blob if in_tail else base + level
                off, comp, size = blobs[bi]
                f.seek(off)
                found[n] = (dict(name=n, package=os.path.basename(pkg), vk_format=fmt, width=w, height=h,
                                 mips=mips, tail_mips=tail, level=(level - loose) if in_tail else level,
                                 in_tail=in_tail, tail_first_level=loose, compression=comp, size=size),
                            f.read(size))
        print(f"{os.path.basename(pkg)}: {len(hits)} texture(s)")

    out = a.out or os.path.splitext(a.list)[0] + ".rtxb"
    entries, blob, pos = [], bytearray(), 0
    for n in want:
        if n in found:
            e, data = found[n]
            e["offset"] = pos
            entries.append(e)
            blob += data
            pos += len(data)
    meta = json.dumps(entries).encode()
    with open(out, "wb") as fh:
        fh.write(b"RTXB" + struct.pack("<I", len(meta)) + meta + bytes(blob))
    missing = [n for n in want if n not in found]
    print(f"wrote {out}: {len(entries)} texture(s), {len(blob) / 1e6:.1f} MB")
    for n in missing:
        print("  not found:", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
