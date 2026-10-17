"""Command line: python -m vmf2rbx build --vmf MAP.vmf [--bsp MAP.bsp] [--game GAMEDIR] -o out.htmp"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vmf2rbx", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="translate a map into a .htmp package")
    b.add_argument("--vmf", required=True)
    b.add_argument("--bsp", help="compiled map, for lights (texture lights + point/spot)")
    b.add_argument("--game", help="game folder containing gameinfo.txt, for materials/textures")
    b.add_argument("--region", nargs=6, type=float, metavar=("MINX", "MINY", "MINZ", "MAXX", "MAXY", "MAXZ"),
                   help="only brushes touching this box (Source units)")
    b.add_argument("--no-verify", action="store_true", help="skip the per-brush decomposition self-check")
    b.add_argument("--rtx-usda", help="a Remix mod's replacement layer (.usda): PBR maps for the materials it "
                   "replaces. Its textures come from --rtxb, or else from loose .dds files next to it")
    b.add_argument("--rtxb", help="Portal RTX textures pulled by tools/rtx_extract.py")
    b.add_argument("--no-props", action="store_true", help="leave out prop_static/dynamic/physics models")
    b.add_argument("--no-seal", action="store_true",
                   help="no light blockers (thick hidden walls in the void behind the map, against light leaks)")
    b.add_argument("--seal-depth", type=float, default=6.0, help="light blocker thickness in studs (default 6)")
    b.add_argument("-o", "--out", required=True)
    b.add_argument("--report", help="also write the build report as JSON")
    r = sub.add_parser("rtx-list", help="list the Portal RTX textures a map needs (input for tools/rtx_extract.py)")
    r.add_argument("--vmf", required=True)
    r.add_argument("--game", required=True, help="game folder containing gameinfo.txt")
    r.add_argument("--usda", required=True, help="Portal RTX's rtx-remix/mods/gameReadyAssets/portal_rtx_replacements.usda")
    r.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "rtx-list":
        return rtx_list(a)

    from srctools import Keyvalues
    from srctools.vmf import VMF
    from .package import write_package
    from .pipeline import BuildOptions, build

    print("## reading map", file=sys.stderr, flush=True)
    with open(a.vmf, encoding="utf-8", errors="replace") as fh:
        vmf = VMF.parse(Keyvalues.parse(fh))
    fsys = None
    if a.game:
        from .assets import open_game
        fsys = open_game(a.game)
    bsp = open(a.bsp, "rb").read() if a.bsp else None
    region = (a.region[:3], a.region[3:]) if a.region else None
    def progress(msg):
        print(f"## {msg}", file=sys.stderr, flush=True)
    res = build(vmf, fsys, bsp, BuildOptions(region=region, textures=fsys is not None,
                                             lights=bsp is not None, verify=not a.no_verify,
                                             rtx_usda=a.rtx_usda, rtxb=a.rtxb, props=not a.no_props,
                                             seal_leaks=not a.no_seal, seal_depth=a.seal_depth,
                                             progress=progress))
    progress("writing package")
    import os
    res.meta["map"] = os.path.splitext(os.path.basename(a.vmf))[0].removesuffix("_d")
    size = write_package(a.out, res.meta, res.images, res.blobs)
    print(f"wrote {a.out} ({size / 1e6:.2f} MB)")
    print(json.dumps(res.report, indent=1, default=str))
    if a.report:
        with open(a.report, "w") as fh:
            json.dump(res.report, fh, indent=1, default=str)
    return 1 if res.report["problems"] else 0


def rtx_list(a) -> int:
    from srctools import Keyvalues
    from srctools.vmf import VMF
    from .assets import open_game
    from .brushes import load_vmf_brushes
    from .rtx import CHANNELS, match_materials, parse_replacements
    with open(a.vmf, encoding="utf-8", errors="replace") as fh:
        vmf = VMF.parse(Keyvalues.parse(fh))
    used = {s.material for b in load_vmf_brushes(vmf) for s in b.visible_sides}
    from .assets import _read
    from .props import build_props
    fsys = open_game(a.game)
    models, *_ = build_props(vmf, fsys, lambda p: _read(fsys, p))
    used = sorted(used | {sm["material"] for subs in models.values() for sm in subs})
    with open(a.usda, encoding="utf-8", errors="replace") as fh:
        reps = parse_replacements(fh.read())
    matched = match_materials(fsys, used, reps)
    names = sorted({m[ch] for m in matched.values() for ch in CHANNELS if ch in m})
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(f"# Portal RTX textures for {os.path.basename(a.vmf)}: {len(matched)} of {len(used)} materials\n")
        for m in sorted(matched):
            fh.write(f"# {m} -> mat_{matched[m]['hash']}\n")
        fh.write("\n".join(names) + "\n")
    print(f"{len(matched)} of {len(used)} materials have RTX replacements; {len(names)} textures -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
