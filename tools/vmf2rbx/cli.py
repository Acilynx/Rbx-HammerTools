"""Command line: python -m vmf2rbx build --vmf MAP.vmf [--bsp MAP.bsp] [--game GAMEDIR] -o out.htmp"""
from __future__ import annotations

import argparse
import json
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
    b.add_argument("-o", "--out", required=True)
    b.add_argument("--report", help="also write the build report as JSON")
    a = ap.parse_args(argv)

    from srctools import Keyvalues
    from srctools.vmf import VMF
    from .package import write_package
    from .pipeline import BuildOptions, build

    with open(a.vmf, encoding="utf-8", errors="replace") as fh:
        vmf = VMF.parse(Keyvalues.parse(fh))
    fsys = None
    if a.game:
        from .assets import open_game
        fsys = open_game(a.game)
    bsp = open(a.bsp, "rb").read() if a.bsp else None
    region = (a.region[:3], a.region[3:]) if a.region else None
    res = build(vmf, fsys, bsp, BuildOptions(region=region, textures=fsys is not None,
                                             lights=bsp is not None, verify=not a.no_verify))
    size = write_package(a.out, res.meta, res.images)
    print(f"wrote {a.out} ({size / 1e6:.2f} MB)")
    print(json.dumps(res.report, indent=1, default=str))
    if a.report:
        with open(a.report, "w") as fh:
            json.dump(res.report, fh, indent=1, default=str)
    return 1 if res.report["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
