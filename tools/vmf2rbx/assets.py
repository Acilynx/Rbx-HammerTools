"""Game file access and material resolution.

Uses the game's gameinfo.txt search paths (via srctools) so DLC/mod folders override the base
game exactly like the engine does, and materials from other games mounted by the mod (Portal
mounts HL2) resolve too.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO

from PIL import Image
from srctools import Keyvalues

TEXTURE_PARAMS = ("$basetexture", "$basetexture2", "$bumpmap", "$normalmap", "$envmapmask",
                  "$phongexponenttexture", "$selfillummask", "$detail", "$blendmodulatetexture")


def open_game(game_dir: str):
    """FileSystemChain for a game folder (the one containing gameinfo.txt).

    Tries srctools' full gameinfo handling first (which can also mount other Steam apps), and
    falls back to reading the SearchPaths block directly, so a copied/partial install or a
    machine without Steam still works."""
    try:
        from srctools.game import Game
        return Game(game_dir).get_filesystem()
    except Exception:
        return _search_paths_filesystem(game_dir)


def _search_paths_filesystem(game_dir: str):
    import glob
    import os
    from srctools.filesys import FileSystemChain, RawFileSystem, VPKFileSystem
    game_dir = os.path.abspath(game_dir)
    root = os.path.dirname(game_dir)
    with open(os.path.join(game_dir, "gameinfo.txt"), encoding="utf-8", errors="replace") as fh:
        kv = Keyvalues.parse(fh)
    paths = kv.find_key("GameInfo").find_key("FileSystem").find_key("SearchPaths")
    chain = FileSystemChain()
    seen = set()
    for entry in paths:
        if "game" not in entry.name.lower():
            continue
        p = entry.value.replace("|gameinfo_path|", game_dir + "/").replace("|all_source_engine_paths|", root + "/")
        p = os.path.normpath(p if os.path.isabs(p) else os.path.join(root, p))
        candidates = []
        if p.endswith("*"):
            base = p[:-1]
            candidates = sorted(glob.glob(os.path.join(base, "*_dir.vpk"))) + sorted(
                d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d))
        elif p.lower().endswith(".vpk"):
            dir_vpk = p[:-4] + "_dir.vpk"
            candidates = [dir_vpk if os.path.exists(dir_vpk) else p]
        else:
            candidates = [p]
        for c in candidates:
            if c in seen or not os.path.exists(c):
                continue
            seen.add(c)
            chain.add_sys(VPKFileSystem(c) if c.lower().endswith(".vpk") else RawFileSystem(c))
    return chain


def _read(fsys, path: str) -> bytes | None:
    try:
        f = fsys[path]
    except (KeyError, FileNotFoundError):
        return None
    try:
        with f.open_bin() as fh:
            return fh.read()
    except (FileNotFoundError, OSError):
        return None          # e.g. listed in a VPK directory whose archive isn't present


@dataclass
class MaterialInfo:
    name: str
    shader: str = ""
    params: dict = field(default_factory=dict)
    patched_from: str = ""
    missing: bool = False

    @property
    def base_texture(self) -> str | None:
        return self.params.get("$basetexture")

    @property
    def keeps_alpha(self) -> bool:
        """Base alpha is transparency only for translucent/alpha-tested materials. Otherwise it's
        a mask for something else ($basealphaenvmapmask, $selfillum, ...) and must be dropped."""
        return self.params.get("$translucent") == "1" or self.params.get("$alphatest") == "1"

    @property
    def self_illum(self) -> bool:
        return self.params.get("$selfillum") == "1"


def resolve_material(fsys, name: str, _depth=0) -> MaterialInfo:
    data = _read(fsys, f"materials/{name}.vmt")
    if data is None:
        return MaterialInfo(name, missing=True)
    kv = Keyvalues.parse(data.decode("utf-8", "replace"))
    root = list(kv)[0]
    params = {c.name.lower(): c.value for c in root if not c.has_children()}
    if root.real_name.lower() == "patch" and _depth < 8:
        inc = params.get("include", "").lower().replace("\\", "/")
        inc = inc.removeprefix("materials/").removesuffix(".vmt")
        base = resolve_material(fsys, inc, _depth + 1)
        merged = dict(base.params)
        for blk in root:
            if blk.has_children() and blk.name.lower() in ("replace", "insert"):
                merged.update({c.name.lower(): c.value for c in blk})
        return MaterialInfo(name, base.shader, merged, inc, base.missing)
    return MaterialInfo(name, root.real_name, params)


def load_base_image(fsys, mat: MaterialInfo, frame: int = 0) -> Image.Image | None:
    """Base texture as RGBA, with alpha applied according to the material's flags."""
    from srctools.vtf import VTF
    tex = mat.base_texture
    if not tex:
        return None
    data = _read(fsys, f"materials/{tex.lower().replace(chr(92), '/')}.vtf")
    if data is None:
        return None
    vtf = VTF.read(BytesIO(data))
    img = vtf.get(frame=min(frame, vtf.frame_count - 1)).to_PIL().convert("RGBA")
    if not mat.keeps_alpha:
        img.putalpha(255)
    return img
