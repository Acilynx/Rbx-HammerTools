"""Game file access and material resolution.

Uses the game's gameinfo.txt search paths (via srctools) so DLC/mod folders override the base
game exactly like the engine does, and materials from other games mounted by the mod (Portal
mounts HL2) resolve too.
"""
from __future__ import annotations

import re

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
        chain = Game(game_dir).get_filesystem()
    except Exception:
        chain = _search_paths_filesystem(game_dir)
    return _mount_dir_vpks(chain)


def with_bsp_pakfile(fsys, bsp_bytes: bytes):
    """Put the map's embedded files (the BSP pakfile lump: per-map material copies, cubemaps,
    custom textures) in front of the game's search path, as the engine does."""
    import io
    import zipfile
    from srctools.filesys import FileSystemChain, ZipFileSystem
    from .lights import bsp_lump
    try:
        data, _ = bsp_lump(bsp_bytes, 40)
        zf = zipfile.ZipFile(io.BytesIO(data))
    except Exception:
        return fsys
    out = FileSystemChain()
    out.add_sys(ZipFileSystem("<bsp pakfile>", zf))
    for fs, prefix in fsys.systems:
        out.add_sys(fs, prefix)
    return out


def _mount_dir_vpks(chain):
    """The engine also mounts pak01_dir.vpk inside every plain-folder "Game" path; Portal 2's
    gameinfo lists only "|gameinfo_path|." and relies on that. Loose files keep priority."""
    import os
    from srctools.filesys import FileSystemChain, RawFileSystem, VPKFileSystem
    mounted = {os.path.normcase(os.path.abspath(fs.path)) for fs, _ in chain.systems
               if isinstance(fs, VPKFileSystem)}
    out = FileSystemChain()
    for fs, prefix in chain.systems:
        out.add_sys(fs, prefix)
        if isinstance(fs, RawFileSystem):
            vpk = os.path.join(fs.path, "pak01_dir.vpk")
            if os.path.isfile(vpk) and os.path.normcase(os.path.abspath(vpk)) not in mounted:
                out.add_sys(VPKFileSystem(vpk))
                mounted.add(os.path.normcase(os.path.abspath(vpk)))
    return out


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
        return (self.params.get("$translucent") == "1" or self.params.get("$alphatest") == "1"
                or self.additive)

    @property
    def additive(self) -> bool:
        return self.params.get("$additive") == "1"

    @property
    def self_illum(self) -> bool:
        return self.params.get("$selfillum") == "1"

    @property
    def liquid(self) -> bool:
        return self.shader.lower() == "water"

    @property
    def energy(self) -> bool:
        """Portal 2's SolidEnergy fields (fizzlers): the base texture is a data mask (flow
        masks in its channels), the colour comes from $flow_vortex_color."""
        return self.shader.lower() == "solidenergy"

    @property
    def refract_tint(self) -> list[int] | None:
        """For Refract-shader glass (no base texture; Source draws the scene behind it, tinted
        and distorted): the tint as 0-255 RGB. Water: its fog colour. Energy fields: their vortex
        colour, scaled to full brightness. None for other shaders."""
        if self.energy:
            nums = [float(x) for x in re.findall(r"[-\d.]+", self.params.get("$flow_vortex_color", "[0.6 2 2.5]"))[:3]]
            if len(nums) < 3 or max(nums) <= 0:
                nums = [0.6, 2.0, 2.5]
            return [int(round(255 * c / max(nums))) for c in nums]
        if self.liquid:
            nums = [float(x) for x in re.findall(r"[-\d.]+", self.params.get("$fogcolor", "{ 60 80 90 }"))[:3]]
            return [int(round(min(255, max(0, c)))) for c in nums] if len(nums) == 3 else [60, 80, 90]
        if self.shader.lower() != "refract":
            return None
        nums = [float(x) for x in re.findall(r"[-\d.]+", self.params.get("$refracttint", "1 1 1"))[:3]]
        if len(nums) < 3:
            return [255, 255, 255]
        if "{" in self.params.get("$refracttint", ""):      # {r g b} is 0-255, [r g b] is 0-1
            return [int(round(min(255, max(0, c)))) for c in nums]
        return [int(round(255 * min(1, max(0, c)))) for c in nums]


_KV_TOKEN = re.compile(r'"([^"]*)"?|([{}])|([^\s{}"]+)')


def parse_vmt_lenient(text: str) -> Keyvalues:
    """KeyValues the way Source's own loader reads them, for VMTs srctools rejects (38 of
    Portal's 5999 materials). Source accepts:
      "$envmaptint" .6 .6 .6     -> key, value ".6", then the stray tokens pair up ("Cannot
                                    have multiple names on the same line" in srctools)
      a missing closing brace    -> the file just ends
      /* ... */, stray ";"       -> skipped here (Source would read them as junk keys)
    Unquoted [$X360]-style conditionals after a value are dropped."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    toks = []
    for q, brace, word in _KV_TOKEN.findall(text):
        if brace:
            toks.append(brace)
        elif word:
            if word.startswith("[") or word == ";":
                continue
            toks.append(("s", word))
        else:
            toks.append(("s", q))
    pos = 0

    def block() -> list:
        nonlocal pos
        items = []
        while pos < len(toks):
            t = toks[pos]
            pos += 1
            if t == "}":
                break
            if t == "{":                      # brace with no key: read and drop
                block()
                continue
            key = t[1]
            nxt = toks[pos] if pos < len(toks) else "}"
            if nxt == "{":
                pos += 1
                items.append(Keyvalues(key, block()))
            elif nxt == "}":
                items.append(Keyvalues(key, ""))
            else:
                pos += 1
                items.append(Keyvalues(key, nxt[1]))
        return items

    return Keyvalues.root(*block())


def parse_vmt(text: str) -> Keyvalues:
    try:
        return Keyvalues.parse(text, allow_escapes=False)   # "models\props\box..." isn't \b
    except Exception:
        return parse_vmt_lenient(text)


def resolve_material(fsys, name: str, _depth=0) -> MaterialInfo:
    data = _read(fsys, f"materials/{name}.vmt")
    if data is None:
        return MaterialInfo(name, missing=True)
    kv = parse_vmt(data.decode("utf-8", "replace"))
    blocks = [b for b in kv if b.has_children()]
    if not blocks:
        return MaterialInfo(name, missing=True)
    root = blocks[0]
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


def load_selfillum_mask(fsys, mat: MaterialInfo) -> Image.Image | None:
    """Source $selfillum mask as a greyscale image: $selfillummask's red channel if given,
    else the base texture's alpha (which load_base_image throws away for opaque materials)."""
    from srctools.vtf import VTF
    tex = mat.params.get("$selfillummask") or mat.base_texture
    if not tex:
        return None
    data = _read(fsys, f"materials/{tex.lower().replace(chr(92), '/')}.vtf")
    if data is None:
        return None
    img = VTF.read(BytesIO(data)).get().to_PIL().convert("RGBA")
    return img.getchannel("R" if "$selfillummask" in mat.params else "A")


ADDITIVE_OPACITY = 0.6     # full white additive pixel -> this opacity (set by eye)


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
    if mat.additive:
        # Additive materials add their colour to what's behind them: black adds nothing, bright
        # adds a lot, whatever the alpha says. Roblox can't blend additively, so opacity comes
        # from brightness instead. Using the VTF alpha (or 1 - alpha) got either the vault glass
        # or the box dropper's tube wrong in Studio; brightness gets both right.
        lum = img.convert("L")
        img.putalpha(lum.point(lambda v: round(v * ADDITIVE_OPACITY)))
    elif not mat.keeps_alpha:
        img.putalpha(255)
    return img
