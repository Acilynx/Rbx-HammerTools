"""Finding things on the user's PC, for the converter window: Steam libraries, the Source games
installed in them, and the compiled map that belongs to a decompiled VMF.

Steam keeps its install path in the registry (HKCU\\Software\\Valve\\Steam, SteamPath) and its
other library folders in <steam>/steamapps/libraryfolders.vdf ("path" entries). A game lives in
<library>/steamapps/common/<install dir>/<mod dir>, which holds gameinfo.txt and maps/.
"""
from __future__ import annotations

import os
import re

# display name -> (Steam install dir, mod dir)
GAMES = {
    "Portal": ("Portal", "portal"),
    "Portal 2": ("Portal 2", "portal2"),
    "Half-Life 2": ("Half-Life 2", "hl2"),
    "Half-Life 2: Episode One": ("Half-Life 2", "episodic"),
    "Half-Life 2: Episode Two": ("Half-Life 2", "ep2"),
}

DEFAULT_STEAM = [r"C:\Program Files (x86)\Steam", r"C:\Program Files\Steam"]


def _registry_steam() -> list[str]:
    try:
        import winreg
    except ImportError:
        return []
    out = []
    for hive, key, name in ((winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam", "SteamPath"),
                            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam", "InstallPath")):
        try:
            with winreg.OpenKey(hive, key) as k:
                out.append(os.path.normpath(winreg.QueryValueEx(k, name)[0]))
        except OSError:
            pass
    return out


def library_paths(vdf_text: str) -> list[str]:
    """"path" entries of a libraryfolders.vdf (backslashes are escaped there)."""
    return [os.path.normpath(p.replace("\\\\", "\\")) for p in re.findall(r'"path"\s+"([^"]+)"', vdf_text)]


def steam_libraries(extra_roots=()) -> list[str]:
    roots = list(extra_roots) + _registry_steam() + DEFAULT_STEAM
    libs = []
    for r in roots:
        if not os.path.isdir(r):
            continue
        cands = [r]
        try:
            with open(os.path.join(r, "steamapps", "libraryfolders.vdf"), encoding="utf-8", errors="replace") as fh:
                cands += library_paths(fh.read())
        except OSError:
            pass
        for c in cands:
            key = os.path.normcase(os.path.abspath(c))
            if os.path.isdir(os.path.join(c, "steamapps")) and key not in {os.path.normcase(os.path.abspath(x)) for x in libs}:
                libs.append(c)
    return libs


def find_games(libraries) -> dict[str, str]:
    """{display name: game folder with gameinfo.txt} for the games in GAMES that are installed."""
    out = {}
    for name, (install, mod) in GAMES.items():
        for lib in libraries:
            d = os.path.join(lib, "steamapps", "common", install, mod)
            if os.path.isfile(os.path.join(d, "gameinfo.txt")):
                out[name] = d
                break
    return out


def map_stem(vmf_path: str) -> str:
    """BSPSource names its output <map>_d.vmf."""
    stem = os.path.splitext(os.path.basename(vmf_path))[0]
    return stem[:-2] if stem.lower().endswith("_d") else stem


def guess_map(vmf_path: str, games: dict[str, str]) -> tuple[str | None, str | None]:
    """(game name, compiled .bsp path) for a VMF: the BSP next to the VMF, else the first game
    whose maps/ folder has a BSP of the same name. Either can be None."""
    stem = map_stem(vmf_path)
    beside = os.path.join(os.path.dirname(vmf_path), stem + ".bsp")
    for name, d in games.items():
        p = os.path.join(d, "maps", stem + ".bsp")
        if os.path.isfile(p):
            return name, p
    return None, (beside if os.path.isfile(beside) else None)


def default_output(vmf_path: str) -> str:
    return os.path.join(os.path.dirname(vmf_path), map_stem(vmf_path) + ".htmp")
