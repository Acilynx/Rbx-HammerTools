"""Finding Steam games and the BSP for a VMF (for the converter window)."""
import os

from vmf2rbx import locate

VDF = r'''"libraryfolders"
{
	"0"
	{
		"path"		"C:\\Program Files (x86)\\Steam"
		"apps" { "400" "123" }
	}
	"1"
	{
		"path"		"D:\\Games\\SteamLibrary"
	}
}'''


def test_library_paths_unescape():
    got = locate.library_paths(VDF)
    assert [p.replace("/", "\\") for p in got] == [r"C:\Program Files (x86)\Steam", r"D:\Games\SteamLibrary"]


def test_map_stem_drops_bspsource_suffix():
    assert locate.map_stem("/x/escape_00_d.vmf") == "escape_00"
    assert locate.map_stem("/x/my_map.vmf") == "my_map"


def _game(root, install, mod, maps=()):
    d = os.path.join(root, "steamapps", "common", install, mod)
    os.makedirs(os.path.join(d, "maps"))
    open(os.path.join(d, "gameinfo.txt"), "w").close()
    for m in maps:
        open(os.path.join(d, "maps", m + ".bsp"), "w").close()
    return d


def test_find_games_and_guess_map(tmp_path):
    lib = str(tmp_path / "lib")
    p1 = _game(lib, "Portal", "portal", ["escape_00"])
    p2 = _game(lib, "Portal 2", "portal2", ["sp_a1_intro3"])
    games = locate.find_games([lib])
    assert games == {"Portal": p1, "Portal 2": p2}
    vmf = str(tmp_path / "decomp" / "sp_a1_intro3_d.vmf")
    assert locate.guess_map(vmf, games) == ("Portal 2", os.path.join(p2, "maps", "sp_a1_intro3.bsp"))
    assert locate.guess_map(str(tmp_path / "x_d.vmf"), games) == (None, None)
    assert locate.default_output(vmf) == str(tmp_path / "decomp" / "sp_a1_intro3.htmp")


def test_py_launcher_list_and_python_choice():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    import importlib
    try:
        gui = importlib.import_module("converter_gui")
    except ImportError:                       # no tkinter on this machine
        import pytest
        pytest.skip("tkinter not installed")
    text = (" -V:3.15 *        C:\\Users\\l\\AppData\\Local\\Programs\\Python\\Python315\\python.exe\n"
            " -V:3.13          C:\\Users\\l\\AppData\\Local\\Programs\\Python\\Python313\\python.exe\n"
            " -3.11-64        C:\\Python311\\python.exe\n")
    got = gui.parse_py_list(text)
    assert [v for v, _ in got] == [(3, 15), (3, 13), (3, 11)]
    assert got[1][1].endswith("Python313\\python.exe")
    # the 3.15 the launcher picked has no packages, 3.13 does -> 3.13
    assert gui.choose_python(got, has=lambda p: "313" in p) == (got[1][1], True)
    # nothing has them -> install into a supported version, preferring 3.13, never 3.15
    assert gui.choose_python(got, has=lambda p: False) == (got[1][1], False)
    assert gui.choose_python([((3, 15), "x.exe")], has=lambda p: False) == (None, False)
