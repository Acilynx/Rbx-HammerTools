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
