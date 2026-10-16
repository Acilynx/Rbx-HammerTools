# Rbx-HammerTools

Tools for building Roblox maps the Source/Hammer way.

| Part | What it is | Where it runs |
|---|---|---|
| `tools/vmf2rbx` | Translator: Hammer map (VMF + compiled BSP) → a map package | Your PC (Python) |
| `plugin/` | Roblox Studio side: the importer that builds a map package into the place (to be merged with the Hammer Tools building plugin) | Roblox Studio |

Scale throughout: **128 Hammer units = 10 studs** (`1 unit = 0.078125 studs`).

## What the translator does

1. **Geometry.** Every brush becomes Roblox parts:
   - boxes (axis-aligned or rotated) → one `Part`
   - prisms (chamfered walls, wedges, round pillars) → an exact split into `Part`s and `WedgePart`s
   - anything else → a placeholder for now (CSG fallback planned)
2. **Textures.** Every visible face gets a `Texture` whose tiling and offset reproduce Hammer's
   texture alignment exactly. Each face uses one of eight orientations of its image
   (rotation/mirroring is baked into the image, because `Texture.Rotation` and `UVScale`
   can't do it correctly). Faces whose texture runs at an angle the tiling can't express get
   their own baked image instead (see `docs/texture-rules.md`).
3. **Lights.** Texture lights are read back from the compiled BSP (VRAD bakes them in as
   `emit_surface` world lights) and become `SurfaceLight`s; point/spot lights use VRAD's own
   intensity and falloff. The importer sets up indoor lighting (no sun/sky light, dim
   ambient) and gives textures a flat roughness map so surfaces are matte.

Coordinates: Source is Z-up, Roblox is Y-up. The mapping is `(x, y, z) → (x, z, −y)`.
Swapping Y and Z alone would mirror the map.

## No game content in this repo

Maps, textures and other assets come from **your own game install** and are generated
locally into `out/`. The `.gitignore` blocks VMF/BSP/VPK/VTF/VMT files, extracted images
and built packages. Tests use synthetic brushes only.

## Usage (PC side)

The easy way: double-click **`Convert Map.bat`**. It opens a window where you pick the
decompiled `.vmf`; it finds the game folder (from your Steam libraries) and the matching
`.bsp` itself, and offers to install the Python packages it needs on first run. Needs Python 3
from python.org (which includes tkinter).

From the command line:

```
pip install -e tools            # or: pip install -r tools/requirements.txt, then run from tools/
python -m vmf2rbx build ^
    --vmf path/to/map.vmf --bsp path/to/map.bsp ^
    --game "C:/Program Files (x86)/Steam/steamapps/common/Portal/portal" ^
    -o out/map.htmp
```

* `--vmf` is the map source (for a shipped map, decompile the BSP with BSPSource)
* `--bsp` is optional; it supplies the lights
* `--game` is the folder with `gameinfo.txt`; materials resolve through its search paths
* `--region MINX MINY MINZ MAXX MAXY MAXZ` (Source units) imports only part of a map
* every decomposed brush is self-checked (volume + point sampling); failures are listed under
  `problems` in the printed report

For Portal 1 maps with Portal with RTX installed, the build can also use its PBR materials;
see `docs/portal-rtx.md`.

Then in Studio: install the importer (see `plugin/ImporterPlugin.server.lua`), click
**Import map** and pick `out/map.htmp`.

Run the tests with `cd tools && python -m pytest`.

## Status

Proven by hand on Portal's `testchmb_a_00` (relaxation vault area): geometry, texture
alignment (verified against Source seam positions) and texture lights. Not yet done:
overlays, props (MDL), entities/I-O (waits on the game engine layer), CSG fallback,
publish mode (Open Cloud uploads).
