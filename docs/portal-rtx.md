# Portal RTX materials (PBR source for Portal 1 maps)

Portal with RTX ships RTX Remix replacement materials for Portal 1's textures: albedo,
normal, roughness, metallic, emissive and sometimes height maps. For Portal 1 maps they are a
better PBR source than guessing maps from the original VTFs.

## How a Remix material is keyed

`rtx-remix/mods/gameReadyAssets/portal_rtx_replacements.usda` has one
`over "mat_<HASH>"` block per replaced material. The hash is the **XXH64 (seed 0) of mip 0
of the original VTF's pixel data**, exactly as stored in the VTF (DXT blocks for compressed
textures). Checked on Portal's own `portal_pak_dir.vpk`: 153 of the 318 replaced materials
match a VTF this way with no false hits; the rest come from HL2 content and models.

So matching is exact, not perceptual: hash the base texture's top mip, look up `mat_<HASH>`.
On testchmb_a_00 that covers 33 of the map's 53 materials (all the concrete panels, floors,
ceilings, metal walls, signage and lights; not the dev/tool-ish ones, the clock digits or
materials whose base texture lives in HL2's VPKs, which aren't staged yet).

Inside a block, textures are `asset inputs:<channel>_texture = @./SubUSDs/textures/X.dds@`
with channels `diffuse`, `normalmap` (octahedral or tangent-space, see `inputs:encoding`),
`reflectionroughness`, `metallic`, `emissive_mask`, `height`, `transmittance`. A replacement
keeps the original's UVs, so the replacement image tiles exactly like the VTF it replaces
(only the resolution differs).

## Where the images are: Remix `.pkg` packages

The `.dds` paths don't exist as files. They are inside `mod_00.pkg` … `mod_12.pkg`
(~13 GB total) in dxvk-remix's asset package format (`src/dxvk/rtx_render/rtx_asset_package.h`):

* header: `u32 magic 0xBAADD00D`, `u32 version 1`, `u64 dictOffset`
* at `dictOffset`: `u16 assetCount`, `u16 blobCount`, then `assetCount` × 20-byte asset
  descriptors (`u16 nameIdx, u8 type, u8 vkFormat, u16 width, u16 height, u16 depth,
  u16 numMips, u16 numTailMips, u16 arraySize, u16 baseBlobIdx, u16 tailBlobIdx`), then
  `blobCount` × 16-byte blob descriptors (`u64: offset:40 | compression:8 | flags:8`,
  `u32 size`, `u32 crc32`), then NUL-separated names (`SubUSDs\textures\X.dds`)
* loose mips are blobs `baseBlobIdx + level`; the small mips are packed into `tailBlobIdx`
* formats are Vulkan BCn (`145` = BC7, `135` = BC2, `139` = BC4, `141` = BC5, …)
* blobs with `compression != 0` are **GDeflate** tile streams (RTX IO): 8-byte header
  (`04 FB 00 00`, then a u32 with the tile count in bits 4–13 and the last tile's size in
  bits 14–31), `numTiles` u32 tile offsets (entry 0 = size of the last tile), then tiles that
  each inflate to 64 KiB with `libdeflate_gdeflate_decompress` (NVIDIA's libdeflate fork,
  as used by Microsoft's DirectStorage GDeflate)

Decoding a 2048² BC7 albedo at mip 2 (512²) is a few hundred KB of GDeflate and works.

## What's still open

* Most packages are larger than what can be copied to a cloud session, so the extraction has
  to run on the PC with the game installed. That needs a GDeflate decoder that runs there
  (a pure-Python one, or a small prebuilt library).
* The Roblox side: `Texture` has `NormalMapContent`, `RoughnessMapContent`,
  `MetalnessMapContent` and an emissive mask. Remix normals are often octahedral and
  OpenGL-handed (`_OTH`), so they need converting to Roblox's tangent-space convention.
