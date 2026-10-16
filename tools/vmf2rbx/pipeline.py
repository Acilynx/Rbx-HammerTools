"""VMF (+ BSP + game files) -> map package contents."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .brushes import Brush, load_vmf_brushes
from .coords import point_to_roblox
from .geometry import (PartSpec, brush_to_parts, brush_volume, inside_brush, inside_piece,
                       near_surface, piece_volume)
from .textures import texture_part


@dataclass
class BuildOptions:
    region: tuple | None = None        # (min_xyz, max_xyz) in Source units; keep brushes touching it
    textures: bool = True
    lights: bool = True
    verify: bool = True                # volume + point test every decomposed brush
    texlights_as_neon: bool = True     # texture-light faces: Neon + light colour instead of an image
    bake_faces: bool = True            # faces at odd texture angles get their own baked image
    rtx_usda: str | None = None        # Portal RTX replacement layer (portal_rtx_replacements.usda)
    rtxb: str | None = None            # textures pulled by tools/rtx_extract.py
    rtx_albedo: bool = True            # use the RTX albedo instead of the original texture
    props: bool = True                 # prop_static / prop_dynamic / prop_physics as meshes
    overlays: bool = True              # info_overlay decals as thin parts


@dataclass
class BuildResult:
    meta: dict
    images: dict
    report: dict = field(default_factory=dict)
    blobs: dict = field(default_factory=dict)      # raw byte blobs (prop meshes), zstd in the package


def _part_record(b: Brush, p: PartSpec, faces) -> dict:
    return dict(id=b.id, piece=p.piece, cls=p.cls, owner=b.owner, entity=b.entity_name,
                mat=b.dominant_material(), kind=p.kind,
                c=[round(float(x), 6) for x in p.center], s=[round(float(x), 6) for x in p.size],
                r=[round(float(x), 6) for x in p.rot.flatten()],
                faces=[[f.face, f.material, f.swap, f.flip_x, f.flip_y,
                        round(f.studs_per_tile_u, 6), round(f.studs_per_tile_v, 6),
                        round(f.offset_u, 6), round(f.offset_v, 6)] + ([f.image] if f.image else [])
                       for f in faces])


MAX_IMAGE = 1024
EMISSIVE_PER_RTX = 0.15                 # Roblox EmissiveStrength per unit of Remix emissive_intensity
EMISSIVE_MIN, EMISSIVE_MAX = 0.5, 4.0   # (set by eye; Remix signs use 7-20)

VOLUME_WARN = 1e-3   # relative; decompiled VMFs carry float noise (planes ~0.03 deg off parallel)
VOLUME_FAIL = 1e-2


def verify_decomposition(b: Brush, parts: list[PartSpec], rng, samples=2000, warnings=None) -> str | None:
    """None if the parts reproduce the brush exactly (within decompile noise), else a message."""
    if len(parts) < 1 or parts[0].cls == "Placeholder" or parts[0].kind == "box_axis":
        return None
    want = brush_volume(b)
    got = sum(piece_volume(p) for p in parts)
    rel = abs(want - got) / max(1.0, want)
    if rel > VOLUME_FAIL:
        return f"volume {got:.4f} != {want:.4f}"
    if rel > VOLUME_WARN:
        if warnings is not None:
            warnings.append(f"brush {b.id}: volume off by {100 * rel:.2f}% (near-box from decompile noise?)")
        return None
    lo, hi = point_to_roblox(b.verts).min(0), point_to_roblox(b.verts).max(0)
    X = rng.uniform(lo, hi, size=(samples, 3))
    inb = inside_brush(b, X)
    cnt = sum(inside_piece(p, X).astype(int) for p in parts)
    bad = ((inb & (cnt != 1)) | (~inb & (cnt > 0))) & ~near_surface(b, X)
    # seams between pieces carry ~1e-6 stud float noise; allow a vanishing fraction
    if bad.sum() > max(1, samples // 1000):
        return f"{int(bad.sum())}/{samples} sample points misplaced"
    return None


def build(vmf, fsys=None, bsp_bytes: bytes | None = None, opts: BuildOptions = BuildOptions()) -> BuildResult:
    brushes = load_vmf_brushes(vmf)
    if opts.region:
        lo, hi = map(np.asarray, opts.region)
        brushes = [b for b in brushes if np.all(b.verts.max(0) > lo) and np.all(b.verts.min(0) < hi)]
    rng = np.random.default_rng(0)
    rep = Counter()
    problems, warnings = [], []
    built: list[tuple[Brush, list[PartSpec]]] = []
    for b in brushes:
        if b.is_tool_only:
            rep["skipped tool-only brush"] += 1
            continue
        parts = brush_to_parts(b)
        rep[f"brush {parts[0].kind} -> {parts[0].cls if parts[0].cls == 'Placeholder' else str(len(parts)) + ' part(s)'}"] += 1
        if opts.verify:
            err = verify_decomposition(b, parts, rng, warnings=warnings)
            if err:
                problems.append(f"brush {b.id}: {err}")
        built.append((b, parts))

    # --- materials and images
    from . import assets
    used = sorted({s.material for b, _ in built for s in b.visible_sides})

    # --- props (meshes per model; their materials join the texture loading below)
    prop_models, prop_blobs, prop_instances, prop_missing = {}, {}, [], []
    if opts.props and fsys is not None:
        from .props import build_props
        prop_models, prop_blobs, prop_instances, prep, prop_missing = build_props(
            vmf, fsys, lambda p: assets._read(fsys, p), opts.region)
        for k, v in prep.items():
            rep[f"props: {k}"] += v
    prop_mats = sorted({sm["material"] for subs in prop_models.values() for sm in subs})
    overlay_mats = sorted({(e["material"] or "").replace("\\", "/").lower()
                           for e in vmf.entities if e["classname"] == "info_overlay"} - {""}) if opts.overlays else []
    used = sorted(set(used) | set(prop_mats) | set(overlay_mats))
    mats, images, sizes, avg_colors = {}, {}, {}, {}
    if opts.textures and fsys is not None:
        for m in used:
            info = assets.resolve_material(fsys, m)
            mats[m] = info
            img = None if info.missing else assets.load_base_image(fsys, info)
            if img is None:
                rep["material without loadable texture"] += 1
                continue
            sizes[m] = (img.width, img.height)          # texel space: always the original size
            if max(img.size) > MAX_IMAGE:                # EditableImages are at most 1024 px
                k = MAX_IMAGE / max(img.size)
                img = img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))), Image.Resampling.LANCZOS)
            images[m] = img
            avg_colors[m] = [int(round(c)) for c in np.asarray(img.convert("RGB"), float).reshape(-1, 3).mean(0)]

    # --- Portal RTX replacements: PBR maps (and albedo) for the materials it covers
    pbr, pbr_const = {}, {}
    if opts.rtxb and opts.rtx_usda and fsys is not None and images:
        from . import rtx
        with open(opts.rtx_usda, encoding="utf-8", errors="replace") as fh:
            reps = rtx.parse_replacements(fh.read())
        matched = rtx.match_materials(fsys, sorted(images), reps)
        # Portal RTX remodelled some props (new meshes, new UV layouts). Their materials point at
        # newly authored textures (T_Fixture_CubeDropperArms_...) that don't fit the original
        # model's UVs. Textures remastered from the original keep its hash in the file name
        # (EC997E9AC116F8A3.dds, ..._albedo.dds) and keep its layout. Brush faces always keep
        # their UVs in Remix, so this only applies to materials used by props alone.
        brush_mats = {s.material for b, _ in built for s in b.visible_sides}
        for m in list(matched):
            if m in brush_mats:
                continue
            h = matched[m]["hash"]
            paths = [v for k, v in matched[m].items() if k in rtx.CHANNELS]
            if not any(h in p.upper() for p in paths):
                del matched[m]
                rep["prop materials skipped: RTX remodel textures"] += 1
        texs = rtx.read_rtxb(opts.rtxb)
        for m, info in matched.items():
            def get(ch):
                return texs.get(rtx.norm(info[ch])) if ch in info else None
            d = get("diffuse")
            if d is not None and opts.rtx_albedo:
                img = d.image.copy()
                if not mats[m].keeps_alpha:
                    img.putalpha(255)
                images[m] = img                       # sizes[m] stays the original's: texel space
                avg_colors[m] = [int(round(c)) for c in np.asarray(img.convert("RGB"), float).reshape(-1, 3).mean(0)]
            maps, consts = {}, {}
            n = get("normalmap")
            if n is not None:
                maps["n"] = rtx.encode_components(rtx.normal_components(n.image, info.get("encoding", rtx.OCTAHEDRAL)))
            for ch, k in (("reflectionroughness", "r"), ("metallic", "m")):
                t = get(ch)
                if t is not None:
                    g = rtx.gray_map(t.image)
                    if isinstance(g, int):
                        consts[k] = g
                    else:
                        maps[k] = g
            e = get("emissive_mask")
            if e is not None:
                g = rtx.gray_map(e.image.convert("L").convert("RGBA"))
                strength = min(EMISSIVE_MAX, max(EMISSIVE_MIN, info.get("emissive_intensity", 7.0) * EMISSIVE_PER_RTX))
                if isinstance(g, int):
                    if g > 8:
                        consts["e"], consts["es"] = g, strength
                else:
                    maps["e"], consts["es"] = g, strength
            if maps or consts or d is not None:
                pbr[m], pbr_const[m] = maps, dict(consts, rtx=info["hash"])
                rep["materials with Portal RTX maps"] += 1

    # --- glow: Source $selfillum where Portal RTX didn't supply an emissive mask
    for m in list(images):
        info = mats.get(m)
        if info is None or not info.self_illum or "e" in pbr.get(m, {}) or "e" in pbr_const.get(m, {}):
            continue
        mask = assets.load_selfillum_mask(fsys, info)
        if mask is None:
            continue
        from . import rtx as _rtx
        g = _rtx.gray_map(Image.merge("RGBA", (mask, mask, mask, mask)))
        if isinstance(g, int):
            if g <= 8:
                continue
            pbr_const.setdefault(m, {})["e"] = g
        else:
            pbr.setdefault(m, {})["e"] = g
        pbr_const.setdefault(m, {})["es"] = 1.0
        rep["materials glowing ($selfillum)"] += 1

    # --- lights (also tells us which materials are texture lights)
    light_rep = None
    if opts.lights and bsp_bytes:
        from .lights import build_lights, read_world_lights
        ids = {b.id for b, _ in built}
        region = opts.region
        def keep_point(p):
            if not region:
                return True
            src = np.array([p["position"][0], -p["position"][2], p["position"][1]]) * 12.8
            return bool(np.all(src > region[0]) and np.all(src < region[1]))
        all_brushes = load_vmf_brushes(vmf)
        light_rep = build_lights(read_world_lights(bsp_bytes), all_brushes,
                                 keep_brush=lambda b: b.id in ids, keep_point=keep_point)
    texlight_mats = set(light_rep.texlights) if light_rep else set()

    # --- textures per face
    parts_out = []
    baked = {}
    for b, parts in built:
        for p in parts:
            faces = []
            if sizes:
                faces, skipped = texture_part(b, p, sizes, images, baked if opts.bake_faces else None, pbr)
                if opts.texlights_as_neon:
                    faces = [f for f in faces if f.material not in texlight_mats]
                for why, m in skipped:
                    rep[f"face skipped: {why}"] += 1
                rep["textured faces"] += len(faces)
                rep["baked faces"] += sum(1 for f in faces if f.image)
            parts_out.append(_part_record(b, p, faces))

    # --- overlays: thin see-through parts in front of the surfaces they decorate
    if opts.overlays:
        from .overlays import overlay_parts
        ov, _, orep = overlay_parts(vmf, sizes, opts.region)
        for k, v in orep.items():
            rep[k] += v
        for i, (p, ft, info) in enumerate(ov):
            parts_out.append(dict(id=-(i + 1), piece=0, cls="Part", owner="info_overlay", entity=info["name"],
                                  mat=info["material"], kind="overlay",
                                  c=[round(float(x), 6) for x in p.center], s=[round(float(x), 6) for x in p.size],
                                  r=[round(float(x), 6) for x in p.rot.flatten()],
                                  faces=[[ft.face, ft.material, ft.swap, ft.flip_x, ft.flip_y,
                                          round(ft.studs_per_tile_u, 6), round(ft.studs_per_tile_v, 6),
                                          round(ft.offset_u, 6), round(ft.offset_v, 6)]] if ft else []))
    face_imgs = {(f[9] if len(f) > 9 else f[1]) for pr in parts_out for f in pr["faces"]} | set(prop_mats)
    images = {m: img for m, img in images.items() if m in face_imgs}
    for m in list(images):
        for k, arr in pbr.get(m, {}).items():
            images[f"{m}#{k}"] = arr                  # importer: <image key>#n / #r / #m
    images.update({k: img for k, img in baked.items() if k.split("#")[0] in face_imgs})

    spawn = None
    for e in vmf.entities:
        if e["classname"] == "info_player_start":
            spawn = point_to_roblox([float(x) for x in e["origin"].split()]).round(4).tolist()
            break

    meta = dict(
        map=getattr(vmf, "map_name", "") or "",
        studs_per_unit=0.078125,
        spawn=spawn,
        parts=parts_out,
        materials={m: dict(alpha=mats[m].keeps_alpha if m in mats else False,
                           selfillum=mats[m].self_illum if m in mats else False,
                           color=avg_colors.get(m),          # fallback for faces left untextured
                           texlight=light_rep.texlights.get(m) if light_rep else None,
                           glass=mats[m].refract_tint if m in mats else None,   # Refract glass: tint
                           pbr=pbr_const.get(m))     # flat roughness/metalness values, RTX hash
                   for m in used},
        lights=dict(surface=light_rep.surface, points=light_rep.points) if light_rep else dict(surface=[], points=[]),
        models=prop_models,
        props=prop_instances,
    )
    report = dict(counts=dict(sorted(rep.items())), problems=problems, warnings=warnings,
                  parts=Counter(p["cls"] for p in parts_out),
                  images=len(images),
                  variants=len({(f[1], f[2], f[3], f[4]) for pr in parts_out for f in pr["faces"]}),
                  unmatched_light_patches=light_rep.unmatched_patches if light_rep else None,
                  dark_lights_skipped=light_rep.dark_lights if light_rep else None)
    report["missing_models"] = prop_missing
    return BuildResult(meta, images, report, prop_blobs)
