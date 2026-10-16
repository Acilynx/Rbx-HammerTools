"""VMF (+ BSP + game files) -> map package contents."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

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


@dataclass
class BuildResult:
    meta: dict
    images: dict
    report: dict = field(default_factory=dict)


def _part_record(b: Brush, p: PartSpec, faces) -> dict:
    return dict(id=b.id, piece=p.piece, cls=p.cls, owner=b.owner, entity=b.entity_name,
                mat=b.dominant_material(), kind=p.kind,
                c=[round(float(x), 6) for x in p.center], s=[round(float(x), 6) for x in p.size],
                r=[round(float(x), 6) for x in p.rot.flatten()],
                faces=[[f.face, f.material, f.swap, f.flip_x, f.flip_y,
                        round(f.studs_per_tile_u, 6), round(f.studs_per_tile_v, 6),
                        round(f.offset_u, 6), round(f.offset_v, 6)] + ([f.image] if f.image else [])
                       for f in faces])


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
    mats, images, sizes, avg_colors = {}, {}, {}, {}
    if opts.textures and fsys is not None:
        for m in used:
            info = assets.resolve_material(fsys, m)
            mats[m] = info
            img = None if info.missing else assets.load_base_image(fsys, info)
            if img is None:
                rep["material without loadable texture"] += 1
                continue
            images[m] = img
            sizes[m] = (img.width, img.height)
            avg_colors[m] = [int(round(c)) for c in np.asarray(img.convert("RGB"), float).reshape(-1, 3).mean(0)]

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
                faces, skipped = texture_part(b, p, sizes, images, baked if opts.bake_faces else None)
                if opts.texlights_as_neon:
                    faces = [f for f in faces if f.material not in texlight_mats]
                for why, m in skipped:
                    rep[f"face skipped: {why}"] += 1
                rep["textured faces"] += len(faces)
                rep["baked faces"] += sum(1 for f in faces if f.image)
            parts_out.append(_part_record(b, p, faces))
    face_imgs = {(f[9] if len(f) > 9 else f[1]) for pr in parts_out for f in pr["faces"]}
    images = {m: img for m, img in images.items() if m in face_imgs}
    images.update({k: img for k, img in baked.items() if k in face_imgs})

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
                           texlight=light_rep.texlights.get(m) if light_rep else None)
                   for m in used},
        lights=dict(surface=light_rep.surface, points=light_rep.points) if light_rep else dict(surface=[], points=[]),
    )
    report = dict(counts=dict(sorted(rep.items())), problems=problems, warnings=warnings,
                  parts=Counter(p["cls"] for p in parts_out),
                  images=len(images),
                  variants=len({(f[1], f[2], f[3], f[4]) for pr in parts_out for f in pr["faces"]}),
                  unmatched_light_patches=light_rep.unmatched_patches if light_rep else None)
    return BuildResult(meta, images, report)
