"""Props (prop_static / prop_dynamic / prop_physics) -> mesh blobs + placed instances.

Each distinct (model, skin, bodygroup) is converted once into one mesh per material, in Roblox
axes and studs, centred on its own bounding box (Roblox MeshParts are easiest to place that
way). Instances reference those meshes with a CFrame.

Mesh blob layout (zstd-compressed in the package): float32 positions (n*3), float32 normals
(n*3), float32 uvs (n*2, v down the image, same as Roblox), uint32 triangles (m*3) wound
counter-clockwise seen from outside (Source stores them clockwise).
"""
from __future__ import annotations

import numpy as np

from .coords import SPU, point_to_roblox, to_roblox
from .models import angle_matrix, load_model, submeshes

PROP_CLASSES = ("prop_static", "prop_dynamic", "prop_dynamic_override", "prop_physics",
                "prop_physics_override", "prop_physics_multiplayer")

# Entities whose model comes from their class, not a "model" keyvalue (Portal 2)
IMPLICIT_MODELS = {
    "prop_testchamber_door": "models/props/portal_door_combined.mdl",
    "npc_security_camera": "models/props/security_camera.mdl",
    "npc_personality_core": "models/npcs/personality_sphere/personality_sphere.mdl",
    "prop_button": "models/props/switch001.mdl",
    "prop_under_button": "models/props_underground/underground_testchamber_button.mdl",
    "prop_floor_button": "models/props/portal_button.mdl",
    "prop_floor_cube_button": "models/props/box_socket.mdl",
    "prop_floor_ball_button": "models/props/ball_button.mdl",
    "prop_laser_catcher": "models/props/laser_catcher_center.mdl",
    "env_portal_laser": "models/props/laser_emitter_center.mdl",
    "prop_wall_projector": "models/props/wall_emitter.mdl",
    "prop_tractor_beam": "models/props/tractor_beam_emitter.mdl",
}
CUBE_MODELS = {0: "models/props/metal_box.mdl", 1: "models/props/metal_box.mdl", 2: "models/props/reflection_cube.mdl"}

C = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])     # Source -> Roblox axes (x, z, -y)


def entity_model(e) -> str | None:
    if e["model"]:
        return e["model"]
    cls = e["classname"]
    if cls == "prop_weighted_cube":
        return CUBE_MODELS.get(int(float(e["CubeType"] or 0)))
    return IMPLICIT_MODELS.get(cls)


def _floats(s: str, n: int) -> list[float]:
    try:
        v = [float(x) for x in (s or "").split()]
    except ValueError:
        v = []
    return (v + [0.0] * n)[:n]


def mesh_blob(pos, nrm, uv, tris) -> bytes:
    return (np.ascontiguousarray(pos, "<f4").tobytes() + np.ascontiguousarray(nrm, "<f4").tobytes()
            + np.ascontiguousarray(uv, "<f4").tobytes() + np.ascontiguousarray(tris, "<u4").tobytes())


def _bbox_iou(P, Q) -> float:
    lo, hi, qlo, qhi = P.min(0), P.max(0), Q.min(0), Q.max(0)
    inter = np.prod(np.clip(np.minimum(hi, qhi) - np.maximum(lo, qlo), 0, None))
    return float(inter / max(np.prod(hi - lo) + np.prod(qhi - qlo) - inter, 1e-9))


def model_hulls(mdl: bytes, phy: bytes | None, positions: np.ndarray) -> list:
    """The model's .phy convex pieces in the same space as its mesh (Source units). The pieces
    are stored in their bone's space; one-bone props need that bone's rest transform (measured:
    62 of sp_a1_intro3's 76 collision models line up with it, 2 without). Pieces that line up
    neither way are left out."""
    from .models import rest_pose
    from .phy import parse_phy_faces
    if not phy:
        return []
    pieces = parse_phy_faces(phy)
    if not pieces:
        return []
    Q = np.concatenate([p for p, _ in pieces])
    W = rest_pose(mdl, bone_to_model=True)
    cands = [(np.eye(3), np.zeros(3))] + ([(W[0][:, :3], W[0][:, 3])] if W else [])
    best = max(cands, key=lambda m: _bbox_iou(positions, Q @ m[0].T + m[1]))
    if _bbox_iou(positions, Q @ best[0].T + best[1]) < 0.3:
        return []
    return [(p @ best[0].T + best[1], t) for p, t in pieces]


def build_props(vmf, fsys, read, region=None, geom: dict | None = None, hulls: dict | None = None):
    """Returns (models: {key: [submesh records]}, blobs: {name: bytes}, instances: [..], report).
    `geom`, if given, collects each model's submeshes as {key: [(material, positions in the
    model's own space (studs, Roblox axes), triangles counter-clockwise)]} (for shadow stand-ins)."""
    from collections import Counter
    rep = Counter()
    exists = lambda m: read(f"materials/{m}.vmt") is not None
    models, blobs, instances, missing = {}, {}, [], set()
    for e in vmf.entities:
        cls = e["classname"]
        model = entity_model(e)
        if not model or (cls not in PROP_CLASSES and cls not in IMPLICIT_MODELS and cls != "prop_weighted_cube"):
            continue
        origin = np.array(_floats(e["origin"], 3))
        if region is not None:
            lo, hi = map(np.asarray, region)
            if not (np.all(origin > lo) and np.all(origin < hi)):
                continue
        path = model.replace("\\", "/").lower()
        stem = path[:-4] if path.endswith(".mdl") else path
        skin = int(float(e["skin"] or 0))
        body = int(float(e["SetBodyGroup"] or e["body"] or 0))
        key = f"{stem}|{skin}|{body}"
        if key not in models:
            files = [read(stem + ext) for ext in (".mdl", ".vvd", ".dx90.vtx")]
            if any(f is None for f in files):
                missing.add(stem)
                models[key] = None
            else:
                try:
                    md = load_model(*files, name=stem, body=body)
                    if hulls is not None and md.meshes:
                        allp = np.concatenate([m[1] for m in md.meshes])
                        hs = model_hulls(files[0], read(stem + ".phy"), allp)
                        if hs:
                            hulls[key] = [(to_roblox(p) * SPU, t) for p, t in hs]
                    subs = []
                    for i, sm in enumerate(submeshes(md, skin, exists)):
                        P = to_roblox(sm.positions) * SPU
                        N = to_roblox(sm.normals)
                        lo, hi = P.min(0), P.max(0)
                        centre = (lo + hi) / 2
                        name = f"mesh:{key}:{i}"
                        uv = sm.uvs - np.floor(sm.uvs.min(0))      # whole tiles, near 0 (see brushmesh.near_zero)
                        blobs[name] = mesh_blob(P - centre, N, uv, sm.triangles[:, ::-1])
                        if geom is not None:
                            geom.setdefault(key, []).append((sm.material, P, sm.triangles[:, ::-1]))
                        subs.append(dict(blob=name, material=sm.material, n=len(P), m=len(sm.triangles),
                                         centre=[round(float(x), 6) for x in centre]))
                    models[key] = subs
                    rep["models converted"] += 1
                except Exception as ex:                                  # pragma: no cover
                    rep[f"model failed: {type(ex).__name__}"] += 1
                    models[key] = None
        if models[key] is None:
            rep["props with a missing/failed model"] += 1
            continue
        A = angle_matrix(*_floats(e["angles"], 3))
        R = C @ A @ C.T
        solid = (e["solid"] or "6") != "0"
        color = [int(x) for x in _floats(e["rendercolor"] or "255 255 255", 3)]
        instances.append(dict(model=key, cls=cls, name=e["targetname"] or "", solid=solid,
                              c=[round(float(x), 6) for x in point_to_roblox(origin)],
                              r=[round(float(x), 6) for x in R.T.flatten()],
                              color=color, parent=e["parentname"] or ""))
        rep[f"{cls}"] += 1
    rep["missing models"] = len(missing)
    return {k: v for k, v in models.items() if v}, blobs, instances, rep, sorted(missing)
