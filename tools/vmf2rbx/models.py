"""Source models (.mdl + .vvd + .dx90.vtx) -> triangle meshes per material.

Reads the rest pose of LOD 0, first model of every body part (the default bodygroup), split by
material, with the skin family a prop asks for. Positions are model space in Source units, as
stored in the .vvd (already in bind pose, so static and resting dynamic props come out right).

Formats (Source SDK studio.h / optimize.h; Portal 1 uses MDL v44, VVD v4, VTX v7):
* .mdl: header with texture names, material directories, skin table and body parts ->
  models -> meshes. A mesh names its material slot and its range of vertices.
* .vvd: the vertices (48 bytes: bone weights, position, normal, uv), optionally reordered per
  LOD by a fixup table.
* .vtx: per LOD, strip groups of triangles indexing into a mesh's vertices.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SubMesh:
    material: str               # material path as resolved, e.g. "models/props/lab_chair/lab_chair"
    positions: np.ndarray       # (n, 3) float, Source model space
    normals: np.ndarray         # (n, 3)
    uvs: np.ndarray             # (n, 2), v runs down the image
    triangles: np.ndarray       # (m, 3) int, Source winding


@dataclass
class ModelData:
    name: str
    textures: list[str]                              # material slot names
    material_dirs: list[str]
    skins: list[list[int]]                           # skin family -> slot remap
    meshes: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=list)
    # (slot, positions, normals, uvs, triangles)
    hull: tuple = ((0, 0, 0), (0, 0, 0))


def _cstr(b: bytes, off: int) -> str:
    end = b.index(b"\0", off)
    return b[off:end].decode("latin-1")


def parse_mdl(mdl: bytes):
    if mdl[:4] != b"IDST":
        raise ValueError("not an MDL")
    version = struct.unpack_from("<i", mdl, 4)[0]
    name = mdl[12:76].split(b"\0")[0].decode("latin-1")
    hull_min = struct.unpack_from("<3f", mdl, 104)
    hull_max = struct.unpack_from("<3f", mdl, 116)
    (tex_n, tex_off, dir_n, dir_off, skin_refs, skin_fams, skin_off,
     bp_n, bp_off) = struct.unpack_from("<9i", mdl, 204)
    textures = [_cstr(mdl, tex_off + 64 * i + struct.unpack_from("<i", mdl, tex_off + 64 * i)[0])
                for i in range(tex_n)]
    dirs = [_cstr(mdl, struct.unpack_from("<i", mdl, dir_off + 4 * i)[0]) for i in range(dir_n)]
    skins = [list(struct.unpack_from(f"<{skin_refs}h", mdl, skin_off + 2 * skin_refs * f)) for f in range(skin_fams)]
    bodyparts = []
    for b in range(bp_n):
        bo = bp_off + 16 * b
        _, nmodels, base, model_idx = struct.unpack_from("<4i", mdl, bo)
        models = []
        for m in range(nmodels):
            mo = bo + model_idx + 148 * m
            nmesh, mesh_idx, nverts, vert_idx = struct.unpack_from("<4i", mdl, mo + 72)
            meshes = []
            for k in range(nmesh):
                me = mo + mesh_idx + 116 * k
                material, _, mnverts, vert_off = struct.unpack_from("<4i", mdl, me)
                meshes.append((material, vert_off, mnverts))
            models.append((vert_idx // 48, nverts, meshes))
        bodyparts.append((base, models))
    return dict(version=version, name=name, textures=textures, dirs=dirs, skins=skins,
                bodyparts=bodyparts, hull=(hull_min, hull_max))


# ------------------------------------------------------------------ rest pose (sequence 0, frame 0)
#
# The engine draws a prop with the bones of its first sequence at frame 0, which isn't always
# the bind pose the .vvd is stored in (many Portal props have a root that the idle animation
# turns 90 degrees). So each vertex is skinned: v' = sum(w_i * BoneToModel_i * PoseToBone_i * v).

ANIM_RAWPOS, ANIM_RAWROT, ANIM_ANIMPOS, ANIM_ANIMROT, ANIM_DELTA, ANIM_RAWROT2 = 1, 2, 4, 8, 16, 32


def _quat_matrix(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _euler_quat(a):
    """Source AngleQuaternion(RadianEuler): x = roll, y = pitch, z = yaw."""
    sr, cr = np.sin(a[0] / 2), np.cos(a[0] / 2)
    sp, cp = np.sin(a[1] / 2), np.cos(a[1] / 2)
    sy, cy = np.sin(a[2] / 2), np.cos(a[2] / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


def _quat48(b, o):
    x, y, zw = struct.unpack_from("<3H", b, o)
    qx, qy, qz = (x - 32768) / 32768, (y - 32768) / 32768, ((zw & 0x7FFF) - 16384) / 16384
    w = np.sqrt(max(0.0, 1 - qx * qx - qy * qy - qz * qz))
    return qx, qy, qz, -w if zw & 0x8000 else w


def _quat64(b, o):
    v = int.from_bytes(b[o:o + 8], "little")
    qx = ((v & 0x1FFFFF) - 1048576) / 1048576.5
    qy = (((v >> 21) & 0x1FFFFF) - 1048576) / 1048576.5
    qz = (((v >> 42) & 0x1FFFFF) - 1048576) / 1048576.5
    w = np.sqrt(max(0.0, 1 - qx * qx - qy * qy - qz * qz))
    return qx, qy, qz, -w if v >> 63 else w


def _first_anim_value(b, ptr_base, offs, scale):
    """Frame-0 values of a compressed (RLE) animation track triple; 0 where there's no track."""
    out = []
    for k in range(3):
        if offs[k] == 0:
            out.append(0.0)
        else:
            out.append(struct.unpack_from("<h", b, ptr_base + offs[k] + 2)[0] * scale[k])
    return np.array(out)


def rest_pose(mdl: bytes, bone_to_model: bool = False) -> list[np.ndarray]:
    """Per bone, the 3x4 skinning matrix (BoneToModel * PoseToBone) for sequence 0, frame 0.
    bone_to_model=True: just BoneToModel (for data stored in bone space, like .phy hulls)."""
    nb, bo = struct.unpack_from("<2i", mdl, 156)
    bones = []
    for i in range(nb):
        b = bo + 216 * i
        parent = struct.unpack_from("<i", mdl, b + 4)[0]
        pos = np.array(struct.unpack_from("<3f", mdl, b + 32))
        quat = struct.unpack_from("<4f", mdl, b + 44)
        rot = np.array(struct.unpack_from("<3f", mdl, b + 60))
        posscale = np.array(struct.unpack_from("<3f", mdl, b + 72))
        rotscale = np.array(struct.unpack_from("<3f", mdl, b + 84))
        p2b = np.array(struct.unpack_from("<12f", mdl, b + 96)).reshape(3, 4)
        bones.append(dict(parent=parent, pos=pos, quat=quat, rot=rot, posscale=posscale,
                          rotscale=rotscale, p2b=p2b))
    # sequence 0 -> its first animation -> per-bone frame 0
    nseq, seq_off = struct.unpack_from("<2i", mdl, 188)
    nanim, anim_off = struct.unpack_from("<2i", mdl, 180)
    if nseq and nanim:
        animindexindex = struct.unpack_from("<i", mdl, seq_off + 60)[0]
        a = struct.unpack_from("<h", mdl, seq_off + animindexindex)[0]
        ad = anim_off + 100 * a
        animblock, animindex = struct.unpack_from("<2i", mdl, ad + 52)
        if animblock == 0 and animindex:
            p = ad + animindex
            while True:
                bone, flags, nxt = struct.unpack_from("<BBh", mdl, p)
                if bone >= nb:
                    break
                bd = bones[bone]
                data = p + 4
                delta = bool(flags & ANIM_DELTA)
                if flags & ANIM_RAWROT2:
                    bd["quat"] = _quat64(mdl, data); rot_size = 8
                elif flags & ANIM_RAWROT:
                    bd["quat"] = _quat48(mdl, data); rot_size = 6
                elif flags & ANIM_ANIMROT:
                    offs = struct.unpack_from("<3h", mdl, data)
                    e = _first_anim_value(mdl, data, offs, bd["rotscale"])
                    bd["quat"] = _euler_quat(e if delta else e + bd["rot"]); rot_size = 6
                else:
                    rot_size = 0
                    if delta:
                        bd["quat"] = (0.0, 0.0, 0.0, 1.0)
                pdata = data + rot_size
                if flags & ANIM_RAWPOS:
                    bd["pos"] = np.frombuffer(mdl, "<f2", count=3, offset=pdata).astype(float)
                elif flags & ANIM_ANIMPOS:
                    offs = struct.unpack_from("<3h", mdl, pdata)
                    v = _first_anim_value(mdl, pdata, offs, bd["posscale"])
                    bd["pos"] = v if delta else v + bd["pos"]
                elif delta:
                    bd["pos"] = np.zeros(3)
                if nxt == 0:
                    break
                p += nxt
    world = []
    for bd in bones:
        m = np.eye(4)
        m[:3, :3] = _quat_matrix(bd["quat"])
        m[:3, 3] = bd["pos"]
        if bd["parent"] >= 0:
            m = world[bd["parent"]] @ m
        world.append(m)
    if bone_to_model:
        return [m[:3, :] for m in world]
    out = []
    for bd, m in zip(bones, world):
        p2b = np.eye(4)
        p2b[:3, :] = bd["p2b"]
        out.append((m @ p2b)[:3, :])
    return out


def bone_names(mdl: bytes) -> list[str]:
    nb, bo = struct.unpack_from("<2i", mdl, 156)
    return [_cstr(mdl, bo + 216 * i + struct.unpack_from("<i", mdl, bo + 216 * i)[0]) for i in range(nb)]


def parse_vvd(vvd: bytes) -> np.ndarray:
    """LOD 0 vertices as a structured array (pos, normal, uv)."""
    if vvd[:4] != b"IDSV":
        raise ValueError("not a VVD")
    nlods = struct.unpack_from("<i", vvd, 12)[0]
    lod_verts = struct.unpack_from("<8i", vvd, 16)
    nfix, fix_off, data_off = struct.unpack_from("<3i", vvd, 48)
    dt = np.dtype([("w", "<3f"), ("b", "u1", 4), ("pos", "<3f"), ("n", "<3f"), ("uv", "<2f")])
    if nfix == 0:
        return np.frombuffer(vvd, dt, count=lod_verts[0], offset=data_off)
    parts = []
    for i in range(nfix):
        lod, src, count = struct.unpack_from("<3i", vvd, fix_off + 12 * i)
        if lod >= 0:                    # fixups for LOD 0 are the ones with lod >= 0
            parts.append(np.frombuffer(vvd, dt, count=count, offset=data_off + 48 * src))
    return np.concatenate(parts)


# VTX strip group / strip record sizes. Models from MDL v49 on (Portal 2, CS:GO) add
# numTopologyIndices + topologyOffset to both (+8 bytes each). The strip size is confirmed on
# Portal 2 data (the old size misreads meshes with several strips); the strip group size follows
# the same change in the engine headers.
VTX_LAYOUTS = {"old": (25, 27), "v49": (33, 35)}


def parse_vtx(vtx: bytes, mdl_version: int = 44):
    """LOD 0 triangles: [bodypart][model][mesh] -> (orig_vertex_ids (k,), triangles (m, 3)).
    Picks the record layout by model version, and falls back to the other one if the data
    doesn't check out (an index outside its strip group's vertices)."""
    first = "v49" if mdl_version >= 49 else "old"
    for name in (first, "old" if first == "v49" else "v49"):
        try:
            return _parse_vtx(vtx, *VTX_LAYOUTS[name])
        except (ValueError, struct.error, IndexError):
            continue
    raise ValueError("VTX layout not recognised")


def _parse_vtx(vtx: bytes, sg_size: int, st_size: int):
    version, = struct.unpack_from("<i", vtx, 0)
    if version != 7:
        raise ValueError(f"unsupported VTX version {version}")
    nbp, bp_off = struct.unpack_from("<2i", vtx, 28)
    out = []
    for b in range(nbp):
        bo = bp_off + 8 * b
        nmodels, model_off = struct.unpack_from("<2i", vtx, bo)
        models = []
        for m in range(nmodels):
            mo = bo + model_off + 8 * m
            nlods, lod_off = struct.unpack_from("<2i", vtx, mo)
            lo = mo + lod_off                               # LOD 0
            nmesh, mesh_off, _ = struct.unpack_from("<2if", vtx, lo)
            meshes = []
            for k in range(nmesh):
                me = lo + mesh_off + 9 * k
                nsg, sg_off = struct.unpack_from("<2i", vtx, me)
                ids, tris = [], []
                base = 0
                for g in range(nsg):
                    so = me + sg_off + sg_size * g
                    nv, v_off, ni, i_off, ns, s_off = struct.unpack_from("<6i", vtx, so)
                    orig = np.array([struct.unpack_from("<H", vtx, so + v_off + 9 * i + 4)[0] for i in range(nv)],
                                    dtype=np.int64)
                    idx = np.frombuffer(vtx, "<u2", count=ni, offset=so + i_off).astype(np.int64)
                    if ni and idx.max() >= nv:
                        raise ValueError("index outside the strip group")
                    for s in range(ns):
                        st = so + s_off + st_size * s
                        sni, si_off, _, _, _, flags = struct.unpack_from("<4ihB", vtx, st)
                        if si_off < 0 or si_off + sni > ni:
                            raise ValueError("strip outside the strip group")
                        seg = idx[si_off:si_off + sni]
                        if flags & 2:                       # triangle strip
                            t = [(seg[i], seg[i + 1], seg[i + 2]) if i % 2 == 0 else (seg[i + 1], seg[i], seg[i + 2])
                                 for i in range(len(seg) - 2)]
                            t = np.array([x for x in t if len({*x}) == 3], dtype=np.int64).reshape(-1, 3)
                        else:
                            t = seg.reshape(-1, 3)
                        tris.append(t + base)
                    ids.append(orig)
                    base += nv
                meshes.append((np.concatenate(ids) if ids else np.zeros(0, np.int64),
                               np.concatenate(tris) if tris else np.zeros((0, 3), np.int64)))
            models.append(meshes)
        out.append(models)
    return out


def load_model(mdl: bytes, vvd: bytes, vtx: bytes, name: str = "", body: int = 0) -> ModelData:
    """`body`: the bodygroup combination (entity "SetBodyGroup"/"body"); 0 = first model of each
    body part."""
    info = parse_mdl(mdl)
    verts = parse_vvd(vvd).copy()
    strips = parse_vtx(vtx, info["version"])
    skin = rest_pose(mdl)
    if skin:
        P = np.zeros((len(verts), 3)); N = np.zeros((len(verts), 3))
        pos, nrm = verts["pos"].astype(float), verts["n"].astype(float)
        for k in range(3):
            w = verts["w"][:, k].astype(float)
            bi = verts["b"][:, k].astype(int)
            active = (k < verts["b"][:, 3]) & (w > 0)
            for b in np.unique(bi[active]):
                sel = active & (bi == b)
                M = skin[min(b, len(skin) - 1)]
                P[sel] += w[sel, None] * (pos[sel] @ M[:, :3].T + M[:, 3])
                N[sel] += w[sel, None] * (nrm[sel] @ M[:, :3].T)
        unweighted = verts["b"][:, 3] == 0
        P[unweighted], N[unweighted] = pos[unweighted], nrm[unweighted]
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
        verts["pos"], verts["n"] = P, N
    md = ModelData(name or info["name"], info["textures"], info["dirs"], info["skins"], hull=info["hull"])
    for b, (base, models) in enumerate(info["bodyparts"]):
        if not models:
            continue
        choice = (body // max(1, base)) % len(models)
        vstart, _, meshes = models[choice]
        for k, (slot, vert_off, _) in enumerate(meshes):
            ids, tris = strips[b][choice][k]
            if len(tris) == 0:
                continue
            v = verts[vstart + vert_off + ids]
            md.meshes.append((slot, v["pos"].astype(float), v["n"].astype(float), v["uv"].astype(float), tris))
    return md


def resolve_slot_material(md: ModelData, slot: int, skin: int, exists) -> str:
    """Material path for a mesh's slot under a skin, searching the model's material dirs the
    way the engine does (first directory with a matching .vmt wins)."""
    fam = md.skins[skin] if 0 <= skin < len(md.skins) else (md.skins[0] if md.skins else None)
    tex = md.textures[fam[slot] if fam else slot]
    tex = tex.replace("\\", "/").lower()
    for d in md.material_dirs:
        cand = (d.replace("\\", "/").lower().strip("/") + "/" + tex).strip("/")
        if exists(cand):
            return cand
    return tex


def submeshes(md: ModelData, skin: int, exists) -> list[SubMesh]:
    """The model's meshes merged per material for one skin."""
    groups: dict[str, list] = {}
    for slot, pos, nrm, uv, tris in md.meshes:
        groups.setdefault(resolve_slot_material(md, slot, skin, exists), []).append((pos, nrm, uv, tris))
    out = []
    for mat, items in groups.items():
        P, N, U, T, base = [], [], [], [], 0
        for pos, nrm, uv, tris in items:
            P.append(pos); N.append(nrm); U.append(uv); T.append(tris + base)
            base += len(pos)
        out.append(SubMesh(mat, np.concatenate(P), np.concatenate(N), np.concatenate(U), np.concatenate(T)))
    return out


def angle_matrix(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Source's AngleMatrix (mathlib): QAngle in degrees -> 3x3 rotation, columns = the
    entity's forward, left, up axes in world space."""
    p, y, r = np.radians([pitch, yaw, roll])
    sp, cp, sy, cy, sr, cr = np.sin(p), np.cos(p), np.sin(y), np.cos(y), np.sin(r), np.cos(r)
    return np.array([
        [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
        [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
        [-sp, sr * cp, cr * cp],
    ])
