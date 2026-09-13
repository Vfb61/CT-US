"""Dataset orchestration: create the {CT, US, T_CT->US} pseudo-paired set.

Each "case" is loaded once (CT + fused tissue map + liver mesh + smooth scatter
noise) and then produces many independent slices whose probe poses, transducer
parameters, render parameters and deformations are randomised.  Every sample is
exported to a folder with the US image, the resliced CT / segmentation and the
ground-truth rigid transform files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
from scipy import ndimage

from . import anatomy as an
from . import deformation as df
from . import fs_utils as fsu
from . import geometry as geo
from . import io_utils as io
from . import pose as pose_mod
from . import render as render_mod
from . import speckle as sp
from .physic_sim import PhysicsProxy

# ----------------------------------------------------------------------------
# Case context
# ----------------------------------------------------------------------------

class CaseContext:
    """Pre-processed volume context for one CT case."""

    def __init__(self, ct: np.ndarray, tissue: np.ndarray, affine: np.ndarray,
                 crop_box, liver_mask: np.ndarray | None, scatter: np.ndarray,
                 cam: str, structure: np.ndarray | None = None,
                 noise_seed: int | None = None):
        self.ct = ct
        self.tissue = tissue
        self.affine = affine
        self.crop_box = crop_box
        self.liver_mask = liver_mask
        self.scatter = scatter
        self.case = cam
        # 可定位结构掩膜（血管腔/壁 + 肝包膜）。用于拒绝"视野内只有均匀肝实质"
        # 的位姿——那种平面上 CT 灰度与标签都近似常数，不含任何位姿信息。
        self.structure = structure
        # scatter 噪声场的种子。**必须落盘**：它是影响 us.png 的随机量之一，
        # 不记录就无法精确复现前向模型（实测重渲染只有 NCC 0.53）。
        self.noise_seed = noise_seed

    def probe(self) -> pose_mod.LiverSurface:
        if self.liver_mask is None:
            liver_mask = self.tissue == an.T_LIVER
        else:
            liver_mask = self.liver_mask
        return pose_mod.LiverSurface(liver_mask, self.affine)


def make_noise_volume(shape, rng, scale: int = 8, amp: float = 0.25) -> np.ndarray:
    """Smooth multiplicative heterogeneity used to modulate the scatter field."""
    lo = 3
    small = rng.normal(-1, 1, (lo, lo, lo)).astype(np.float32)
    full = ndimage.zoom(small, (shape[0] / lo, shape[1] / lo, shape[2] / lo), order=1)
    full = full[:shape[0], :shape[1], :shape[2]]
    full -= full.mean()
    return np.exp(amp * full).astype(np.float32)


def _iso_resample(ct, labels_liver, labels_vessel, affine, target_spacing):
    """Resample CT + labels to isotropic target_spacing (mm)."""
    pixdim = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    zoom = pixdim / float(target_spacing)
    # cap memory: total output voxels ≤ 64M
    out_voxels = int(np.prod(ct.shape) * np.prod(zoom))
    if out_voxels > 64_000_000:
        scale = (64_000_000 / out_voxels) ** (1.0 / 3)
        zoom = np.clip(zoom, 0.1, None) * scale
        print(f"  [iso] capped output voxels to {64_000_000}, spacing → {pixdim / zoom}")
    ct_r = ndimage.zoom(ct, zoom, order=1).astype(np.float32)
    aff_r = affine.copy()
    aff_r[:3, :3] = affine[:3, :3] / zoom[:, None]
    lr = ndimage.zoom(labels_liver.astype(np.int16), zoom, order=0).astype(np.int16) if labels_liver is not None else None
    lv = ndimage.zoom(labels_vessel.astype(np.int16), zoom, order=0).astype(np.int16) if labels_vessel is not None else None
    return ct_r, lr, lv, aff_r


def classify_liver_from_ct(ct, tissue) -> np.ndarray:
    """Estimate a liver-like mask for cases without liver labels: the largest
    soft-tissue connected component co-located with the CT necrotic range."""
    body = tissue == an.T_BODY_WALL
    lab, n = ndimage.label(body)
    if n == 0:
        return np.zeros(tissue.shape, dtype=bool)
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    big = np.where(sizes == sizes.max())[0][0] + 1
    return lab == big


def load_case(task_dir: str, split: str = "imagesTr", case_name: str = "",
              has_liver_label: bool = True, has_vessel_label: bool = True,
              derive_vessels: bool = True, crop_margin_vox: int = 12,
              rng=None, iso_spacing=None, noise_seed: int | None = None) -> CaseContext:
    """Load and fuse one case into a CaseContext.

    iso_spacing: if set (mm), resample CT+labels to isotropic voxel size.
    noise_seed:  显式指定 scatter 噪声场的种子（**会被落盘**，使前向模型可精确复现）。
                 `rng` 仍用于其它需要随机性的地方。
    """
    task_dir = Path(task_dir)
    img_dir = task_dir / split
    lbl_dir = task_dir / split.replace("images", "labels")
    case_name = case_name or io.list_cases(task_dir, split)[0]
    is_liver_task = case_name.startswith("liver_")
    is_vessel_task = case_name.startswith("hepaticvessel_")

    ct, affine, info = io.load_volume(img_dir / (case_name + ".nii.gz"))

    labels_liver = labels_vessel = None
    lp = lbl_dir / (case_name + ".nii.gz")

    if is_liver_task and lp.exists():
        labels_liver = np.asarray(io.load_volume(lp)[0], dtype=np.int16)
    if is_vessel_task and lp.exists():
        labels_vessel = np.asarray(io.load_volume(lp)[0], dtype=np.int16)

    # isotropic resampling BEFORE tissue building
    if iso_spacing is not None:
        ct, labels_liver, labels_vessel, affine = _iso_resample(
            ct, labels_liver, labels_vessel, affine, iso_spacing)

    tissue = an.build_tissue_map(ct, labels_liver=labels_liver,
                                 labels_vessel=labels_vessel,
                                 derive_vessels=derive_vessels)

    liver_mask = None
    if labels_liver is not None:
        liver_mask = labels_liver == 1
    elif (tissue == an.T_LIVER).sum() > 20000:  # explicit liver region present
        liver_mask = tissue == an.T_LIVER
    else:
        # Task08-style: largest soft-tissue component serves as the modelled organ
        liver_mask = classify_liver_from_ct(ct, tissue)
        tissue[liver_mask] = an.T_LIVER
        if labels_vessel is not None:
            lum_top = labels_vessel == 1
            tissue[lum_top] = an.T_VESSEL
            tissue[an.vessel_wall_rim(lum_top)] = an.T_VESSEL_WALL

    ct_c, liver_c, aff_c, box = io.crop_and_recenter(ct, affine, liver_mask, crop_margin_vox)
    tissue_c = tissue[box]

    # scatter 噪声场的种子必须确定且可复现：优先用显式 noise_seed，
    # 否则从传入的 rng 抽一个并记录下来（旧实现直接从 rng 消耗，种子无法回溯）。
    if noise_seed is None:
        _r = rng if rng is not None else np.random.default_rng()
        noise_seed = int(_r.integers(0, 2 ** 31 - 1))
    noise = make_noise_volume(ct_c.shape, np.random.default_rng(int(noise_seed)))
    scatter = an.scatter_volume(tissue_c, spatial_noise=noise)
    structure = build_structure_mask(tissue_c)

    return CaseContext(ct_c, tissue_c, aff_c, box, liver_c, scatter, case_name,
                       structure=structure, noise_seed=int(noise_seed))


def build_structure_mask(tissue: np.ndarray) -> np.ndarray:
    """「可定位结构」掩膜：血管腔 + 血管壁 + 肝包膜（肝/非肝界面）。

    这些是扫描平面上唯一随位姿**剧烈变化**的内容；其余肝实质在 CT 上几乎均匀
    （HU 近似常数），平面上横移数毫米图像几乎不变 ⇒ 不含位姿信息。
    """
    liver = tissue == an.T_LIVER
    if liver.any():
        ero = ndimage.binary_erosion(liver, structure=ndimage.generate_binary_structure(3, 1),
                                     iterations=1)
        capsule = liver & ~ero
    else:
        capsule = np.zeros(tissue.shape, dtype=bool)
    return (capsule
            | (tissue == an.T_VESSEL)
            | (tissue == an.T_VESSEL_WALL))


def plane_structure_fraction(ctx: CaseContext, probe: geo.Probe, pose: dict,
                             stride: int = 4, deform=None) -> float:
    """扫描平面上「结构像素」占比（0..1），用于位姿筛选（见 build_structure_mask）。"""
    if ctx.structure is None or not ctx.structure.any():
        return 1.0
    world = probe.world_grid(pose["face"], pose["u"], pose["v"], pose["w"], deform=deform)
    sub = np.ascontiguousarray(world[::stride, ::stride])
    if sub.size == 0:
        return 1.0
    sp = probe.copy(nx=sub.shape[0], nz=sub.shape[1])
    s = sp.resample(ctx.structure.astype(np.float32), ctx.affine, sub, cval=0.0, order=0)
    return float((s > 0.5).mean())


# ----------------------------------------------------------------------------
# Sample generation
# ----------------------------------------------------------------------------

def sample_probe_geometry(ctx: CaseContext, rng, param=None) -> tuple[geo.Probe, dict]:
    """Sample probe + pose, accepting only poses that cover the liver well.

    `param["min_structure_frac"]`（默认 0.0 = 历史行为）额外要求扫描平面上
    「可定位结构」（血管腔/壁 + 肝包膜）占比达到阈值。**这是本轮最关键的数据
    侧改动**：肝实质在 CT 上近似均匀，若平面落在实质内部，CT slab 与 US 都不含
    随位姿变化的信息，网络无输入可学。筛选后每个样本都至少经过一些血管/包膜。
    """
    param = param or {}
    kind_pool = param.get("kinds", ["convex", "convex", "linear"])
    liver_surf = ctx.probe()
    min_struct = float(param.get("min_structure_frac", 0.0))
    best = None

    for _ in range(60):
        kind = str(rng.choice(kind_pool))
        freq = float(rng.uniform(param.get("freq_min", 2.5), param.get("freq_max", 5.0)))
        usec = param.get("usec", {})
        if kind == "linear":
            pr = geo.build_probe({
                "kind": "linear",
                "nx": int(usec.get("nx", rng.choice([320, 384, 448]))),
                "nz": int(usec.get("nz", rng.choice([240, 280, 320]))),
                "dx": float(rng.uniform(0.35, 0.7)),
                "dz": float(rng.uniform(0.35, 0.55)),
                "near": float(rng.uniform(1.0, 4.0)),
                "freq": freq,
            })
        else:
            pr = geo.build_probe({
                "kind": "convex",
                "nx": int(usec.get("nx", rng.choice([320, 384, 448]))),
                "nz": int(usec.get("nz", rng.choice([240, 280, 320]))),
                "fov_angle": float(np.deg2rad(rng.uniform(60.0, 85.0))),
                "dz": float(rng.uniform(0.35, 0.55)),
                "near": float(rng.uniform(1.0, 4.0)),
                "radius": float(rng.uniform(45.0, 75.0)),
                "freq": freq,
            })
        pose = pose_mod.sample_probe_pose(liver_surf, rng)
        frac, max_d = pose_mod.plane_liver_coverage(liver_surf, pr, pose)
        frac_min = param.get("frac_min", 0.10)
        if not (frac >= frac_min
                and max_d >= param.get("min_depth_fraction", 0.25) * pr.depth_span):
            continue
        if min_struct <= 0.0:
            return pr, pose
        sf = plane_structure_fraction(ctx, pr, pose)
        if best is None or sf > best[0]:
            best = (sf, pr, pose)
        if sf >= min_struct:
            return pr, pose
    if best is not None:
        # 宁可用「结构最多」的候选，也不要退化成空视野
        return best[1], best[2]
    # last-resort: return the first candidate anyway
    pr = geo.build_probe({"kind": "linear", "nx": 320, "nz": 280, "dz": 0.5,
                          "dx": 0.55, "freq": 3.5})
    pose = pose_mod.sample_probe_pose(ctx.probe(), rng)
    return pr, pose


def resample_planes(ctx: CaseContext, probe: geo.Probe, pose: dict,
                    deform=None) -> dict:
    """Resample CT/tissue/scatter along the scan plane."""
    world = probe.world_grid(pose["face"], pose["u"], pose["v"], pose["w"],
                             deform=deform)
    ct_plane = probe.resample(ctx.ct, ctx.affine, world, cval=-1024.0, order=1)
    tissue_plane = probe.resample(ctx.tissue, ctx.affine, world, cval=an.T_AIR, order=0)
    scatter_plane = probe.resample(ctx.scatter, ctx.affine, world, cval=an.TISSUE_PROPS[an.T_AIR].scatter, order=1)
    return {"world": world, "ct": ct_plane, "tissue": tissue_plane,
            "scatter": scatter_plane}


def plane_affine_true(probe: geo.Probe, pose: dict) -> tuple[np.ndarray, float]:
    """切片数据的**正确** NIfTI affine：(col, row, 0, 1) -> CT 世界坐标(mm)。

    背景：切片数组是 (nx, nz) 的**扫描平面像素网格**，不是体素网格，因此不能把
    `us_to_ct`（US 物理 mm 坐标 -> CT 世界）直接当 affine 用——它既没有像素间距
    缩放，也没有 `(n-1)/2` 居中与 `near` 偏移（旧实现就是这么写的，导致像素(0,0)
    偏 68 mm、远端角点偏 334 mm）。

    本函数返回 (A, max_err_mm)：
      - linear：解析精确。i 方向 = u·dx，j 方向 = w·dz。
      - convex：**扇形网格在物理上共面，但在 (i,j) 坐标下是非线性的**（弧长 ~ r·sin(a)），
        因此任何单一 affine 都只是近似。这里取「(0,0) 处的一阶差分 + 精确原点」，
        保证近场中心邻域精确，`max_err_mm` 报告全图最大偏差。
        **需要精确几何时请用 `world_grid()`（本仓库已把整张网格落盘到
        `geometry.npz`：`dx_mm/dz_mm/nu/nv/kind/near/radius/fov_angle/face/u/v/w`）。**

    A 满足：p_world = A @ [col, row, 0, 1]^T
    """
    face = np.asarray(pose["face"], dtype=np.float64)
    u = np.asarray(pose["u"], dtype=np.float64)
    v = np.asarray(pose["v"], dtype=np.float64)
    w = np.asarray(pose["w"], dtype=np.float64)
    nx, nz = int(probe.nx), int(probe.nz)

    A = np.eye(4, dtype=np.float64)
    if probe.kind != "convex":
        A[:3, 0] = u * probe.dx
        A[:3, 1] = w * probe.dz
        A[:3, 2] = -v                      # 与 us_to_ct 的第三列一致，保持右手系
        A[:3, 3] = face + w * probe.near - u * ((nx - 1) / 2.0) * probe.dx
        return A, 0.0

    world = probe.world_grid(face, u, v, w, deform=None)     # (nx, nz, 3)
    cols = np.arange(nx, dtype=np.float64)
    rows = np.arange(nz, dtype=np.float64)
    cc, rr = np.meshgrid(cols, rows, indexing="ij")
    design = np.stack([cc.ravel(), rr.ravel(), np.ones(cc.size)], axis=1)   # (N,3)
    coeff, *_ = np.linalg.lstsq(design, world.reshape(-1, 3), rcond=None)   # (3,3)
    # 一阶项取「原点处的真实一阶差分」，保证 affine 在像素 (0,0) 及其邻域精确；
    # 最小二乘的系数是全局平均方向，在扇形上会有明显偏差。
    if nx >= 2:
        coeff[0] = world[1, 0] - world[0, 0]
    if nz >= 2:
        coeff[1] = world[0, 1] - world[0, 0]
    coeff[2] = world[0, 0]                       # i=j=0 -> 精确
    A[:3, 0] = coeff[0]
    A[:3, 1] = coeff[1]
    A[:3, 3] = coeff[2]
    # 第三列取平面法向（保持右手系 + det=+1）
    n = np.cross(A[:3, 0], A[:3, 1])
    nn = np.linalg.norm(n)
    n = n / nn if nn > 1e-12 else -v
    if np.dot(n, -v) < 0:
        n = -n
    A[:3, 2] = n
    fitted = (design @ coeff).reshape(nx, nz, 3)
    err = float(np.abs(fitted - world).max())
    return A, err


def make_transform(probe: geo.Probe, pose: dict) -> dict:
    """Ground-truth rigid transforms US(3D, y=z=0 plane) <-> CT world."""
    u = np.asarray(pose["u"], dtype=np.float64)
    v = np.asarray(pose["v"], dtype=np.float64)
    w = np.asarray(pose["w"], dtype=np.float64)
    face = np.asarray(pose["face"], dtype=np.float64)
    us_to_ct = np.eye(4)
    us_to_ct[:3, 0] = u      # x_us : lateral
    us_to_ct[:3, 1] = w      # y_us : depth
    us_to_ct[:3, 2] = -v     # z_us : elevation (negated to keep det=+1)
    us_to_ct[:3, 3] = face
    ct_to_us = io.affine_inverse(us_to_ct)
    slice_affine, slice_affine_err = plane_affine_true(probe, pose)
    return {
        "us_to_ct": us_to_ct,
        "ct_to_us": ct_to_us,
        "slice_affine": slice_affine,
        "slice_affine_err_mm": slice_affine_err,
        "face": face, "u": u, "v": v, "w": w,
        "dx_mm": probe.dx_effective,
        "dz_mm": probe.dz,
        "nu": probe.nx, "nv": probe.nz,
    }


def generate_sample(ctx: CaseContext, index: int, rng, param: dict | None = None,
                    global_params: dict | None = None,
                    compute_reference: bool = True,
                    sid: str | None = None, render_seed: int | None = None) -> dict:
    """Create one {CT, US, T} sample.

    `render_seed`：**本样本渲染用的种子**。给定后渲染只由它决定（与调用顺序无关），
    并被落盘，使 `us.png` 可以被**精确复现**。旧实现把病例级 `rng` 一路传进
    `render_bmode`，于是某个样本的散斑实现取决于它前面生成过多少样本——既不可
    复现，也无法在磁盘上核对。

    `global_params["n_speckle"]`（默认 1）：同一位姿渲染几次**独立散斑实现**。
    审计实测：单次实现与 12 次平均的 `mean|Δ|` 达图像均值的 **56.2%**，而位姿信号
    只有 33.6%（1mm 横移）——噪声是信号的 2–9 倍。给出多次实现后，下游既可以用
    平均图降噪，也可以学到"对散斑实现不变"这件事（旧数据只给一帧，无法学到）。
    """
    param = param or {}
    global_params = global_params or {}
    probe, pose = sample_probe_geometry(ctx, rng, param)
    deform_params = df.random_deform_params(
        probe, rng,
        enabled=not global_params.get("no_deform", False),
        prob=global_params.get("deform_prob"),
        resp_amp_range=global_params.get("resp_amp_range", (0.5, 4.5)))
    deform = df.build_deformation(probe, deform_params, rng)
    planes = resample_planes(ctx, probe, pose, deform)
    render_params = dict(global_params.get("render", {}))

    n_speckle = max(1, int(global_params.get("n_speckle", 1) or 1))
    if render_seed is None:
        render_seed = int(np.random.default_rng().integers(0, 2 ** 31 - 1))
    renders = []
    for k in range(n_speckle):
        rr = np.random.default_rng(int(render_seed) + 7919 * k)
        renders.append(render_mod.render_bmode(planes["ct"], planes["tissue"],
                                               planes["scatter"], probe, render_params,
                                               rr, want_envelope=True))
    out = renders[0]
    # 散斑平均图：直接可用的"降噪观测"。实测（scripts/ab_observation.py）
    # 与固定对数压缩参考叠加时，梯度域真值盆地 prominence 从 0.111 提到 0.606。
    if n_speckle > 1:
        mean_bmode = np.mean([r["bmode"] for r in renders], axis=0)
        out_mean = {"bmode": mean_bmode.astype(np.float32),
                    "uint8": render_mod.sp.to_uint8(mean_bmode)}
    else:
        out_mean = None
    transform = make_transform(probe, pose)
    sample = {
        "index": index,
        "sid": sid or f"sample_{index:06d}",
        "case": ctx.case,
        "probe": probe,
        "pose": pose,
        "deform_params": deform_params,
        "planes": planes,
        "render": out,
        "render_all": renders,
        "render_mean": out_mean,
        "render_seed": int(render_seed),
        "noise_seed": ctx.noise_seed,
        "transform": transform,
        "params": {"render": out["params"], "deform": deform_params,
                   "probe": _probe_json(probe), "pose": _pose_json(pose),
                   "render_seed": int(render_seed),
                   "noise_seed": (int(ctx.noise_seed) if ctx.noise_seed is not None else None),
                   "n_speckle": int(n_speckle)},
    }
    if compute_reference:
        qa = PhysicsProxy(planes["ct"], planes["tissue"], planes["scatter"], probe)
        sample["reference"] = qa.bmode()
    return sample


def _probe_json(probe: geo.Probe) -> dict:
    """探头参数落盘。

    注意 `dx` 只对 linear 有意义（convex 时它是被忽略的占位值）；下游应当使用
    `dx_eff`（横向有效像素间距，= lateral_span / nx），它与 `transform.npz["dx_mm"]`
    完全一致。`dx_eff` 缺失会让下游退回默认值并把平移量纲算错 1.4–4 倍。
    """
    return dict(kind=probe.kind, nx=probe.nx, nz=probe.nz, dx=probe.dx,
                dx_eff=float(probe.dx_effective),
                lateral_span=float(probe.lateral_span),
                depth_span=float(probe.depth_span),
                fov_angle=float(probe.fov_angle), dz=probe.dz,
                near=probe.near, radius=probe.radius, freq=probe.freq)


def _pose_json(pose: dict) -> dict:
    out = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in pose.items()}
    out.pop("kind", None)
    return out


def generate_volume(ctx: CaseContext, rng, n_elev: int = 32,
                    elev_spacing: float = 1.0,
                    param: dict | None = None,
                    global_params: dict | None = None) -> dict:
    """Generate a 3D US volume by sweeping along the elevation direction.

    Returns dict with:
      - us_vol:   (n_elev, nz, nx) float32 [0,1]
      - ct_vol:   (n_elev, nz, nx) float32 HU
      - seg_vol:  (n_elev, nz, nx) int16 组织标签
      - transforms: 逐层 GT 变换（每层一个，含 slice_affine）
      - transform:  中心层变换（向后兼容）
      - vol_affine: 整个堆叠体积的 NIfTI affine（(col,row,elev,1) -> CT 世界 mm）
      - n_elev / elev_spacing / elev_axis / deform_params
      - probe / pose / case
    """
    param = param or {}
    global_params = global_params or {}

    # Sample probe geometry (center slice)
    probe, pose = sample_probe_geometry(ctx, rng, param)

    # Elevation positions (offset from center)
    elev_offsets = (np.arange(n_elev) - (n_elev - 1) / 2) * elev_spacing

    # 形变参数**只抽一次**，整卷共用（旧实现在循环内每层重抽，导致相邻层拿到
    # 互相独立的形变、层间跳变可达数 mm，3D 体积在 elevation 方向不连续）
    deform_params = df.random_deform_params(
        probe, rng,
        enabled=not global_params.get("no_deform", False),
        prob=global_params.get("deform_prob"),
        resp_amp_range=global_params.get("resp_amp_range", (0.5, 4.5)))
    deform = df.build_deformation(probe, deform_params, rng)

    us_slices = []
    ct_slices = []
    tissue_slices = []
    transforms = []

    for i, offset in enumerate(elev_offsets):
        # Shift face along elevation direction
        slice_pose = dict(pose)
        slice_pose["face"] = pose["face"] + pose["v"] * offset

        planes = resample_planes(ctx, probe, slice_pose, deform)

        render_params = dict(global_params.get("render", {}))
        out = render_mod.render_bmode(planes["ct"], planes["tissue"],
                                      planes["scatter"], probe, render_params, rng,
                                      want_envelope=True)

        us_slices.append(out["uint8"].astype(np.float32) / 255.0)
        ct_slices.append(planes["ct"].astype(np.float32))
        tissue_slices.append(planes["tissue"].astype(np.int16))
        transforms.append(make_transform(probe, slice_pose))

    # Stack into 3D volumes: (n_elev, nz, nx)
    us_vol = np.stack(us_slices, axis=0)
    ct_vol = np.stack(ct_slices, axis=0)
    seg_vol = np.stack(tissue_slices, axis=0)

    # 体积级变换：elevation 轴 = v，步长 elev_spacing，居中
    vol_affine, vol_affine_err = plane_affine_true(
        probe, dict(pose, face=pose["face"] - pose["v"] * (n_elev - 1) / 2.0 * elev_spacing))
    # 把「层」方向从平面法向换成真实的 elevation 轴 v
    vol_affine[:3, 2] = np.asarray(pose["v"], dtype=np.float64) * float(elev_spacing)

    return {
        "us_vol": us_vol,
        "ct_vol": ct_vol,
        "seg_vol": seg_vol,
        "n_elev": int(n_elev),
        "elev_spacing": float(elev_spacing),
        "elev_axis": np.asarray(pose["v"], dtype=np.float64),
        "vol_affine": vol_affine,
        "vol_affine_err_mm": vol_affine_err,
        "transforms": transforms,          # 逐层 GT（旧实现只给中心层）
        "deform_params": deform_params,    # 整卷共用
        "transform": transforms[len(transforms) // 2],   # 兼容：中心层变换
        "probe": probe,
        "pose": pose,
        "case": ctx.case,
    }


# ----------------------------------------------------------------------------
# Writer / index
# ----------------------------------------------------------------------------

class DatasetWriter:
    """写出一个数据集分片（label 级目录）。

    与旧实现的差别（修 bug）：
      * 所有落盘都是**原子写**（tmp + os.replace），避免并发下读到半个文件；
      * `volumes/` 的写入用**跨进程文件锁**串行化（旧实现只有进程内 set，
        多进程时会重复写同一个体积文件）；
      * 支持 `skip_existing` 做**幂等重跑**；
      * 索引不再用 `open(...,"a")` 追加（多进程无锁追加会产生陈旧重复行），
        而是在 `finish()` 时**按 sid 去重后原子重写**。
    """

    def __init__(self, out_dir: str, name: str = "ct2us", skip_existing: bool = False):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.vol_dir = self.out_dir / "volumes"
        self.vol_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.out_dir / f"{name}_index.jsonl"
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.skip_existing = bool(skip_existing)
        self._written_volumes: set[str] = set()
        self._rows: list[dict] = []
        try:
            self._rows.extend(fsu.read_index_jsonl(self.index_path))
        except Exception:
            pass

    def ensure_volume(self, ctx: CaseContext) -> str:
        """Persist the case CT crop + tissue map once per case, return path."""
        rel = f"volumes/{ctx.case}_ct.nii.gz"
        path = self.out_dir / rel
        seg_path = self.out_dir / f"volumes/{ctx.case}_seg.nii.gz"
        if rel in self._written_volumes:
            return rel
        if path.exists() and seg_path.exists():
            self._written_volumes.add(rel)
            return rel
        # 跨进程串行化：同一 case 可能被不同 worker 同时处理
        with fsu.file_lock(path):
            if not (path.exists() and seg_path.exists()):
                io.save_nifti(path, ctx.ct, ctx.affine, like=None)
                io.save_nifti(seg_path, ctx.tissue.astype(np.int16), ctx.affine, like=None)
        self._written_volumes.add(rel)
        return rel

    def sample_dir(self, sid: str) -> Path:
        return self.out_dir / sid

    def is_complete(self, sid: str) -> bool:
        """样本目录是否已完整（用于幂等重跑）。"""
        d = self.sample_dir(sid)
        return (d / "params.json").exists() and (d / "us.png").exists()

    def finish(self) -> Path:
        """按 sid 去重后原子重写索引，返回索引路径。"""
        if self._rows:
            fsu.write_index_jsonl(self.index_path, self._rows)
        elif not self.index_path.exists():
            fsu.atomic_write_text(self.index_path, "")
        return self.index_path

    def write(self, ctx: CaseContext, sample: dict, split: str = "train") -> dict:
        sid = sample.get("sid", f"sample_{sample['index']:06d}")
        d = self.sample_dir(sid)
        if self.skip_existing and self.is_complete(sid):
            meta = json.loads((d / "params.json").read_text(encoding="utf-8"))
            self._rows.append(meta)
            return meta

        d.mkdir(parents=True, exist_ok=True)
        tr = sample["transform"]
        # 切片用**平面 affine**（不是 us_to_ct）
        slice_affine = tr.get("slice_affine", tr["us_to_ct"])
        io.save_nifti(d / "ct_slice.nii.gz", sample["planes"]["ct"], slice_affine, like=None)
        io.save_nifti(d / "seg_slice.nii.gz", sample["planes"]["tissue"].astype(np.int16),
                      slice_affine, like=None)
        np.savez_compressed(
            d / "transform.npz",
            us_to_ct=tr["us_to_ct"], ct_to_us=tr["ct_to_us"],
            slice_affine=slice_affine,
            face=tr["face"], u=tr["u"], v=tr["v"], w=tr["w"],
            dx_mm=tr["dx_mm"], dz_mm=tr["dz_mm"],
            nu=np.int32(tr.get("nu", 0)), nv=np.int32(tr.get("nv", 0)),
            kind=str(sample["probe"].kind),
            near=np.float64(sample["probe"].near),
            radius=np.float64(sample["probe"].radius),
            fov_angle=np.float64(sample["probe"].fov_angle),
        )
        fsu.atomic_write_bytes(d / "us.png", _png_bytes(sample["render"]["uint8"]))
        # 同一姿态的多次**独立散斑实现**（`us_01.png` …）与它们的平均图。
        # 单帧观测的散斑噪声是位姿信号的 2–9 倍；给出多帧后下游才能：
        #   1) 用 `us_mean.png` 直接降噪；
        #   2) 学到"对散斑实现不变"的度量（旧数据只给一帧，学不到）。
        files = {"us": "us.png"}
        allr = sample.get("render_all") or []
        for k, r in enumerate(allr[1:], start=1):
            name = f"us_{k:02d}.png"
            fsu.atomic_write_bytes(d / name, _png_bytes(r["uint8"]))
            files[f"us_{k:02d}"] = name
        if sample.get("render_mean") is not None:
            fsu.atomic_write_bytes(d / "us_mean.png",
                                   _png_bytes(sample["render_mean"]["uint8"]))
            files["us_mean"] = "us_mean.png"
        if "reference" in sample:
            fsu.atomic_write_bytes(d / "reference_us.png", _png_bytes(sample["reference"]["uint8"]))
            files["reference_us"] = "reference_us.png"

        # 说明：整张扫描平面世界网格（convex 的精确几何）**不落盘**——它由
        # `params.probe + params.pose` 完全确定，可用
        # `ct2us.geometry.world_grid_from_meta(probe_json, pose_json)` 即时重建
        # （落盘约 1MB/样本，会让数据集体积翻倍）。

        volume_rel = self.ensure_volume(ctx)
        meta = {
            "sid": sid, "split": split, "case": sample.get("case", ""),
            "volume": volume_rel, "params": sample["params"],
            "render_seed": sample.get("render_seed"),
            "noise_seed": sample.get("noise_seed"),
            "n_speckle": len(sample.get("render_all") or [1]),
            "files": dict(files, ct_slice="ct_slice.nii.gz",
                          seg_slice="seg_slice.nii.gz",
                          transform="transform.npz"),
        }
        fsu.atomic_write_json(d / "params.json", meta)
        self._rows.append(meta)
        return meta


def _png_bytes(arr: np.ndarray) -> bytes:
    from PIL import Image
    import io as _io
    buf = _io.BytesIO()
    Image.fromarray(np.asarray(arr, dtype=np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()