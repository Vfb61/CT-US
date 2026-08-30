#!/usr/bin/env python
"""CT-US registration: coarse pose regression + differentiable LNCC refinement.

The synthetic dataset provides the ground-truth rigid transform T_CT->US for
every US slice.  The pipeline is two-stage:

  1. Coarse stage: 3D-2D pose regression network
       CT volume (3D crop) -> 3D encoder
       US slice (2D)       -> 2D encoder
       fusion -> MLP -> 6-DoF pose (translation + axis-angle rotation)
     supervised by (a) grouped translation/rotation MSE with normalised
     translation scale, (b) a modality-consistent label objective (the CT
     tissue-label volume is resliced at the predicted pose with the true probe
     fan geometry and aligned with the US segmentation head), (c) a direct
     tissue-class segmentation auxiliary head against the stored seg_slice,
     (d) an angle hinge penalty.

  2. Fine stage (inference / evaluation): differentiable label-consistency
     refinement.  A pose candidate is refined by gradient descent of the
     resliced-label vs US-segmentation objective, coarse-to-fine.  Confidence
     derives from the best agreement and the spread of the refined candidates.

The reslice is *fan-accurate*: it reproduces the exact probe world-grid used
during generation (linear AND convex), so for a rigid sample the ground-truth
transform reslices to the stored CT/seg slices pixel-exactly, giving the
refiner a well-conditioned valley.  Sparse evaluation tools:

  python scripts/evaluate_registration.py --index ... --oracle_seg

Evaluation reports TRE (mean/median, %<5mm / %<10mm), capture-range of the
refiner, and per-frame latency.

Requirements: torch>=2.0 (CPU is enough).  Run e.g.

    python train/baseline_registration.py \
        --index outputs/pairs_rigid/.../*_index.jsonl --epochs 40

The module is import-safe without torch (guarded), so the rest of the repo
runs on CPU-only numpy/scipy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except Exception as e:  # noqa: BLE001
    HAS_TORCH = False
    _TORCH_ERR = e

LOSS_W, LOSS_H = 384, 256
LEVEL_WEIGHTS = (0.25, 1.0, 1.0)
SEG_CLASSES = 7          # matches tissue-map codes 0..6 used by anatomy
N_SEG = 4                # classes actually supervised (0..3: bg/body/bone/liver)

# ----------------------------------------------------------------------------
# Rigid-frame helpers (numpy data prep / torch graph)
# ----------------------------------------------------------------------------

def axis_angle_to_matrix(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
    ])


def matrix_to_axis_angle(R):
    R = np.asarray(R, dtype=np.float64)
    angle = math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if angle < 1e-8:
        return np.zeros(3), 0.0
    s = math.sin(angle)
    if s < 1e-8:  # angle ~ pi
        d = np.sqrt(np.maximum((np.diag(R) + 1) / 2, 0))
        a = d * np.sign([R[2, 1] - R[1, 2] if d[0] < 0.5 else 1.0,
                         R[0, 2] - R[2, 0] if d[1] < 0.5 else 1.0,
                         R[1, 0] - R[0, 1] if d[2] < 0.5 else 1.0])
        a /= (np.linalg.norm(a) + 1e-12)
        return a, angle
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v / (2 * s), angle


def affine_to_vector(M):
    R = M[:3, :3]
    t = M[:3, 3]
    axis, angle = matrix_to_axis_angle(R)
    return np.concatenate([t, np.asarray(axis) * angle])


def vector_to_affine(v):
    v = np.asarray(v, dtype=np.float64)
    R = axis_angle_to_matrix(v[3:6], np.linalg.norm(v[3:6]))
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = v[:3]
    return M


def rotation_hinge_np(v, max_deg=25.0):
    ang = np.linalg.norm(v[3:6]) * 180.0 / math.pi
    return max(0.0, ang - max_deg)


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------

def _read_index(index_files):
    rows = []
    for f in index_files:
        f = Path(f)
        base = f.parent
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["_base"] = str(base)
                rows.append(row)
    keys = list(dict.fromkeys(r["sid"] for r in rows))
    by_key = {r["sid"]: r for r in rows}
    return [by_key[k] for k in keys]


def load_sample(row):
    """Load one sample row into numpy arrays.

    Returns dict with ct (D,H,W), affine_ct(4,4), us (H,W) float [0,1],
    seg (H,W) int tissue labels, us_to_ct, ct_to_us, probe params.
    """
    import nibabel as nib
    from PIL import Image

    base = Path(row["_base"])
    ct_path = base / row["volume"]
    img = nib.load(str(ct_path))
    ct = np.asarray(img.get_fdata(), dtype=np.float32)
    affine_ct = np.asarray(img.affine, dtype=np.float64)

    us_img = Image.open(base / row["sid"] / row["files"]["us"]).convert("L")
    us = np.asarray(us_img, dtype=np.float32) / 255.0

    seg_img = nib.load(str(base / row["sid"] / row["files"]["seg_slice"]))
    seg = np.asarray(seg_img.get_fdata(), dtype=np.int64)

    tf = np.load(base / row["sid"] / row["files"]["transform"])
    us_to_ct = tf["us_to_ct"].astype(np.float64)
    ct_to_us = tf["ct_to_us"].astype(np.float64)

    seg_vol_path = base / row["volume"].replace("_ct.nii.gz", "_seg.nii.gz")
    seg_vol = np.asarray(nib.load(str(seg_vol_path)).get_fdata(), dtype=np.int64)

    pr = row["params"]["probe"]
    return {
        "ct": ct, "affine_ct": affine_ct, "us": us, "seg": seg, "seg_vol": seg_vol,
        "us_to_ct": us_to_ct, "ct_to_us": ct_to_us,
        "nx": int(pr["nx"]), "nz": int(pr["nz"]),
        "dx": float(pr.get("dx_eff", pr.get("dx", 0.5))), "dz": float(pr["dz"]),
        "probe": pr,
    }


def preprocess_ct(ct, out_shape=(64, 96, 96), window=(-150.0, 250.0)):
    """Window HU to [0,1] and resize the crop to a fixed shape."""
    lo, hi = window
    ct = np.clip((np.asarray(ct, dtype=np.float32) - lo) / (hi - lo), 0, 1)
    from scipy.ndimage import zoom
    z = (out_shape[0] / ct.shape[0], out_shape[1] / ct.shape[1], out_shape[2] / ct.shape[2])
    ct = zoom(ct, z, order=1)
    return ct.astype(np.float32)


def tre(gt, pred, nx, nz, dx, dz, n_pts=200, rng=None):
    """Target registration error (mm) between GT and predicted transforms."""
    rng = rng if rng is not None else np.random.default_rng(0)
    i = rng.uniform(0, nx - 1, n_pts)
    j = rng.uniform(0, nz - 1, n_pts)
    x = (i - (nx - 1) / 2) * dx
    y = j * dz
    pts = np.stack([x, y, np.zeros_like(x), np.ones_like(x)], axis=0)
    p_gt = (gt[:3, :] @ pts).T  # (n,3)
    p_pd = (pred[:3, :] @ pts).T
    return float(np.mean(np.linalg.norm(p_gt - p_pd, axis=1)))


def trans_scale_of(data, fallback=150.0):
    """Characteristic translation magnitude (mm) for loss normalisation."""
    vals = [np.linalg.norm(np.asarray(s["us_to_ct"])[:3, 3]) for s in data]
    if not vals:
        return fallback
    return float(np.median(vals))


# ----------------------------------------------------------------------------
# Networks
# ----------------------------------------------------------------------------

class PoseRegNet(nn.Module):
    def __init__(self, latent=128):
        super().__init__()
        c = 8
        self.enc3d = nn.Sequential(
            nn.Conv3d(1, c, 4, stride=2, padding=1), nn.BatchNorm3d(c), nn.ReLU(),
            nn.Conv3d(c, 2 * c, 4, stride=2, padding=1), nn.BatchNorm3d(2 * c), nn.ReLU(),
            nn.Conv3d(2 * c, 4 * c, 4, stride=2, padding=1), nn.BatchNorm3d(4 * c), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.enc2d = nn.Sequential(
            nn.Conv2d(1, c, 4, stride=2, padding=1), nn.BatchNorm2d(c), nn.ReLU(),
            nn.Conv2d(c, 2 * c, 4, stride=2, padding=1), nn.BatchNorm2d(2 * c), nn.ReLU(),
            nn.Conv2d(2 * c, 4 * c, 4, stride=2, padding=1), nn.BatchNorm2d(4 * c), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(8 * c, latent), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(latent, 6),
        )
        self.seg = SegHead()

    def forward(self, ct, us):
        z3 = self.enc3d(ct).flatten(1)
        z2 = self.enc2d(us).flatten(1)
        pose = self.head(torch.cat([z3, z2], dim=1))
        seg = self.seg(us)
        return pose, seg


class SegHead(nn.Module):
    """2D tissue-class head on the US slice (auxiliary supervision)."""

    def __init__(self, nc=SEG_CLASSES):
        super().__init__()
        c = 8
        self.conv = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(),
            nn.Conv2d(c, 2 * c, 3, padding=1), nn.BatchNorm2d(2 * c), nn.ReLU(),
            nn.Conv2d(2 * c, 2 * c, 3, padding=1), nn.BatchNorm2d(2 * c), nn.ReLU(),
        )
        self.out = nn.Conv2d(2 * c, nc, 1)

    def forward(self, us):
        f = self.conv(us)
        return self.out(F.interpolate(f, size=(LOSS_H, LOSS_W), mode="bilinear", align_corners=True))


# ----------------------------------------------------------------------------
# Differentiable reslice + LNCC appearance loss
# ----------------------------------------------------------------------------

def _pose_cols_from_matrix(M):
    """Recover the probe pose (face,u,v,w) from a T_US->CT matrix.

    us_to_ct columns where u=lateral, w=beam, -v=elevation:  u=M[:,0],
    w=M[:,1], v=-M[:,2], face=M[:,3].  Re-orthonormalised for the fan grid.
    Returns torch tensors.
    """
    face = M[:3, 3]
    w0 = M[:3, 1]
    u0 = M[:3, 0]
    v0 = -M[:3, 2]
    w = w0 / (w0.norm() + 1e-12)
    u = u0 - (u0 @ w) * w
    u = u / (u.norm() + 1e-12)
    v = torch.linalg.cross(w, u)
    v = v / (v.norm() + 1e-12)
    return u, v, w, face


def _rot3_axis(vv, ang, wdir):
    """Rodrigues rotation of wdir about axis vv by ang (torch, vectorised)."""
    vv = vv.expand(wdir.shape)
    c = torch.cos(ang)
    s = torch.sin(ang)
    cross = torch.linalg.cross(vv, wdir, dim=-1)
    dot = (vv * wdir).sum(-1, keepdim=True)
    return c[..., None] * wdir + s[..., None] * cross + (1.0 - c)[..., None] * dot * vv


def _plane_world_torch(sample, u, v, w, face, device, cols=None, rows=None):
    """World coordinate (nrow? ncol x nrow, 3) of the scan plane pixels.

    Replicates ct2us.geometry.Probe.world_grid (deform=None) exactly, which is
    what generation used for linear AND convex probes.  cols/rows are optional
    subsampled pixel indices (row = depth index).
    """
    pr = sample["probe"]
    kind = pr["kind"]
    nx0 = int(pr["nx"])
    nz0 = int(pr["nz"])
    if cols is None:
        cols = torch.arange(nx0, device=device, dtype=torch.float32)
    if rows is None:
        rows = torch.arange(nz0, device=device, dtype=torch.float32)
    if kind == "convex":
        fov = float(pr["fov_angle"])
        radius = float(pr["radius"])
        near = float(pr["near"])
        dz2 = float(pr["dz"])
        angles = (cols - (nx0 - 1) / 2) * (fov / (nx0 - 1))
        dirs = _rot3_axis(v, angles, w[None, :].expand(cols.shape[0], 3))  # (ncol,3)
        apex = face - w * radius
        radii = radius + near + rows * dz2  # (nrow,)
        world = apex[None, None, :] + dirs[:, None, :] * radii[None, :, None]
    else:
        dx2 = float(pr["dx"])
        near = float(pr["near"])
        dz2 = float(pr["dz"])
        lat = (cols - (nx0 - 1) / 2) * dx2  # (ncol,)
        dep = near + rows * dz2              # (nrow,)
        world = (face[None, None, :]
                 + u[None, None, :] * lat[:, None, None]
                 + w[None, None, :] * dep[None, :, None])
    return world  # (ncol,nrow,3)


def reslice_volume(sample, vol, M, device=None, cols=None, rows=None,
                   cval=0.0) -> torch.Tensor:
    """Differentiable bilinear reslice of a volume on the fan scan-plane grid.

    sample: dict from load_sample (needs affine_ct, probe); vol: numpy volume.
    M: 4x4 T_US->CT (numpy or torch, may carry gradients).
    cols/rows: optional subsampled pixel indices.  Out-of-grid pixels become
    cval.  Returns (1,1,ncol,nrow).
    """
    dev = device or torch.device("cpu")
    if not torch.is_tensor(M):
        M = torch.tensor(np.asarray(M, dtype=np.float64), dtype=torch.float32, device=dev)
    u, v, w, face = _pose_cols_from_matrix(M)
    world = _plane_world_torch(sample, u, v, w, face, dev, cols=cols, rows=rows)

    vol_t = torch.tensor(vol, dtype=torch.float32, device=dev)
    a = torch.tensor(sample["affine_ct"], dtype=torch.float32, device=dev)
    Rinv = torch.linalg.inv(a[:3, :3])
    vox = world @ Rinv.T - (Rinv @ a[:3, 3])  # (ncol,nrow,3)

    shape = torch.tensor(vol_t.shape, dtype=torch.float32, device=dev)
    g = 2.0 * vox / (shape - 1) - 1.0
    oob = ((g < -1.0) | (g > 1.0)).any(dim=-1)  # (ncol,nrow)
    g = g[..., [2, 1, 0]].reshape(1, 1, world.shape[0], world.shape[1], 3)
    slab = F.grid_sample(vol_t[None, None], g, mode="bilinear",
                         padding_mode="zeros", align_corners=True).squeeze(2)
    if cval != 0.0:
        slab = torch.where(oob.unsqueeze(0).unsqueeze(0),
                           torch.full_like(slab, cval), slab)
    return slab


def reslice_pose(sample, M, device=None, cols=None, rows=None) -> torch.Tensor:
    """Differentiable reslice of the CT volume at pose M (fan-accurate)."""
    return reslice_volume(sample, sample["ct"], M, device=device, cols=cols,
                          rows=rows, cval=-1024.0)


def reslice_label(sample, M, device=None, cols=None, rows=None) -> torch.Tensor:
    """Differentiable reslice of the tissue-label volume (fan-accurate).

    Bilinear sampling of the label volume yields a smooth label-blend map,
    differentiable in pose — the modality-consistent target (soft labels).
    """
    return reslice_volume(sample, sample["seg_vol"], M, device=device, cols=cols,
                          rows=rows, cval=0.0)


def seg_consistency_loss(logits_slab, label_map, mask):
    """Alignment of a predicted label map with a (smooth) label map.

    logits_slab: (1,C,nrow,ncol) class logits from the US segmentation head.
    label_map:   (nrow,ncol) soft label blend resliced from the CT label volume.
    mask:        (nrow,ncol) valid-pixel weights.
    """
    p = torch.softmax(logits_slab, dim=1)          # (1,C,nrow,ncol)
    cls = torch.arange(logits_slab.shape[1], device=logits_slab.device,
                       dtype=logits_slab.dtype)
    pred_map = torch.einsum("c,chw->hw", cls, p[0])  # (nrow,ncol)
    m = mask.float()
    d = (pred_map - label_map) * m
    return (d * d).sum() / m.sum().clamp(min=1.0)


def label_mask(label_map, us_slab):
    """Valid-pixel mask for the seg-consistency loss."""
    return ((label_map > 0.25) & (us_slab > 0.05)).float()


def reslice_ct(ct, affine_ct, us_to_ct, nx, nz, dx, dz, device=None,
               probe=None) -> torch.Tensor:
    """Differentiable reslicing of the CT volume onto the US pixel grid.

    Preferred over the linear-grid version below: uses the true probe fan
    geometry when a probe dict is supplied (matches generation exactly, so the
    GT transform reproduces the US slice).  Returns (1,1,nx,nz).
    """
    if probe is not None:
        sample = {"ct": ct, "affine_ct": affine_ct, "probe": probe}
        return reslice_pose(sample, us_to_ct, device=device)
    dev = device or torch.device("cpu")
    i = torch.arange(nx, dtype=torch.float32, device=dev)
    j = torch.arange(nz, dtype=torch.float32, device=dev)
    gi, gj = torch.meshgrid(i, j, indexing="ij")
    x_us = (gi - (nx - 1) / 2) * dx
    y_us = gj * dz
    ones = torch.ones_like(x_us)
    src = torch.stack([x_us, y_us, torch.zeros_like(x_us), ones], dim=-1)  # (nx,nz,4)
    M = torch.tensor(us_to_ct, dtype=torch.float32, device=dev)
    world = torch.einsum("ij,kli->klj", M, src)[..., :3]  # (nx,nz,3)

    a = torch.tensor(affine_ct, dtype=torch.float32, device=dev)
    Rinv = torch.linalg.inv(a[:3, :3])
    vox = torch.einsum("ij,klj->kli", Rinv, world) - (Rinv @ a[:3, 3])

    shape = torch.tensor(ct.shape, dtype=torch.float32, device=dev)
    g = 2.0 * vox / (shape - 1) - 1.0  # (nx,nz,3) in (i,j,k) = (D,H,W) order
    g = torch.stack([g[..., 2], g[..., 1], g[..., 0]], dim=-1)  # -> (x,y,z)
    ct_t = torch.tensor(ct, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(0)
    sampled = F.grid_sample(ct_t, g.unsqueeze(0).unsqueeze(0),
                            mode="bilinear", padding_mode="zeros", align_corners=True)
    return sampled.squeeze(2)  # (1,1,nx,nz)


def appearance_loss(pred_ct_slice, us):
    """Multi-scale local-normalised cross-correlation (LNCC) dissimilarity.

    pred_ct_slice, us: (1,1,H,W).  Output near 0 with good local alignment.
    """
    loss = 0.0
    for k, w in enumerate(LEVEL_WEIGHTS):
        ks = 2 ** (k + 1) + 1
        p = F.avg_pool2d(pred_ct_slice, ks, stride=1, padding=ks // 2)
        u = F.avg_pool2d(us, ks, stride=1, padding=ks // 2)
        d0 = pred_ct_slice - p
        d1 = us - u
        cov = (d0 * d1).mean()
        v0 = d0.pow(2).mean()
        v1 = d1.pow(2).mean()
        denom = (v0 * v1).clamp(min=1e-8).sqrt() + 1e-6
        loss = loss + w * (1.0 - cov / denom)
    return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)


def lncc_similarity(loss):
    """Map a /1,2,1/ LNCC dissimilarity to a similarity in [0,1]."""
    v = float(torch.clamp(1.0 - loss, min=0.0))
    return min(v, 1.0)


def _norm_to_vector(vn, ts):
    """Un-normalise a refined [t/ts, rot] vector to raw 6-dof."""
    return torch.cat([vn[:3] * ts, vn[3:]])


def _subsample(n0, scale):
    """Evenly-spaced subsampled pixel indices over the full grid."""
    k = max(1, int(round(n0 * scale)))
    return np.unique(np.linspace(0, n0 - 1, k).round().astype(np.int64))


def refine_pose(sample, init_affine, seg_fn, steps=None, lr=2e-2,
                scales=((0.5, 12), (1.0, 25)), device=None):
    """Gradient refinement of a pose against the US slice.

    Uses the modality-consistent label-space objective: the CT tissue-label
    volume is resliced at the candidate pose (differentiable, fan-accurate) and
    aligned with the segmentation-head prediction on the US image.  Coarse-to-
    fine on subsampled grids.  Returns final affine, dissimilarity, similarity
    and per-level losses.  `steps` (optional) overrides the per-level counts
    with a single full-resolution schedule.
    """
    if not HAS_TORCH:
        raise RuntimeError("torch required for refine_pose")
    if steps is not None:
        scales = ((1.0, steps),)
    dev = device or torch.device("cpu")
    us = torch.tensor(sample["us"], dtype=torch.float32, device=dev)
    us_t = us.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits_full = seg_fn(us_t)  # (1,C,LOSS_H,LOSS_W): fixed inference head

    ts = trans_scale_of([sample])
    v0 = affine_to_vector(init_affine)
    param = torch.tensor(np.concatenate([v0[:3] / ts, v0[3:]]), dtype=torch.float32,
                         device=dev, requires_grad=True)

    def score(M, cols, rows):
        lab = reslice_label(sample, M, device=dev, cols=cols, rows=rows)[0, 0]  # (nrow,ncol)
        us_s = F.interpolate(us_t, size=(lab.shape[0], lab.shape[1]),
                             mode="bilinear", align_corners=True)[0, 0]
        Ls = F.interpolate(logits_full, size=(lab.shape[0], lab.shape[1]),
                           mode="bilinear", align_corners=True)
        return seg_consistency_loss(Ls, lab, label_mask(lab, us_s))

    opt = torch.optim.Adam([param], lr=lr)
    per_level = []
    for scale, n in scales:
        cols = torch.tensor(_subsample(sample["nx"], scale), device=dev, dtype=torch.int64)
        rows = torch.tensor(_subsample(sample["nz"], scale), device=dev, dtype=torch.int64)
        for _ in range(n):
            opt.zero_grad()
            M_t = _vector_to_affine_torch(_norm_to_vector(param, ts))
            loss = score(M_t, cols, rows)
            loss.backward()
            opt.step()
        per_level.append(float(loss.detach()))

    final = vector_to_affine(_norm_to_vector(param.detach(), ts).cpu().numpy())
    with torch.no_grad():
        lab = reslice_label(sample, final, device=dev)[0, 0]
        us_full = us_t[0, 0]
        L = F.interpolate(logits_full, size=(lab.shape[0], lab.shape[1]),
                          mode="bilinear", align_corners=True)
        best_loss = seg_consistency_loss(L, lab, label_mask(lab, us_full))
    return {"affine": final, "loss": float(best_loss),
            "sim": 1.0 / (1.0 + float(best_loss)),
            "levels": per_level}


def refine_top_k(sample, candidates, seg_fn, k=None, **kw):
    """Refine up to k top candidates and return the best by final similarity."""
    cands = list(candidates)
    if k is not None and len(cands) > k:
        cands = cands[:k]
    res = [refine_pose(sample, c, seg_fn, **kw) for c in cands]
    res.sort(key=lambda r: -r["sim"])
    return res[0], res


def _vector_to_affine_torch(v):
    """Differentiable 6-vector [t, axis*angle] -> 4x4 (as torch tensor 4x4)."""
    t = v[:3]
    rod = v[3:6]
    ang = rod.norm()
    ax = rod / (ang + 1e-8)
    x, y, z = ax[0], ax[1], ax[2]
    c, s = torch.cos(ang), torch.sin(ang)
    C = 1 - c
    R = torch.stack([
        torch.stack([c + x * x * C, x * y * C - z * s, x * z * C + y * s]),
        torch.stack([y * x * C + z * s, c + y * y * C, y * z * C - x * s]),
        torch.stack([z * x * C - y * s, z * y * C + x * s, c + z * z * C]),
    ])
    M = torch.eye(4, device=v.device, dtype=v.dtype)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


# ----------------------------------------------------------------------------
# Training / evaluation
# ----------------------------------------------------------------------------

def make_batch(items, device):
    max_nx = max(it["nx"] for it in items)
    max_nz = max(it["nz"] for it in items)
    ct_b, us_b, seg_b = [], [], []
    gt_v = []
    gt_m = []
    metas = []
    for it in items:
        ct = preprocess_ct(it["ct"])
        ct_b.append(ct[None])
        us = np.pad(it["us"], ((0, max_nx - it["nx"]), (0, max_nz - it["nz"])))
        us_b.append(us[None])
        seg = np.pad(it["seg"].astype(np.int64),
                     ((0, max_nx - it["nx"]), (0, max_nz - it["nz"])), constant_values=0)
        seg_b.append(seg[None])
        vec = affine_to_vector(it["us_to_ct"])
        gt_v.append(vec)
        gt_m.append(it["us_to_ct"])
        metas.append(it)
    ct = torch.tensor(np.stack(ct_b), dtype=torch.float32, device=device)
    us = torch.tensor(np.stack(us_b), dtype=torch.float32, device=device)
    seg = torch.tensor(np.stack(seg_b), dtype=torch.long, device=device)
    gt = torch.tensor(np.stack(gt_v), dtype=torch.float32, device=device)
    return ct, us, seg, gt, gt_m, metas


def train(args):
    if not HAS_TORCH:
        print("torch not installed; install with:  pip install torch")
        print(f"diagnostic: {_TORCH_ERR}")
        return
    import torch.optim as optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    rows = _read_index(args.index)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(rows)
    n_val = max(1, int(args.val_frac * len(rows)))
    val_rows, train_rows = rows[:n_val], rows[n_val:]

    train_data = [load_sample(r) for r in train_rows]
    val_data = [load_sample(r) for r in val_rows]
    ts = trans_scale_of(train_data)
    print(f"train={len(train_data)}  val={len(val_data)}  device={device}  "
          f"trans_scale={ts:.1f}mm")

    model = PoseRegNet().to(device)
    seg_fn = model.seg
    opt = optim.Adam(model.parameters(), lr=args.lr)
    sched = optim.lr_scheduler.StepLR(opt, step_size=max(1, args.epochs // 3), gamma=0.5)
    ce = nn.CrossEntropyLoss(ignore_index=-1)

    def seg_loss(pred_seg, seg):
        tar = F.interpolate(seg.float(), size=(LOSS_H, LOSS_W),
                            mode="nearest").long().squeeze(1)
        return ce(pred_seg, tar)

    def run_epoch(data, train_mode=False):
        model.train(train_mode)
        idx = np.arange(len(data))
        if train_mode:
            rng.shuffle(idx)
        tot_pose = tot_app = tot_seg = 0.0
        tre_list = []
        rot_list = []
        trans_list = []
        steps = 0
        for s in range(0, len(idx), args.batch):
            batch = idx[s:s + args.batch]
            items = [data[i] for i in batch]
            ct, us, seg, gt, gt_m, metas = make_batch(items, device)
            pred, seg_pred = model(ct, us)

            # grouped pose loss: translation (scaled mm) and rotation separately
            t_gt = gt[:, :3] / ts
            t_pr = pred[:, :3].detach().cpu().numpy()
            t_gtn = gt[:, :3].cpu().numpy()
            trans_batch = float(np.mean(np.linalg.norm(t_pr - t_gtn, axis=1)))
            pose_loss = args.w_trans * F.mse_loss(pred[:, :3] / ts, t_gt) \
                + args.w_rot * F.mse_loss(pred[:, 3:], gt[:, 3:])
            rot_pen = torch.relu(pred[:, 3:].norm(dim=1) * 180.0 / math.pi
                                 - args.rot_hinge_deg).mean()

            app_loss = 0.0
            for b, meta in enumerate(metas):
                pr = vector_to_affine(pred[b].detach().cpu().numpy())
                lab = reslice_label(meta, pr, device=device)[0, 0]  # (nz,nx)
                us_i = torch.tensor(meta["us"], dtype=torch.float32, device=device)
                us_it = us_i.unsqueeze(0).unsqueeze(0)
                us_full = us_it[0, 0]
                lg = seg_fn(us_it)  # (1,C,LOSS_H,LOSS_W)
                L = F.interpolate(lg, size=(lab.shape[0], lab.shape[1]),
                                  mode="bilinear", align_corners=True)
                app_loss = app_loss + seg_consistency_loss(L, lab, label_mask(lab, us_full))
            app_loss = app_loss / max(1, len(metas))

            sl = seg_loss(seg_pred, seg)
            loss = pose_loss + args.w_app * app_loss + args.w_seg * sl + args.w_rotpen * rot_pen

            if train_mode:
                opt.zero_grad()
                loss.backward()
                opt.step()

            tot_pose += pose_loss.item()
            tot_app += app_loss.item()
            tot_seg += sl.item()
            steps += 1
            for b, meta in enumerate(metas):
                pr = vector_to_affine(pred[b].detach().cpu().numpy())
                tre_list.append(tre(meta["us_to_ct"], pr, meta["nx"], meta["nz"],
                                    meta["dx"], meta["dz"]))
                rot_list.append(float(np.linalg.norm(pred[b, 3:].detach().cpu().numpy()) * 180.0 / math.pi))
                trans_list.append(trans_batch)
        if train_mode:
            sched.step()
        return (tot_pose / max(1, steps), tot_app / max(1, steps),
                tot_seg / max(1, steps), float(np.mean(tre_list)))

    best_tre = float("inf")
    for ep in range(1, args.epochs + 1):
        tp, ta, tse, tret = run_epoch(train_data, train_mode=True)
        vp, va, vse, trev = run_epoch(val_data, train_mode=False)
        print(f"epoch {ep:3d}  train pose={tp:.4f} app={ta:.4f} seg={tse:.3f} TRE={tret:.2f}mm | "
              f"val pose={vp:.4f} app={va:.4f} seg={vse:.3f} TRE={trev:.2f}mm")
        if trev < best_tre:
            best_tre = trev
            torch.save(model.state_dict(), args.checkpoint)
            print(f"  saved best -> {args.checkpoint}")
    print(f"best val TRE = {best_tre:.2f} mm")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--checkpoint", default="outputs/baseline_reg.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--w_trans", type=float, default=0.5)
    ap.add_argument("--w_rot", type=float, default=1.0)
    ap.add_argument("--w_app", type=float, default=0.5)
    ap.add_argument("--w_seg", type=float, default=0.05)
    ap.add_argument("--w_rotpen", type=float, default=0.05)
    ap.add_argument("--rot_hinge_deg", type=float, default=25.0)
    args = ap.parse_args()
    if not HAS_TORCH:
        print("torch not installed; install with:  pip install torch")
        print(f"diagnostic: {_TORCH_ERR}")
        return
    train(args)


if __name__ == "__main__":
    main()