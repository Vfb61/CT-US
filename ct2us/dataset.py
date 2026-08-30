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
                 cam: str):
        self.ct = ct
        self.tissue = tissue
        self.affine = affine
        self.crop_box = crop_box
        self.liver_mask = liver_mask
        self.scatter = scatter
        self.case = cam

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
              rng=None) -> CaseContext:
    """Load and fuse one case into a CaseContext."""
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

    rng = rng if rng is not None else np.random.default_rng()
    noise = make_noise_volume(ct_c.shape, rng)
    scatter = an.scatter_volume(tissue_c, spatial_noise=noise)

    return CaseContext(ct_c, tissue_c, aff_c, box, liver_c, scatter, case_name)


# ----------------------------------------------------------------------------
# Sample generation
# ----------------------------------------------------------------------------

def sample_probe_geometry(ctx: CaseContext, rng, param=None) -> tuple[geo.Probe, dict]:
    """Sample probe + pose, accepting only poses that cover the liver well."""
    param = param or {}
    kind_pool = param.get("kinds", ["convex", "convex", "linear"])
    liver_surf = ctx.probe()

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
        if frac >= frac_min and max_d >= param.get("min_depth_fraction", 0.25) * pr.depth_span:
            return pr, pose
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
    return {
        "us_to_ct": us_to_ct,
        "ct_to_us": ct_to_us,
        "face": face, "u": u, "v": v, "w": w,
        "dx_mm": probe.dx_effective,
        "dz_mm": probe.dz,
        "nu": probe.nx, "nv": probe.nz,
    }


def generate_sample(ctx: CaseContext, index: int, rng, param: dict | None = None,
                    global_params: dict | None = None,
                    compute_reference: bool = True,
                    sid: str | None = None) -> dict:
    """Create one {CT, US, T} sample."""
    param = param or {}
    global_params = global_params or {}
    probe, pose = sample_probe_geometry(ctx, rng, param)
    deform_params = df.random_deform_params(probe, rng, enabled=not global_params.get("no_deform", False))
    deform = df.build_deformation(probe, deform_params, rng)
    planes = resample_planes(ctx, probe, pose, deform)
    render_params = dict(global_params.get("render", {}))
    out = render_mod.render_bmode(planes["ct"], planes["tissue"],
                                  planes["scatter"], probe, render_params, rng,
                                  want_envelope=True)
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
        "transform": transform,
        "params": {"render": out["params"], "deform": deform_params,
                   "probe": _probe_json(probe), "pose": _pose_json(pose)},
    }
    if compute_reference:
        qa = PhysicsProxy(planes["ct"], planes["tissue"], planes["scatter"], probe)
        sample["reference"] = qa.bmode()
    return sample


def _probe_json(probe: geo.Probe) -> dict:
    return dict(kind=probe.kind, nx=probe.nx, nz=probe.nz, dx=probe.dx,
                fov_angle=float(probe.fov_angle), dz=probe.dz,
                near=probe.near, radius=probe.radius, freq=probe.freq)


def _pose_json(pose: dict) -> dict:
    out = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in pose.items()}
    out.pop("kind", None)
    return out


# ----------------------------------------------------------------------------
# Writer / index
# ----------------------------------------------------------------------------

class DatasetWriter:
    def __init__(self, out_dir: str, name: str = "ct2us"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.vol_dir = self.out_dir / "volumes"
        self.vol_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.out_dir / f"{name}_index.jsonl"
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self.index_path.write_text("", encoding="utf-8")
        self._written_volumes = set()

    def ensure_volume(self, ctx: CaseContext) -> str:
        """Persist the case CT crop + tissue map once per case, return path."""
        rel = f"volumes/{ctx.case}_ct.nii.gz"
        path = self.out_dir / rel
        if rel not in self._written_volumes:
            io.save_nifti(path, ctx.ct, ctx.affine, like=None)
            io.save_nifti(self.out_dir / f"volumes/{ctx.case}_seg.nii.gz",
                          ctx.tissue.astype(np.int16), ctx.affine, like=None)
            self._written_volumes.add(rel)
        return rel

    def write(self, ctx: CaseContext, sample: dict, split: str = "train") -> dict:
        sid = sample.get("sid", f"sample_{sample['index']:06d}")
        d = self.out_dir / sid
        d.mkdir(parents=True, exist_ok=True)
        us_img = sample["render"]["uint8"]
        io.save_nifti(d / "ct_slice.nii.gz",
                      sample["planes"]["ct"], sample["transform"]["us_to_ct"],
                      like=None)
        io.save_nifti(d / "seg_slice.nii.gz",
                      sample["planes"]["tissue"].astype(np.int16),
                      sample["transform"]["us_to_ct"], like=None)
        np.savez_compressed(d / "transform.npz",
                            us_to_ct=sample["transform"]["us_to_ct"],
                            ct_to_us=sample["transform"]["ct_to_us"],
                            face=sample["transform"]["face"],
                            u=sample["transform"]["u"],
                            v=sample["transform"]["v"],
                            w=sample["transform"]["w"],
                            dx_mm=sample["transform"]["dx_mm"],
                            dz_mm=sample["transform"]["dz_mm"])
        (d / "us.png").write_bytes(_png_bytes(us_img))
        volume_rel = self.ensure_volume(ctx)
        meta = {
            "sid": sid, "split": split, "case": sample.get("case", ""),
            "volume": volume_rel, "params": sample["params"], "files": {
                "us": "us.png", "ct_slice": "ct_slice.nii.gz",
                "seg_slice": "seg_slice.nii.gz", "transform": "transform.npz",
            },
        }
        if "reference" in sample:
            ref = sample["reference"]["uint8"]
            (d / "reference_us.png").write_bytes(_png_bytes(ref))
            meta["files"]["reference_us"] = "reference_us.png"
        io.save_json(d / "params.json", meta)
        with open(self.index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        return meta

    def finish(self):
        return self.index_path


def _png_bytes(arr: np.ndarray) -> bytes:
    from PIL import Image
    import io as _io
    buf = _io.BytesIO()
    Image.fromarray(np.asarray(arr, dtype=np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()