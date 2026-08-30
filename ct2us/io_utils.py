"""NIfTI IO, affine helpers and scan-plane resampling."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates


# ----------------------------------------------------------------------------
# NIfTI loading
# ----------------------------------------------------------------------------

def _clean_files(path: Path):
    """Drop macOS '._' metadata junk left by decompression."""
    if not path.exists():
        return []
    return sorted(
        [p for p in path.iterdir() if p.suffix in (".nii", ".gz", ".nii.gz")
         and not p.name.startswith("._") and not p.name.startswith(".")]
    )


def load_volume(path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load a NIfTI file and return (data RAS-canonical float32, voxel affine, header info).

    Returns data with shape (i, j, k) matching the affine columns, i.e. the
    nibabel native array layout.
    """
    path = Path(path)
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    affine = np.asarray(img.affine, dtype=np.float64)
    pixdim = np.asarray(img.header["pixdim"][1:4], dtype=np.float64)
    info = {
        "shape": data.shape,
        "pixdim": pixdim,
        "affine": affine,
        "orient": "".join(nib.aff2axcodes(affine)),
    }
    return data, affine, info


def save_nifti(path, data: np.ndarray, affine: np.ndarray, like: Optional[str] = None):
    """Write a NIfTI file, optionally inheriting metadata from a reference file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if like is not None:
        ref = nib.load(str(like))
        img = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine, header=ref.header.copy())
    else:
        img = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine)
    nib.save(img, str(path))
    return path


def list_cases(root: Path, split: str = "imagesTr") -> list[str]:
    """List case names (basenames without extension) of the MSD-style folder."""
    root = Path(root)
    d = root / split
    names = [p.name for p in _clean_files(d)]
    return [n[:-len(".nii.gz")] if n.endswith(".nii.gz") else n for n in names]


# ----------------------------------------------------------------------------
# Affine helpers
# ----------------------------------------------------------------------------

def affine_inverse(affine: np.ndarray) -> np.ndarray:
    """Inverse of a 4x4 affine (uses only linear part, translation handled)."""
    R = affine[:3, :3]
    t = affine[:3, 3]
    Rinv = np.linalg.inv(R)
    out = np.eye(4)
    out[:3, :3] = Rinv
    out[:3, 3] = -Rinv @ t
    return out


def world_to_voxel_grid(affine: np.ndarray, world_coords: np.ndarray) -> np.ndarray:
    """Map (..., 3) world coordinates (RAS mm) to voxel index coordinates.

    The returned array has the same shape as world_coords and is ready for
    scipy.ndimage.map_coordinates (axis order matches the volume array).
    """
    inv = affine_inverse(affine)
    xyz = np.asarray(world_coords, dtype=np.float64)
    shape = xyz.shape
    flat = xyz.reshape(-1, 3)
    ones = np.ones((flat.shape[0], 1), dtype=np.float64)
    homo = np.concatenate([flat, ones], axis=1)
    vox = (inv @ homo.T).T[:, :3]
    return vox.reshape(shape)


def resample_plane(volume: np.ndarray, affine: np.ndarray,
                   coords: np.ndarray, order: int = 1,
                   cval: float = -1024.0) -> np.ndarray:
    """Resample a volume on an arbitrary set of voxel coordinates.

    coords: (3, N) voxel-index coordinates (axis0->volume axis0, ...).
    Returns: (N,) array of sampled values, out-of-range -> cval.
    """
    vol = np.asarray(volume, dtype=np.float32)
    cval = float(cval)
    if vol.size == 0:
        return np.full(coords.shape[1], cval, dtype=np.float32)
    sampled = map_coordinates(vol, coords, order=order, mode="constant", cval=cval,
                              prefilter=True)
    return np.asarray(sampled, dtype=np.float32)


def bounding_box(mask: np.ndarray, margin_vox: int = 10) -> tuple[slice, slice, slice]:
    """Bounding box of a boolean/nonzero mask with margin, clamped to volume."""
    idx = np.argwhere(np.asarray(mask) > 0)
    if idx.size == 0:
        raise ValueError("mask is empty")
    lo = idx.min(axis=0)
    hi = idx.max(axis=0) + 1
    lo = np.maximum(0, lo - margin_vox)
    hi = np.minimum(np.asarray(mask.shape), hi + margin_vox)
    return (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))


def crop_and_recenter(volume: np.ndarray, affine: np.ndarray, mask: np.ndarray,
                      margin_vox: int = 10):
    """Crop volume+label to a mask bounding box, recentering the affine.

    Returns (crop_vol, crop_mask, crop_affine, box). The crop_affine maps voxel
    index in the cropped grid to the same world space as the original affine.
    """
    box = bounding_box(mask, margin_vox)
    crop_vol = volume[box]
    crop_mask = mask[box]
    offset = np.array([b.start for b in box], dtype=np.float64)
    crop_affine = affine.copy()
    crop_affine[:3, 3] = affine[:3, 3] + affine[:3, :3] @ offset
    return crop_vol, crop_mask, crop_affine, box


# ----------------------------------------------------------------------------
# Plane helpers shared by geometry and dataset export
# ----------------------------------------------------------------------------

def plane_affine(origin, x_axis, y_axis, z_axis):
    """Build a 4x4 affine whose columns are the three image axes + origin."""
    M = np.eye(4)
    M[:3, 0] = x_axis
    M[:3, 1] = y_axis
    M[:3, 2] = z_axis
    M[:3, 3] = origin
    return M


def reslice_to_plane(volume: np.ndarray, affine: np.ndarray,
                     px_world: np.ndarray, cval=-1024.0) -> np.ndarray:
    """Resample a volume onto a grid of world-space points.

    px_world: (Ny, Nx, 3) world coordinates (RAS mm).
    Returns: (Ny, Nx) sampled values.
    """
    vox = world_to_voxel_grid(affine, px_world.reshape(-1, 3))
    sampled = resample_plane(volume, affine, vox.T, order=1, cval=cval)
    return sampled.reshape(px_world.shape[:2])


def normalize_hu(ct_plane: np.ndarray, wl=40.0, ww=400.0, clip=True) -> np.ndarray:
    """Windowing normalization of HU -> [0,1]."""
    lo = wl - ww * 0.5
    hi = wl + ww * 0.5
    out = (np.asarray(ct_plane, dtype=np.float32) - lo) / (hi - lo)
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=lambda o: float(o))
    return path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)