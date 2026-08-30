"""Probe pose sampling on the liver capsule surface.

Mimics intraoperative open-surgery ultrasound: the transducer is pressed onto
the liver capsule, so the probe face is placed a small stand-off above the
capsule and the beam axis points into the organ.  Randomisation of position,
tilt, lateral roll and stand-off produces diverse scanning geometries with the
anatomy remaining reliably inside the field of view.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.measure import marching_cubes

from .geometry import orthonormal_frame, rotate_about


class LiverSurface:
    """Pre-extracted capsule mesh + utilities for pose sampling."""

    def __init__(self, liver_mask: np.ndarray, affine: np.ndarray):
        m = np.asarray(liver_mask, dtype=np.float32) - 0.5
        try:
            verts, faces, normals, _ = marching_cubes(m, level=0.0)
        except Exception:  # noqa: BLE001
            verts, faces, normals = self._fallback(liver_mask)
        # voxel indices -> world mm
        self.verts_world = np.asarray(liver_mask.shape) * 0
        world = affine[:3, :3] @ verts.T + affine[:3, 3:4]
        self.verts_world = world.T.astype(np.float64)
        # outward normals (already in voxel frame) -> world (rotate only)
        self.normals_world = (affine[:3, :3] @ normals.T).T
        for k in range(self.normals_world.shape[0]):
            n = np.linalg.norm(self.normals_world[k])
            if n > 1e-9:
                self.normals_world[k] /= n
        self.liver_mask = np.asarray(liver_mask, dtype=bool)
        self.affine = np.asarray(affine, dtype=np.float64)
        if self.verts_world.shape[0] == 0:
            raise ValueError("Liver mask produced an empty surface mesh")

    @staticmethod
    def _fallback(liver_mask):
        idx = np.argwhere(np.asarray(liver_mask) > 0)
        if idx.size == 0:
            raise ValueError("empty liver mask")
        cx = idx.mean(axis=0)
        d = idx - cx
        r = np.linalg.norm(d, axis=1)
        verts = d[r > np.percentile(r, 98)]
        return verts, np.array([[0, 1, 2]]), verts / (np.linalg.norm(verts, axis=1, keepdims=True) + 1e-9)

    # -- sampling -----------------------------------------------------------
    def sample_point(self, rng=None) -> tuple[np.ndarray, np.ndarray]:
        rng = rng if rng is not None else np.random.default_rng()
        if rng.random() < 0.35:
            # surface voxel with high local curvature -> capsule / dome
            dist = ndimage.distance_transform_edt(np.logical_not(self.liver_mask))
            ring = (dist > 0.0) & (dist < 3.0)
            candidates = np.argwhere(ring)
            if candidates.shape[0] == 0:
                candidates = np.argwhere(self.liver_mask)
            idx = candidates[rng.integers(0, candidates.shape[0])]
            p_vox = idx.astype(np.float64) + rng.random(3)
            p_world = self.affine[:3, :3] @ p_vox + self.affine[:3, 3]
            # normal from distance gradient
            g = np.stack(np.gradient(dist.astype(np.float32)), axis=0)
            n = g[:, idx[0], idx[1], idx[2]]
            nn = np.linalg.norm(n)
            n_world = self.affine[:3, :3] @ (n / max(nn, 1e-9))
            nn = np.linalg.norm(n_world)
            return p_world, n_world / max(nn, 1e-9)
        k = rng.integers(0, self.verts_world.shape[0])
        return self.verts_world[k].copy(), self.normals_world[k].copy()


def sample_probe_pose(liver_surf: LiverSurface, rng=None) -> dict:
    """Randomly sample a probe pose targeting the liver capsule.

    Returns dict with keys: face, u, v, w, standoff, tilt, roll, kind.
    face is the transducer face centre (world mm), w the unit beam axis into
    the liver, u the lateral (array) axis and v the elevation axis.
    """
    rng = rng if rng is not None else np.random.default_rng()
    p_world, n_world = liver_surf.sample_point(rng)

    standoff = rng.uniform(3.0, 14.0)          # distance surface->face (mm)
    face = p_world + n_world * standoff         # outside the liver
    w = -n_world                                # beam points into the liver

    tilt = np.deg2rad(rng.uniform(0.0, 26.0))
    if rng.random() < 0.5:
        tilt = 0.0                              # keep many near-normal poses
    roll = rng.uniform(0.0, 2 * np.pi)          # random rotation about beam

    # random tangent -> lateral axis
    u, v, _ = orthonormal_frame(w, rng=rng)
    # roll
    Rr = rotate_about(w, roll)
    u = Rr @ u
    v = np.cross(w, u)
    v /= np.linalg.norm(v)
    # tilt the whole frame by `tilt` in a random plane containing w
    tilt_axis = v.copy()
    Rr2 = rotate_about(tilt_axis, tilt)
    w = Rr2 @ w
    u = Rr2 @ u
    v = Rr2 @ v

    return {
        "face": face, "u": u, "v": v, "w": w,
        "standoff": float(standoff), "tilt": float(np.degrees(tilt)),
        "roll": float(roll), "kind": str(liver_surf.liver_mask.dtype),
    }


def plane_liver_coverage(liver_surf: LiverSurface, probe,
                         pose: dict, coarse: int = 3) -> tuple[float, float]:
    """Measure how much liver is inside a candidate scan plane.

    Returns (fraction_of_image_pixels, max_liver_depth_mm).  Used to reject
    poses whose plane only skims the organ.
    """
    from .geometry import build_probe
    from .io_utils import world_to_voxel_grid
    pr = probe.copy(nx=probe.nx // coarse, nz=probe.nz // coarse)
    world = pr.world_grid(pose["face"], pose["u"], pose["v"], pose["w"])
    look = world.reshape(-1, 3)
    vox = world_to_voxel_grid(liver_surf.affine, look)
    ix, iy, iz = vox[:, 0].round().astype(int), vox[:, 1].round().astype(int), vox[:, 2].round().astype(int)
    shp = np.array(liver_surf.liver_mask.shape)
    ok = (ix >= 0) & (ix < shp[0]) & (iy >= 0) & (iy < shp[1]) & (iz >= 0) & (iz < shp[2])
    c0, c1, c2 = np.clip(ix, 0, shp[0] - 1), np.clip(iy, 0, shp[1] - 1), np.clip(iz, 0, shp[2] - 1)
    img = np.where(ok, liver_surf.liver_mask[c0, c1, c2], False).reshape(pr.nx, pr.nz)
    frac = img.mean()
    depth_positions = np.where(img.any(axis=0))[0]
    max_d = 0.0
    if depth_positions.size:
        max_d = probe.depth_offsets()[depth_positions[-1]]
    return float(frac), float(max_d)