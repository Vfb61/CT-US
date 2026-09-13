"""Probe geometry: scan plane construction and ray-resampling into a volume.

Defines the geometric mapping between the CT anatomy space and the ultrasound
scan space.  A probe pose (face point + lateral/beam/elevation axes) combined
with transducer parameters produces a world-space coordinate for every B-mode
pixel; uniform resampling then yields the CT/tissue/scatter planes that the
renderer consumes.  The same frame defines the ground-truth affine
T_CT<->US  stored in the dataset.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from .io_utils import (world_to_voxel_grid, resample_plane, plane_affine,
                       affine_inverse)


# ----------------------------------------------------------------------------
# Rigid-frame helpers
# ----------------------------------------------------------------------------

def orthonormal_frame(beam: np.ndarray, lateral: np.ndarray | None = None,
                      rng=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build right-handed probe frame (u_lateral, v_elevation, w_beam).

    w_beam points into the body.  If lateral is None a random perpendicular
    direction is chosen.
    """
    rng = rng if rng is not None else np.random.default_rng()
    w = np.asarray(beam, dtype=np.float64).reshape(3)
    w = w / np.linalg.norm(w)
    if lateral is None:
        ref = np.array([1.0, 0.0, 0.0]) if abs(w[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(w, ref)
        u = u / np.linalg.norm(u)
    else:
        u = np.asarray(lateral, dtype=np.float64).reshape(3)
        u = u - np.dot(u, w) * w
        n = np.linalg.norm(u)
        if n < 1e-6:
            ref = np.array([1.0, 0.0, 0.0]) if abs(w[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            u = np.cross(w, ref)
            u = u / np.linalg.norm(u)
        else:
            u = u / n
    v = np.cross(w, u)
    v = v / np.linalg.norm(v)
    return u, v, w


def rotate_about(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix (3x3)."""
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    return np.array([
        [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
    ])


# ----------------------------------------------------------------------------
# Probe model
# ----------------------------------------------------------------------------

class Probe:
    """Ultrasound transducer geometry.

    Parameters (all physical units mm or rad):
      kind        'linear' | 'convex'
      nx, nz      image resolution (lateral samples, depth samples)
      dx          lateral pitch (mm) for linear
      fov_angle   angular coverage (rad) for convex
      nz_depth    depth axis length per pixel (mm)
      near        near-field offset (mm) from face
      radius      virtual apex radius behind the face for convex (mm)
      freq        centre frequency (MHz), used by renderer/calibration
    """

    def __init__(self, kind="convex", nx=320, nz=280, dx=0.55,
                 fov_angle=np.deg2rad(78.0), dz=0.5, near=2.0,
                 radius=60.0, freq=3.5):
        self.kind = kind
        self.nx = int(nx)
        self.nz = int(nz)
        self.dx = float(dx)
        self.dz = float(dz)
        self.fov_angle = float(fov_angle)
        self.near = float(near)
        self.radius = float(radius)
        self.freq = float(freq)
        if kind == "convex":
            self.lateral_span = 2 * np.sin(0.5 * self.fov_angle) * self.radius
            self.dx_effective = self.lateral_span / self.nx
        else:
            self.lateral_span = float(nx) * float(dx)
            self.dx_effective = self.dx
        self.depth_span = self.near + self.nz * self.dz

    # -- geometry tables ----------------------------------------------------
    def depth_offsets(self):
        """Depth (mm) along each image row from the near field."""
        return self.near + np.arange(self.nz) * self.dz

    def lateral_offsets(self):
        """Lateral (mm) of each image column relative to the face centre."""
        if self.kind == "convex":
            angles = (np.arange(self.nx) - (self.nx - 1) / 2) * (self.fov_angle / (self.nx - 1))
            return np.sin(angles) * self.radius
        return (np.arange(self.nx) - (self.nx - 1) / 2) * self.dx

    def pixel_affine(self, face, u, v, w) -> np.ndarray:
        """4x4 affine mapping (x_us_lateral, y_us_depth, z_us_elev, 1) -> world.

        This is the geometric ground truth T_US->CT for a iso-tropic pixel
        spacing; both linear and convex share this local linear map only at the
        pixel grid (convex is handled by the grid itself).
        """
        return plane_affine(face, u, w, v)

    # -- world grid ---------------------------------------------------------
    def world_grid(self, face, u, v, w, deform=None):
        """World-space coordinate of every B-mode pixel.

        deform: optional callable(lat_mm, depth_mm) -> (dlat, ddepth) 3-vector
                displacement added to the ideal (lat, depth) position, in probe
                local coordinates (lateral/beam/elevation).
        Returns ndarray (nx, nz, 3).
        """
        la = self.lateral_offsets()
        de = self.depth_offsets()
        if self.kind == "convex":
            angles = (np.arange(self.nx) - (self.nx - 1) / 2) * (self.fov_angle / (self.nx - 1))
            dirs = []
            for a in angles:
                dirs.append(rotate_about(v, a) @ w)
            dirs = np.stack(dirs, axis=0)  # (nx, 3)
            apex = face - w * self.radius
            L, D = np.meshgrid(np.arange(self.nx), np.arange(self.nz), indexing="ij")
            radii = self.radius + self.near + D * self.dz
            # (nx, nz, 3)
            world = apex[None, None, :] + dirs[:, None, :] * radii[:, :, None]
        else:
            lat = la
            d = de[None, :]
            world = face[None, None, :] + u[None, None, :] * lat[:, None, None] \
                + w[None, None, :] * d[:, :, None]
        if deform is not None:
            latg, depg = np.meshgrid(la, de, indexing="ij")
            disp = deform(latg, depg)  # (nx, nz, 3) in probe local coords
            world = world + (u[None, None, :] * disp[..., 0][..., None]
                             + w[None, None, :] * disp[..., 1][..., None]
                             + v[None, None, :] * disp[..., 2][..., None])
        return world

    def resample(self, volume, affine, world_grid, cval=-1024.0, order: int = 1):
        """Resample a volume onto the pixel world grid -> (nx, nz) plane."""
        look = world_grid.reshape(-1, 3)
        vox = world_to_voxel_grid(affine, look)
        s = resample_plane(np.asarray(volume, dtype=np.float32), affine, vox.T, order, cval)
        return s.reshape(self.nx, self.nz)

    def copy(self, **kwargs):
        p = Probe(
            kind=kwargs.get("kind", self.kind),
            nx=kwargs.get("nx", self.nx),
            nz=kwargs.get("nz", self.nz),
            dx=kwargs.get("dx", self.dx),
            fov_angle=kwargs.get("fov_angle", self.fov_angle),
            dz=kwargs.get("dz", self.dz),
            near=kwargs.get("near", self.near),
            radius=kwargs.get("radius", self.radius),
            freq=kwargs.get("freq", self.freq),
        )
        for k, v in kwargs.items():
            if hasattr(p, k):
                setattr(p, k, v)
        return p


def probe_from_meta(meta: dict) -> Probe:
    """从 `transform.npz` / `params.json["probe"]` 的字段重建 Probe。

    关键点：`probe.dx` 对 convex 是**无意义的占位值**（例如 0.55，而真值是 0.1686），
    真正可信的是 `dx_eff = lateral_span / nx`。因此这里优先用 meta 里的
    `lateral_span`（缺失时用 `dx_eff * nx` 反推），再据此设置 `dx`，
    避免下游把占位值当成横向像素间距。
    """
    kind = str(meta.get("kind", "convex"))
    nx = int(meta.get("nx", 320))
    nz = int(meta.get("nz", 280))
    dx = float(meta.get("dx", 0.55))
    span = meta.get("lateral_span", None)
    dx_eff = meta.get("dx_eff", None)
    if span is None and dx_eff is not None:
        span = float(dx_eff) * nx
    if span is not None:
        span = float(span)
        if kind == "convex":
            # 已知 fov/radius 时反解 dx 无意义；直接用 span 设置 lateral_span，
            # 并把 dx 设为等价 pitch 以便下游按 lateral_span/nx 取值。
            dx = span / nx
        else:
            dx = span / nx
    p = Probe(kind=kind, nx=nx, nz=nz, dx=dx,
              fov_angle=float(meta.get("fov_angle", np.deg2rad(78.0))),
              dz=float(meta.get("dz", 0.5)),
              near=float(meta.get("near", 2.0)),
              radius=float(meta.get("radius", 60.0)),
              freq=float(meta.get("freq", 3.5)))
    if span is not None:
        p.lateral_span = span
        p.dx_effective = span / nx
    return p


def world_grid_from_meta(meta: dict, pose: dict):
    """由 meta(含 probe 字段) + pose(face/u/v/w) 重建扫描平面世界网格 (nx, nz, 3)。

    这是**精确**几何来源（convex 的网格在 (i,j) 下非线性，无法用 affine 表达）。
    """
    p = probe_from_meta(meta)
    face = np.asarray(pose["face"], dtype=np.float64)
    u = np.asarray(pose["u"], dtype=np.float64)
    v = np.asarray(pose["v"], dtype=np.float64)
    w = np.asarray(pose["w"], dtype=np.float64)
    return p.world_grid(face, u, v, w, deform=None)


def build_probe(param: dict | None = None) -> Probe:
    """Create a Probe from an (optionally partially filled) parameter dict."""
    defaults = dict(kind="convex", nx=320, nz=280, dx=0.55,
                    fov_angle=np.deg2rad(78.0), dz=0.5, near=2.0,
                    radius=60.0, freq=3.5)
    if param:
        defaults.update(param)
    if isinstance(defaults["fov_angle"], (list, tuple)):
        defaults["fov_angle"] = np.deg2rad(defaults["fov_angle"][0])
    return Probe(**defaults)