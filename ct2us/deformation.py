"""Tissue deformation between pre-operative CT and intra-operative ultrasound.

Implementations produce a displacement field in probe-local coordinates
(lateral, beam/depth, elevation).  Two mechanisms:

  * Respiratory motion — a smooth sinusoidally modulated translation whose
    amplitude peaks at the probe centre and decays with depth;
  * Probe compression — the transducer presses into the tissue, so points
    between the surface and a depth L0 are displaced toward the probe.

The displacement field is evaluated on the (lateral x depth) image grid and
passed to geometry.Probe.world_grid, i.e. sampling coordinates are warped while
the probe itself stays fixed — exactly what a live intra-operative sweep sees.
"""

from __future__ import annotations

import numpy as np


def build_deformation(probe, params: dict, rng=None) -> callable:
    """Return deform(lat_mm, depth_mm) -> (nx, nz, 3) displacement (mm).

    lat / depth are (nx, nz) grids.  params keys:
      resp_amp_mm      max respiratory displacement along beam (default 0)
      resp_phase_rad   phase (default random)
      compression_mm   max probe-compression displacement (default 0)
      compress_depth_mm depth (mm) over which compression acts (default 0.35*depth)
    """
    rng = rng if rng is not None else np.random.default_rng()
    resp = float(params.get("resp_amp_mm", 0.0))
    phase = float(params.get("resp_phase_rad", rng.uniform(0, 2 * np.pi)))
    comp = float(params.get("compression_mm", 0.0))
    L0 = float(params.get("compress_depth_mm", 0.35 * probe.depth_span))
    jitter = float(params.get("lateral_jitter_mm", 0.0))

    lat_scale = max(probe.lateral_span * 0.5, 1e-3)
    depth_scale = max(probe.depth_span, 1e-3)

    def deform(lat, depth):
        lat = np.asarray(lat, dtype=np.float64)
        depth = np.asarray(depth, dtype=np.float64)
        d_lat = np.zeros_like(lat)
        d_dep = np.zeros_like(depth)
        d_ele = np.zeros_like(lat)

        if resp > 0:
            env = np.exp(-0.5 * (lat / lat_scale) ** 2) \
                * np.exp(-0.5 * (depth / depth_scale) ** 2)
            d_dep += resp * np.sin(phase) * env

        if comp > 0 and L0 > 0:
            # displacement is negative (toward the probe) inside the cap
            frac = np.clip(depth / L0, 0.0, 1.0)
            cap = np.where(depth <= L0, 1.0, 0.0)
            env = np.exp(-0.5 * (lat / (0.6 * lat_scale)) ** 2)
            d_dep += (-comp) * cap * (1.0 - frac) * env

        if jitter > 0:
            d_lat += jitter * np.exp(-0.5 * (depth / depth_scale) ** 2)

        return np.stack([d_lat, d_dep, d_ele], axis=-1)

    return deform


def random_deform_params(probe, rng=None, enabled=True) -> dict:
    """Sample deformation parameters for one generated slice."""
    rng = rng if rng is not None else np.random.default_rng()
    if not enabled or rng.random() < 0.35:
        return {"resp_amp_mm": 0.0, "compression_mm": 0.0,
                "resp_phase_rad": 0.0, "lateral_jitter_mm": 0.0}
    p = {}
    p["resp_amp_mm"] = float(rng.uniform(0.5, 4.5))
    p["compression_mm"] = float(rng.uniform(0.0, 6.0))
    p["resp_phase_rad"] = float(rng.uniform(0, 2 * np.pi))
    p["lateral_jitter_mm"] = float(rng.uniform(0.0, 0.6))
    return p