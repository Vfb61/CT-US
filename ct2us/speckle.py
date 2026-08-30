"""Speckle field synthesis and envelope/log signal operators.

Speckle is generated in the scan plane with an anisotropic two-dimensional
Gaussian pass-band filter applied to analytic white noise, which reproduces the
Rayleigh envelope statistics of fully-developed speckle whose axial/lateral
correlation lengths follow the transducer point-spread function.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def correlated_complex(shape: tuple[int, int], sigma_lat: float, sigma_ax: float,
                       rng) -> np.ndarray:
    """Anisotropically correlated complex Gaussian -> Rayleigh envelope (nx, nz)."""
    nx, nz = shape
    z = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    zf = gaussian_filter(z.real, (sigma_lat, sigma_ax)) + 1j * gaussian_filter(z.imag, (sigma_lat, sigma_ax))
    return np.abs(zf).astype(np.float32)


def generate_speckle(shape: tuple[int, int], strength_map: np.ndarray,
                     sigma_lat: float = 4.0, sigma_ax: float = 1.2,
                     rng=None) -> np.ndarray:
    """Multiplicative speckle field of unit mean.

    strength_map: (nx, nz) in [0,1] controlling local speckle contrast
    (low inside anechoic lumina / cystic regions).  sigma_lat / sigma_ax are
    correlation widths in pixels along lateral / axial directions.

    Returns a float32 array with mean approx 1.
    """
    rng = rng if rng is not None else np.random.default_rng()
    env = correlated_complex(shape, sigma_lat, sigma_ax, rng)
    m = env / (env.mean() + 1e-9)          # unity-mean Rayleigh envelope
    sm = np.clip(np.asarray(strength_map, dtype=np.float32), 0.0, 1.0)
    out = 1.0 + sm * (m - 1.0)
    return np.clip(out, 1e-3, 8.0).astype(np.float32)


def time_gain_comp(db_per_cm: float = 0.6, freq: float = 3.5,
                   depth_cm: np.ndarray | None = None, nz: int = 240,
                   dz_cm: float = 0.05) -> np.ndarray:
    """Time-gain-compensation gain profile along the depth axis.

    Returns a (nz,) linear gain; parameter db_per_cm is the nominal tissue
    attenuation in dB/cm that the TGC attempts to compensate at freq MHz.
    """
    if depth_cm is None:
        depth_cm = np.arange(nz) * dz_cm
    depth_cm = np.asarray(depth_cm, dtype=np.float64)
    alpha_per_cm = db_per_cm * freq * 0.25  # DGC slope in dB/cm (reduced to avoid over-boost)
    gain_db = alpha_per_cm * depth_cm
    gain_db = np.clip(gain_db, 0.0, 46.0)
    return (10.0 ** (gain_db / 20.0)).astype(np.float32)


def log_compress(env: np.ndarray, dynamic_range_db: float = 60.0,
                 pct: float = 99.0, gamma: float = 1.0) -> np.ndarray:
    """Envelope -> [0,1] log-compressed B-mode intensity.

    The 100-DR-dB reference is placed at the pct-th percentile of the envelope
    (in the log domain), which makes the displayed image robust to global gain.
    """
    env = np.asarray(env, dtype=np.float64)
    env = np.maximum(env, 1e-12)
    ref = np.percentile(env, pct)
    dB = 20.0 * np.log10(env / max(ref, 1e-12))
    out = (dB + dynamic_range_db) / dynamic_range_db
    out = np.clip(out, 0.0, 1.0)
    if abs(gamma - 1.0) > 1e-6:
        out = np.power(out, gamma)
    return out.astype(np.float32)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0) * 255).astype(np.uint8)


def depth_attenuation_db(alpha_line: np.ndarray, dz_cm: float, freq: float) -> np.ndarray:
    """Two-way depth attenuation loss (linear multiplier) along one beam line.

    alpha_line: (nz,) per-pixel attenuation dB/(cm*MHz); dz_cm spacing.
    """
    cum = np.cumsum(np.asarray(alpha_line, dtype=np.float64) * dz_cm * freq)
    return (10.0 ** (-cum / 20.0)).astype(np.float32)