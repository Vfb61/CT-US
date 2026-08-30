"""Acoustic artefact operators applied in the scan-plane envelope domain.

All operators are pure 2D transforms that take a linear envelope image (and
per-column tissue parameters) and return a modified envelope image, so they can
be toggled / parameterised independently.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter1d

EPS = 1e-9


def acoustic_shadow(env: np.ndarray, alpha_line: np.ndarray,
                    specular_mask: np.ndarray, dz_cm: float, freq: float,
                    shadow_db: float = 18.0, spec_block_db: float = 26.0) -> np.ndarray:
    """Shadow cast behind strong reflectors and highly attenuating tissue.

    alpha_line: (nx, nz) per-pixel attenuation dB/(cm*MHz)
    specular_mask: (nx, nz) boolean strong specular reflectors (e.g. capsule)
    """
    path_db = alpha_line * dz_cm * freq            # per-pixel two-way dB
    path_db = np.clip(path_db, 0.0, 1.0)
    block = path_db + specular_mask * (spec_block_db * dz_cm)
    cum = np.cumsum(block, axis=1)
    shadow = 10.0 ** (-np.clip(cum, 0.0, shadow_db) / 20.0)
    return (env * shadow).astype(env.dtype)


def posterior_enhancement(env: np.ndarray, alpha_line: np.ndarray,
                          ref_alpha: float, dz_cm: float, freq: float,
                          max_boost_db: float = 9.0) -> np.ndarray:
    """Posterior enhancement behind low-attenuation (anechoic) structures.

    A column whose integrated attenuation is below a reference tissue column
    receives a depth-dependent boost once the deficit accumulates.
    """
    deficit_db = np.cumsum((ref_alpha - alpha_line) * dz_cm * freq, axis=1)
    deficit_db = np.clip(deficit_db, 0.0, max_boost_db)
    boost = 10.0 ** (deficit_db / 20.0)
    return (env * boost).astype(env.dtype)


def edge_shadow(env: np.ndarray, shadow_img: np.ndarray,
                lat_px: float = 3.0, strength: float = 0.55) -> np.ndarray:
    """Dark edge wedges produced at the lateral margins of anechoic / shadowed
    structures (refraction & out-of-plane scatter loss).
    """
    smooth = gaussian_filter(np.asarray(shadow_img, dtype=np.float64), (lat_px, 2.0))
    lap = np.gradient(np.gradient(smooth, axis=0), axis=0)
    wedge = -np.clip(lap, 0.0, None) * strength
    wedge = gaussian_filter(wedge, (1.0, 3.0))
    return (env / (1.0 + wedge)).astype(env.dtype)


def reverberation(env: np.ndarray, surface_specular: np.ndarray,
                  depth_mm: np.ndarray, spacing_mm: float = 6.0,
                  n_ghosts: int = 3, decay: float = 0.62,
                  gain: float = 0.35) -> np.ndarray:
    """Reverberation: ghost echoes at multiples of a fixed spacing below strong
    near-surface reflectors (e.g. probe-tissue interface, capsule).
    """
    out = np.array(env, dtype=np.float64, copy=True)
    shift = int(round(spacing_mm / (depth_mm[1] - depth_mm[0]))) if len(depth_mm) > 1 else 8
    surface = np.asarray(surface_specular, dtype=np.float64)
    for k in range(1, n_ghosts + 1):
        kshift = k * shift
        if kshift >= env.shape[1] - 2:
            break
        ghost = env[:, :-kshift]
        surf = surface[:, :-kshift]
        out[:, kshift:] += (ghost * surf * (decay ** k) * gain)
    return np.clip(out, 0.0, None).astype(env.dtype)


def additive_noise(env: np.ndarray, snr_db: float = 38.0, rng=None) -> np.ndarray:
    """Signal-dependent thermal/RF noise in the envelope domain.

    snr_db is the reference signal-to-noise ratio at the envelope peak.
    """
    rng = rng if rng is not None else np.random.default_rng()
    peak = np.max(env) + EPS
    sigma = peak * 10.0 ** (-snr_db / 20.0)
    return (env + rng.normal(0, sigma, env.shape)).astype(env.dtype)


def speckle_denoise_lumen(env: np.ndarray, lumen_mask: np.ndarray,
                          strength: float = 0.6) -> np.ndarray:
    """Spatial smoothing inside large anechoic regions (partially developed,
    low-contrast speckle inside vessel lumina / cysts)."""
    if np.count_nonzero(lumen_mask) == 0:
        return env
    smooth = gaussian_filter(env, (3.0, 2.5))
    out = np.where(lumen_mask, (1 - strength) * env + strength * smooth, env)
    return out.astype(env.dtype)


def lateral_resolution(env: np.ndarray, sigma_lat: float, sigma_ax: float) -> np.ndarray:
    """PSF-like lateral/axial blur applied to the RF/envelope image."""
    return gaussian_filter(env, (sigma_lat, sigma_ax)).astype(env.dtype)