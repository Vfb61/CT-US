"""B-mode renderer: from resampled scan planes to a synthetic ultrasound image.

The renderer works entirely in the scan plane (lateral x depth).  Inputs are
the resampled CT, tissue label and scatter planes plus the attenuation field;
output is a linear envelope image which is then log-compressed and quantised,
optionally with the full set of acoustic artefacts enabled.

Parameter dictionary (render):
  global_gain       base backscatter multiplier
  alpha_scale       multiplies the tissue attenuation field
  tgc_slope_db_cm   TGC slope (dB per cm) at the probe frequency
  dr_db             display dynamic range (dB)
  gamma             display gamma
  speckle_strength  [0,1] global speckle contrast
  speckle_lat_px    speckle lateral correlation width (px)
  speckle_ax_px     speckle axial correlation width (px)
  specular_gain     capsule/interface specular echo gain
  wall_echo_gain    vessel-wall echo gain
  shadow_db         max shadow depth (dB)
  spec_block_db     specular interface extra attenuation (dB/cm)
  posterior_db      max posterior enhancement (dB)
  edge_shadow_w     edge-shadow strength
  reverb_gain       reverberation gain (0 disables)
  noise_snr_db      additive noise SNR (dB)
  lumen_denoise     smoothing of speckle inside lumina (0..1)
"""

from __future__ import annotations

import numpy as np

from . import anatomy as an
from . import artifacts as art
from . import speckle as sp


DEFAULT_PARAMS = {
    "global_gain": 2.2,
    "alpha_scale": 1.0,
    "tgc_slope_db_cm": 0.7,
    "dr_db": 62.0,
    "gamma": 1.0,
    "speckle_strength": 0.7,
    "speckle_lat_px": 5.0,
    "speckle_ax_px": 1.3,
    "specular_gain": 1.2,
    "wall_echo_gain": 1.5,
    "shadow_db": 20.0,
    "spec_block_db": 26.0,
    "posterior_db": 8.0,
    "edge_shadow_w": 0.5,
    "reverb_gain": 0.3,
    "noise_snr_db": 38.0,
    "lumen_denoise": 0.6,
    "post_blur": (0.5, 0.8),
}


def _fill(params, defaults):
    out = dict(defaults)
    if params:
        out.update({k: v for k, v in params.items() if v is not None})
    return out


def _alpha_plane(tissue_plane: np.ndarray, alpha_scale: float) -> np.ndarray:
    a = np.zeros(tissue_plane.shape, dtype=np.float32)
    for code, props in an.TISSUE_PROPS.items():
        a[tissue_plane == code] = props.alpha_db_cm_mhz
    return a * alpha_scale


def _specular_echo(ct_plane: np.ndarray, tissue_plane: np.ndarray,
                   params) -> np.ndarray:
    """Specular interface echoes from tissue boundaries oriented normal to the
    beam (strong) and from the CT gradient (capsule / tumour / wall rims)."""
    nx, nz = tissue_plane.shape
    spec = np.zeros((nx, nz), dtype=np.float32)

    # tissue-boundary map: label changes along depth (beam) or laterally
    grad_lab = np.abs(np.gradient(tissue_plane.astype(np.float32), axis=1)) > 0.5
    grad_lat = np.abs(np.gradient(tissue_plane.astype(np.float32), axis=0)) > 0.5

    # reflectivity: interfaces perpendicular to the beam echo strongest.
    # Beam direction is along the depth axis (axis=1).
    # normal in (lat, depth) plane:
    gl = np.gradient(tissue_plane.astype(np.float32), axis=0)
    gd = np.gradient(tissue_plane.astype(np.float32), axis=1)
    gn = np.sqrt(gl ** 2 + gd ** 2 + 1e-9)
    ref = np.abs(gd) / gn                    # aligned with beam (0..1)

    tissue = tissue_plane
    ref_spec = np.zeros((nx, nz), dtype=np.float32)
    for code, props in an.TISSUE_PROPS.items():
        ref_spec[tissue == code] = props.specular
    spec[grad_lab] = ref_spec[grad_lab] * ref[grad_lab] * params["specular_gain"]
    spec[grad_lat] = ref_spec[grad_lat] * ref[grad_lat] * params["specular_gain"] * 0.5

    # capsule rim: strong echo on both sides of the liver boundary
    liver = tissue_plane == an.T_LIVER
    non_liver = ~liver
    edge_lat = np.logical_xor(liver[:, :-1], liver[:, 1:])
    edge_lat = np.pad(edge_lat, ((0, 0), (0, 1)), constant_values=False)
    edge_dep = np.logical_xor(liver[:-1, :], liver[1:, :])
    edge_dep = np.pad(edge_dep, ((0, 1), (0, 0)), constant_values=False)
    spec[edge_lat] = np.maximum(spec[edge_lat], 1.4 * params["specular_gain"])
    spec[edge_dep] = np.maximum(spec[edge_dep], 1.8 * params["specular_gain"])

    # vessel walls
    wall = tissue_plane == an.T_VESSEL_WALL
    spec[wall] = np.maximum(spec[wall], params["wall_echo_gain"])

    # CT-gradient-based capsule line (double-line bright rim)
    ctg = np.abs(np.gradient(ct_plane, axis=1)) + np.abs(np.gradient(ct_plane, axis=0))
    strong = (ctg > np.percentile(ctg, 92.0)) & (liver | non_liver)
    spec[strong] = np.maximum(spec[strong], params["specular_gain"] * 0.5)

    return spec


def _strength_map(tissue_plane: np.ndarray, params) -> np.ndarray:
    """Local speckle contrast: high in parenchyma, low in lumina/cysts."""
    s = np.ones(tissue_plane.shape, dtype=np.float32) * params["speckle_strength"]
    s[tissue_plane == an.T_VESSEL] *= 0.30
    s[tissue_plane == an.T_TUMOR] *= 0.85
    s[tissue_plane == an.T_AIR] *= 0.10
    s[tissue_plane == an.T_BONE] *= 0.20
    return np.clip(s, 0.0, 1.0)


def render_bmode(ct_plane: np.ndarray, tissue_plane: np.ndarray,
                 scatter_plane: np.ndarray, probe,
                 params: dict | None = None, rng=None,
                 want_envelope: bool = True):
    """Render one synthetic B-mode slice.

    ct_plane / tissue_plane / scatter_plane have shape (nx, nz) as produced by
    geometry.Probe.resample.
    probe: geometry.Probe instance carrying freq / pixel spacing / depth axis.
    Returns dict with keys: bmode (float 0..1), uint8, envelope, params, and
    tissue/alphas used (for calibration).
    """
    p = _fill(params, DEFAULT_PARAMS)
    rng = rng if rng is not None else np.random.default_rng()

    alpha = _alpha_plane(tissue_plane, p["alpha_scale"])
    spec = _specular_echo(ct_plane, tissue_plane, p)

    # 1) base backscatter envelope
    base = np.asarray(scatter_plane, dtype=np.float32) * p["global_gain"]
    env = base + spec

    # 2) multiplicative speckle
    smap = _strength_map(tissue_plane, p)
    m = sp.generate_speckle(env.shape, smap,
                            sigma_lat=p["speckle_lat_px"],
                            sigma_ax=p["speckle_ax_px"], rng=rng)
    env = env * m

    # 3) two-way depth attenuation
    dz_cm = probe.dz / 10.0
    att = np.zeros_like(env)
    cum = np.cumsum(alpha * dz_cm * probe.freq, axis=1)
    att = 10.0 ** (-np.clip(cum, 0.0, 90.0) / 10.0)
    env = env * att

    # 4) TGC (linear-domain compensation, applied mildly)
    depths = probe.depth_offsets() / 10.0  # cm
    gain_db = p["tgc_slope_db_cm"] * depths
    tgc = 10.0 ** (np.clip(gain_db, 0.0, 40.0) / 20.0)
    env = env * tgc[None, :]

    # 5) artefacts
    spec_mask = spec > 0.5 * max(spec.max(), 1e-6)
    env = art.acoustic_shadow(env, alpha, spec_mask, dz_cm, probe.freq,
                              shadow_db=p["shadow_db"], spec_block_db=p["spec_block_db"])
    env = art.posterior_enhancement(env, alpha, ref_alpha=0.7,
                                    dz_cm=dz_cm, freq=probe.freq,
                                    max_boost_db=p["posterior_db"])

    shadow_ref = art.acoustic_shadow(np.ones_like(env), alpha, spec_mask,
                                     dz_cm, probe.freq,
                                     shadow_db=p["shadow_db"], spec_block_db=p["spec_block_db"])
    env = art.edge_shadow(env, shadow_ref, lat_px=3.0, strength=p["edge_shadow_w"])

    if p["reverb_gain"] > 0:
        near = depths < 2.5
        surf = spec.max(axis=1, keepdims=True) > 0.5 * max(spec.max(), 1e-6)
        surf2d = np.broadcast_to(surf[:, 0:1], env.shape) & near[None, :]
        env = art.reverberation(env, surf2d.astype(np.float32),
                                probe.depth_offsets(), spacing_mm=6.0,
                                n_ghosts=3, decay=0.62, gain=p["reverb_gain"])

    lumen = tissue_plane == an.T_VESSEL
    if p["lumen_denoise"] > 0:
        env = art.speckle_denoise_lumen(env, lumen, strength=p["lumen_denoise"])

    env = art.additive_noise(env, snr_db=p["noise_snr_db"], rng=rng)

    # 6) PSF-like final blur + envelope/log compression
    if p["post_blur"] is not None:
        env = art.lateral_resolution(env, p["post_blur"][0], p["post_blur"][1])

    bmode = sp.log_compress(env, dynamic_range_db=p["dr_db"], gamma=p["gamma"])
    out = {
        "bmode": bmode.astype(np.float32),
        "uint8": sp.to_uint8(bmode),
        "envelope": env,
        "alpha_plane": alpha,
        "tissue_plane": tissue_plane,
        "specular_plane": spec,
        "params": p,
    }
    if not want_envelope:
        out.pop("envelope", None)
    return out