"""Realism calibration between the fast renderer and a physics reference.

Implements the six comparison axes from the scheme:

    Speckle Statistics, Depth Attenuation, Tissue Contrast, Boundary Echo,
    Vessel Appearance, Acoustic Shadow.

Each axis is a function of (envelope/uint8 image + tissue plane) reducing to a
small number of scalar statistics.  ``metric_distance`` collapses two metric
dicts into one scalar; ``calibrate_renderer`` performs a lightweight random
search over a subset of render parameters to minimise it.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from . import anatomy as an


# ----------------------------------------------------------------------------
# ROI helpers
# ----------------------------------------------------------------------------

def _roi(tissue_plane: np.ndarray, code: int, erode: int = 3) -> np.ndarray:
    from scipy import ndimage
    m = tissue_plane == code
    if erode > 0:
        structure = ndimage.generate_binary_structure(2, 2)
        m = ndimage.binary_erosion(m, structure, iterations=erode)
    return m


def _safe_mean(x, roi):
    vals = x[roi]
    if vals.size == 0:
        return 0.0
    return float(np.mean(vals))


def _safe_std(x, roi):
    vals = x[roi]
    if vals.size == 0:
        return 0.0
    return float(np.std(vals))


def _fit_slope_db_cm(depth_cm, log_env_row):
    m = np.isfinite(log_env_row) & (np.abs(log_env_row) < 1e12)
    if m.sum() < 4:
        return 0.0
    return float(np.polyfit(depth_cm[m], log_env_row[m], 1)[0] * 20.0 / np.log(10.0))


# ----------------------------------------------------------------------------
# Six metric axes
# ----------------------------------------------------------------------------

def speckle_statistics(env: np.ndarray, tissue: np.ndarray, **kw) -> dict:
    roi = _roi(tissue, an.T_LIVER, erode=6)
    if roi.sum() < 50:
        return {"speckle_snr": 0.0, "rayleigh_sigma_norm": 0.0,
                "axial_corr_mm": 0.0, "lateral_corr_mm": 0.0}
    e = _safe_mean(env, roi)
    es = _safe_std(env, roi)
    env_r = env[roi]
    sigma = float(np.sqrt(np.mean((env_r - e) ** 2 + 1e-12) / 2)) if env_r.size else 0.0
    # autocorrelation lengths (mm): normalise to unit mean, high-pass in 2-D
    x = np.where(roi, env.astype(np.float64), 0.0)
    x = x - gaussian_filter(x, (5.0, 8.0))
    cax = _corr_len(x, axis=1)
    clat = _corr_len(x, axis=0)
    return {
        "speckle_snr": float(es / (e + 1e-9)),
        "rayleigh_sigma_norm": float(sigma / (e + 1e-9)),
        "axial_corr_mm": cax,
        "lateral_corr_mm": clat,
    }


def _corr_len(x: np.ndarray, axis: int, max_lag: int = 40, n_lines: int = 24) -> float:
    """First zero-crossing of the line autocorrelation averaged over lines.

    axis=1: correlations along depth; axis=0: along lateral.  Returns the lag
    (in pixels) where the mean normalised autocorrelation first crosses zero.
    """
    if x.ndim != 2 or min(x.shape) < 8:
        return 0.0
    same = x if axis == 1 else x.T
    n_lines = min(n_lines, same.shape[0])
    idx = np.linspace(0, same.shape[0] - 1, n_lines).astype(int)
    ks = np.arange(1, min(max_lag, same.shape[1] - 2))
    corrs = []
    for li in idx:
        y = same[li]
        y = y - y.mean()
        denom = np.sum(y * y) + 1e-12
        if denom < 1e-12:
            continue
        cs = np.array([np.sum(y[:-k] * y[k:]) / denom for k in ks])
        corrs.append(cs)
    if not corrs:
        return 0.0
    c = np.mean(corrs, axis=0)
    for k, v in zip(ks, c):
        if v < 0:
            return float(k)
    return float(max_lag)


def depth_attenuation(env: np.ndarray, tissue: np.ndarray, depth_mm: np.ndarray, **kw) -> dict:
    roi = _roi(tissue, an.T_LIVER, erode=3)
    env_masked = np.where(roi, env, np.nan).astype(np.float64)
    row = np.nanmean(env_masked, axis=0)
    valid = np.zeros_like(depth_mm, dtype=bool)
    valid[:] = np.isfinite(row)
    lo, hi = int(0.1 * env.shape[1]), int(0.9 * env.shape[1])
    valid[lo:hi] = False
    if valid.sum() < 4:
        return {"attenuation_db_cm": 0.0}
    log_row = np.log(np.maximum(row[valid], 1e-12))
    dm = depth_mm[valid] / 10.0
    slope = float(np.polyfit(dm, log_row, 1)[0] * 20.0 / np.log(10.0))
    return {"attenuation_db_cm": slope}


def tissue_contrast(env: np.ndarray, tissue: np.ndarray, **kw) -> dict:
    liver = _roi(tissue, an.T_LIVER, erode=5)
    tu = _roi(tissue, an.T_TUMOR, erode=3)
    ve = _roi(tissue, an.T_VESSEL, erode=2)

    def db(a, b):
        ma = _safe_mean(np.log(env + 1e-12), a)
        mb = _safe_mean(np.log(env + 1e-12), b)
        if not np.isfinite(ma) or not np.isfinite(mb):
            return float("nan")
        return float(-20.0 * (ma - mb) / np.log(10.0))

    return {
        "tumor_vs_liver_db": db(tu, liver),
        "liver_vs_vessel_db": db(ve, liver),
    }


def boundary_echo(env: np.ndarray, tissue: np.ndarray, **kw) -> dict:
    """Peak intensity and FWHM thickness of the capsule/cancell boundary echo."""
    liver = tissue == an.T_LIVER
    edge = np.zeros_like(liver)
    edge[:-1] |= (liver[:-1] != liver[1:])
    edge[:, :-1] |= (liver[:, :-1] != liver[:, 1:])
    vals = env[edge]
    if vals.size == 0:
        return {"boundary_peak": 0.0, "boundary_fwhm_mm": 0.0}
    peak = float(np.percentile(vals, 99))
    # FWHM across a horizontal profile at the row of the peak
    r, c = np.unravel_index(np.argmax(env * edge), env.shape)
    prof = env[r, :]
    half = 0.5 * peak
    above = prof >= half
    if above.sum() < 2:
        fwhm = 0.0
    else:
        idx = np.where(above)[0]
        fwhm = float(idx.max() - idx.min())
    return {"boundary_peak": peak, "boundary_fwhm_mm": fwhm}


def vessel_appearance(env: np.ndarray, tissue: np.ndarray, **kw) -> dict:
    lum = _roi(tissue, an.T_VESSEL, erode=1)
    wall = tissue == an.T_VESSEL_WALL
    liver = _roi(tissue, an.T_LIVER, erode=3)
    l = _safe_mean(env, lum)
    w = _safe_mean(env, wall)
    p = _safe_mean(env, liver)
    if p <= 1e-12 or l <= 1e-12 or not np.isfinite(l) or not np.isfinite(p):
        lumd = float("nan")
    else:
        lumd = float(min(60.0, -20.0 * np.log(l / p) / np.log(10.0)))
    wr = float(w / (p + 1e-9)) if (np.isfinite(w) and np.isfinite(p)) else float("nan")
    wr = float(min(10.0, wr)) if np.isfinite(wr) else float("nan")
    return {
        "lumen_depth_db": lumd,
        "wall_echo_ratio": wr,
    }


def acoustic_shadow(env: np.ndarray, tissue: np.ndarray, spec: np.ndarray, **kw) -> dict:
    """Intensity loss behind strong specular reflectors relative to neighbours."""
    if spec.size == 0:
        return {"shadow_db": 0.0, "shadow_width_px": 0.0}
    strong = spec > 0.5 * max(spec.max(), 1e-6)
    cols = np.where(strong.max(axis=1))[0]
    if cols.size == 0:
        return {"shadow_db": 0.0, "shadow_width_px": 0.0}
    nz = env.shape[1]
    d0 = int(0.15 * nz)
    d1 = int(0.55 * nz)
    # average env in the window, excluding vessel branches that also darken
    win = env[:, d0:d1]
    in_col = win[cols]
    ref_cols = []
    for c in cols:
        for off in (12, -12):
            cc = c + off
            if 0 <= cc < env.shape[0] and cc not in cols:
                ref_cols.append(cc)
    if not ref_cols:
        return {"shadow_db": 0.0, "shadow_width_px": 0.0}
    ref = np.mean([win[cc] for cc in ref_cols])
    inc = np.mean(in_col)
    db = float(20.0 * np.log((inc + 1e-12) / (ref + 1e-12)) / np.log(10.0))
    width = float(np.median(np.diff(cols))) if len(cols) > 1 else 0.0
    return {"shadow_db": db, "shadow_width_px": width}


def metrics(image: np.ndarray, tissue: np.ndarray, depth_mm: np.ndarray,
            spec: np.ndarray | None = None) -> dict:
    """Full 6-axis metric dict for a linear-envelope image.

    The envelope is normalised to its 99th percentile so that the metrics are
    invariant to the global gain scale of the imaging chain.
    """
    env = np.asarray(image, dtype=np.float64)
    p99 = np.percentile(env, 99) + 1e-12
    env = env / p99
    m = {}
    m.update(speckle_statistics(env, tissue))
    m.update(depth_attenuation(env, tissue, depth_mm))
    m.update(tissue_contrast(env, tissue))
    m.update(boundary_echo(env, tissue))
    m.update(vessel_appearance(env, tissue))
    m.update(acoustic_shadow(env, tissue, spec if spec is not None else np.zeros_like(env)))
    return m


# ----------------------------------------------------------------------------
# Comparison + parameter calibration
# ----------------------------------------------------------------------------

def _norm(x, ref):
    return float(np.abs(x)) / (float(np.abs(ref)) + 1e-9)


def metric_distance(m1: dict, m2: dict, weights: dict | None = None) -> float:
    """Scalar distance between two metric dicts (relative L1 with defaults)."""
    w = weights or {
        "speckle_snr": 1.0,
        "rayleigh_sigma_norm": 0.5,
        "axial_corr_mm": 0.3,
        "lateral_corr_mm": 0.3,
        "attenuation_db_cm": 1.0,
        "liver_vs_vessel_db": 1.0,
        "tumor_vs_liver_db": 0.8,
        "boundary_peak": 0.5,
        "lumen_depth_db": 1.0,
        "wall_echo_ratio": 0.5,
        "shadow_db": 0.7,
    }
    keys = [k for k in w if k in m1 and k in m2]
    numer = 0.0
    denom = 0.0
    for k in keys:
        a, b = m1[k], m2[k]
        if not (np.isfinite(a) and np.isfinite(b) and abs(b) > 1e-12):
            continue
        numer += w[k] * abs(a - b) / abs(b)
        denom += w[k]
    if denom <= 0:
        return float("inf")
    return float(numer / denom * 100.0)


SCALAR_METRICS = [
    "speckle_snr", "rayleigh_sigma_norm", "axial_corr_mm", "lateral_corr_mm",
    "attenuation_db_cm", "liver_vs_vessel_db", "tumor_vs_liver_db",
    "boundary_peak", "lumen_depth_db", "wall_echo_ratio", "shadow_db",
]


def calibrate_renderer(render_fn, probe, geometry_inputs, reference_metrics: dict,
                       iterations: int = 40, rng=None, **fixed) -> dict:
    """Random multi-start search over render parameters minimising the distance
    to a physics reference metric set.

    render_fn(params) must return a dict with 'envelope' and 'tissue_plane'
    (as produced by render.render_bmode).  geometry_inputs are forwarded to it.
    """
    rng = rng if rng is not None else np.random.default_rng()
    best = None
    best_dist = float("inf")

    def candidate():
        c = {
            "speckle_strength": float(rng.uniform(0.3, 0.95)),
            "alpha_scale": float(rng.uniform(0.6, 1.5)),
            "specular_gain": float(rng.uniform(0.6, 1.8)),
            "shadow_db": float(rng.uniform(10.0, 30.0)),
            "posterior_db": float(rng.uniform(4.0, 12.0)),
            "global_gain": float(rng.uniform(0.8, 3.0)),
        }
        for k, v in fixed.items():
            c[k] = v
        return c

    for _ in range(iterations):
        p = candidate()
        out = render_fn(p)
        env = out["envelope"] if "envelope" in out else np.asarray(out.get("bmode"), dtype=np.float64)
        tissue = out["tissue_plane"]
        m = metrics(env, tissue, probe.depth_offsets(),
                    out.get("specular_plane"))
        d = metric_distance(m, reference_metrics)
        if d < best_dist:
            best_dist = d
            best = {"params": p, "metrics": m, "distance": d}
    return {"best": best, "distance": best_dist}