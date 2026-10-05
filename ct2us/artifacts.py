"""Acoustic artefact operators applied in the scan-plane envelope domain.

All operators are pure 2D transforms that take a linear envelope image (and
per-column tissue parameters) and return a modified envelope image, so they can
be toggled / parameterised independently.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter1d

EPS = 1e-9


def acoustic_shadow(env: np.ndarray, specular_mask: np.ndarray, dz_cm: float,
                    shadow_db: float = 18.0, spec_block_db: float = 26.0) -> np.ndarray:
    """Shadow cast behind strong reflectors (bone surface, gas, dense capsule).

    **不再接收 `alpha_line`。** 组织衰减已由 `render_bmode` 步骤⑤用
    `cum = cumsum(alpha * dz_cm * freq)` 统一施加过一次，而本函数原先对**同一个 α**
    再累积一遍，属于重复计入，且按列累积、不受 TGC 抵消。后果按组织分：

        肝实质 α=0.70 → 第二遍最多再扣 shadow_db(20 dB)，被压暗
        骨     α=6.00 → 第二遍最多再扣 20 dB，被压暗
        血管腔 α=0.16 → 第二遍几乎不扣，**相对被抬高**

    净效果是**把管腔相对抬亮**、抹掉"暗腔亮壁"——实测（关噪声）腔 0.260 vs
    肝 0.176，极性在源头就是反的。

    佐证：姊妹函数 `posterior_enhancement` 用的是相对于参考组织的**差额**
    `(ref_alpha - alpha_line)`，本函数却用绝对值；同一文件里处理同一件事的两个
    函数写法不对称，本身就是 bug 的特征。

    骨与气的声影改由 `specular_mask` 路径产生（骨 specular 0.95、气 0.90，都是强
    反射体），它们自身的高衰减仍由步骤⑤正确计入一次。

    specular_mask: (nx, nz) boolean strong specular reflectors (e.g. capsule)
    """
    cum = np.cumsum(specular_mask * (spec_block_db * dz_cm), axis=1)
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


def noise_sigma(env: np.ndarray, snr_db: float = 38.0, ref_pct: float = 99.0) -> float:
    """加性噪声的 σ = 组织参考包络（`ref_pct` 分位数）× 10^(-snr_db/20)。

    单独抽出来是为了让 `render_bmode` 能**在不复制公式的前提下**回传 σ，
    供"腔信号 vs 噪声底余量"这一诊断量使用。
    """
    ref = float(np.percentile(env, ref_pct)) + EPS
    return ref * 10.0 ** (-snr_db / 20.0)


def additive_noise(env: np.ndarray, snr_db: float = 38.0, rng=None,
                   ref_pct: float = 99.0) -> np.ndarray:
    """加性热噪声（包络域）。

    `snr_db` 是**组织参考信号与噪声底之比**，参考量取 `ref_pct` 分位数。

    **参考量必须用分位数而不是 max**：max 是镜面尖峰，可达实质水平的 10~30 倍，
    于是同样一个 `snr_db` 在不同位姿下对应完全不同的噪声底，既不可比也不可复现。
    更严重的是它把噪声底推到**高于血管腔自身的包络**：实测同一平面，噪声把
    「腔 − 肝」的差从 6.2 dB 压到 0.6 dB。零均值噪声本身不改均值，但它埋掉信号后，
    步骤⑪的对数压缩会把低信号区的灰度抬到噪声底对应值——腔的信息就是这样丢的。
    """
    rng = rng if rng is not None else np.random.default_rng()
    sigma = noise_sigma(env, snr_db, ref_pct)
    return (env + rng.normal(0, sigma, env.shape)).astype(env.dtype)


def speckle_denoise_lumen(env: np.ndarray, lumen_mask: np.ndarray,
                          strength: float = 0.6) -> np.ndarray:
    """Spatial smoothing inside large anechoic regions (partially developed,
    low-contrast speckle inside vessel lumina / cysts).

    **掩膜内归一化（normalized convolution）**，这是本函数的关键：
    早先的实现是 `smooth = gaussian_filter(env, (3.0, 2.5))`，对**整幅图**
    模糊后再按掩膜混合。腔径只有 3~6 体素，而模糊核 (3.0, 2.5) 与腔同量级，
    邻域里占压倒性权重的是**紧贴的亮壁**（specular 0.85、wall_echo_gain 1.5）
    与肝实质（scatter 0.50）。结果是 `strength=0.6` 把腔外亮度灌进腔内，
    抬亮管腔、抹掉"暗腔亮壁"对比，且灌入量随局部散斑起伏 ⇒ 对比度**符号逐层翻转**，
    全卷平均后互相抵消（实测 98% 腔体素落在背景 5~95% 区间内）。

    归一化后 `smooth` 只由腔内体素加权平均得到，腔外体素权重为 0，
    因此仅平滑腔内自身的散斑，不引入腔外亮度。
    """
    if np.count_nonzero(lumen_mask) == 0:
        return env
    m = np.asarray(lumen_mask, dtype=np.float64)
    num = gaussian_filter(np.asarray(env, dtype=np.float64) * m, (3.0, 2.5))
    den = gaussian_filter(m, (3.0, 2.5))
    # 掩膜外 / 掩膜模糊后仍为 0 的位置用自身兜底，避免除零产生 NaN
    smooth = np.where(den > 1e-6, num / np.maximum(den, 1e-9), env)
    out = np.where(lumen_mask, (1 - strength) * env + strength * smooth, env)
    return out.astype(env.dtype)


def lateral_resolution(env: np.ndarray, sigma_lat: float, sigma_ax: float) -> np.ndarray:
    """PSF-like lateral/axial blur applied to the RF/envelope image."""
    return gaussian_filter(env, (sigma_lat, sigma_ax)).astype(env.dtype)