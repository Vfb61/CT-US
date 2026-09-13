"""Anatomy: label fusion, tissue classification and acoustic parameter tables.

Tissue classification consumes one CT case plus optional MSD-style labels
(Task03: liver=1, cancer=2; Task08: vessel=1, tumour=2).  Structures not
covered by labels (ribs, body wall, air) are recovered from CT HU heuristics.

Every tissue has a small set of acoustic properties used both by the fast
renderer and by the physics simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

# ----------------------------------------------------------------------------
# Tissue catalogue
# ----------------------------------------------------------------------------

T_AIR = 0
T_BODY_WALL = 1
T_BONE = 2
T_LIVER = 3
T_TUMOR = 4
T_VESSEL = 5
T_VESSEL_WALL = 6

TISSUE_NAMES = {
    T_AIR: "air",
    T_BODY_WALL: "body_wall",
    T_BONE: "bone",
    T_LIVER: "liver",
    T_TUMOR: "tumor",
    T_VESSEL: "vessel_lumen",
    T_VESSEL_WALL: "vessel_wall",
}


@dataclass(frozen=True)
class TissueProps:
    """Acoustic / echographic properties of one tissue type."""

    speed_mps: float      # sound speed m/s
    density: float        # kg/m^3
    alpha_db_cm_mhz: float  # attenuation dB / (cm * MHz)
    scatter: float        # B-mode backscatter amplitude (mean)
    specular: float       # specular reflectivity coefficient
    name: str


TISSUE_PROPS = {
    T_AIR:      TissueProps(331.0, 1.18, 0.0, 0.02, 0.90, "air"),
    T_BODY_WALL: TissueProps(1540.0, 1060.0, 0.90, 0.60, 0.25, "body_wall"),
    T_BONE:     TissueProps(3200.0, 1900.0, 6.00, 1.10, 0.95, "bone"),
    T_LIVER:    TissueProps(1570.0, 1060.0, 0.70, 0.50, 0.18, "liver"),
    T_TUMOR:    TissueProps(1560.0, 1045.0, 0.85, 0.22, 0.15, "tumor"),
    T_VESSEL:   TissueProps(1570.0, 1060.0, 0.16, 0.04, 0.05, "vessel_lumen"),
    T_VESSEL_WALL: TissueProps(1565.0, 1075.0, 0.80, 0.85, 0.65, "vessel_wall"),
}


# ----------------------------------------------------------------------------
# HU heuristics for unlabelled anatomy
# ----------------------------------------------------------------------------

def classify_body_structures(ct: np.ndarray, liver_mask: np.ndarray | None,
                             param: dict | None = None) -> np.ndarray:
    """HU-based fallback classification for air / bone / soft tissue.

    Returns an int16 volume with values among the T_* codes.  Voxels covered by
    liver_mask (or any provided organ mask) are left as T_AIR 0 so that label
    priors can over-write them afterwards.
    """
    p = {
        "air_hu": -850.0,
        "bone_hu": 260.0,
        "soft_min_hu": -220.0,
        "soft_max_hu": 280.0,
    }
    if param:
        p.update(param)
    out = np.zeros(ct.shape, dtype=np.int16) + T_AIR
    body = ct > p["soft_min_hu"]
    body &= ct < p["soft_max_hu"]
    bone = ct >= p["bone_hu"]
    air = ct < p["air_hu"]
    out[bone] = T_BONE
    out[air] = T_AIR
    out[body] = T_BODY_WALL
    if liver_mask is not None:
        m = np.asarray(liver_mask, dtype=bool)
        out[m] = T_BODY_WALL  # provisional, labels over-write later
    return out


# ----------------------------------------------------------------------------
# Vessel / duct detection from HU when no vessel label is available
# ----------------------------------------------------------------------------

def detect_vessels_heuristic(ct: np.ndarray, liver_mask: np.ndarray | None,
                             min_hu: float | None = None, max_hu: float | None = None,
                             relative_offset: float = 45.0,
                             min_vol_vox: int = 120) -> np.ndarray:
    """Crude anechoic-tubular detection used only when vessel labels are absent.

    If min_hu/max_hu are None they are derived relative to the liver-parenchyma
    median HU (vessels are typically 15-45 HU darker than parenchyma, which
    covers both contrast and non-contrast phases).

    Returns a boolean volume of candidate vessel lumina.
    """
    out = np.zeros(ct.shape, dtype=bool)
    base = 0.0
    if liver_mask is not None and min_hu is None:
        liver_v = ct[np.asarray(liver_mask, dtype=bool)]
        if liver_v.size:
            base = float(np.median(liver_v))
    if min_hu is None:
        min_hu = base - relative_offset
    if max_hu is None:
        max_hu = base - 12.0
    within = ct > min_hu
    within &= ct < max_hu
    if liver_mask is not None:
        within &= np.asarray(liver_mask, dtype=bool)
    if not within.any():
        return out
    lab, n = ndimage.label(within)
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = sizes >= min_vol_vox
    keep[0] = False
    out = keep[lab]
    # small opening to drop speckle-sized blobs
    out = ndimage.binary_opening(out, structure=np.ones((1, 3, 3)))
    return out


# ----------------------------------------------------------------------------
# Wall rim construction
# ----------------------------------------------------------------------------

def vessel_wall_rim(lumen: np.ndarray, wall_thickness_vox: int = 2,
                    preserve_lumen: bool = True,
                    min_lumen_vox_for_wall: int = 200) -> np.ndarray:
    """Wall ring around vessel lumina.

    **历史缺陷（已修）**：旧实现 `wall = dilation(lumen, t) & ~erosion(lumen, 1)`
    会把管腔**外圈全部吃掉**并把它们标成高回声的管壁。对细血管（CT 上肝内血管
    半径常只有 1–3 voxel）`erosion(lumen,1)` 几乎为空，于是**整条血管都被改成
    "亮壁"**：实测 8887 个管腔体素只剩 89 个。后果是超声里**血管不再是无回声
    （黑）的管状结构，而是一堆亮斑**——而血管树恰恰是肝内最可靠、最能定位的
    解剖标志；同时 `T_VESSEL` 标签几乎为空，任何依赖血管标签的监督/匹配都失效。

    新规则（`preserve_lumen=True`，默认）：
      * 管腔**完整保留**；
      * 只有**体积 ≥ `min_lumen_vox_for_wall`** 的连通管腔才在外侧生成
        `wall_thickness_vox` 厚的管壁环；
      * 细血管不生成管壁（真实超声里也分辨不出壁），整体作为无回声管腔。

    这样血管始终是「暗腔 + 亮壁」，与 CT/标签一致，才可能提供位姿信息。

    `preserve_lumen=False` 可复现历史行为（仅用于对照实验）。
    """
    lumen = np.asarray(lumen, dtype=bool)
    struct = ndimage.generate_binary_structure(3, 2)
    if not preserve_lumen:
        dil = ndimage.binary_dilation(lumen, structure=struct, iterations=wall_thickness_vox)
        ero = ndimage.binary_erosion(lumen, structure=struct, iterations=1)
        return dil & ~ero

    if not lumen.any():
        return np.zeros_like(lumen)
    if min_lumen_vox_for_wall > 0:
        lab, n = ndimage.label(lumen, structure=struct)
        if n == 0:
            return np.zeros_like(lumen)
        sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        keep = np.zeros(n + 1, dtype=bool)
        keep[1:] = sizes >= min_lumen_vox_for_wall
        big = keep[lab]
        if not big.any():
            return np.zeros_like(lumen)
    else:
        big = lumen
    dil = ndimage.binary_dilation(big, structure=struct, iterations=wall_thickness_vox)
    return dil & ~big


# ----------------------------------------------------------------------------
# Main entry: build the tissue map
# ----------------------------------------------------------------------------

def build_tissue_map(ct: np.ndarray, labels_liver: np.ndarray | None = None,
                     labels_vessel: np.ndarray | None = None,
                     derive_vessels: bool = True,
                     param: dict | None = None) -> np.ndarray:
    """Fuse all available labels + HU heuristics into a single tissue map.

    ct:            HU volume (float32)
    labels_liver:  optional; 1=liver, 2=cancer
    labels_vessel: optional; 1=vessel lumen, 2=tumour
    derive_vessels: fallback HU-based detection when no vessel label is present

    Returns an int16 volume with T_* codes.
    """
    shape = ct.shape
    labelled = np.zeros(shape, dtype=bool)

    # 1) HU-based defaults
    if labels_liver is not None:
        liver_mask = np.asarray(labels_liver, dtype=np.uint8) == 1
    else:
        liver_mask = None
    tissue = classify_body_structures(ct, liver_mask, param)

    # 2) over-write with organ labels
    if labels_liver is not None:
        lab = np.asarray(labels_liver, dtype=np.uint8)
        liver = lab == 1
        cancer = lab == 2
        tissue[liver] = T_LIVER
        tissue[cancer] = T_TUMOR
        labelled |= liver | cancer

    # 3) vessels
    if labels_vessel is not None:
        labv = np.asarray(labels_vessel, dtype=np.uint8)
        lumen = labv == 1
        tum = labv == 2
        tissue[lumen] = T_VESSEL
        tissue[tum] = T_TUMOR
        wall = vessel_wall_rim(lumen)
        tissue[wall] = T_VESSEL_WALL
        labelled |= lumen | tum | wall
    elif derive_vessels and liver_mask is not None:
        lumenv = detect_vessels_heuristic(ct, liver_mask)
        tissue[lumenv] = T_VESSEL
        wall = vessel_wall_rim(lumenv)
        tissue[wall] = T_VESSEL_WALL
        labelled |= lumenv | wall

    # 4) small cleanup only on the **HU-derived** voxels. 中值滤波会抹掉
    #    1–2 voxel 厚的细血管腔/壁；标签派生的体素必须原样保留，否则血管
    #    这一唯一的细解剖标志会被"清理"掉（正是历史数据失去定位信息的原因之一）。
    filtered = ndimage.median_filter(tissue.astype(np.int16), size=(1, 3, 3))
    tissue = np.where(labelled, tissue, filtered)
    return tissue.astype(np.int16)


# ----------------------------------------------------------------------------
# Continuous scatter field
# ----------------------------------------------------------------------------

def scatter_volume(tissue: np.ndarray,
                   spatial_noise: np.ndarray | None = None) -> np.ndarray:
    """Mean backscatter field per voxel, modulated by optional smooth noise.

    spatial_noise: float array same shape, exp-normalised multiplicative
    variation used to give continua a slightly heterogeneous texture.
    """
    s = np.zeros(tissue.shape, dtype=np.float32)
    for code, props in TISSUE_PROPS.items():
        s[tissue == code] = props.scatter
    if spatial_noise is not None:
        s = s * spatial_noise
    return s.astype(np.float32)


def acoustic_fields(tissue: np.ndarray,
                    spatial_noise: np.ndarray | None = None):
    """Build volumetric acoustic parameter fields {c, rho, alpha, S}.

    Returns dict of float32 arrays with the same shape as tissue.
    S is the scatter amplitude field, c sound speed (m/s), rho density,
    alpha attenuation (dB/(cm*MHz)).
    """
    c = np.zeros(tissue.shape, dtype=np.float32)
    rho = np.zeros(tissue.shape, dtype=np.float32)
    alpha = np.zeros(tissue.shape, dtype=np.float32)
    s = np.zeros(tissue.shape, dtype=np.float32)
    for code, props in TISSUE_PROPS.items():
        m = tissue == code
        c[m] = props.speed_mps
        rho[m] = props.density
        alpha[m] = props.alpha_db_cm_mhz
        s[m] = props.scatter
    if spatial_noise is not None:
        s = s * spatial_noise
    return {"sound_speed": c, "density": rho, "alpha": alpha, "scatter": s}