"""超声侧血管点云可行性检查：能否从 3D 超声体里提出血管中心线？

为什么这是生死线
----------------
项目路线是 **CT 点云 ↔ 超声点云** 配准，血管是主要配准基元。
CT 侧已验证可行（`check_vessel_centerline.py`：主干连通，5/6 病例最长分量占骨架 >50%）。
所以现在成败取决于：**超声侧能不能提出血管点云。**

此前已实测否掉一条**错误思路**：拿单帧 B 超去"逐像素预测 CT 组织标签"，
血管 IoU = 0.000（US 强度按组织类别几乎不可分）。但那不是点云框架下的正确思路 ——
血管在超声里是**无回声（暗）的管状结构**，应该用**暗管状结构检测**
（Frangi vesselness 作用在"暗度"上），而不是逐像素分类。

本脚本（不下载、不训练）
------------------------
1. 用生成端 `generate_volume` 生成一个 **3D 超声体**（多帧扫掠）；
   **位姿必须按血管专项筛选**（`min_vessel_frac`），否则默认采样出来的平面
   一个血管都不穿 —— 实测用合体结构掩膜（含肝包膜）筛出的 32 层卷里血管体素 = 0。
2. **GT**：从同一卷的 `seg_vol`（组织标签）取血管腔+壁 → 骨架化 → 最大连通分量
3. **预测**：`1 - us_vol` → Frangi(3D 暗管增强) → 阈值 → 骨架化 → 最大连通分量
4. 两者在**同一 US 索引空间**里比较（不需要配准），报召回/精度/中心线距离

判据：召回 > 50% 且中心线误差 < 2mm ⇒ 超声侧点云可行；否则需换基元
（改用肝表面、或引入 Doppler、或调整生成端让血管更显影）。

用法
----
    python scripts/check_us_vessel_points.py --case hepaticvessel_002 --n-elev 32
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                     # noqa: BLE001
    pass

from ct2us import anatomy as an                        # noqa: E402
from ct2us import dataset as ds                        # noqa: E402


def skeletonize(mask: np.ndarray) -> np.ndarray:
    from skimage.morphology import skeletonize as sk
    return sk(mask.astype(bool))


def largest_component(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage
    lab, n = ndimage.label(mask, structure=np.ones((3, 3, 3), int))
    if n == 0:
        return mask
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def frangi_dark(vol: np.ndarray, sigmas):
    """暗管增强：直接在超声体上取**暗脊**（血管无回声）。

    **坑**：`gamma` 必须留默认 `None`。手动给 gamma=15.0 会把输出压成**全零**
    （实测合成暗管：gamma=15 时管中心响应 0.0000，默认 gamma=None 时 0.7476）。
    `gamma` 是结构度归一化尺度，应当由数据自适应决定，不能拍一个数。
    """
    from skimage.filters import frangi
    return frangi(np.asarray(vol, np.float32), sigmas=sigmas, alpha=0.5, beta=0.5,
                  black_ridges=True)


def dist_to_mask(pts: np.ndarray, mask: np.ndarray, spacing):
    from scipy import ndimage
    dt = ndimage.distance_transform_edt(~mask, sampling=spacing)
    return dt[tuple(pts.T)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--liver", default=r"dataset\Task03_Liver\Task03_Liver")
    ap.add_argument("--vessel", default=r"dataset\Task08_HepaticVessel\Task08_HepaticVessel")
    ap.add_argument("--case", default="hepaticvessel_002")
    ap.add_argument("--n-elev", type=int, default=32)
    ap.add_argument("--elev-spacing", type=float, default=1.0)
    ap.add_argument("--iso-mm", type=float, default=1.0)
    ap.add_argument("--min-vessel-frac", type=float, default=0.01,
                    help="位姿血管专项筛选：要求扫描平面上血管像素占比达到阈值。"
                         "**点云路线必需** —— 用合体结构掩膜筛出的平面一个血管都不穿")
    ap.add_argument("--probe", default="convex,convex,linear",
                    help="探头类型池。**点云路线建议用 linear 或 linear 为主**："
                         "凸阵的横向视野只有 ±(radius·sin(fov/2)) ≈ 54–68mm，"
                         "盖不住血管树（实测 GT 骨架因此只剩 64 个点）；"
                         "线阵横向可达 nx·dx ≈ 150–300mm")
    ap.add_argument("--thresholds", default="90,95,98,99.5")
    ap.add_argument("--out", default="_probe/us_vessel_points.json")
    args = ap.parse_args()

    task_dir = args.vessel if args.case.startswith("hepaticvessel_") else args.liver
    root = Path(task_dir)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root
    rng = np.random.default_rng(0)
    ctx = ds.load_case(str(root), "imagesTr", case_name=args.case, noise_seed=12345)
    print(f"[us-check] 病例 {ctx.case}  tissue: "
          f"vessel_lumen={int((ctx.tissue == an.T_VESSEL).sum())} "
          f"vessel_wall={int((ctx.tissue == an.T_VESSEL_WALL).sum())}  "
          f"vessel_mask={int(ctx.vessel.sum())}")

    gp = {"no_deform": True,
          "render": {"log_ref": 60.0, "speckle_strength": 0.35, "speckle_lat_px": 3.0}}
    out = ds.generate_volume(ctx, rng, n_elev=args.n_elev,
                             elev_spacing=args.elev_spacing,
                             param={"min_vessel_frac": args.min_vessel_frac,
                                    "kinds": [k.strip() for k in args.probe.split(",")]},
                             global_params=gp)
    us = np.asarray(out["us_vol"], np.float32)
    seg = np.asarray(out["seg_vol"], np.int16)
    probe = out["probe"]

    # **轴序陷阱**：`generate_volume` 的文档字符串写 (n_elev, nz, nx)，但实现是
    # `np.stack(tissue_slices)`，而 `probe.resample` 返回 `reshape(nx, nz)`，
    # 所以实际是 **(n_elev, nx, nz)**。按文档写会搞错各向同性缩放的轴。
    n_elev, n_ax1, n_ax2 = us.shape
    ax1_is_x = (n_ax1 == probe.nx)
    sp_ax1 = probe.dx_effective if ax1_is_x else probe.dz
    sp_ax2 = probe.dz if ax1_is_x else probe.dx_effective
    hist = {int(k): int(v) for k, v in zip(*np.unique(seg, return_counts=True))}
    print(f"  US 体 {us.shape}（轴序 (elev, {'nx' if ax1_is_x else 'nz'}, "
          f"{'nz' if ax1_is_x else 'nx'})）值域 [{us.min():.3f},{us.max():.3f}]")
    print(f"  体素间距: elev {args.elev_spacing} / 轴1 {sp_ax1:.4f} / 轴2 {sp_ax2:.4f} mm")
    print("  seg_vol 标签直方图: " +
          ", ".join(f"{k}:{an.TISSUE_NAMES.get(k, '?')}={v}" for k, v in sorted(hist.items())))

    from scipy import ndimage
    zoom = (args.elev_spacing / args.iso_mm, sp_ax1 / args.iso_mm, sp_ax2 / args.iso_mm)
    us_i = ndimage.zoom(us, zoom, order=1)
    seg_i = np.rint(ndimage.zoom(seg.astype(np.float32), zoom, order=0)).astype(np.int16)
    spacing = (args.iso_mm, args.iso_mm, args.iso_mm)
    print(f"  各向同性后 {us_i.shape} @ {args.iso_mm}mm")

    gt_mask = (seg_i == an.T_VESSEL) | (seg_i == an.T_VESSEL_WALL)
    print(f"  GT 血管体素={int(gt_mask.sum())}")
    if gt_mask.sum() < 50:
        print("  [中止] 该卷没有覆盖到血管；请提高 --min-vessel-frac 或换病例")
        return 1
    gt_skel = largest_component(skeletonize(gt_mask))
    gt_pts = np.argwhere(gt_skel)
    print(f"  GT 骨架（最大连通分量）={len(gt_pts)} 点")

    fr = frangi_dark(us_i, sigmas=(1.0, 2.0, 3.0, 4.0))
    print(f"  Frangi 暗管度量: min={fr.min():.4f} max={fr.max():.4f} mean={fr.mean():.5f}")

    rows = []
    print(f"\n  {'阈值%':>7}{'骨架点':>9}{'召回@2mm':>10}{'精度@2mm':>10}"
          f"{'距离中位':>10}{'距离P90':>9}")
    for t in [float(x) for x in args.thresholds.split(",")]:
        thr = np.percentile(fr, t)
        pm = largest_component(skeletonize(fr >= thr))
        if pm.sum() == 0:
            print(f"  {t:>7.1f}{0:>9}{'-':>10}{'-':>10}{'-':>10}{'-':>9}")
            continue
        pred_pts = np.argwhere(pm)
        d_gt = dist_to_mask(gt_pts, pm, spacing)
        d_pr = dist_to_mask(pred_pts, gt_skel, spacing)
        rec = float((d_gt <= 2.0).mean())
        pre = float((d_pr <= 2.0).mean())
        print(f"  {t:>7.1f}{len(pred_pts):>9}{rec:>10.3f}{pre:>10.3f}"
              f"{np.median(d_gt):>10.2f}{np.percentile(d_gt, 90):>9.2f}")
        rows.append({"threshold_pct": t, "pred_pts": int(len(pred_pts)),
                     "recall_2mm": rec, "precision_2mm": pre,
                     "dist_median_mm": float(np.median(d_gt)),
                     "dist_p90_mm": float(np.percentile(d_gt, 90))})

    print(f"\n{'=' * 92}\n判读（容差 2mm）")
    if rows:
        best = max(rows, key=lambda r: r["recall_2mm"])
        ok = best["recall_2mm"] > 0.5 and best["dist_median_mm"] < 2.0
        print(f"  最佳：阈值 {best['threshold_pct']}%  召回 {best['recall_2mm']:.3f}  "
              f"精度 {best['precision_2mm']:.3f}  距离中位 {best['dist_median_mm']:.2f} mm")
        print(f"  ⇒ 超声侧血管点云: {'可行' if ok else '不可行 / 需换方案'}"
              f"（判据 召回>0.5 且 距离中位<2mm）")
    if args.out:
        p = Path(args.out)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[1] / p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"case": ctx.case, "us_shape": list(us.shape),
                                 "iso_shape": list(us_i.shape),
                                 "label_hist": hist,
                                 "gt_points": int(len(gt_pts)), "rows": rows},
                                ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  写入 {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
