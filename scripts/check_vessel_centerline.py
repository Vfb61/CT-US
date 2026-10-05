"""CT 侧血管中心线可行性检查：5mm 层厚下血管能不能提出连续中心线？

背景
----
项目路线已明确为 **CT 点云 ↔ 超声点云** 配准（血管为主要配准基元）。
因此"能不能从 CT 提出**连续**的血管中心线"是整条链的**第一个生死关**。

实测事实：MSD Task08（HepaticVessel，带血管标签）的**原始 z 层厚是 5.0 mm**，
而肝内血管直径通常 2–5 mm。5mm 层厚下血管在 z 方向可能断开，
骨架化会得到一堆碎片 → 提不出连续中心线 → "血管关键点"路线在 CT 侧先断。

本脚本检查（不训练、不下载，全用现有数据）：
  1. 血管标签的体素数、原始层厚
  2. 在**原始 5mm** 与**插值到 1mm** 两种情况下分别做 3D 骨架化
  3. 骨骼的连通分量数、最长分量长度、分量大小分布
  4. 用距离变换估计血管半径分布（半径 < 1 voxel 说明太细，骨架化必然碎片化）

判读：若 1mm 插值后最长连通分量仍只占总长的很小比例 ⇒ CT 侧需要换方案
（改用 HU 阈值在 1mm 的 Task03 上提血管，或换血管质量更好的 CT）。

用法
----
    python scripts/check_vessel_centerline.py --volumes 6
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

from ct2us import io_utils as io                       # noqa: E402


def _zoom(vol, zoom, order):
    from scipy import ndimage
    return ndimage.zoom(vol.astype(np.float32), zoom, order=order)


def skeletonize(mask: np.ndarray) -> np.ndarray:
    """3D 骨架化；优先 skimage，缺失时退回形态学细化。"""
    try:
        from skimage.morphology import skeletonize as sk_skel
        return sk_skel(mask.astype(bool))
    except Exception:                                  # noqa: BLE001
        from scipy import ndimage
        # 退化方案：反复腐蚀到只剩中轴（近似），仅用于对比
        out = mask.astype(bool).copy()
        prev = out
        for _ in range(12):
            er = ndimage.binary_erosion(prev, np.ones((3, 3, 3), bool))
            if not er.any():
                break
            prev = er
        return prev


def comp_stats(skel: np.ndarray, spacing_mm: tuple[float, float, float]):
    from scipy import ndimage
    lab, n = ndimage.label(skel, structure=np.ones((3, 3, 3), int))
    if n == 0:
        return dict(n=0, total_vox=int(skel.sum()), longest_vox=0, longest_mm=0.0,
                    top10=[], frac_longest=0.0)
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    order = np.argsort(-sizes)
    longest = int(sizes[order[0]])
    # 长度用体素数 × 平均体素步长近似（对角步长按 sqrt(3) 折中取 1.4）
    step = float(np.mean(spacing_mm)) * 1.4
    return dict(n=int(n), total_vox=int(skel.sum()), longest_vox=longest,
                longest_mm=longest * step,
                top10=[int(sizes[i]) for i in order[:10]],
                frac_longest=float(longest / max(skel.sum(), 1)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vessel", default=r"dataset\Task08_HepaticVessel\Task08_HepaticVessel",
                    help="Task08 目录（带血管标签）")
    ap.add_argument("--split", default="imagesTr")
    ap.add_argument("--volumes", type=int, default=6)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--target-mm", type=float, default=1.0)
    ap.add_argument("--out", default="_probe/vessel_centerline.json")
    args = ap.parse_args()

    root = Path(args.vessel)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root
    names = io.list_cases(root, args.split)[args.first:args.first + args.volumes]
    if not names:
        print(f"[check] {root} 下没有病例"); return 1
    print(f"[check] {len(names)} 个病例: {names}")

    rows = []
    for name in names:
        lp = root / args.split.replace("images", "labels") / (name + ".nii.gz")
        if not lp.exists():
            print(f"  [skip] {name}: 无标签"); continue
        lab, aff, _ = io.load_volume(lp)
        lab = np.asarray(lab)
        zooms = np.linalg.norm(np.asarray(aff)[:3, :3], axis=0)
        # Task08 惯例：1=血管腔, 2=肿瘤
        lumen = (np.rint(lab) == 1)
        n_vox = int(lumen.sum())
        print(f"\n{'=' * 92}\n{name}: shape={lab.shape} 原生间距={np.round(zooms, 3)} mm  "
              f"血管腔体素={n_vox}  占体积 {100.0 * n_vox / lab.size:.3f}%")
        if n_vox < 100:
            print("  [skip] 血管体素太少"); continue

        # ---- 血管半径分布（原始分辨率下的距离变换）----
        from scipy import ndimage
        dt = ndimage.distance_transform_edt(lumen, sampling=zooms)
        radii = dt[lumen]
        r_med = float(np.median(radii)); r_p90 = float(np.percentile(radii, 90))
        print(f"  血管半径估计: 中位 {r_med:.2f} mm  P90 {r_p90:.2f} mm  "
              f"最大 {float(radii.max()):.2f} mm")
        print(f"  ⇒ 原始层厚 {zooms[2]:.2f} mm 与中位半径之比 = "
              f"{zooms[2] / max(r_med, 1e-6):.2f}（>2 说明 z 向严重欠采样）")

        res = {"case": name, "native_mm": zooms.tolist(), "lumen_vox": n_vox,
               "radius_med_mm": r_med, "radius_p90_mm": r_p90}

        # ---- 原始 5mm 下骨架化 ----
        sk0 = skeletonize(lumen)
        c0 = comp_stats(sk0, tuple(zooms))
        print(f"  [原始 {zooms[2]:.2f}mm] 骨架体素={c0['total_vox']}  "
              f"连通分量={c0['n']}  最长分量={c0['longest_vox']} 体素"
              f"（占 {100 * c0['frac_longest']:.1f}%）")
        print(f"               最大 10 个分量: {c0['top10']}")
        res["native_skel"] = c0

        # ---- 插值到 target-mm 各向同性后骨架化 ----
        zoom = zooms / args.target_mm
        lum_r = (_zoom(lumen, zoom, order=0) > 0.5)
        sk1 = skeletonize(lum_r)
        spacing = tuple([args.target_mm] * 3)
        c1 = comp_stats(sk1, spacing)
        print(f"  [插值 {args.target_mm:.2f}mm] 血管体素={int(lum_r.sum())}  "
              f"骨架体素={c1['total_vox']}  连通分量={c1['n']}  "
              f"最长分量={c1['longest_vox']} 体素 ≈ {c1['longest_mm']:.0f} mm"
              f"（占 {100 * c1['frac_longest']:.1f}%）")
        print(f"               最大 10 个分量: {c1['top10']}")
        res["iso_skel"] = c1
        rows.append(res)

    print(f"\n{'=' * 92}\n汇总")
    if rows:
        f0 = np.asarray([r["native_skel"]["frac_longest"] for r in rows])
        f1 = np.asarray([r["iso_skel"]["frac_longest"] for r in rows])
        n0 = np.asarray([r["native_skel"]["n"] for r in rows])
        n1 = np.asarray([r["iso_skel"]["n"] for r in rows])
        l1 = np.asarray([r["iso_skel"]["longest_mm"] for r in rows])
        rm = np.asarray([r["radius_med_mm"] for r in rows])
        print(f"  血管中位半径: 中位 {np.median(rm):.2f} mm")
        print(f"  骨架最长分量占比: 原始 {np.median(f0) * 100:.1f}%  ->  "
              f"插值 {args.target_mm:g}mm 后 {np.median(f1) * 100:.1f}%")
        print(f"  连通分量数: 原始 中位 {np.median(n0):.0f}  ->  插值后 中位 {np.median(n1):.0f}")
        print(f"  插值后最长分量长度: 中位 {np.median(l1):.0f} mm")
        ok = int((f1 > 0.5).sum())
        print(f"  ⇒ 「最长分量占骨架一半以上」的病例: {ok}/{len(rows)}")
        print("  判读：占比高 + 分量数少 ⇒ 能提出连续血管树，路线可行；"
              "占比低 + 分量数多 ⇒ 血管碎片化，CT 侧需换方案")
    if args.out:
        p = Path(args.out)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[1] / p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  写入 {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
