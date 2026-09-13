"""根源测量：把"位姿信号"与"散斑噪声"分开量化，并逐一看哪种相似度能救。

核心问题
--------
模板匹配的真值峰值**真的存在**，但 prominence 只有 0.005–0.075（基线 ~0.8）。
为什么这么浅？本脚本把图像方差拆成三部分：

  * **位姿信号** `S`：同一病例、不同位姿的**期望图像**（多散斑平均）之间的差异
  * **散斑噪声** `N`：同一位姿、不同散斑实现之间的差异
  * 若 `N ≫ S`，则任何"逐像素比较"的目标函数都必然浅——**这就是根源**

并且直接检验**候选修复**：局部归一化 / 高通 / 梯度幅值 / 裁剪视野，
看哪一种能把 prominence 显著抬起来。

用法
----
    python scripts/measure_similarity_options.py --liver <Task03> --volumes 2
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

from ct2us import dataset as ds                       # noqa: E402
from ct2us import io_utils as io                      # noqa: E402
from ct2us import render as render_mod                # noqa: E402


# --------------------------------------------------------------------------
# 相似度算子（都作用在两张同样大小的图上）
# --------------------------------------------------------------------------
def _ncc(a, b, mask=None):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if mask is not None:
        a = a[mask]
        b = b[mask]
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


def sim_global(a, b):
    return _ncc(a, b)


def _highpass(x, sigma=6.0):
    from scipy import ndimage
    return x - ndimage.gaussian_filter(x, sigma)


def sim_highpass(a, b):
    return _ncc(_highpass(a), _highpass(b))


def sim_gradmag(a, b):
    from scipy import ndimage
    ga = np.hypot(*np.gradient(ndimage.gaussian_filter(a, 1.0)))
    gb = np.hypot(*np.gradient(ndimage.gaussian_filter(b, 1.0)))
    return _ncc(ga, gb)


def sim_local(a, b, block=16):
    """分块局部 NCC 的均值——**抹掉全局亮度/对比度偏置**，只看局部结构。"""
    nx, nz = a.shape
    vals = []
    for i in range(0, nx - block + 1, block):
        for j in range(0, nz - block + 1, block):
            pa = a[i:i + block, j:j + block]
            pb = b[i:i + block, j:j + block]
            if pa.std() < 1e-6 or pb.std() < 1e-6:
                continue
            vals.append(_ncc(pa, pb))
    return float(np.mean(vals)) if vals else 0.0


def _center_mask(shape, frac=0.6):
    """只保留中间 `frac` 比例的视野——排除近场体壁那种"整体亮度"主导区。"""
    m = np.zeros(shape, dtype=bool)
    nx, nz = shape
    ix = int(nx * (1 - frac) / 2)
    iz = int(nz * (1 - frac) / 2)
    m[ix:nx - ix, iz:nz - iz] = True
    return m


SIMILARITIES = {
    "global_ncc": sim_global,
    "highpass_ncc": sim_highpass,
    "gradmag_ncc": sim_gradmag,
    "local_ncc16": sim_local,
    "center60_ncc": lambda a, b: _ncc(a, b, _center_mask(a.shape, 0.6)),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--liver", required=True)
    ap.add_argument("--volumes", type=int, default=2)
    ap.add_argument("--poses", type=int, default=2)
    ap.add_argument("--n-speckle", type=int, default=24,
                    help="估计期望图像的散斑平均次数（越大越准，也越慢）")
    ap.add_argument("--span-mm", type=float, default=6.0)
    ap.add_argument("--steps", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-structure-frac", type=float, default=0.03)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    names = io.list_cases(Path(args.liver), "imagesTr")[:args.volumes]
    results = []
    for ci, name in enumerate(names):
        rng = np.random.default_rng(args.seed + 1000 * ci)
        ctx = ds.load_case(args.liver, "imagesTr", case_name=name)
        print(f"\n{'=' * 96}\n病例 {ctx.case}")

        for k in range(args.poses):
            probe, pose = ds.sample_probe_geometry(ctx, rng,
                                                   {"min_structure_frac": args.min_structure_frac})
            planes = ds.resample_planes(ctx, probe, pose, deform=None)

            # --- 同一位姿的 n_speckle 次实现 -> 期望图像 + 散斑方差 ---
            imgs = []
            for s in range(args.n_speckle):
                r = np.random.default_rng(args.seed * 7919 + 13 * s + k)
                out = render_mod.render_bmode(planes["ct"], planes["tissue"],
                                              planes["scatter"], probe, None, r,
                                              want_envelope=True)
                imgs.append(out["bmode"].astype(np.float64))
            mean_img = np.mean(imgs, axis=0)
            speckle_var = float(np.mean([np.mean((im - mean_img) ** 2) for im in imgs]))
            g0 = imgs[0]                                  # 单次实现（= "观测量"）

            # --- 位姿方差：沿 u / w 扫描的期望图像差异 ---
            u = np.asarray(pose["u"], np.float64)
            w = np.asarray(pose["w"], np.float64)
            print(f"\n  位姿 {k}: kind={probe.kind} nx={probe.nx} nz={probe.nz} "
                  f"dx_eff={probe.dx_effective:.4f} dz={probe.dz:.3f} | "
                  f"平面结构占比={ds.plane_structure_fraction(ctx, probe, pose):.5f}")

            for axname, axis in (("u-lat", u), ("w-dep", w)):
                offs = np.linspace(-args.span_mm, args.span_mm, args.steps)
                means = []
                for off in offs:
                    p2 = dict(pose)
                    p2["face"] = np.asarray(pose["face"], np.float64) + axis * float(off)
                    pl = ds.resample_planes(ctx, probe, p2, deform=None)
                    acc = None
                    for s in range(max(4, args.n_speckle // 3)):
                        r = np.random.default_rng(args.seed * 104729 + 17 * s + k)
                        o = render_mod.render_bmode(pl["ct"], pl["tissue"], pl["scatter"],
                                                    probe, None, r, want_envelope=True)
                        b = o["bmode"].astype(np.float64)
                        acc = b if acc is None else acc + b
                    means.append(acc / max(4, args.n_speckle // 3))
                pose_var = float(np.mean([np.mean((m - mean_img) ** 2) for m in means]))
                ratio = speckle_var / max(pose_var, 1e-12)

                # --- 相似度选项：单次实现(位姿0) vs 各位姿的期望图像 ---
                row = {}
                for nm, fn in SIMILARITIES.items():
                    vals = np.asarray([fn(m, g0) for m in means])
                    i0 = int(np.argmin(np.abs(offs)))
                    ends = 0.5 * (vals[0] + vals[-1])
                    row[nm] = {"prominence": float(vals[i0] - ends),
                               "prominence_rel": float((vals[i0] - ends)
                                                       / (abs(vals[i0]) + 1e-9)),
                               "curve": vals.tolist()}
                print(f"    {axname}: 散斑方差={speckle_var:.5f} 位姿方差={pose_var:.5f} "
                      f"⇒ 噪声/信号 = {ratio:.1f}x")
                for nm, d in row.items():
                    print(f"      {nm:14s} prominence={d['prominence']:+.4f} "
                          f"(相对 {d['prominence_rel'] * 100:+.1f}%)  "
                          f"曲线 " + " ".join(f"{v:+.3f}" for v in d["curve"]))
                results.append({"case": ctx.case, "pose": k, "axis": axname,
                                "speckle_var": speckle_var, "pose_var": pose_var,
                                "noise_over_signal": ratio, "sims": row})

    if results:
        print(f"\n{'=' * 96}\n汇总（{len(results)} 条扫描）")
        r = np.asarray([x["noise_over_signal"] for x in results])
        print(f"  噪声/信号 比: 中位 {np.median(r):.2f}x  范围 {r.min():.2f}–{r.max():.2f}x")
        for nm in SIMILARITIES:
            p = np.asarray([x["sims"][nm]["prominence"] for x in results])
            print(f"  {nm:14s} prominence 中位 {np.median(p):+.4f}  "
                  f"最小 {p.min():+.4f}  最大 {p.max():+.4f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        print(f"\n写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
