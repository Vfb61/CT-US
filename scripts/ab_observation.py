"""阶段 A 的廉价前置验证：观测端的哪些改动能真正加深盆地？

背景
----
`E:\\new\\_scratch\\basin_similarity.py` 实测（磁盘真实观测 + 生成端模板，±6mm 扫描）：

    graw（灰度）       prominence 中位 +0.0131   峰值在 0 只有 4/12
    gradmag（梯度幅值） prominence 中位 +0.0910   峰值在 0 只有 6/12

即：**梯度域是对的方向，但即使最好的相似度，真值峰值也有一半的扫描偏掉**。
原因回到根源 1：观测是**单次散斑实现**，其幅度是位姿信号的 2–9 倍。

因此本脚本做 A/B，逐项检验观测端的哪个改动能把"峰值在 0"的比例推上去：

  * `base`      当前默认（散斑 0.7、lat_px 5、每图百分位归一化、观测=单次实现）
  * `obs3`      观测 = 3 次独立实现的平均（模拟 A2：同一位姿多散斑）
  * `lowspk`    `speckle_strength=0.35, speckle_lat_px=3`（A4）
  * `fixref`    `log_ref` 固定为绝对参考（A3，去掉每图百分位归一化）
  * `all`       以上全部叠加

判读：哪个组合能把"峰值在 0"推到接近 12/12、prominence 明显抬升，
就按它去改生成端（而不是凭直觉改）。

用法
----
    python scripts/ab_observation.py --liver <Task03> --volumes 2 --poses 2
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


def ncc(a, b):
    a = np.asarray(a, np.float64).ravel() - np.mean(a)
    b = np.asarray(b, np.float64).ravel() - np.mean(b)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


def gradmag(x, sigma=1.0):
    from scipy import ndimage
    gx, gy = np.gradient(ndimage.gaussian_filter(np.asarray(x, np.float64), sigma))
    return np.hypot(gx, gy)


# 各变体的渲染参数覆盖
VARIANTS = {
    "base":    {},
    "obs3":    {},                                     # 只改观测（多实现平均）
    "lowspk":  {"speckle_strength": 0.35, "speckle_lat_px": 3.0},
    "fixref":  {"log_ref": 60.0},
    "all":     {"speckle_strength": 0.35, "speckle_lat_px": 3.0, "log_ref": 60.0},
}
OBS_REPEATS = {"base": 1, "obs3": 3, "lowspk": 1, "fixref": 1, "all": 3}


def render(ctx, probe, pose, params, seed, want_mean_of=1):
    """渲染 `want_mean_of` 次独立实现；返回 (平均图, 单次图)。"""
    planes = ds.resample_planes(ctx, probe, pose, deform=None)
    acc = None
    first = None
    for k in range(want_mean_of):
        r = np.random.default_rng(seed + 7919 * k)
        out = render_mod.render_bmode(planes["ct"], planes["tissue"], planes["scatter"],
                                      probe, params, r, want_envelope=True)
        b = out["bmode"].astype(np.float64)
        acc = b if acc is None else acc + b
        if k == 0:
            first = b
    return acc / want_mean_of, first


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--liver", required=True)
    ap.add_argument("--volumes", type=int, default=2)
    ap.add_argument("--poses", type=int, default=2)
    ap.add_argument("--span-mm", type=float, default=6.0)
    ap.add_argument("--steps", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-structure-frac", type=float, default=0.03)
    ap.add_argument("--log-refs", default=None,
                    help="逗号分隔的固定 log_ref 取值，逐个测（标定用）")
    ap.add_argument("--spans", default=None,
                    help="逗号分隔的扫描半宽 (mm)，粗尺度与细尺度都要成峰")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.log_refs:
        VARIANTS.clear()
        OBS_REPEATS.clear()
        for v in [float(x) for x in args.log_refs.split(",")]:
            key = f"ref{v:g}"
            VARIANTS[key] = {"log_ref": v}
            OBS_REPEATS[key] = 1
        print(f"标定模式：逐个测试 log_ref = {list(VARIANTS)}")
    spans = [float(x) for x in args.spans.split(",")] if args.spans else [args.span_mm]

    names = io.list_cases(Path(args.liver), "imagesTr")[:args.volumes]
    stats = {v: {"graw": [], "gradmag": [], "peak0_graw": 0, "peak0_gradmag": 0, "n": 0,
                 "clipped": []}
             for v in VARIANTS}
    stats_span = {sp: {v: {"peak0": 0, "n": 0, "prom": []} for v in VARIANTS}
                  for sp in spans}

    for ci, name in enumerate(names):
        rng = np.random.default_rng(args.seed + 1000 * ci)
        ctx = ds.load_case(args.liver, "imagesTr", case_name=name)
        print(f"\n病例 {ctx.case}")
        for k in range(args.poses):
            probe, pose = ds.sample_probe_geometry(
                ctx, rng, {"min_structure_frac": args.min_structure_frac})
            u = np.asarray(pose["u"], np.float64)
            w = np.asarray(pose["w"], np.float64)
            for v, params in VARIANTS.items():
                obs, _ = render(ctx, probe, pose, params, args.seed * 131 + 17 * k,
                                OBS_REPEATS[v])
                stats[v]["clipped"].append(float((obs <= 1e-6).mean()))
                for axname, axis in (("u", u), ("w", w)):
                    for sp in spans:
                        offs = np.linspace(-sp, sp, args.steps)
                        curves = {"graw": [], "gradmag": []}
                        for off in offs:
                            p2 = dict(pose)
                            p2["face"] = (np.asarray(pose["face"], np.float64)
                                          + axis * float(off))
                            tmpl, _ = render(ctx, probe, p2, params,
                                             args.seed * 331 + 29 * k, 8)
                            curves["graw"].append(ncc(tmpl, obs))
                            curves["gradmag"].append(ncc(gradmag(tmpl), gradmag(obs)))
                        i0 = int(np.argmin(np.abs(offs)))
                        for m in ("graw", "gradmag"):
                            vv = np.asarray(curves[m])
                            prom = float(vv[i0] - 0.5 * (vv[0] + vv[-1]))
                            if sp == spans[0]:
                                stats[v][m].append(prom)
                                if int(np.argmax(vv)) == i0:
                                    stats[v][f"peak0_{m}"] += 1
                            if m == "gradmag":
                                stats_span[sp][v]["prom"].append(prom)
                                if int(np.argmax(vv)) == i0:
                                    stats_span[sp][v]["peak0"] += 1
                                stats_span[sp][v]["n"] += 1
                        if v in ("base", "all") and axname == "u" and sp == spans[0]:
                            print(f"  位姿{k} {v:8s} u  gradmag " +
                                  " ".join(f"{x:+.3f}" for x in curves["gradmag"]))
                    stats[v]["n"] += 1

    print(f"\n{'=' * 96}\n汇总（prominence 中位越大越好；峰值在 0 的比例越高越好）")
    for sp in spans:
        print(f"  --- 扫描半宽 ±{sp:g} mm（gradmag）---")
        print(f"  {'变体':<9}{'prom 中位':>12}{'峰值@0':>10}")
        for v in VARIANTS:
            s = stats_span[sp][v]
            n = max(s["n"], 1)
            print(f"  {v:<9}{np.median(s['prom']):>+12.4f}{s['peak0']:>7d}/{n:<3d}")
    print(f"\n  --- 灰度基线对照（±{spans[0]:g}mm）---")
    print(f"  {'变体':<9}{'graw prom':>12}{'gradmag prom':>14}"
          f"{'峰值@0 graw':>14}{'峰值@0 grad':>14}{'裁剪像素比':>12}")
    for v in VARIANTS:
        s = stats[v]
        n = max(s["n"], 1)
        print(f"  {v:<9}{np.median(s['graw']):>+12.4f}{np.median(s['gradmag']):>+14.4f}"
              f"{s['peak0_graw']:>10d}/{n:<3d}{s['peak0_gradmag']:>10d}/{n:<3d}"
              f"{np.median(s['clipped']):>12.3f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {v: {k: (list(map(float, val)) if isinstance(val, list) else val)
                 for k, val in stats[v].items()} for v in VARIANTS},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
