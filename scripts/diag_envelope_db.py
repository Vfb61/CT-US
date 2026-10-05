"""定位"肝实质为什么被压到灰度 0.08"——把包络的动态范围按组织类别摊开。

背景
----
诊断图显示：肝实质落在 US 强度 0.084~0.136，血管对比度只有 ~0.04（绝对量级）。
`speckle.log_compress` 的映射是

    out = 1 + dB / dr_db,   dB = 20*log10(env / ref)

`ref` 默认 = **本图包络的 99 百分位**。于是 out=0.084 意味着实质比"全图最亮的 1%"
低约 57 dB。真实 B 模式里，实质与最亮镜面反射一般只差 20~35 dB。
如果实测确实是 ~57 dB，那根因就是**镜面回声相对弥散散射过强**，
而不是"血管太细/散斑太强"——修的地方完全不同。

本脚本只回答一个问题：**包络顶部（p99）是谁撑起来的，各类组织在它下面多少 dB。**

用法
----
    python scripts/diag_envelope_db.py --case hepaticvessel_001 --pose 1
    python scripts/diag_envelope_db.py --case hepaticvessel_001 --set dr_db=40
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ct2us import anatomy as an                      # noqa: E402
from ct2us import dataset as ds                      # noqa: E402
from ct2us import io_utils as io                     # noqa: E402
from ct2us import render as render_mod               # noqa: E402

CAPTURED: list[dict] = []
_ORIG = render_mod.render_bmode


def _cap(*a, **kw):
    o = _ORIG(*a, **kw)
    CAPTURED.append(o)
    return o


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vessel", default=r"dataset\Task08_HepaticVessel\Task08_HepaticVessel")
    ap.add_argument("--case", default="hepaticvessel_001")
    ap.add_argument("--pose", type=int, default=1, help="该病例的第几个位姿")
    ap.add_argument("--probe", default="linear")
    ap.add_argument("--n-elev", type=int, default=32)
    ap.add_argument("--elev-spacing", type=float, default=1.0)
    ap.add_argument("--min-vessel-frac", type=float, default=0.01)
    ap.add_argument("--set", action="append", default=[],
                    help="覆盖渲染参数，如 --set dr_db=40 --set specular_gain=0.3")
    args = ap.parse_args()

    rp: dict = {}
    for kv in args.set:
        k, _, v = kv.partition("=")
        try:
            rp[k.strip()] = float(v)
        except ValueError:
            rp[k.strip()] = v

    render_mod.render_bmode = _cap
    root = ROOT_DIR / args.vessel
    print(f"[diag] root={root}  case={args.case}  render覆盖={rp or '（默认）'}")

    ctx = ds.load_case(str(root), "imagesTr", case_name=args.case, noise_seed=2000)
    rng = np.random.default_rng(1000 + args.pose)
    for _ in range(args.pose + 1):
        out = ds.generate_volume(
            ctx, rng, n_elev=args.n_elev, elev_spacing=args.elev_spacing,
            param={"min_vessel_frac": args.min_vessel_frac, "kinds": [args.probe]},
            global_params={"no_deform": True, "render": rp})
    render_mod.render_bmode = _ORIG

    if not CAPTURED:
        print("[diag] 没有捕获到渲染调用"); return 1
    p = CAPTURED[-1]["params"]
    dr = float(p["dr_db"])
    print(f"[diag] 捕获 {len(CAPTURED)} 层；dr_db={dr}  log_ref={p.get('log_ref')}  "
          f"global_gain={p['global_gain']}  specular_gain={p['specular_gain']}")

    # **必须跨全部层汇总**：log_compress 每层自适应取该层 p99，
    # 只看一层会得出与数据集（体积均值）相反的结论。
    env = np.concatenate([np.asarray(c["envelope"], np.float64).ravel() for c in CAPTURED])
    tis = np.concatenate([np.asarray(c["tissue_plane"]).ravel() for c in CAPTURED])
    spec = np.concatenate([np.asarray(c["specular_plane"], np.float64).ravel()
                           for c in CAPTURED])
    # 逐层自适应的参考值：每层各自的 p99
    refs = np.concatenate([
        np.full(np.asarray(c["envelope"]).size,
                np.percentile(np.asarray(c["envelope"], np.float64), 99.0)
                if p.get("log_ref") is None else float(p["log_ref"]))
        for c in CAPTURED])
    ref = float(np.median(refs))
    print(f"[diag] 汇总量 {env.size} 体素 / {len(CAPTURED)} 层；"
          f"逐层 p99 参考值中位={ref:.4g}（范围 {refs.min():.4g}~{refs.max():.4g}）")

    # 深度剖面：肝亮度是否随深度漂移（TGC 是否补够了衰减）
    deps = np.concatenate([
        np.broadcast_to(np.arange(np.asarray(c["envelope"]).shape[1])[None, :],
                        np.asarray(c["envelope"]).shape).ravel()
        for c in CAPTURED])

    def db(x):
        """标量版：用逐层参考值的中位数换算（用于汇总打印）。"""
        return 20.0 * np.log10(float(np.maximum(x, 1e-12)) / max(ref, 1e-12))

    def bm(x):
        return float(np.clip((db(x) + dr) / dr, 0.0, 1.0))

    qs = [50, 90, 99, 99.9, 100]
    print(f"\n包络分位（dB re p99，p99={ref:.4g}）: " +
          "  ".join(f"p{q}={db(np.percentile(env, q)):+.1f}" for q in qs))
    print(f"  即全图动态范围 p50->max = {db(env.max()) - db(np.percentile(env, 50)):.1f} dB")

    print(f"\n{'类别':<14}{'体素':>8}{'包络中位dB':>12}{'包络p99dB':>11}"
          f"{'→灰度中位':>10}{'镜面占比':>10}")
    for code, name in sorted(an.TISSUE_NAMES.items() if hasattr(an, "TISSUE_NAMES")
                             else [(k, str(k)) for k in range(7)]):
        m = tis == code
        n = int(m.sum())
        if n < 30:
            continue
        e = env[m]
        print(f"{name:<14}{n:>8d}{db(np.median(e)):>+12.1f}{db(np.percentile(e, 99)):>+11.1f}"
              f"{bm(np.median(e)):>10.3f}{(spec[m] > 0).mean():>9.0%}")

    print("\n亮度 vs 深度（TGC 是否补足衰减；理想应大致平坦）:")
    nb = 8
    edges = np.linspace(0, deps.max() + 1, nb + 1)
    liv = tis == an.T_LIVER
    lum = tis == an.T_VESSEL
    for i in range(nb):
        m = (deps >= edges[i]) & (deps < edges[i + 1])
        ml, mv = liv & m, lum & m
        s = f"  深度 {int(edges[i]):>4d}~{int(edges[i + 1]):>4d} px  "
        if ml.sum() >= 200:
            s += (f"肝 n={int(ml.sum()):>8d} {db(np.median(env[ml])):>+7.1f} dB "
                  f"→{bm(np.median(env[ml])):.3f}")
        if mv.sum() >= 50:
            s += (f"   腔 n={int(mv.sum()):>7d} {db(np.median(env[mv])):>+7.1f} dB "
                  f"→{bm(np.median(env[mv])):.3f}")
        print(s)

    # ---- 判据表第 4 行：腔信号 vs 噪声底余量 ----
    sig = [c.get("noise_sigma") for c in CAPTURED if c.get("noise_sigma")]
    if sig:
        sg = float(np.median(sig))
        e_lum = float(np.median(env[lum])) if lum.sum() >= 50 else float("nan")
        e_liv = float(np.median(env[liv])) if liv.sum() >= 200 else float("nan")
        print(f"\n噪声底 σ = {sg:.4g}（逐层中位；参考量 = 该层包络 p99）")
        if np.isfinite(e_lum):
            margin = 20.0 * np.log10(max(e_lum, 1e-12) / max(sg, 1e-12))
            print(f"  腔包络中位 {e_lum:.4g}  ⇒ 腔高于噪声底 {margin:+.1f} dB   "
                  f"[判据：≥ +10 dB]")
        if np.isfinite(e_liv):
            print(f"  肝包络中位 {e_liv:.4g}  ⇒ 肝高于噪声底 "
                  f"{20.0 * np.log10(max(e_liv, 1e-12) / max(sg, 1e-12)):+.1f} dB")
    else:
        print("\n[注意] 未取到 noise_sigma（渲染端未回传），噪声底余量判据跳过")

    top = env >= refs
    print("\n撑起 p99 的体素属于哪类（这就是把实质压黑的元凶）:")
    for code in np.unique(tis[top]):
        nm = an.TISSUE_NAMES.get(code, str(code)) if hasattr(an, "TISSUE_NAMES") else str(code)
        print(f"  {nm:<14}{(tis[top] == code).sum():>7d}  ({(tis[top] == code).mean():>5.1%})")
    print(f"  其中镜面分量>0 的占 {(spec[top] > 0).mean():.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
