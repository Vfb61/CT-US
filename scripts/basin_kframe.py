"""K 帧加权相似度盆地测试：把"同一位姿的多次散斑实现"用起来。

动机
----
阶段 A 让每个样本落盘了 K=3 个**独立散斑实现**（`us.png` / `us_01.png` / `us_02.png`）
与它们的平均图。目前只用了平均图——但 K 帧还能提供**逐像素方差**，而方差本身就是
一张"可靠性图"：

  * 位姿稳定、结构清晰的像素：各帧一致 ⇒ 方差小
  * 被散斑主导的像素：各帧差异大 ⇒ 方差大

把匹配加权成 `1/var`，等于**自动把权重压到结构上、把散斑压掉**。
这不是新数据，只是把阶段 A 已经落盘的信息用起来；而且**部署时同样可用**——
手术导航中探头会持续采集，同一位置本来就能拿到连续多帧。

判据
----
对每个样本沿已知物理位移扫描，比较
  * `plain`    : 梯度域 NCC(tmpl, 平均图)                     ← 当前基线
  * `wvar`     : 加权梯度域 NCC，权重 = 1/(跨帧方差 + eps)
  * `wvar_log` : 权重 = 1/(log(1+var) + eps)
看 prominence 与"峰值是否落在真值"。

用法
----
    python scripts/basin_kframe.py --shard outputs/pairs_v2 --n 8 --span-mm 6
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
from ct2us import render as render_mod                # noqa: E402


def gradmag(x, sigma=1.0):
    from scipy import ndimage
    gx, gy = np.gradient(ndimage.gaussian_filter(np.asarray(x, np.float64), sigma))
    return np.hypot(gx, gy)


def ncc(a, b):
    a = np.asarray(a, np.float64).ravel() - np.mean(a)
    b = np.asarray(b, np.float64).ravel() - np.mean(b)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


def wncc(a, b, w):
    """加权相关系数（权重 w >= 0）。"""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    w = np.asarray(w, np.float64)
    wa = w.sum()
    if wa <= 1e-12:
        return 0.0
    am = (w * a).sum() / wa
    bm = (w * b).sum() / wa
    da, db = a - am, b - bm
    num = (w * da * db).sum()
    den = np.sqrt((w * da * da).sum() * (w * db * db).sum())
    return float(num / den) if den > 1e-12 else 0.0


def load_png(p):
    from PIL import Image
    with Image.open(p) as im:
        return np.asarray(im.convert("L"), dtype=np.float64) / 255.0


def build_ctx(meta):
    task = (r"E:\build\dataset\Task03_Liver\Task03_Liver"
            if meta["case"].startswith("liver_")
            else r"E:\build\dataset\Task08_HepaticVessel\Task08_HepaticVessel")
    return ds.load_case(task, "imagesTr", case_name=meta["case"],
                        noise_seed=meta.get("noise_seed"))


def template(ctx, probe, pose, params):
    """确定性期望模板（散斑关掉；其余参数含 log_ref 与观测一致）。"""
    p = dict(params)
    p["speckle_strength"] = 0.0
    planes = ds.resample_planes(ctx, probe, pose, deform=None)
    out = render_mod.render_bmode(planes["ct"], planes["tissue"], planes["scatter"],
                                  probe, p, np.random.default_rng(0), want_envelope=True)
    return out["bmode"].astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", default="outputs/pairs_v2")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--span-mm", type=float, default=6.0)
    ap.add_argument("--steps", type=int, default=7)
    ap.add_argument("--eps-frac", type=float, default=0.05,
                    help="权重正则：eps = eps_frac * mean(var)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.shard)
    if not root.is_absolute():
        root = Path(r"E:\build") / root
    dirs = sorted(d for d in root.rglob("*") if (d / "params.json").exists())
    step = max(1, len(dirs) // args.n)
    picked = dirs[::step][:args.n]
    print(f"[kframe] {len(dirs)} 样本，测 {len(picked)} 个，±{args.span_mm:g}mm")

    stats = {k: {"prom": [], "peak0": 0} for k in ("plain", "wvar", "wvar_log")}
    varfrac = []
    for d in picked:
        meta = json.loads((d / "params.json").read_text(encoding="utf-8"))
        pr = meta["params"]
        probe = ds.geo.probe_from_meta(pr["probe"])
        base = {k: np.asarray(pr["pose"][k], np.float64) for k in ("face", "u", "v", "w")}
        params = dict(pr["render"])
        ctx = build_ctx(meta)
        frames = [load_png(d / n) for n in ("us.png", "us_01.png", "us_02.png")
                  if (d / n).exists()]
        if len(frames) < 2:
            print(f"  [skip] {meta['sid']} 只有 {len(frames)} 帧")
            continue
        gm = [gradmag(f) for f in frames]
        og_mean = np.mean(gm, axis=0)
        og_var = np.var(gm, axis=0)
        eps = args.eps_frac * float(og_var.mean()) + 1e-12
        w1 = 1.0 / (og_var + eps)
        w2 = 1.0 / (np.log1p(og_var / (og_var.mean() + 1e-12)) + 1.0)
        varfrac.append(float(og_var.mean() / (og_mean.mean() ** 2 + 1e-12)))

        for axname, key in (("u-lat", "u"), ("w-dep", "w")):
            axis = base[key]
            offs = np.linspace(-args.span_mm, args.span_mm, args.steps)
            cs = {k: [] for k in stats}
            for off in offs:
                p2 = dict(base)
                p2["face"] = base["face"] + axis * float(off)
                tg = gradmag(template(ctx, probe, p2, params))
                cs["plain"].append(ncc(tg, og_mean))
                cs["wvar"].append(wncc(tg, og_mean, w1))
                cs["wvar_log"].append(wncc(tg, og_mean, w2))
            i0 = int(np.argmin(np.abs(offs)))
            for k, c in cs.items():
                v = np.asarray(c)
                stats[k]["prom"].append(float(v[i0] - 0.5 * (v[0] + v[-1])))
                if int(np.argmax(v)) == i0:
                    stats[k]["peak0"] += 1
            print(f"  {meta['sid']:>22s} {axname}: " + "  ".join(
                f"{k}={np.asarray(c)[i0] - 0.5 * (c[0] + c[-1]):+.4f}" for k, c in cs.items()))
            if axname == "u-lat":
                for k, c in cs.items():
                    print(f"      {k:9s} " + " ".join(f"{x:+.3f}" for x in c))

    print(f"\n{'=' * 92}\nK 帧加权盆地汇总（{len(varfrac)} 样本）")
    n_tot = sum(stats["plain"]["peak0"] for _ in [0]) or 0
    n_sc = len(stats["plain"]["prom"])
    for k in stats:
        v = np.asarray(stats[k]["prom"])
        if v.size:
            print(f"  {k:9s} prominence 中位 {np.median(v):+.4f}  最小 {v.min():+.4f}  "
                  f"峰值在 0: {stats[k]['peak0']}/{n_sc}")
    if varfrac:
        print(f"  跨帧方差/均值² 中位: {np.median(varfrac):.4f} （越大说明散斑越主导）")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {k: {"prom": list(map(float, stats[k]["prom"])), "peak0": stats[k]["peak0"]}
             for k in stats}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
