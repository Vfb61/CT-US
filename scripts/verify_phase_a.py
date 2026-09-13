"""阶段 A 验收：可复现性 + 盆地复验（全部走**磁盘**路径，不是进程内模拟）。

两项验收
--------
1. **可复现性**：用落盘的 `render_seed` / `noise_seed` / `probe` / `pose` 重建
   `CaseContext` 并重渲染，应当**逐像素复原** `us.png`。
   旧数据做不到（实测重渲染只有 `NCC 0.53`；即使关掉散斑也只有 0.68），
   因为 scatter 噪声场与散斑实现都没有落盘。

2. **盆地复验**：以磁盘上的 `us_mean.png`（K 次散斑平均）为观测，
   在候选位姿渲染同参数的期望模板，沿已知物理位移扫描，报告
   灰度域与梯度幅值域的 prominence 与"峰值是否落在 0"。
   这是阶段 A 的**验收指标**，也是决定能否进入阶段 C 的门槛。

用法
----
    python scripts/verify_phase_a.py --shard outputs/pairs_v2 --n 8
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


def ncc(a, b):
    a = np.asarray(a, np.float64).ravel() - np.mean(a)
    b = np.asarray(b, np.float64).ravel() - np.mean(b)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


def gradmag(x, sigma=1.0):
    from scipy import ndimage
    gx, gy = np.gradient(ndimage.gaussian_filter(np.asarray(x, np.float64), sigma))
    return np.hypot(gx, gy)


def load_png(p):
    from PIL import Image
    with Image.open(p) as im:
        return np.asarray(im.convert("L"), dtype=np.float64) / 255.0


def render(ctx, probe, pose, params, seed, k=1):
    planes = ds.resample_planes(ctx, probe, pose, deform=None)
    acc, first = None, None
    for i in range(k):
        out = render_mod.render_bmode(planes["ct"], planes["tissue"], planes["scatter"],
                                      probe, params, np.random.default_rng(seed + 7919 * i),
                                      want_envelope=True)
        b = out["bmode"].astype(np.float64)
        acc = b if acc is None else acc + b
        if i == 0:
            first = b
    return acc / k, first


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", default="outputs/pairs_v2")
    ap.add_argument("--n", type=int, default=8, help="验收样本数")
    ap.add_argument("--span-mm", type=float, default=6.0)
    ap.add_argument("--steps", type=int, default=7)
    ap.add_argument("--tmpl-speckle", type=int, default=8,
                    help="模板平均的实现次数（越小越快）")
    args = ap.parse_args()

    root = Path(args.shard)
    if not root.is_absolute():
        root = Path(r"E:\build") / root
    dirs = sorted(d for d in root.rglob("*") if (d / "params.json").exists())
    if not dirs:
        print(f"[verify] {root} 下没有样本目录")
        return 1
    step = max(1, len(dirs) // args.n)
    picked = dirs[::step][:args.n]
    print(f"[verify] {root}  共 {len(dirs)} 样本，验收 {len(picked)} 个")

    # ---------------- 1) 可复现性 ----------------
    print(f"\n{'=' * 96}\n1) 可复现性：用落盘种子重渲染 vs 磁盘 us.png")
    repro = []
    for d in picked:
        meta = json.loads((d / "params.json").read_text(encoding="utf-8"))
        pr = meta["params"]
        probe = ds.geo.probe_from_meta(pr["probe"])
        pose = {k: v for k, v in pr["pose"].items()}
        for k in ("face", "u", "v", "w"):
            pose[k] = np.asarray(pose[k], np.float64)
        task = r"E:\build\dataset\Task03_Liver\Task03_Liver" if meta["case"].startswith("liver_") \
            else r"E:\build\dataset\Task08_HepaticVessel\Task08_HepaticVessel"
        ctx = ds.load_case(task, "imagesTr", case_name=meta["case"],
                           noise_seed=meta.get("noise_seed"))
        params = dict(pr["render"])
        _, img = render(ctx, probe, pose, params, int(meta["render_seed"]), k=1)
        disk = load_png(d / "us.png")
        e = float(np.mean(np.abs(img - disk)))
        repro.append((meta["sid"], e, ncc(img, disk)))
        print(f"  {meta['sid']:>22s}  mean|Δ|={e:.6f}  NCC={ncc(img, disk):.6f}")
    errs = np.asarray([r[1] for r in repro])
    nccs = np.asarray([r[2] for r in repro])
    ok = int((nccs > 0.999).sum())
    print(f"  ⇒ 逐位复现 {ok}/{len(repro)}（判据 NCC>0.999）；"
          f"mean|Δ| 中位 {np.median(errs):.2e}，max {errs.max():.2e}")
    print("  对照：旧数据（pairs_v1）重渲染只有 NCC 0.53；关掉散斑也只有 0.68")

    # ---------------- 2) 盆地复验 ----------------
    print(f"\n{'=' * 96}\n2) 盆地复验：观测=磁盘 us_mean.png，模板=同参数期望图像")
    rows = []
    for d in picked:
        meta = json.loads((d / "params.json").read_text(encoding="utf-8"))
        pr = meta["params"]
        probe = ds.geo.probe_from_meta(pr["probe"])
        base = {k: np.asarray(pr["pose"][k], np.float64) for k in ("face", "u", "v", "w")}
        task = r"E:\build\dataset\Task03_Liver\Task03_Liver" if meta["case"].startswith("liver_") \
            else r"E:\build\dataset\Task08_HepaticVessel\Task08_HepaticVessel"
        ctx = ds.load_case(task, "imagesTr", case_name=meta["case"],
                           noise_seed=meta.get("noise_seed"))
        params = dict(pr["render"])
        obs = load_png(d / "us_mean.png") if (d / "us_mean.png").exists() else load_png(d / "us.png")
        for axname, key in (("u-lat", "u"), ("w-dep", "w")):
            axis = base[key]
            offs = np.linspace(-args.span_mm, args.span_mm, args.steps)
            cg, cm = [], []
            for off in offs:
                p2 = dict(base)
                p2["face"] = base["face"] + axis * float(off)
                t, _ = render(ctx, probe, p2, params, 12345, k=args.tmpl_speckle)
                cg.append(ncc(t, obs))
                cm.append(ncc(gradmag(t), gradmag(obs)))
            i0 = int(np.argmin(np.abs(offs)))
            row = {}
            for nm, c in (("graw", cg), ("gradmag", cm)):
                v = np.asarray(c)
                row[nm] = float(v[i0] - 0.5 * (v[0] + v[-1]))
                row[f"{nm}_peak0"] = int(np.argmax(v)) == i0
                row[f"{nm}_curve"] = v.tolist()
            rows.append(row)
            print(f"  {meta['sid']:>22s} {axname}: graw prom={row['graw']:+.4f} "
                  f"(peak0={row['graw_peak0']})  gradmag prom={row['gradmag']:+.4f} "
                  f"(peak0={row['gradmag_peak0']})")
            if axname == "u-lat":
                print("      gradmag " + " ".join(f"{x:+.3f}" for x in row["gradmag_curve"]))

    if rows:
        print(f"\n{'=' * 96}\n阶段 A 验收汇总（{len(rows)} 条扫描，±{args.span_mm:g}mm）")
        for nm in ("graw", "gradmag"):
            p = np.asarray([r[nm] for r in rows])
            k0 = sum(1 for r in rows if r[f"{nm}_peak0"])
            print(f"  {nm:8s} prominence 中位 {np.median(p):+.4f}  最小 {p.min():+.4f}  "
                  f"峰值在 0: {k0}/{len(rows)}")
        print(f"  对照（pairs_v1 + 每图百分位归一化，±6mm）："
              f"graw 中位 +0.0131（峰值 4/12），gradmag 中位 +0.0910（峰值 6/12）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
