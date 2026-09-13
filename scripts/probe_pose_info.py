"""量化「扫描平面上到底有多少位姿信息」——训练前的必过关卡。

为什么需要这个脚本
------------------
配准的输出是位姿，但**输入是图像**。位姿没有以坐标形式写在像素里，它只能通过
"什么结构出现在什么位置"被间接携带。所以必须先把「信息够不够」变成可测量的量，
而不是靠训练去试（本项目已经为此烧掉 5 条技术路线、几百 epoch）。

核心测量：**模拟模板匹配盆地（simulated-template matching basin）**
--------------------------------------------------------------------
这是与实际部署同构的最小实验：
  观测 `us_obs` = 在位姿 0 渲染**一次**（一次散斑实现，模拟真实采集）
  模板 `us_tmpl(ξ)` = 在位姿 ξ 渲染 **N 次取平均**（模拟"从 CT 仿真出来的超声"）
  判据 `match(ξ) = NCC(us_tmpl(ξ), us_obs)`

  观测与模板**使用互不相同的散斑种子**，所以 `match(0)` 本身**不是恒为 1**
  （避免了"自己和自己比"的平凡零）。

若 `match(ξ)` 在 ξ=0 处取得全局极大 ⇒ 平面上的位姿**可被识别**，监督回归/匹配
路线有信息可用，可以开训练。
若曲线平坦、或极值不在 0 ⇒ 平面上的信息不足以定位位姿，训练必然无效。

同时报告：
  * `ct_sens(ξ)`  = mean|slab(ξ)-slab(0)| / std(slab(0)) —— 参照物（CT slab）灵敏度。
    肝实质 HU 近似常数，若平面落在实质内部，参照物本身就无信息。
  * `struct_frac` —— 平面内「血管腔/壁 + 肝包膜」像素占比。这是 CT 上唯一随位姿
    剧烈变化的内容（`ct2us.dataset.build_structure_mask`）。
  * `cross` —— 该观测与其他病例模板的最大 NCC（解剖差异造成的"机会水平"）。
    若 `match(0)` 不高于 `cross`，则匹配只是在认"解剖长什么样"，不是认位姿。

用法
----
    python scripts/probe_pose_info.py --liver <Task03> --vessel <Task08> \
        --volumes 2 --poses 2 --min-structure-frac 0.03 \
        --render speckle_strength=0.45 --out _probe/before.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:                                    # 控制台可能是 GBK；避免打印时崩掉
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                       # noqa: BLE001
    pass

from ct2us import anatomy as an          # noqa: E402
from ct2us import dataset as ds          # noqa: E402
from ct2us import io_utils as io         # noqa: E402
from ct2us import render as render_mod   # noqa: E402


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


def down(x: np.ndarray, f: int = 2) -> np.ndarray:
    nx, nz = x.shape
    nx -= nx % f
    nz -= nz % f
    return x[:nx, :nz].reshape(nx // f, f, nz // f, f).mean(axis=(1, 3))


def render_at(ctx, probe, pose, render_params, seeds):
    """在给定位姿渲染若干次散斑实现，返回 (均值图, 首次图, planes)。"""
    planes = ds.resample_planes(ctx, probe, pose, deform=None)
    acc = None
    first = None
    for k, sd in enumerate(seeds):
        rng = np.random.default_rng(int(sd))
        out = render_mod.render_bmode(planes["ct"], planes["tissue"],
                                      planes["scatter"], probe, render_params, rng,
                                      want_envelope=True)
        b = down(out["bmode"].astype(np.float64), 2)
        acc = b if acc is None else acc + b
        if k == 0:
            first = b
    return acc / len(seeds), first, planes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--liver", default=None, help="Task03 liver task dir")
    ap.add_argument("--vessel", default=None, help="Task08 hepatic vessel task dir")
    ap.add_argument("--split", default="imagesTr")
    ap.add_argument("--volumes", type=int, default=2, help="每类任务的病例数")
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--poses", type=int, default=2, help="每病例测的位姿数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iso", type=float, default=None, metavar="MM")
    ap.add_argument("--n-speckle", type=int, default=12,
                    help="模板平均的散斑实现次数")
    ap.add_argument("--span-mm", type=float, default=8.0)
    ap.add_argument("--steps", type=int, default=9)
    ap.add_argument("--min-structure-frac", type=float, default=0.0)
    ap.add_argument("--render", nargs="*", default=[], metavar="K=V",
                    help="渲染参数覆盖，如 speckle_strength=0.45")
    ap.add_argument("--preserve-lumen", type=int, default=1,
                    help="1=保留管腔（新行为），0=复现旧行为（吃掉管腔）")
    ap.add_argument("--out", default=None, help="结果 JSON 路径")
    args = ap.parse_args()

    if not (args.liver or args.vessel):
        ap.error("至少给一个 --liver / --vessel")

    render_params = {}
    for kv in args.render:
        k, _, v = kv.partition("=")
        render_params[k.strip()] = float(v)
    print(f"渲染覆盖: {render_params or '(无，用 DEFAULT_PARAMS)'}")

    if not args.preserve_lumen:                      # 对照实验：复现历史行为
        _orig = an.vessel_wall_rim

        def _legacy(lumen, wall_thickness_vox=2, **kw):
            return _orig(lumen, wall_thickness_vox, preserve_lumen=False)

        an.vessel_wall_rim = _legacy
        print("!! 已切换到历史 vessel_wall_rim（吃掉管腔）")

    jobs = []
    for task_dir in (args.liver, args.vessel):
        if not task_dir:
            continue
        names = io.list_cases(task_dir, args.split)
        for n in names[args.first:args.first + args.volumes]:
            jobs.append((str(task_dir), n))
    print(f"待测病例: {[n for _, n in jobs]}")

    probe_param = {"min_structure_frac": args.min_structure_frac}
    samples = []          # 每个元素: dict(case, pose, tmpl0, obs, curve...)
    cases_ctx = {}

    base_seed = int(args.seed) * 1000003
    for ci, (task_dir, name) in enumerate(jobs):
        rng = np.random.default_rng(args.seed + 1000 * ci)
        try:
            ctx = ds.load_case(task_dir, args.split, case_name=name,
                               iso_spacing=args.iso)
        except Exception as exc:                     # noqa: BLE001
            print(f"[skip] {name}: {exc}")
            continue
        cases_ctx[name] = ctx
        counts = {an.TISSUE_NAMES[c]: int((ctx.tissue == c).sum())
                  for c in an.TISSUE_NAMES}
        n_str = int(ctx.structure.sum()) if ctx.structure is not None else 0
        print(f"\n{'=' * 100}\n病例 {ctx.case}")
        print("  组织体素数: " + "  ".join(f"{k}={v}" for k, v in counts.items()))
        print(f"  可定位结构: {n_str} ({100.0 * n_str / ctx.tissue.size:.4f}% of volume)")

        for k in range(args.poses):
            probe, pose = ds.sample_probe_geometry(ctx, rng, probe_param)
            sf = ds.plane_structure_fraction(ctx, probe, pose)
            u = np.asarray(pose["u"], dtype=np.float64)
            w = np.asarray(pose["w"], dtype=np.float64)
            sd_t = base_seed + 7919 * (ci * args.poses + k)      # 模板种子组
            # 观测：单次实现，种子与模板**完全不同**
            obs_seed = base_seed + 50_000_000 + 104729 * (ci * args.poses + k)
            _, us_obs, planes0 = render_at(ctx, probe, pose, render_params,
                                           seeds=[obs_seed])
            print(f"\n  --- 位姿 {k}: kind={probe.kind} nx={probe.nx} nz={probe.nz} "
                  f"dx_eff={probe.dx_effective:.4f} dz={probe.dz:.3f} "
                  f"near={probe.near:.2f} radius={probe.radius:.1f} "
                  f"fov={np.rad2deg(probe.fov_angle):.1f}deg")
            print(f"      平面结构占比={sf:.5f}  "
                  f"CT 平面 std={float(np.std(planes0['ct'])):.1f} HU")

            rec = {"case": ctx.case, "pose": k, "struct_frac": sf,
                   "probe": ds._probe_json(probe), "axes": {},
                   "slab0": planes0["ct"], "obs": us_obs}

            for axis_name, axis, step in (("u-lat", u, probe.dx_effective * 2.0),
                                          ("w-dep", w, probe.dz * 2.0)):
                offs = np.linspace(-args.span_mm, args.span_mm, args.steps)
                ct_sens, match = [], []
                tmpl0 = None
                for off in offs:
                    p2 = dict(pose)
                    p2["face"] = (np.asarray(pose["face"], dtype=np.float64)
                                  + axis * float(off))
                    usE, _, pl = render_at(
                        ctx, probe, p2, render_params,
                        seeds=[sd_t + 1009 * j for j in range(args.n_speckle)])
                    if abs(off) < 1e-9:
                        tmpl0 = usE
                    ct_sens.append(float(np.mean(np.abs(pl["ct"] - planes0["ct"]))
                                         / (np.std(planes0["ct"]) + 1e-6)))
                    match.append(ncc(usE, us_obs))
                i0 = int(np.argmin(np.abs(offs)))
                match = np.asarray(match)
                kmax = int(np.argmax(match))
                rec["axes"][axis_name] = {
                    "offsets": offs.tolist(), "step_mm": step,
                    "ct_sens": ct_sens, "match": match.tolist(),
                    "match_at0": float(match[i0]),
                    "match_max": float(match.max()),
                    "peak_mm": float(offs[kmax]),
                    "prominence": float(match.max() - match[i0]),
                    "extreme_ratio": float(match.max() / (match[i0] + 1e-9)),
                }
                print(f"    {axis_name} (候选步长 {step:.2f}mm)")
                print(f"      ct_sens: " + " ".join(f"{v:.3f}" for v in ct_sens))
                print(f"      match  : " + " ".join(f"{v:+.3f}" for v in match))
                print(f"      ⇒ match(0)={match[i0]:+.4f}  max={match.max():+.4f} "
                      f"@ {offs[kmax]:+.2f}mm  prominence={match.max() - match[i0]:+.4f}")
            rec["tmpl0"] = tmpl0
            samples.append(rec)

    # ---- 跨病例控制 + 检索 ----
    print(f"\n{'=' * 100}\n跨病例对照（解剖差异造成的\"机会水平\"）与检索")
    if len(samples) >= 2:
        from scipy import ndimage as ndi

        def fit(x, shape=(128, 96)):
            z = (shape[0] / x.shape[0], shape[1] / x.shape[1])
            return ndi.zoom(x, z, order=1)[:shape[0], :shape[1]]

        def vec(x):
            v = fit(x).ravel()
            v = v - v.mean()
            n = np.linalg.norm(v)
            return v / n if n > 1e-12 else v

        obs = np.stack([vec(s["obs"]) for s in samples])
        tmpl = np.stack([vec(s["tmpl0"]) for s in samples])
        S = obs @ tmpl.T                       # (i=obs, j=tmpl) 均为 ξ=0
        diag = np.diag(S).copy()
        off = S.copy()
        np.fill_diagonal(off, -np.inf)
        print(f"  {len(samples)} 个 (病例,位姿) 样本, 统一缩放到 128x96 后比较")
        print(f"  对角（自身模板 ξ=0）  : mean={diag.mean():+.4f} min={diag.min():+.4f} "
              f"max={diag.max():+.4f}")
        print(f"  非对角（其他病例 ξ=0）: mean={off.max(axis=1).mean():+.4f} "
              f"max={off.max():+.4f}")
        print(f"  差距（对角 − 最大非对角）: mean={np.mean(diag - off.max(axis=1)):+.4f}")
        # 检索：每个观测在"全部模板"里的排名（1 = 最好）
        order = np.argsort(-S, axis=1)
        ranks = np.array([int(np.where(order[i] == i)[0][0]) + 1 for i in range(len(S))])
        print(f"  检索 top-1 命中率 = {float((ranks == 1).mean()):.3f} "
              f"（随机 = {1.0 / len(S):.3f}）, 排名 = {ranks.tolist()}")
        # 空间置换对照：保留直方图、破坏空间对应
        shuf = []
        for s in samples:
            t = fit(s["tmpl0"])
            rng = np.random.default_rng(len(shuf) + 12345)
            t = t.reshape(-1)[rng.permutation(t.size)].reshape(t.shape)
            shuf.append(float(vec(t) @ vec(s["obs"])))
        print(f"  空间置换模板 vs 观测 NCC: mean={np.mean(shuf):+.4f} max={np.max(shuf):+.4f}"
              f"  ← 远低于对角则说明信号是**空间对应**而不是灰度直方图")
    else:
        print("  样本数 < 2，跳过")

    # ---- 逐样本汇总 ----
    print(f"\n{'=' * 100}\n逐样本盆地汇总（prominence = match(0) − |ξ|=span 处的 match）")
    print(f"  {'病例':<20}{'位姿':<5}{'轴':<7}{'结构占比':>9}{'match(0)':>10}"
          f"{'端点':>9}{'prominence':>12}")
    proms = []
    for s in samples:
        for an_, d in s["axes"].items():
            m = np.asarray(d["match"])
            i0 = int(np.argmin(np.abs(np.asarray(d["offsets"]))))
            ends = [m[0], m[-1]]
            p = float(m[i0] - np.mean(ends))
            proms.append((d["peak_mm"], p))
            print(f"  {s['case']:<20}{s['pose']:<5}{an_:<7}{s['struct_frac']:>9.5f}"
                  f"{m[i0]:>+10.4f}{np.mean(ends):>+9.4f}{p:>+12.4f}")
    peak_at0 = sum(1 for pk, _ in proms if abs(pk) < 1e-6)
    print(f"  ⇒ {peak_at0}/{len(proms)} 条曲线峰值落在 ξ=0；"
          f"prominence 中位 {np.median([p for _, p in proms]):+.4f}，"
          f"最小 {min(p for _, p in proms):+.4f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        slim = [{k: v for k, v in s.items() if k not in ("slab0", "obs", "tmpl0")}
                for s in samples]
        Path(args.out).write_text(json.dumps(
            {"render_params": render_params,
             "min_structure_frac": args.min_structure_frac,
             "preserve_lumen": bool(args.preserve_lumen),
             "records": slim}, indent=2), encoding="utf-8")
        print(f"\n结果已写入 {args.out}")

    print("\n判读：match(ξ) 必须在 ξ=0 处取全局极大（或 us_sens 在 0 处取极小）；"
          "曲线平坦 ⇒ 平面上没有位姿信息，训练必然无效。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
