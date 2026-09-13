#!/usr/bin/env python
"""Generate the {CT, US_synthetic, T_CT->US} pseudo-paired dataset.

Each (case, sample) writes:
  us.png / ct_slice.nii.gz / seg_slice.nii.gz / transform.npz / params.json
plus one index JSON-lines file per (task, label) shard.

幂等 / 并发（修 bug 后）：
  * 每个 worker 只负责写自己那一份样本目录，**索引由主进程统一原子重写**，
    不再出现多进程无锁 `open(...,"a")` 追加导致的陈旧重复行；
  * `--resume` 跳过已完成的样本（按 `params.json` 判定）；
  * 输出目录会写一个 `manifest.json` 记录完整命令与参数（provenance）。

Examples:
  python scripts/generate_dataset.py \
      --liver dataset/Task03_Liver \
      --vessel dataset/Task08_HepaticVessel \
      --out outputs/pairs --volumes 6 --per_volume 10 --seed 0

  python scripts/generate_dataset.py \
      --data dataset/Task08_HepaticVessel --out outputs/pairs \
      --volumes 10 --per_volume 8 --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ct2us import dataset as ds
from ct2us import fs_utils as fsu
from ct2us import io_utils


def _shard_dir(out_root: str, label: str) -> Path:
    return Path(out_root) / label.replace("\\", "_").replace("/", "_").replace(":", "")


def _stable_seed(label: str, case_name: str, base_seed: int) -> int:
    """与任务顺序无关的稳定种子（旧实现用 len(jobs)，--liver/--vessel 顺序会改变结果）。"""
    h = hashlib.blake2b(f"{label}|{case_name}|{base_seed}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") % (2 ** 31)


def _worker(args):
    """只生成样本目录，返回 (shard_dir, [rows])；不触碰索引文件。"""
    (task_dir, label, name, split, per_volume, seed, out_root,
     global_params, skip_existing) = args
    rng = np.random.default_rng(seed)
    ctx = ds.load_case(task_dir, split, case_name=name,
                       iso_spacing=global_params.get("iso_spacing"),
                       noise_seed=global_params.get("noise_seed"))
    writer = ds.DatasetWriter(_shard_dir(out_root, label), name="index",
                              skip_existing=skip_existing)
    rows = []
    for k in range(per_volume):
        sid = f"{name}_{k:02d}"
        if skip_existing and writer.is_complete(sid):
            continue
        # 每样本一个**确定、可复现**的渲染种子（与前面生成过多少样本无关）。
        # 旧实现把病例级 rng 一路传进 render_bmode，导致同一 sid 的散斑实现
        # 取决于生成顺序 —— 既不可复现，也无法从磁盘核对。
        render_seed = int(global_params.get("render_seed_base", 0)) + seed + 104729 * k
        sample = ds.generate_sample(
            ctx, index=k, rng=rng, global_params=global_params,
            compute_reference=bool(global_params.get("with_reference", False)),
            render_seed=(render_seed if global_params.get("persist_rng", True) else None))
        sample["sid"] = sid
        rows.append(writer.write(ctx, sample, split="train"))
    return str(writer.index_path), rows


def _merge_index(index_path: str, rows) -> int:
    """把新行并入已有索引（按 sid 去重、后写覆盖），原子重写。"""
    existing = []
    p = Path(index_path)
    if p.exists():
        try:
            existing = fsu.read_index_jsonl(p)
        except Exception:
            existing = []
    return fsu.write_index_jsonl(p, existing + list(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", help="single task dir (auto-detects labels)")
    ap.add_argument("--liver", help="Task03 liver task dir")
    ap.add_argument("--vessel", help="Task08 hepatic vessel task dir")
    ap.add_argument("--out", default="outputs/pairs")
    ap.add_argument("--split", default="imagesTr")
    ap.add_argument("--volumes", type=int, default=6, help="unique CT cases used")
    ap.add_argument("--per_volume", type=int, default=10, help="slices per case")
    ap.add_argument("--first", type=int, default=0, help="use cases [first, first+volumes)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--with_reference", action="store_true",
                    help="also render the physics-proxy reference sample")
    ap.add_argument("--no_deform", action="store_true",
                    help="disable breathing/pressure deformation (rigid set)")
    ap.add_argument("--deform_prob", type=float, default=None, metavar="P",
                    help="probability that a sample actually gets a non-zero deformation "
                         "(default: ct2us.deformation default, i.e. 0.65 non-zero)")
    ap.add_argument("--iso", type=float, default=None, metavar="MM",
                    help="resample CT+labels to isotropic spacing (mm) before generation")
    ap.add_argument("--resume", action="store_true",
                    help="skip samples whose directory is already complete")
    ap.add_argument("--force", action="store_true",
                    help="regenerate everything (default behaviour; overrides --resume)")
    # ---- 阶段 A：观测端修复（由 scripts/ab_observation.py 实测驱动）----
    ap.add_argument("--n-speckle", type=int, default=1, metavar="K",
                    help="同一位姿渲染 K 次独立散斑实现，并额外写出 us_mean.png。"
                         "实测单帧散斑噪声是位姿信号的 2–9 倍；K≥3 时梯度域盆地 "
                         "prominence 由 0.111 提升到 0.165（与 fixref 叠加则到 0.606）")
    ap.add_argument("--log-ref", type=float, default=None, metavar="V",
                    help="对数压缩的**固定**参考包络值。默认 None = 每图 99 百分位"
                         "（历史行为，会让位姿->像素映射带全局缩放）。实测固定参考是"
                         "最强的单项修复：梯度域 prominence 0.111 -> 0.361，峰值在 0 达 8/8")
    ap.add_argument("--speckle-strength", type=float, default=None,
                    help="覆盖 render 的 speckle_strength（默认 0.7）")
    ap.add_argument("--speckle-lat-px", type=float, default=None,
                    help="覆盖 render 的 speckle_lat_px（默认 5.0）")
    ap.add_argument("--render-kv", nargs="*", default=[], metavar="K=V",
                    help="任意 render 参数覆盖，如 shadow_db=15")
    args = ap.parse_args()

    tasks = []
    if args.data:
        tasks.append(("data_" + Path(args.data).name, Path(args.data)))
    if args.liver:
        tasks.append(("liver_" + Path(args.liver).name, Path(args.liver)))
    if args.vessel:
        tasks.append(("vessel_" + Path(args.vessel).name, Path(args.vessel)))
    if not tasks:
        ap.error("provide --data or --liver/--vessel")

    skip_existing = bool(args.resume) and not bool(args.force)
    t_start = time.time()

    # ---- 组装 render 覆盖 + 阶段 A 参数 ----
    render_override: dict = {}
    for kv in args.render_kv or []:
        k2, _, v2 = kv.partition("=")
        try:
            render_override[k2.strip()] = float(v2)
        except ValueError:
            render_override[k2.strip()] = v2
    if args.log_ref is not None:
        render_override["log_ref"] = float(args.log_ref)
    if args.speckle_strength is not None:
        render_override["speckle_strength"] = float(args.speckle_strength)
    if args.speckle_lat_px is not None:
        render_override["speckle_lat_px"] = float(args.speckle_lat_px)

    jobs = []
    for label, task_dir in tasks:
        names = io_utils.list_cases(task_dir, args.split)
        for n in names[args.first:args.first + args.volumes]:
            s = _stable_seed(label, n, args.seed)
            jobs.append((str(task_dir), label, n, args.split, args.per_volume,
                         s, args.out,
                         {"with_reference": bool(args.with_reference),
                          "no_deform": bool(args.no_deform),
                          "deform_prob": args.deform_prob,
                          "iso_spacing": args.iso,
                          "render": render_override,
                          "n_speckle": int(args.n_speckle),
                          "persist_rng": True,
                          # 每个 case 一个稳定的 scatter 噪声种子（会被落盘）
                          "noise_seed": s,
                          "render_seed_base": 0},
                         skip_existing))

    print(f"{len(jobs)} case tasks to generate ({'resume' if skip_existing else 'force/overwrite'})")
    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(_worker, jobs))
    else:
        results = [_worker(j) for j in jobs]

    # ---- 索引：全部由主进程原子重写（并发安全的根本保证）----
    by_index: dict[str, list] = {}
    for index_path, rows in results:
        by_index.setdefault(index_path, []).extend(rows)
    for index_path, rows in by_index.items():
        n_unique = _merge_index(index_path, rows)
        print(f"index -> {index_path}  ({len(rows)} new rows, {n_unique} unique)")

    total = sum(len(r) for _, r in results)
    out_res = Path(args.out).resolve()
    print(f"generated {total} new samples -> {out_res}")

    # ---- provenance ----
    manifest = {
        "argv": sys.argv,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "elapsed_sec": round(time.time() - t_start, 1),
        "args": vars(args),
        "n_jobs": len(jobs),
        "n_samples_written": total,
        "shards": sorted(by_index),
    }
    try:
        import nibabel
        manifest["nibabel"] = nibabel.__version__
    except Exception:
        pass
    fsu.atomic_write_json(out_res / "manifest.json", manifest)
    print(f"provenance -> {out_res / 'manifest.json'}")

    indexes = sorted(str(p) for p in out_res.rglob("*_index.jsonl"))
    print("index files:")
    for p in indexes:
        print("  " + p)


if __name__ == "__main__":
    main()
