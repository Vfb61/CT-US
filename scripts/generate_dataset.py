#!/usr/bin/env python
"""Generate the {CT, US_synthetic, T_CT->US} pseudo-paired dataset.

Each (case, sample) writes:
  us.png / ct_slice.nii.gz / seg_slice.nii.gz / transform.npz / params.json
plus an index JSON-lines file.

Examples:
  python scripts/generate_dataset.py \
      --liver dataset/Task03_Liver \
      --vessel dataset/Task08_HepaticVessel \
      --out outputs/pairs --volumes 6 --per_volume 10 --seed 0

  python scripts/generate_dataset.py \
      --data dataset/Task08_HepaticVessel --out outputs/pairs \
      --volumes 10 --per_volume 8
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ct2us import dataset as ds
from ct2us import io_utils


def _worker(args):
    task_dir, label, name, split, per_volume, seed, out_root, global_params = args
    rng = np.random.default_rng(seed)
    ctx = ds.load_case(task_dir, split, case_name=name)
    writer = ds.DatasetWriter(Path(out_root) / label.replace("\\", "_").replace("/", "_").replace(":", ""), name="index")
    metas = []
    for k in range(per_volume):
        sample = ds.generate_sample(ctx, index=k, rng=rng,
                                    global_params=global_params,
                                    compute_reference=bool(global_params.get("with_reference", False)))
        sample["sid"] = f"{name}_{k:02d}"
        meta = writer.write(ctx, sample, split="train")
        metas.append(meta)
    writer.finish()
    return metas


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

    jobs = []
    for label, task_dir in tasks:
        names = io_utils.list_cases(task_dir, args.split)
        for n in names[args.first:args.first + args.volumes]:
            jobs.append((str(task_dir), label, n, args.split, args.per_volume,
                         int((args.seed + len(jobs)) % (2 ** 31)), args.out,
                         {"with_reference": bool(args.with_reference),
                          "no_deform": bool(args.no_deform)}))

    print(f"{len(jobs)} case tasks to generate")
    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(_worker, jobs))
    else:
        results = [_worker(j) for j in jobs]

    total = sum(len(r) for r in results)
    out_res = Path(args.out).resolve()
    print(f"generated {total} samples -> {out_res}")
    indexes = sorted(str(p) for p in out_res.rglob("*_index.jsonl"))
    print("index files:")
    for p in indexes:
        print("  " + p)


if __name__ == "__main__":
    main()