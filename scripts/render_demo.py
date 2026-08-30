#!/usr/bin/env python
"""Quick render demo: generate a handful of synthetic US slices from one or two
CT cases and write a visual preview montage.

Examples:
  python scripts/render_demo.py --data dataset/Task08_HepaticVessel --out outputs/demo
  python scripts/render_demo.py --liver dataset/Task03_Liver --vessel dataset/Task08_HepaticVessel --out outputs/demo --n 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ct2us import calibrate, dataset, io_utils, render


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", help="a single task dir (auto-detects labels)")
    ap.add_argument("--liver", help="Task03 liver task dir")
    ap.add_argument("--vessel", help="Task08 hepatic vessel task dir")
    ap.add_argument("--out", default="outputs/demo", help="output dir")
    ap.add_argument("--n", type=int, default=4, help="samples per case")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="imagesTr")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tasks = []
    if args.data:
        tasks.append((Path(args.data), args.data))
    if args.liver:
        tasks.append((Path(args.liver), args.liver))
    if args.vessel:
        tasks.append((Path(args.vessel), args.vessel))
    if not tasks:
        ap.error("provide --data or --liver/--vessel")

    from matplotlib import pyplot as plt

    cases = []
    seen = set()
    for task_dir, label in tasks:
        names = io_utils.list_cases(task_dir, args.split)
        for n in names[:2]:
            if n in seen:
                continue
            seen.add(n)
            cases.append((task_dir, label, n))

    montage_rows = []
    for task_dir, label, name in cases:
        print(f"[load] {label} / {name}")
        ctx = dataset.load_case(task_dir, args.split, case_name=name)
        for k in range(args.n):
            sample = dataset.generate_sample(ctx, index=k, rng=rng,
                                            compute_reference=True)
            rows = _compose_rows(sample)
            montage_rows.extend(rows)
            for role, arr in [("us", sample["render"]["uint8"]),
                              ("ct", sample["planes"]["ct"]),
                              ("seg", sample["planes"]["tissue"]),
                              ("phys", sample["reference"]["uint8"])]:
                fn = out / f"{name}_{k:02d}_{role}.png"
                _save_gray(fn, arr)
            print(f"  -> sample {k}: shape={sample['render']['uint8'].T.shape} "
                  f"tissues={sorted(set(np.unique(sample['planes']['tissue']).tolist()))}")
            if k == 0:
                _save_metrics(name, sample)

    if montage_rows:
        _montage(montage_rows, out / "preview.png", ncol=len(montage_rows[0]))
    print(f"done -> {out.resolve()}")


def _compose_rows(sample) -> list:
    from matplotlib import pyplot as plt
    cmap = "gray"
    r1 = [("Synthetic US", sample["render"]["uint8"], cmap),
          ("Resliced CT", _norm_ct(sample["planes"]["ct"]), cmap),
          ("Tissue labels", _label_rgb(sample["planes"]["tissue"]), None)]
    r2 = [("Physics proxy", sample["reference"]["uint8"], cmap),
          ("Specular", _norm01(sample["render"]["specular_plane"]), cmap),
          ("Envelope", _norm01(sample["render"]["envelope"]), cmap)]
    return [r1, r2]


def _norm01(a):
    a = np.asarray(a, dtype=np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)


def _norm_ct(a):
    return _norm01(np.asarray(a, dtype=np.float32))


def _label_rgb(seg):
    seg = np.asarray(seg, dtype=np.int32)
    cmap = np.array([
        [0, 0, 0],          # background
        [160, 160, 160],    # body wall
        [255, 255, 255],    # bone
        [90, 130, 80],      # liver
        [200, 60, 60],      # tumour
        [30, 30, 160],      # vessel lumen
        [220, 180, 60],     # vessel wall
    ], dtype=np.float32) / 255.0
    rgb = np.zeros((*seg.shape, 3), dtype=np.float32)
    for i in range(cmap.shape[0]):
        rgb[seg == i] = cmap[i]
    return rgb


def _save_gray(path, arr):
    from PIL import Image
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    Image.fromarray(_uint8(arr), mode="L").save(path)


def _uint8(a):
    a = np.asarray(a, dtype=np.float32)
    lo, hi = a.min(), np.percentile(a, 99.5)
    a = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) * 255
    return a.astype(np.uint8)


def _montage(rows, path, ncol=3, scale=2.0):
    from matplotlib import pyplot as plt
    nrow = len(rows)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol * scale * 0.4,
                                                  4 * nrow * scale * 0.4))
    axes = np.atleast_2d(axes)
    for r, row in enumerate(rows):
        for c, (title, arr, cmap) in enumerate(row):
            ax = axes[r, c]
            disp = arr if arr.ndim == 3 else arr.T
            if cmap is None:
                ax.imshow(disp, interpolation="nearest")
            else:
                ax.imshow(disp, cmap=cmap, interpolation="nearest")
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_metrics(name, sample):

    def _pad(d, sp_):
        return {k: round(float(v), 3) for k, v in d.items()}

    env = sample["render"]["envelope"]
    tissue = sample["planes"]["tissue"]
    dm = calibrate.metrics(env, tissue, sample["probe"].depth_offsets(),
                           sample["render"]["specular_plane"])
    ref = sample["reference"]["envelope"]
    rm = calibrate.metrics(ref, tissue, sample["probe"].depth_offsets(),
                           sample["render"]["specular_plane"])
    dist = calibrate.metric_distance(dm, rm)
    text = f"{name}\nrender : {_pad(dm, 0)}\nphysics: {_pad(rm, 0)}\nmetric distance = {dist:.2f}"
    print(text)


if __name__ == "__main__":
    main()