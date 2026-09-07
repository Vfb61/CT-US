#!/usr/bin/env python
"""Generate 3D CT-US paired volumes for E:\new registration project.

Output format (matching E:\new's data/ directory structure):
  data/ct/
    case_000.nii.gz    # 3D CT volume (D, H, W), HU values
    case_001.nii.gz
  data/us/
    case_000.nii.gz    # 3D US volume (D, H, W), float32 [0,1]
    case_001.nii.gz

E:\new normalization:
  CT: (vol + 200) / 500.0
  US: (vol - vol.min()) / (vol.max() - vol.min())

Examples:
  python scripts/generate_for_new.py \
      --liver dataset/Task03_Liver/Task03_Liver \
      --out data --n_cases 10 --n_elev 32 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ct2us import dataset as ds
from ct2us import io_utils


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--liver", help="Task03 liver task dir")
    ap.add_argument("--vessel", help="Task08 hepatic vessel task dir")
    ap.add_argument("--data", help="single task dir (auto-detects labels)")
    ap.add_argument("--out", default="data", help="output root (creates ct/ and us/ subdirs)")
    ap.add_argument("--n_cases", type=int, default=10, help="number of cases to generate")
    ap.add_argument("--per_case", type=int, default=1, help="volumes per case (different poses)")
    ap.add_argument("--n_elev", type=int, default=32, help="elevation slices per volume")
    ap.add_argument("--elev_spacing", type=float, default=1.0, help="elevation spacing (mm)")
    ap.add_argument("--split", default="imagesTr")
    ap.add_argument("--first", type=int, default=0, help="skip first N cases")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tasks = []
    if args.data:
        tasks.append(("data", Path(args.data)))
    if args.liver:
        tasks.append(("liver", Path(args.liver)))
    if args.vessel:
        tasks.append(("vessel", Path(args.vessel)))
    if not tasks:
        ap.error("provide --data or --liver/--vessel")

    out_root = Path(args.out)
    ct_dir = out_root / "ct"
    us_dir = out_root / "us"
    ct_dir.mkdir(parents=True, exist_ok=True)
    us_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    case_idx = 0

    for label, task_dir in tasks:
        names = io_utils.list_cases(task_dir, args.split)
        for name in names[args.first:args.first + args.n_cases]:
            print(f"[{case_idx:03d}] Loading {label}/{name}...")
            try:
                ctx = ds.load_case(str(task_dir), args.split, case_name=name, rng=rng)
            except Exception as e:
                print(f"  SKIP (load error): {e}")
                continue

            for vol_idx in range(args.per_case):
                sid = f"{label}_{name}_{vol_idx:02d}"
                print(f"  Generating {sid} (n_elev={args.n_elev})...")
                try:
                    vol_data = ds.generate_volume(
                        ctx, rng,
                        n_elev=args.n_elev,
                        elev_spacing=args.elev_spacing,
                    )
                except Exception as e:
                    print(f"  SKIP (gen error): {e}")
                    continue

                # Save CT volume (NIfTI, HU values)
                ct_vol = vol_data["ct_vol"]  # (n_elev, nz, nx)
                # Build affine: elevation->z, depth->y, lateral->x
                # Use identity affine for simplicity (E:\new will resize)
                affine = np.eye(4, dtype=np.float64)
                io_utils.save_nifti(ct_dir / f"{sid}.nii.gz", ct_vol, affine)

                # Save US volume (NIfTI, float32 [0,1])
                us_vol = vol_data["us_vol"]  # (n_elev, nz, nx), float32 [0,1]
                io_utils.save_nifti(us_dir / f"{sid}.nii.gz", us_vol, affine)

                print(f"    CT: {ct_vol.shape}, US: {us_vol.shape}")

            case_idx += 1

    print(f"\nDone. Generated {case_idx} cases -> {out_root.resolve()}")
    print(f"  CT: {ct_dir}")
    print(f"  US: {us_dir}")


if __name__ == "__main__":
    main()
