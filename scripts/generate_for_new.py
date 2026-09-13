#!/usr/bin/env python
"""Generate 3D CT-US paired volumes for downstream registration projects (E:\\new).

Output layout (per case id `<label>_<case>_<vol>`):
    <out>/ct/<sid>.nii.gz        3D CT volume, HU, **正确的 afffine**
    <out>/us/<sid>.nii.gz        3D US volume, float32 [0,1]
    <out>/seg/<sid>.nii.gz       3D 组织标签（int16）
    <out>/meta/<sid>.npz         几何与 GT：us_to_ct / vol_affine / nu / nv / dx_mm / dz_mm
    <out>/meta/<sid>.json        probe / pose / deform / 逐层 GT 列表
    <out>/index.jsonl            样本索引（含 gt 文件相对路径）

⚠️ 与旧版的差别（修 D1）：旧版把 affine 写死 `np.eye(4)` 且**丢掉全部 GT**
（transform / probe / seg / 逐层变换），导致下游只能做自监督、且 CT 与 US 同网格
使"配准答案恒为单位阵"。现在 GT 全部落盘。

下游归一化约定（E:\\new）：
  CT: (vol + 200) / 500.0
  US: (vol - vol.min()) / (vol.max() - vol.min())

Examples:
  python scripts/generate_for_new.py \
      --liver dataset/Task03_Liver/Task03_Liver \
      --out E:/new/data_build --n_cases 10 --n_elev 32 --elev_spacing 1.0 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ct2us import dataset as ds
from ct2us import fs_utils as fsu
from ct2us import io_utils


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--liver", help="Task03 liver task dir")
    ap.add_argument("--vessel", help="Task08 hepatic vessel task dir")
    ap.add_argument("--data", help="single task dir (auto-detects labels)")
    ap.add_argument("--out", default="data", help="output root")
    ap.add_argument("--n_cases", type=int, default=10, help="number of cases to generate")
    ap.add_argument("--per_case", type=int, default=1, help="volumes per case (different poses)")
    ap.add_argument("--n_elev", type=int, default=32, help="elevation slices per volume")
    ap.add_argument("--elev_spacing", type=float, default=1.0, help="elevation spacing (mm)")
    ap.add_argument("--split", default="imagesTr")
    ap.add_argument("--first", type=int, default=0, help="skip first N cases")
    ap.add_argument("--iso", type=float, default=None, metavar="MM",
                    help="resample CT+labels to isotropic spacing (mm) before generation")
    ap.add_argument("--no_deform", action="store_true",
                    help="disable breathing/pressure deformation (rigid set)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="skip already generated sids")
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
    ct_dir, us_dir, seg_dir, meta_dir = (out_root / "ct", out_root / "us",
                                        out_root / "seg", out_root / "meta")
    for d in (ct_dir, us_dir, seg_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    index_path = out_root / "index.jsonl"
    rows = []
    case_idx = 0
    t0 = time.time()

    for label, task_dir in tasks:
        names = io_utils.list_cases(task_dir, args.split)
        for name in names[args.first:args.first + args.n_cases]:
            print(f"[{case_idx:03d}] Loading {label}/{name}...")
            try:
                ctx = ds.load_case(str(task_dir), args.split, case_name=name, rng=rng,
                                   iso_spacing=args.iso)
            except Exception as e:
                print(f"  SKIP (load error): {e}")
                continue

            for vol_idx in range(args.per_case):
                sid = f"{label}_{name}_{vol_idx:02d}"
                if args.resume and (meta_dir / f"{sid}.json").exists():
                    print(f"  SKIP {sid} (exists)")
                    continue
                print(f"  Generating {sid} (n_elev={args.n_elev})...")
                try:
                    v = ds.generate_volume(
                        ctx, rng, n_elev=args.n_elev, elev_spacing=args.elev_spacing,
                        global_params={"no_deform": bool(args.no_deform)})
                except Exception as e:
                    print(f"  SKIP (gen error): {e}")
                    continue

                # CT / US / seg：体素顺序 (n_elev, nz, nx) -> NIfTI 存成 (nx, nz, n_elev)
                affine = np.asarray(v["vol_affine"], dtype=np.float64)
                io_utils.save_nifti(ct_dir / f"{sid}.nii.gz",
                                    np.ascontiguousarray(v["ct_vol"].transpose(2, 1, 0)),
                                    affine, like=None)
                io_utils.save_nifti(us_dir / f"{sid}.nii.gz",
                                    np.ascontiguousarray(v["us_vol"].transpose(2, 1, 0)),
                                    affine, like=None)
                io_utils.save_nifti(seg_dir / f"{sid}.nii.gz",
                                    np.ascontiguousarray(v["seg_vol"].transpose(2, 1, 0).astype(np.int16)),
                                    affine, like=None)

                tr = v["transform"]
                np.savez_compressed(
                    meta_dir / f"{sid}.npz",
                    us_to_ct=tr["us_to_ct"], ct_to_us=tr["ct_to_us"],
                    slice_affine=tr["slice_affine"], vol_affine=affine,
                    face=tr["face"], u=tr["u"], v=tr["v"], w=tr["w"],
                    dx_mm=np.float64(tr["dx_mm"]), dz_mm=np.float64(tr["dz_mm"]),
                    nu=np.int32(tr["nu"]), nv=np.int32(tr["nv"]),
                    n_elev=np.int32(v["n_elev"]),
                    elev_spacing=np.float64(v["elev_spacing"]),
                    elev_axis=np.asarray(v["elev_axis"], dtype=np.float64),
                    slice_us_to_ct=np.stack([t["us_to_ct"] for t in v["transforms"]]),
                )
                meta = {
                    "sid": sid, "split": "train", "case": name, "label": label,
                    "shape_dhw": list(v["us_vol"].shape),
                    "probe": ds._probe_json(v["probe"]),
                    "pose": ds._pose_json(v["pose"]),
                    "deform": v["deform_params"],
                    "n_elev": int(v["n_elev"]),
                    "elev_spacing": float(v["elev_spacing"]),
                    "vol_affine_err_mm": float(v["vol_affine_err_mm"]),
                    "files": {
                        "ct": f"ct/{sid}.nii.gz",
                        "us": f"us/{sid}.nii.gz",
                        "seg": f"seg/{sid}.nii.gz",
                        "meta": f"meta/{sid}.npz",
                    },
                }
                fsu.atomic_write_json(meta_dir / f"{sid}.json", meta)
                rows.append(meta)
                print(f"    CT: {v['ct_vol'].shape}, US: {v['us_vol'].shape}, "
                      f"affine_err={v['vol_affine_err_mm']:.4f}mm")
            case_idx += 1

    if rows:
        fsu.write_index_jsonl(index_path, rows)
    fsu.atomic_write_json(out_root / "manifest.json", {
        "argv": sys.argv,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - t0, 1),
        "args": vars(args),
        "n_samples": len(rows),
    })

    print(f"\nDone. {len(rows)} cases -> {out_root.resolve()}")
    print(f"  ct/us/seg/meta 子目录 + index.jsonl + manifest.json")


if __name__ == "__main__":
    main()
