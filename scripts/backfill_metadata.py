#!/usr/bin/env python
"""回填/修复已有数据集的元信息（不重新生成图像）。

做三件事，全部可原地完成、不需要重跑生成：

  1. **补 `dx_eff` 等 probe 字段** 到 `params.json` 与 `transform.npz`。
     旧数据只存了 `probe.dx`——它对 convex 是无意义占位值（例如 0.55，真值 0.1686），
     下游 `baseline_registration.load_sample` 会退到默认 0.55，使 TRE 被放大 1.37–4.14×。

  2. **索引体检与修复**：按 `sid` 去重（保留最后一行，与运行时语义一致），
     并用 `probe.(nx,nz)` 与磁盘 `us.png` 尺寸对账，剔除明显陈旧的重复行。
     历史数据里 pairs_v1/vessel_* 有 957 行 / 872 唯一 sid。

  3. （可选 `--fix-affine`）把 `ct_slice`/`seg_slice` 的 NIfTI affine 从错误的
     `us_to_ct` 改写成正确的**平面 affine**。注意：这需要重写每个样本的
     NIfTI 文件（pairs_v1 约 2.7 GB 的读+写），耗时较长；且当前没有任何下游代码
     读该 affine，因此默认**不**开启。

用法：
    python scripts/backfill_metadata.py --scan outputs/pairs_v1 --dry-run
    python scripts/backfill_metadata.py --scan outputs/pairs_v1
    python scripts/backfill_metadata.py --scan outputs/pairs_v1 --fix-affine
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ct2us import dataset as ds
from ct2us import fs_utils as fsu
from ct2us import geometry as geo
from ct2us import io_utils


def _load_npz(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


_PROBE_KEYS = ("kind", "nx", "nz", "dx", "dz", "fov_angle", "near", "radius", "freq")


def _probe_with_pose(pr_json: dict, pose_json: dict):
    """用 probe_from_meta 重建 Probe（正确处理 convex 的 dx 占位值问题）。"""
    probe = geo.probe_from_meta(pr_json)
    pose = {k: np.asarray(pose_json[k], dtype=np.float64)
            for k in ("face", "u", "v", "w")}
    return probe, pose


def backfill_sample(sample_dir: Path, params: dict, fix_affine: bool,
                    dry_run: bool, write_geometry: bool = False) -> dict:
    """返回统计 dict。"""
    out = {"sid": params.get("sid", sample_dir.name), "fixed_params": False,
           "fixed_transform": False, "fixed_affine": False, "fixed_geometry": False,
           "slice_affine_err_mm": None, "size_ok": True, "note": ""}

    pr_old = params["params"]["probe"]
    po = params["params"]["pose"]
    probe, pose = _probe_with_pose(pr_old, po)

    # ---------- 1. params.json 补字段 ----------
    pr_new = ds._probe_json(probe)
    merged = dict(pr_old)
    merged.update({k: v for k, v in pr_new.items() if k not in pr_old})
    if any(merged.get(k) != pr_old.get(k) for k in pr_new):
        out["fixed_params"] = True
        if not dry_run:
            params["params"]["probe"] = merged
            fsu.atomic_write_json(sample_dir / "params.json", params)

    # ---------- 2. transform.npz 补字段 ----------
    tf_path = sample_dir / "transform.npz"
    if tf_path.exists():
        tz = _load_npz(tf_path)
        A, err = ds.plane_affine_true(probe, pose)
        out["slice_affine_err_mm"] = err
        if err > 1.0:
            out["note"] = f"平面 affine 近似残差 {err:.2f}mm（convex 扇面网格非线性，见 geometry.npz）"
        newd = {
            "us_to_ct": tz["us_to_ct"], "ct_to_us": tz["ct_to_us"],
            "slice_affine": A,
            "face": tz["face"], "u": tz["u"], "v": tz["v"], "w": tz["w"],
            "dx_mm": np.float64(probe.dx_effective), "dz_mm": np.float64(probe.dz),
            "nu": np.int32(probe.nx), "nv": np.int32(probe.nz),
            "kind": str(probe.kind), "near": np.float64(probe.near),
            "radius": np.float64(probe.radius),
            "fov_angle": np.float64(probe.fov_angle),
        }
        if any(k not in tz for k in newd):
            out["fixed_transform"] = True
            if not dry_run:
                # 注意：np.savez_compressed 传 Path 会自动补 ".npz"，必须给字符串
                tmp = tf_path.with_name("transform.tmp.npz")
                np.savez_compressed(str(tmp), **newd)
                os.replace(tmp, tf_path)

    # ---------- 3. geometry.npz（**默认不写**，见下）----------
    # 说明：整张世界网格可占 1 MB/样本（2072 样本约 2.1 GB）。而它由
    # `probe + pose` 完全确定，用 `ct2us.geometry.world_grid_from_meta()` 可即时重建，
    # 因此默认不落盘，避免数据集体积翻倍。仅在 `--write-geometry` 时写出。
    geom_path = sample_dir / "geometry.npz"
    if write_geometry and not geom_path.exists():
        try:
            world = probe.world_grid(pose["face"], pose["u"], pose["v"],
                                     pose["w"], deform=None)
            out["fixed_geometry"] = True
            if not dry_run:
                tmp = geom_path.with_name("geometry.tmp.npz")
                np.savez_compressed(
                    str(tmp), world_grid=np.asarray(world, dtype=np.float32),
                    kind=str(probe.kind), nx=np.int32(probe.nx), nz=np.int32(probe.nz),
                    dx_mm=np.float64(probe.dx_effective), dz_mm=np.float64(probe.dz),
                    near=np.float64(probe.near), radius=np.float64(probe.radius),
                    fov_angle=np.float64(probe.fov_angle),
                    face=pose["face"], u=pose["u"], v=pose["v"], w=pose["w"],
                )
                os.replace(tmp, geom_path)
                # 顺带把 files 里补上 geometry
                files = params.setdefault("files", {})
                if files.get("geometry") != "geometry.npz":
                    files["geometry"] = "geometry.npz"
                    if not dry_run:
                        fsu.atomic_write_json(sample_dir / "params.json", params)
        except Exception as e:
            out["note"] += f" geometry 失败: {e}"

    # ---------- 3. 可选：重写 NIfTI affine ----------
    if fix_affine:
        A, _ = ds.plane_affine_true(probe, pose)
        for name in ("ct_slice.nii.gz", "seg_slice.nii.gz"):
            p = sample_dir / name
            if not p.exists():
                continue
            data, _, _ = io_utils.load_volume(p)
            # load_volume 返回 (nx,nz) 顺序；存储时转置为 (nz,nx)
            out["fixed_affine"] = True
            if not dry_run:
                fid = data.astype(np.float32 if name.startswith("ct") else np.int16)
                io_utils.save_nifti(p, fid.transpose(1, 0), A, like=None)

    return out


def repair_index(index_path: Path, dry_run: bool, verbose: bool = True) -> dict:
    """按 sid 去重 + 用 us.png 尺寸剔除陈旧行。"""
    rows = fsu.read_index_jsonl(index_path)
    base = index_path.parent
    kept, dropped = [], []
    from PIL import Image
    for r in rows:
        pr = (r.get("params") or {}).get("probe") or {}
        png = base / r["sid"] / r.get("files", {}).get("us", "us.png")
        if not png.exists():
            dropped.append((r["sid"], "样本目录/us.png 缺失"))
            continue
        try:
            with Image.open(png) as im:
                w, h = im.size          # PNG 是 (width=nz, height=nx)
        except Exception as e:
            dropped.append((r["sid"], f"us.png 读取失败: {e}"))
            continue
        size = (h, w)                   # 转回 (nx, nz)，与 probe 一致
        if pr and (int(pr.get("nx", -1)), int(pr.get("nz", -1))) != size:
            dropped.append((r["sid"],
                            f"index(nx,nz)={(pr.get('nx'), pr.get('nz'))} != us.png{size}"))
            continue
        kept.append(r)
    if not dry_run and kept:
        tmp_path = index_path.with_name("index.tmp")
        fsu.atomic_write_text(
            tmp_path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
        tmp_path.replace(index_path)
    if verbose:
        print(f"  {index_path}: {len(rows)} -> {len(kept)} 行（丢弃 {len(dropped)}）")
        for sid, why in dropped[:5]:
            print(f"      - {sid}: {why}")
        if len(dropped) > 5:
            print(f"      ... 其余 {len(dropped) - 5} 条")
    return {"before": len(rows), "after": len(kept), "dropped": dropped}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", nargs="+", required=True,
                    help="dataset root dirs (each containing *_index.jsonl)")
    ap.add_argument("--dry-run", action="store_true", help="只报告，不写盘")
    ap.add_argument("--fix-affine", action="store_true",
                    help="同时重写 ct_slice/seg_slice 的 NIfTI affine（慢，重写全部文件）")
    ap.add_argument("--write-geometry", action="store_true",
                    help="额外落盘整张 world_grid 到 geometry.npz（约 1MB/样本；"
                         "不写也能用 geometry.world_grid_from_meta 精确重建）")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 个样本（调试用）")
    args = ap.parse_args()

    t0 = time.time()
    for root in args.scan:
        root = Path(root)
        index_files = sorted(root.rglob("*_index.jsonl"))
        if not index_files:
            print(f"[skip] {root}: 没有 *_index.jsonl")
            continue
        print(f"\n=== {root} ===")
        for index_path in index_files:
            print(f"--- 索引修复: {index_path.name}")
            repair_index(index_path, dry_run=args.dry_run)

            rows = fsu.read_index_jsonl(index_path)
            if args.limit:
                rows = rows[: args.limit]
            n_params = n_tf = n_aff = n_geom = 0
            approx = []
            for i, r in enumerate(rows):
                sd = index_path.parent / r["sid"]
                if not (sd / "params.json").exists():
                    continue
                try:
                    params = json.loads((sd / "params.json").read_text(encoding="utf-8"))
                    st = backfill_sample(sd, params, args.fix_affine, args.dry_run,
                                         write_geometry=args.write_geometry)
                    n_params += int(st["fixed_params"])
                    n_tf += int(st["fixed_transform"])
                    n_aff += int(st["fixed_affine"])
                    n_geom += int(st["fixed_geometry"])
                    if st["slice_affine_err_mm"] and st["slice_affine_err_mm"] > 1.0:
                        approx.append((st["sid"], st["slice_affine_err_mm"]))
                except Exception as e:
                    print(f"    [error] {r['sid']}: {e}")
                if (i + 1) % 500 == 0:
                    print(f"    ... {i + 1}/{len(rows)}")
            print(f"    补 params.json: {n_params} / {len(rows)}")
            print(f"    补 transform.npz: {n_tf} / {len(rows)}")
            print(f"    补 geometry.npz: {n_geom} / {len(rows)}")
            if args.fix_affine:
                print(f"    重写 NIfTI affine: {n_aff} / {len(rows)}")
            if approx:
                vals = [e for _, e in approx]
                print(f"    注意: {len(approx)} 个 convex 样本的 plane affine 是近似 "
                      f"(残差 {min(vals):.2f}–{max(vals):.2f} mm)；精确几何请读 geometry.npz 的 world_grid")

    print(f"\n完成，用时 {time.time() - t0:.1f}s"
          + ("（dry-run，未写盘）" if args.dry_run else ""))


if __name__ == "__main__":
    main()
