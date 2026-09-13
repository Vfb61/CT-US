#!/usr/bin/env python
"""Self-check that generation does not corrupt position/structure geometry.

For each sample in the dataset index we:
  1. rebuild the scan-plane world grid exactly as generation did (probe + pose,
     no deformation);
  2. re-reslice the stored (cropped) CT volume along that grid;
  3. compare against the CT slice persisted next to the US image;
  4. verify transform.npz's us_to_ct matches the geometry-derived transform
     and has det=+1.

On a rigid ("--no_deform") dataset the CT reslice must be pixel-identical to
the stored slice (RMS ~ 0, sub-mm).  On a deformed dataset the mismatch is
expected and reported, not failed.

Examples:
  python scripts/check_consistency.py \
      --index outputs/pairs_rigid/liver_Task03_Liver/index_index.jsonl \
             outputs/pairs_rigid/vessel_Task08_HepaticVessel/index_index.jsonl
  python scripts/check_consistency.py --scan outputs/pairs2 --strict
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ct2us import dataset as ds
from ct2us import geometry as geo
from ct2us import io_utils


def _rows(index_files):
    """读取索引并按 sid 去重（保留首次出现顺序，值取**最后一次**出现的行）。

    去重是必须的：历史数据里存在多次重跑追加造成的陈旧重复行
    （例如 pairs_v1/vessel_* 的 957 行里只有 872 个唯一 sid，79 条的
    probe.(nx,nz) 与磁盘 us.png 尺寸不符），不去重会拿旧几何去核对新样本。
    """
    order: list[str] = []
    by_key: dict[str, dict] = {}
    for f in index_files:
        f = Path(f)
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["_base"] = str(f.parent)
                row["_index_file"] = str(f)
                sid = str(row.get("sid", ""))
                if sid not in by_key:
                    order.append(sid)
                by_key[sid] = row
    return [by_key[s] for s in order]


def _image_size(path: Path):
    """(nx, nz) of a PNG sample image, or None.

    注意 us.png 是 numpy (nz, nx) 存成的图像，因此 PIL 的 (width, height) 对应
    (nz, nx)；这里转回 (nx, nz) 以便与 probe 的取值直接比较。
    """
    try:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
            return (h, w)
    except Exception:
        return None


_PROBE_KEYS = ("kind", "nx", "nz", "dx", "dz", "fov_angle", "near", "radius", "freq")


def _build_probe(pr_json: dict):
    """用 probe_from_meta 重建（正确处理 convex 的 dx 占位值问题）。"""
    return geo.probe_from_meta(pr_json)


def _is_deformed(row) -> bool:
    d = (row.get("params") or {}).get("deform", {})
    return bool(d.get("resp_amp_mm", 0.0) or d.get("compression_mm", 0.0))


def _torch_check(row) -> dict:
    """Cross-validate torch fan-accurate reslice vs stored slices."""
    try:
        from train.baseline_registration import load_sample, reslice_pose, reslice_label
        import torch
    except Exception as e:  # noqa: BLE001
        return {"sid": row["sid"], "torch_error": str(e)}

    base = Path(row["_base"])
    sample = load_sample(row)
    M_gt = sample["us_to_ct"]

    # CT reslice at GT pose (should match stored ct_slice pixel-exactly)
    with torch.no_grad():
        re_ct = reslice_pose(sample, M_gt).squeeze().cpu().numpy()
    stored_ct, _, _ = io_utils.load_volume(base / row["sid"] / "ct_slice.nii.gz")
    ct_diff = re_ct - stored_ct
    ct_rms = float(np.sqrt(np.mean(np.square(ct_diff))))
    ct_maxabs = float(np.max(np.abs(ct_diff)))

    # Seg reslice at GT pose (bilinear soft labels vs stored nearest-neighbor)
    with torch.no_grad():
        re_seg = reslice_label(sample, M_gt).squeeze().cpu().numpy()
    stored_seg, _, _ = io_utils.load_volume(base / row["sid"] / "seg_slice.nii.gz")
    # reslice_label uses bilinear (soft) labels for differentiability;
    # stored seg_slice was resampled with order=0 (nearest).
    # Geometric position is identical; mismatch at tissue boundaries is expected.
    seg_exact = bool(np.array_equal(re_seg.astype(int), stored_seg.astype(int)))
    seg_rounded = bool(np.all(np.round(re_seg).astype(int) == stored_seg.astype(int)))

    return {
        "sid": row["sid"], "kind": row["params"]["probe"]["kind"],
        "deformed": _is_deformed(row),
        "torch_ct_rms": ct_rms, "torch_ct_maxabs": ct_maxabs,
        "torch_seg_exact": seg_exact, "torch_seg_rounded": seg_rounded,
    }


def _check(row) -> dict:
    base = Path(row["_base"])
    pr = row["params"]["probe"]
    po = row["params"]["pose"]
    probe = _build_probe(pr)
    face = np.asarray(po["face"], dtype=np.float64)
    u = np.asarray(po["u"], dtype=np.float64)
    v = np.asarray(po["v"], dtype=np.float64)
    w = np.asarray(po["w"], dtype=np.float64)

    ct, affine, _ = io_utils.load_volume(base / row["volume"])
    stored_ct, stored_aff, _ = io_utils.load_volume(base / row["sid"] / "ct_slice.nii.gz")
    seg_vol, _, _ = io_utils.load_volume(base / row["volume"].replace("_ct.nii.gz", "_seg.nii.gz"))

    world = probe.world_grid(face, u, v, w, deform=None)
    re_ct = probe.resample(ct, affine, world, cval=-1024.0, order=1)

    diff = re_ct - stored_ct
    rms = float(np.sqrt(np.mean(np.square(diff))))
    rel = rms / (float(np.ptp(stored_ct)) + 1e-6)

    tz = np.load(base / row["sid"] / "transform.npz")
    us_to_ct = tz["us_to_ct"]
    R = us_to_ct[:3, :3]
    det = float(np.linalg.det(R))
    # R 必须正交（旧检查只看 det，无法发现缩放/剪切）
    orth = float(np.max(np.abs(R.T @ R - np.eye(3))))
    derived = ds.make_transform(probe,
                                {"face": face, "u": u, "v": v, "w": w})["us_to_ct"]
    tmatch = float(np.max(np.abs(us_to_ct - derived)))

    # 切片 affine 必须与几何推导的平面 affine 一致（新格式）
    if "slice_affine" in tz.files:
        stored_slice_aff = tz["slice_affine"]
        derived_slice_aff, slice_fit_err = ds.plane_affine_true(
            probe, {"face": face, "u": u, "v": v, "w": w})
        amatch = float(np.max(np.abs(stored_slice_aff - derived_slice_aff)))
    else:
        # 旧格式：affine 被错误地写成 us_to_ct，只报告不通过
        stored_slice_aff = stored_aff
        derived_slice_aff = None
        slice_fit_err = float("nan")
        amatch = float(np.max(np.abs(np.asarray(stored_aff) - us_to_ct))) if stored_aff is not None else 0.0

    # 索引行与磁盘样本的一致性：probe.(nx,nz) 必须等于 us.png 尺寸
    size = _image_size(base / row["sid"] / row["files"]["us"])
    nx, nz = int(pr["nx"]), int(pr["nz"])
    size_ok = (size is None) or (tuple(size) == (nx, nz))

    seg, _, _ = io_utils.load_volume(base / row["sid"] / "seg_slice.nii.gz")
    same_lbls = bool(np.array_equal(seg,
                     probe.resample(seg_vol, affine, world, cval=0.0, order=0)))
    return {
        "sid": row["sid"], "kind": pr["kind"], "rms": rms, "rel": rel,
        "det": det, "orth": orth, "tmatch": tmatch, "amatch": amatch,
        "deformed": _is_deformed(row), "labels_match": same_lbls,
        "size_ok": size_ok,
        "png_size": None if size is None else tuple(int(v) for v in size),
        "probe_nx_nz": (nx, nz),
        "slice_affine_err": slice_fit_err,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", nargs="+", help="index jsonl files")
    ap.add_argument("--scan", help="scan a directory for *_index.jsonl")
    ap.add_argument("--show", action="store_true", help="print every sample line")
    ap.add_argument("--strict", action="store_true",
                    help="treat any rigid-sample RMS above --threshold as failure")
    ap.add_argument("--torch", action="store_true",
                    help="also cross-validate torch fan-accurate reslice vs stored slices")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="RMS (HU) threshold for rigid-sample failure")
    ap.add_argument("--limit", type=int, default=None,
                    help="only check the first N samples (quick smoke check)")
    args = ap.parse_args()

    index_files = []
    if args.scan:
        index_files = sorted(str(p) for p in Path(args.scan).rglob("*_index.jsonl"))
    if args.index:
        index_files.extend(args.index)
    if not index_files:
        ap.error("provide --index or --scan")

    rows = _rows(index_files)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("no samples found")
        return 1

    strict_rows = [r for r in rows if not _is_deformed(r)]
    deformed = len(rows) - len(strict_rows)

    results = []
    for r in rows:
        try:
            res = _check(r)
            if args.torch:
                res.update(_torch_check(r))
            results.append(res)
        except Exception as e:  # noqa: BLE001
            br = {"sid": r["sid"], "error": str(e)}
            results.append(br)
            print(f"ERROR {r['sid']}: {e}")

    ok_rows = [r for r in results if "error" not in r]
    rigid = [r for r in ok_rows if not r["deformed"]]
    failed = [r for r in rigid if r["rms"] > args.threshold]

    if args.show:
        for r in results:
            if "error" in r:
                continue
            tag = "DEFORMED" if r["deformed"] else "  rigid "
            line = (f"{tag} {r['sid']:<24} kind={r['kind']:<6} rms={r['rms']:9.4f} "
                    f"rel={r['rel']:.3e} det={r['det']:+.4f} "
                    f"orth={r['orth']:.1e} Tmatch={r['tmatch']:.2e} "
                    f"aff={r['amatch']:.2e} size_ok={r['size_ok']} "
                    f"labels={r['labels_match']}")
            if "torch_ct_rms" in r:
                line += (f" | torch_ct_rms={r['torch_ct_rms']:.4f} "
                         f"maxabs={r['torch_ct_maxabs']:.4f} "
                         f"seg_exact={r['torch_seg_exact']} rounded={r['torch_seg_rounded']}")
            print(line)

    print(f"samples={len(rows)}  rigid={len(rigid)}  deformed={deformed}  errors={len(results)-len(ok_rows)}")
    if rigid:
        print(f"rigid rms: mean={np.mean([r['rms'] for r in rigid]):.4f} "
              f"max={max(r['rms'] for r in rigid):.4f}")
    allmatch = [r["tmatch"] for r in ok_rows]
    if allmatch:
        print(f"transform match: max diff vs geometry-derived = {max(allmatch):.2e}")
    orths = [r["orth"] for r in ok_rows]
    if orths:
        print(f"rotation orthogonality |R^T R - I|: max={max(orths):.3e} "
              f"(旧检查只看 det，无法发现缩放/剪切)")
    dets = [r["det"] for r in ok_rows]
    if dets:
        print("determinants: min=%.4f max=%.4f" % (min(dets), max(dets)))
    print("labels resample identical to stored seg_slice:",
          sum(1 for r in ok_rows if r["labels_match"]), "/", len(ok_rows))
    bad_size = [r for r in ok_rows if not r["size_ok"]]
    print(f"index-vs-disk size consistency: {len(ok_rows) - len(bad_size)}/{len(ok_rows)} ok"
          + (f"  ← {len(bad_size)} 条索引行的 (nx,nz) 与 us.png 不符（陈旧行或旧格式）" if bad_size else ""))
    for r in bad_size[:5]:
        print(f"    {r['sid']}: index says {r['probe_nx_nz']}, us.png is {r['png_size']}")
    sae = [r["slice_affine_err"] for r in ok_rows if np.isfinite(r.get("slice_affine_err", np.nan))]
    if sae:
        print(f"slice affine 平面拟合残差(convex): max={max(sae):.4f} mm")

    # torch cross-validation summary
    torch_rows = [r for r in ok_rows if "torch_ct_rms" in r]
    if torch_rows:
        ct_rms = [r["torch_ct_rms"] for r in torch_rows]
        ct_ma = [r["torch_ct_maxabs"] for r in torch_rows]
        seg_exact = sum(1 for r in torch_rows if r["torch_seg_exact"])
        seg_rounded = sum(1 for r in torch_rows if r["torch_seg_rounded"])
        print(f"torch reslice: ct_rms mean={np.mean(ct_rms):.4f} max={max(ct_rms):.4f} | "
              f"ct_maxabs max={max(ct_ma):.4f}")
        print(f"  seg: exact={seg_exact}/{len(torch_rows)} rounded={seg_rounded}/{len(torch_rows)} "
              f"(bilinear vs nearest: boundary mismatch is expected)")

    if args.strict and (failed or len(rigid) == 0):
        print(f"FAIL: {len(failed)} rigid sample(s) above threshold "
              f"({args.threshold}).")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())