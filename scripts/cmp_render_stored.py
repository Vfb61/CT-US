"""把"现场重渲染"与"磁盘上已存的数据集"逐体素对齐，判定到底谁错了。

动机
----
`diag_envelope_db.py` 现场渲染得到 肝实质灰度 0.444、血管腔 0.242；
而磁盘上 `us_vessel_ds_diag` 里的肝实质只有 0.084。相差 22 dB，二者必有一错。
文件时间戳显示 `_diag`(20:41:55) 晚于 `gen_us_vessel_dataset.py`(20:41:29)，
所以**不是陈旧数据**。必须定位真实差异，否则后面所有判断都站在沙子上。

用法
----
    python scripts/cmp_render_stored.py --case hepaticvessel_001 --poses 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ct2us import anatomy as an                      # noqa: E402
from ct2us import dataset as ds                      # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vessel", default=r"dataset\Task08_HepaticVessel\Task08_HepaticVessel")
    ap.add_argument("--stored", default=r"outputs\us_vessel_ds_diag\samples")
    ap.add_argument("--case", default="hepaticvessel_001")
    ap.add_argument("--poses", type=int, default=2)
    ap.add_argument("--probe", default="linear")
    ap.add_argument("--n-elev", type=int, default=32)
    ap.add_argument("--elev-spacing", type=float, default=1.0)
    ap.add_argument("--min-vessel-frac", type=float, default=0.01)
    args = ap.parse_args()

    root = ROOT_DIR / args.vessel
    stored_dir = ROOT_DIR / args.stored
    # 与 gen_us_vessel_dataset.py 完全一致的种子与调用方式
    ci = 0
    rng = np.random.default_rng(1000 + ci)
    ctx = ds.load_case(str(root), "imagesTr", case_name=args.case, noise_seed=2000 + ci)
    gp = {"no_deform": True, "render": {}}

    for k in range(args.poses):
        out = ds.generate_volume(
            ctx, rng, n_elev=args.n_elev, elev_spacing=args.elev_spacing,
            param={"min_vessel_frac": args.min_vessel_frac, "kinds": [args.probe]},
            global_params=gp)
        fresh = np.asarray(out["us_vol"], np.float32)
        seg = np.asarray(out["seg_vol"], np.int16)
        sid = f"{args.case}_{k:02d}"
        f = stored_dir / f"{sid}.npz"
        if not f.exists():
            print(f"{sid}: 磁盘无此文件，跳过对比"); continue
        z = np.load(f)
        old = z["us"].astype(np.float32) / 255.0
        oldseg = z["seg"]

        print(f"\n===== {sid} =====")
        print(f"  形状  现场={fresh.shape}  磁盘={old.shape}")
        if fresh.shape == old.shape:
            d = np.abs(fresh - old)
            print(f"  逐体素 |差|: 中位={np.median(d):.5f}  p99={np.percentile(d, 99):.5f} "
                  f" 最大={d.max():.5f}   完全相同={np.array_equal(fresh, old)}")
        print(f"  组织标签一致: {np.array_equal(seg, oldseg)}"
              + ("" if seg.shape == oldseg.shape else "（形状不同）"))

        def per_label(vol, lb, ref):
            m = ref == lb
            return vol[m].mean() if m.sum() > 20 else float("nan")

        print(f"  {'类别':<14}{'现场':>10}{'磁盘':>10}")
        for code in (an.T_AIR, an.T_BODY_WALL, an.T_BONE, an.T_LIVER,
                     an.T_TUMOR, an.T_VESSEL, an.T_VESSEL_WALL):
            nm = an.TISSUE_NAMES.get(code, str(code)) if hasattr(an, "TISSUE_NAMES") else str(code)
            print(f"  {nm:<14}{per_label(fresh, code, seg):>10.4f}"
                  f"{per_label(old, code, oldseg):>10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
