"""生成「3D 超声体 + 血管 GT」数据集 —— 训练超声侧血管分割网络的输入。

在这条链里的位置
----------------
    3D 超声体 → [3D U-Net 血管分割] → 血管掩膜 → 骨架化 → 中心线点云
    CT                                        → 同上 → 中心线点云
                                              ↓
                                        点云配准（粗 → 非刚性）

本项目要训练的是**超声侧**那一步。CT 侧已验证可行
（`check_vessel_centerline.py`：主干连通，5/6 病例最长分量占骨架 >50%）。

为什么必须用合成数据
--------------------
真实术中超声拿不到**稠密血管 GT**（只有稀疏分叉点）。合成数据能同时给出
`us_vol` 与逐体素血管标签，所以是唯一能训练/评估分割网络的手段。

已修的两个数据侧机制（否则一轮里一个血管都没有）
------------------------------------------------
* **位姿血管专项筛选** `min_vessel_frac`：合体结构掩膜里肝包膜占多数，
  按它筛出的平面贴着肝表面、**一个血管都不穿**（实测 32 层卷里血管体素 = 0）。
* **探头视野**：凸阵横向只有 ±(radius·sin(fov/2)) ≈ 54–68mm，盖不住血管树
  （GT 骨架只剩 64 点）；线阵横向 ~139mm，GT 血管体素多 4.5 倍。

每个样本落盘（`<sid>.npz`）
---------------------------
    us      (n_elev, nx, nz) uint8     合成 B-mode，[0,255]
    vessel  (n_elev, nx, nz) uint8     血管腔+壁的 GT 标签（0/1）
    pts_mm  (N, 3) float32             血管中心线点云（US 物理 mm，含半径在第 4 列）
    probe / pose                        几何（用于把点云映回世界坐标）

用法
----
    python scripts/gen_us_vessel_dataset.py --cases 6 --poses 4 --out outputs/us_vessel_ds
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                     # noqa: BLE001
    pass

from ct2us import anatomy as an                        # noqa: E402
from ct2us import dataset as ds                        # noqa: E402
from ct2us import io_utils as io                       # noqa: E402


def skeletonize(mask: np.ndarray) -> np.ndarray:
    from skimage.morphology import skeletonize as sk
    return sk(mask.astype(bool))


def prune_skeleton(skel: np.ndarray, min_branch_vox: int = 12, rounds: int = 3) -> np.ndarray:
    """剪掉骨架上的短毛刺（迭代式：反复去掉只有一个邻居的端点）。

    骨架化必然产生大量假分支，不剪掉的话点云里全是噪声分支，
    而且 CT / US 两侧的毛刺分布不同 ⇒ 两侧点云不可比。
    """
    from scipy import ndimage
    k = np.ones((3, 3, 3), np.uint8)
    s = skel.copy()
    for _ in range(rounds):
        # 端点 = 自身为 1 且 3x3x3 邻域和为 2（只连着一个邻居）
        nb = ndimage.convolve(s.astype(np.uint8), k, mode="constant") - s
        ends = s & (nb <= 1)
        # 只剥 min_branch_vox 轮，避免把主干也吃掉
        s = s & ~ends
        # 恢复：把被剥掉的端点中"离主干很近"的补回来是没必要的——
        # 迭代剥 min_branch_vox 次即等价于剪掉长度 < min_branch_vox 的分支
    # 剥完后把剩下的按 min_branch_vox-1 次膨胀找回主干厚度（骨架本就是 1 体素宽）
    return s


def centerline_cloud(vessel: np.ndarray, spacing: tuple[float, float, float],
                     min_comp_vox: int = 30, resample_mm: float = 1.5,
                     prune_vox: int = 8):
    """血管掩膜 -> 中心线点云（US 物理 mm）+ 每点半径。

    步骤：骨架化 -> 剪毛刺 -> 只留够大的连通分量 -> 等弧长（最小间距）重采样。
    """
    from scipy import ndimage
    if not vessel.any():
        return np.zeros((0, 4), np.float32)
    skel = skeletonize(vessel)
    # 剪毛刺：迭代剥端点（每次剥掉当前所有端点）会同时缩短主干，
    # 所以限制轮数，并保留最后做一次"连通分量筛选"来去掉孤立短枝
    for _ in range(prune_vox):
        nb = ndimage.convolve(skel.astype(np.uint8), np.ones((3, 3, 3), np.uint8),
                              mode="constant") - skel
        skel = skel & ~(nb <= 1)

    lab, n = ndimage.label(skel, structure=np.ones((3, 3, 3), int))
    if n == 0:
        return np.zeros((0, 4), np.float32)
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    keep = np.where(sizes >= min_comp_vox)[0] + 1
    if keep.size == 0:
        keep = np.array([int(np.argmax(sizes)) + 1])
    sel = np.isin(lab, keep)

    # 半径：在**原始掩膜**上做距离变换（不是骨架上），取骨架点处的值
    dt = ndimage.distance_transform_edt(vessel, sampling=spacing)
    pts_idx = np.argwhere(sel)
    radii = dt[tuple(pts_idx.T)]
    pts_mm = pts_idx.astype(np.float64) * np.asarray(spacing, np.float64)

    # 等弧长重采样：贪心取"与已选点最小间距 >= resample_mm"的点
    rng = np.random.default_rng(0)
    order = rng.permutation(len(pts_mm))
    chosen = []
    for i in order:
        p = pts_mm[i]
        if all(np.linalg.norm(p - pts_mm[j]) >= resample_mm for j in chosen):
            chosen.append(i)
    chosen = np.asarray(chosen, dtype=int)
    out = np.concatenate([pts_mm[chosen], radii[chosen, None]], axis=1)
    return out.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vessel", default=r"dataset\Task08_HepaticVessel\Task08_HepaticVessel")
    ap.add_argument("--liver", default=r"dataset\Task03_Liver\Task03_Liver")
    ap.add_argument("--task", choices=["vessel", "liver"], default="vessel",
                    help="vessel=Task08（带血管标签）；liver=Task03（只有肝脏标签）")
    ap.add_argument("--cases", type=int, default=6)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--poses", type=int, default=4, help="每病例生成多少个体积")
    ap.add_argument("--n-elev", type=int, default=32)
    ap.add_argument("--elev-spacing", type=float, default=1.0)
    ap.add_argument("--probe", default="linear", help="探头类型（点云路线建议 linear）")
    ap.add_argument("--min-vessel-frac", type=float, default=0.01)
    # **不要覆盖 `log_ref`**。它原本被固定成 60，而包络自身的逐层 p99 中位数只有
    # 3.0（001）/ 4.2（002），固定值等于整体压低 20*log10(60/3.5) ≈ 25 dB，
    # 把肝实质和血管腔一起推到灰度下限（肝 0.107 vs 腔 0.108，对比度归零）——
    # 这是"血管在合成超声里看不见"的直接原因。
    # 固定参考的唯一动机是让**不同位姿间绝对亮度可比**，从而支持基于强度的目标函数；
    # 但点云路线用的是"分割→中心线→点云配准"，不做强度配准，
    # 因而不需要该性质，保留渲染器默认（逐图自适应 p99）即可。
    ap.add_argument("--render", default="speckle_strength=0.35,speckle_lat_px=3")
    ap.add_argument("--out", default="outputs/us_vessel_ds")
    args = ap.parse_args()

    root = Path(args.vessel if args.task == "vessel" else args.liver)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parents[1] / out_dir
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    rp = {}
    for kv in args.render.split(","):
        k, _, v = kv.partition("=")
        try:
            rp[k.strip()] = float(v)
        except ValueError:
            rp[k.strip()] = v
    gp = {"no_deform": True, "render": rp}

    names = io.list_cases(root, "imagesTr")[args.first:args.first + args.cases]
    print(f"[gen] {len(names)} 病例 × {args.poses} 体位 = {len(names) * args.poses} 个体积")
    print(f"[gen] 渲染参数 {rp}  探头 {args.probe}  最小血管占比 {args.min_vessel_frac}")

    rows, t0 = [], time.time()
    for ci, name in enumerate(names):
        rng = np.random.default_rng(1000 + ci)
        ctx = ds.load_case(str(root), "imagesTr", case_name=name,
                           noise_seed=2000 + ci)
        n_lum = int((ctx.tissue == an.T_VESSEL).sum())
        n_vm = int(ctx.vessel.sum())
        print(f"\n{name}: vessel_lumen={n_lum} vessel_mask={n_vm}")
        if n_vm < 200:
            print("  [skip] 该病例没有血管标签（Task03 只有肝脏）"); continue
        for k in range(args.poses):
            out = ds.generate_volume(ctx, rng, n_elev=args.n_elev,
                                     elev_spacing=args.elev_spacing,
                                     param={"min_vessel_frac": args.min_vessel_frac,
                                            "kinds": [args.probe]},
                                     global_params=gp)
            us = np.asarray(out["us_vol"], np.float32)
            seg = np.asarray(out["seg_vol"], np.int16)
            probe = out["probe"]
            n_elev, n_ax1, n_ax2 = us.shape
            ax1_is_x = (n_ax1 == probe.nx)
            sp = (args.elev_spacing,
                  probe.dx_effective if ax1_is_x else probe.dz,
                  probe.dz if ax1_is_x else probe.dx_effective)
            ves = ((seg == an.T_VESSEL) | (seg == an.T_VESSEL_WALL))
            sid = f"{name}_{k:02d}"
            # **职责边界**：build 是"数据合成"侧，只落盘 us + 血管掩膜 + 几何。
            # 「掩膜 -> 中心线点云」是"分割->配准"链的一环，属于 E:\new
            # （`models/vessel_cloud.py`，训练与推理共用同一份实现）。
            # 早期版本在这里顺带算了点云，会造成**两份实现**、训练/推理不一致的隐患。
            np.savez_compressed(
                out_dir / "samples" / f"{sid}.npz",
                us=(np.clip(us, 0, 1) * 255).astype(np.uint8),
                vessel=ves.astype(np.uint8),
                # **同时落盘完整组织标签**：只存合并的 vessel 掩膜不够用 ——
                # 血管腔（暗）与血管壁（亮）的对比度完全不同，混在一起会互相抵消，
                # 导致"血管可辨识度"诊断给出误导性的结论。体积一样大，直接存 seg。
                seg=seg.astype(np.uint8),
                # **落盘 CT 平面**（HU，int16）。渲染器诊断需要它做对照：
                # 只有把"同一几何下的 CT / 组织标签 / 合成 US"三者并排看，
                # 才能判断某个亮暗结构是解剖造成的还是渲染算子造成的。
                # 不存的话每次诊断都得重新渲染一遍体积，且无法与已存样本逐体素对齐。
                ct=np.clip(out["ct_vol"], -1024, 3071).astype(np.int16),
                spacing=np.asarray(sp, np.float32),
            )
            rows.append(dict(sid=sid, case=name, shape=list(us.shape),
                             spacing=[float(x) for x in sp],
                             vessel_vox=int(ves.sum()),
                             probe=ds._probe_json(probe),
                             pose={kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                                   for kk, vv in out["pose"].items() if kk != "kind"},
                             vol_affine=np.asarray(out["vol_affine"]).tolist(),
                             elev_spacing=float(out["elev_spacing"])))
            print(f"  {sid}: 形状 {us.shape} 间距 {np.round(sp, 3)}  "
                  f"血管体素 {int(ves.sum())}  耗时 {time.time() - t0:.0f}s")

    (out_dir / "index.json").write_text(json.dumps(
        {"n": len(rows), "args": vars(args), "samples": rows},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{'=' * 92}\n[gen] 完成 -> {out_dir}")
    if rows:
        print(f"  样本 {len(rows)}")
        print(f"  血管体素中位 {np.median([r['vessel_vox'] for r in rows]):.0f}  "
              f"最少 {min(r['vessel_vox'] for r in rows)}")
        # 几何齐全性自检：**点云构建与 GT 对齐都依赖 vol_affine / pose**，
        # 缺了它们下游只能重建数据（早期版本就没写，导致 80 个样本白生成）
        have_geo = sum(1 for r in rows if r.get("vol_affine") is not None)
        print(f"  带几何（vol_affine/pose）的样本: {have_geo}/{len(rows)}"
              f"{'' if have_geo == len(rows) else '  !!! 下游需要它，请用当前脚本重生成'}")
    print(f"  总耗时 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
