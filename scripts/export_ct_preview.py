"""把原始 CT 体数据导成可直接看的 PNG / GIF，并统计血管与肝实质的 HU 关系。

用途
----
在比较"合成超声 vs 真实解剖"之前，先得看清 CT 本身：
  · 血管在 CT 里到底是**亮的（增强期）还是暗的（平扫）**—— 这决定了合成超声
    里"暗腔"是不是正确的目标外观；
  · 原始层厚有多大 —— 层厚 5 mm 的体数据重采样到 0.4 mm 的超声网格会产生
    体素化台阶（在合成图上表现为硬边方块）；
  · 血管标注的分布，用来挑有血管的层面做对照。

方向约定：`vol[x, y, z]` 的 z 为头脚方向，导出的是**轴位层** `vol[:, :, k]`，
显示时近端（前腹壁）朝上。

用法
----
    python scripts/export_ct_preview.py --cases hepaticvessel_001 hepaticvessel_002
    python scripts/export_ct_preview.py --n-cases 4 --n-slices 8
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

DEFAULT_ROOT = r"dataset\Task08_HepaticVessel\Task08_HepaticVessel"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--cases", nargs="*", default=[],
                    help="病例名（不含扩展名）；默认自动取前 n-cases 个")
    ap.add_argument("--n-cases", type=int, default=4)
    ap.add_argument("--n-slices", type=int, default=8)
    ap.add_argument("--wl", type=float, default=60.0, help="窗位 HU")
    ap.add_argument("--ww", type=float, default=400.0, help="窗宽 HU")
    ap.add_argument("--out", default="", help="默认 preview/ct/<数据集名>")
    ap.add_argument("--dpi", type=int, default=110)
    args = ap.parse_args()

    import nibabel as nib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from PIL import Image

    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    root = os.path.join(ROOT_DIR, args.root) if not os.path.isabs(args.root) else args.root
    img_dir, lab_dir = os.path.join(root, "imagesTr"), os.path.join(root, "labelsTr")
    if not os.path.isdir(img_dir):
        print(f"[ct] 找不到 {img_dir}"); return 1

    if args.cases:
        cases = list(args.cases)
    else:
        names = sorted(f for f in os.listdir(img_dir)
                       if f.endswith(".nii.gz") and not f.startswith("._"))
        cases = [f[: -len(".nii.gz")] for f in names[: args.n_cases]]

    outdir = args.out or os.path.join(ROOT_DIR, "preview", "ct")
    os.makedirs(outdir, exist_ok=True)
    lo, hi = args.wl - args.ww / 2.0, args.wl + args.ww / 2.0
    print(f"[ct] 窗位/窗宽 = {args.wl:.0f}/{args.ww:.0f} HU  → 显示区间 [{lo:.0f}, {hi:.0f}]")
    print(f"[ct] 输出目录 {outdir}\n")

    made, summary = [], []
    for case in cases:
        ip = os.path.join(img_dir, case + ".nii.gz")
        lp = os.path.join(lab_dir, case + ".nii.gz")
        if not os.path.exists(ip):
            print(f"  [skip] 无 {ip}"); continue
        img = nib.load(ip)
        vol = np.asarray(img.dataobj, dtype=np.float32)
        # NIfTI 的 affine 前三列是体素轴方向；取各轴体素尺寸（mm）
        sp = tuple(float(x) for x in np.sqrt((img.affine[:3, :3] ** 2).sum(axis=0)))

        lab = None
        if os.path.exists(lp):
            lab = np.asarray(nib.load(lp).dataobj)
        liver = (lab == 1) if lab is not None else None
        ves = (lab == 2) if lab is not None else None

        print(f"{case}: 形状 {vol.shape}  体素尺寸 {tuple(round(s, 3) for s in sp)} mm")
        if lab is not None:
            uniq = np.unique(lab)
            print(f"    标签取值 {uniq.tolist()}  (Task08: 1=肝 2=肝血管)")
            if ves is not None and ves.any() and liver is not None and liver.any():
                hv, hl = vol[ves], vol[liver]
                # 血管壁邻域 = 血管外扩一圈减去血管本身，作为"局部背景"
                from scipy import ndimage
                ring = ndimage.binary_dilation(ves, iterations=2) & ~ves
                hr = vol[ring]
                print(f"    HU 中位：血管腔 {np.median(hv):.0f}  邻域 {np.median(hr):.0f}  "
                      f"肝实质 {np.median(hl):.0f}  ⇒ 血管比邻域"
                      f"{'暗' if np.median(hv) < np.median(hr) else '亮'} "
                      f"{abs(np.median(hv) - np.median(hr)):.0f} HU")

        # 选层：优先含有血管的层，按包含量挑选并沿 z 均匀铺开
        nz = vol.shape[2]
        if ves is not None and ves.any():
            cnt = ves.reshape(-1, nz).sum(axis=0)
            cand = np.where(cnt >= max(20, 0.05 * cnt.max()))[0]
        else:
            cand = np.arange(nz)
        if len(cand) < args.n_slices:
            cand = np.arange(nz)
        ks = np.unique(cand[np.linspace(0, len(cand) - 1, args.n_slices).astype(int)])

        def win(a):
            return np.clip((a - lo) / (hi - lo), 0, 1)

        nc = len(ks)
        fig, axes = plt.subplots(2, nc, figsize=(2.4 * nc, 5.4), squeeze=False)
        for j, k in enumerate(ks):
            sl = vol[:, :, k]
            axes[0][j].imshow(win(sl).T, cmap="gray", vmin=0, vmax=1,
                              origin="lower", aspect="equal")
            axes[0][j].set_title(f"z={k}  血管 {int(ves[:, :, k].sum()) if ves is not None else 0}",
                                 fontsize=8)
            axes[1][j].imshow(win(sl).T, cmap="gray", vmin=0, vmax=1,
                              origin="lower", aspect="equal")
            if liver is not None and liver[:, :, k].any():
                axes[1][j].contour(liver[:, :, k].T.astype(float), levels=[0.5],
                                   colors="lime", linewidths=0.8)
            if ves is not None and ves[:, :, k].any():
                axes[1][j].contour(ves[:, :, k].T.astype(float), levels=[0.5],
                                   colors="red", linewidths=0.9)
            for r in (0, 1):
                axes[r][j].set_xticks([]); axes[r][j].set_yticks([])
        fig.suptitle(f"{case}  轴位 CT   窗 {args.wl:.0f}/{args.ww:.0f} HU   "
                     f"体素 {tuple(round(s, 2) for s in sp)} mm   "
                     f"下排：绿=肝 红=肝血管", fontsize=11)
        plt.tight_layout(rect=(0, 0, 1, 0.94))
        p = os.path.join(outdir, f"{case}_ct.png")
        plt.savefig(p, dpi=args.dpi, facecolor="white"); plt.close(fig)
        made.append(p)

        # 逐层动画（轴位扫过全卷）
        frames = []
        for k in range(nz):
            sl = win(vol[:, :, k]).T
            rgb = np.stack([sl] * 3, -1)
            if liver is not None and liver[:, :, k].any():
                m = liver[:, :, k].T
                rgb[m] = 0.6 * rgb[m] + 0.4 * np.array([0.1, 1.0, 0.2])
            if ves is not None and ves[:, :, k].any():
                m = ves[:, :, k].T
                rgb[m] = 0.3 * rgb[m] + 0.7 * np.array([1.0, 0.15, 0.15])
            fi = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
            frames.append(fi.resize((fi.width * 2, fi.height * 2), Image.NEAREST))
        pg = os.path.join(outdir, f"{case}_ct_cine.gif")
        frames[0].save(pg, save_all=True, append_images=frames[1:],
                       duration=120, loop=0)
        made.append(pg)
        summary.append((case, vol.shape, sp))

    print(f"\n[ct] 共写出 {len(made)} 个文件")
    for p in made:
        print("   ", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
