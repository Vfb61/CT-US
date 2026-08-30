#!/usr/bin/env python
"""Evaluation protocol for CT-US registration (A2/A3).

Reports on a held-out index of synthetic pairs:
  - coarse pose-regression TRE statistics (only if a checkpoint is given)
  - LNCC-refined TRE (mean/median/std, success rate %<5mm and %<10mm)
  - capture range: refine from GT perturbed by (translation, rotation) and
    report convergence rate to <5mm
  - confidence diagnostics: best LNCC similarity and top-K translation spread
  - per-frame inference latency (GPU)

Examples:
  python scripts/evaluate_registration.py \
      --index outputs/pairs_rigid/vessel_Task08_HepaticVessel/index_index.jsonl \
      --out outputs/eval/rigid.json --max_samples 12

  # coarse model available:
  python scripts/evaluate_registration.py --index ... --checkpoint outputs/model.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))

import baseline_registration as br


def perturb(affine, d_t=0.0, d_deg=0.0, rng=None):
    """Perturb a rigid transform by translation (mm) and rotation (deg)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    M = np.array(affine, dtype=np.float64)
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis) + 1e-12
    ang = np.deg2rad(d_deg)
    R = br.axis_angle_to_matrix(axis, ang)
    M[:3, :3] = R @ M[:3, :3]
    M[:3, 3] = M[:3, 3] + rng.standard_normal(3) * d_t
    return M


def tre_sample(pred):
    return pred["tre"]


def summarize(tres):
    a = np.asarray(tres, dtype=np.float64)
    if a.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan"),
                "p5mm": float("nan"), "p10mm": float("nan")}
    return {"mean": float(a.mean()), "median": float(np.median(a)), "std": float(a.std()),
            "p5mm": float(np.mean(a < 5.0) * 100.0), "p10mm": float(np.mean(a < 10.0) * 100.0)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/eval/registration.json")
    ap.add_argument("--max_samples", type=int, default=12)
    ap.add_argument("--checkpoint", default="", help="optional coarse pose net")
    ap.add_argument("--k_candidates", type=int, default=1,
                    help="top-K candidates to refine (1 = coarse only)")
    ap.add_argument("--no_coarse", action="store_true",
                    help="skip coarse net; refine from a GT-perturbed init")
    ap.add_argument("--oracle_seg", action="store_true",
                    help="use the stored seg_slice as a perfect US segmentation "
                         "(validates the geometric refiner independent of the head)")
    ap.add_argument("--refine_steps", type=int, default=40)
    ap.add_argument("--capture_t", nargs="+", type=float, default=[5.0, 15.0, 30.0])
    ap.add_argument("--capture_deg", nargs="+", type=float, default=[5.0, 10.0, 20.0])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not br.HAS_TORCH:
        print("torch required")
        return 1
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = br._read_index(args.index)
    rows = rows[: args.max_samples]
    data = [br.load_sample(r) for r in rows]
    print(f"samples={len(data)} device={device}")

    coarse_ok = False
    model = None
    seg_fn = None
    if args.checkpoint and not args.no_coarse:
        model = br.PoseRegNet().to(device)
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        model.eval()
        seg_fn = model.seg
        coarse_ok = True
    else:
        # no coarse model: use a randomly-initialised seg head so the
        # refinement objective is at least well-defined on the US side
        model = br.PoseRegNet().to(device)
        seg_fn = model.seg

    # candidates builder
    def coarse_candidates(sample):
        if coarse_ok:
            ct = br.preprocess_ct(sample["ct"])
            us = sample["us"]
            nx, nz = sample["nx"], sample["nz"]
            ct = torch.tensor(ct[None][None], dtype=torch.float32, device=device)
            us = torch.tensor(us[None][None], dtype=torch.float32, device=device)
            with torch.no_grad():
                pred, _ = model(ct, us)
            v = pred[0].cpu().numpy()
            return [br.vector_to_affine(v)]
        return []

    def seg_fn_builder(sample):
        """Return a segmentation function for the current sample."""
        if args.oracle_seg:
            seg_i = torch.tensor(sample["seg"], dtype=torch.float32, device=device)

            def oracle(us_t):
                lbl = torch.nn.functional.interpolate(
                    seg_i[None, None], size=(br.LOSS_H, br.LOSS_W),
                    mode="nearest").long()[0, 0]
                oh = torch.nn.functional.one_hot(lbl, num_classes=br.SEG_CLASSES).float()
                return torch.permute(oh, (2, 0, 1))[None] * 100.0 - 50.0

            return oracle
        return seg_fn

    results = []
    tot_refine_s = 0.0
    conv_matrix = {}
    rng = np.random.default_rng(args.seed)

    for s in data:
        rec = {"sid": getattr(s, "sid", ""), "kind": ""}
        # coarse prediction, if available
        coarse_tre = None
        if coarse_ok:
            cands = coarse_candidates(s)
            if cands:
                coarse_tre = br.tre(s["us_to_ct"], cands[0], s["nx"], s["nz"], s["dx"], s["dz"])
            rec["coarse_tre"] = coarse_tre

        # refinement from the coarse candidate (or GT-perturbed init)
        init = None
        if candidates := coarse_candidates(s):
            init = candidates[0]
        else:
            init = perturb(s["us_to_ct"], d_t=8.0, d_deg=6.0, rng=rng)

        t0 = time.perf_counter()
        best, allres = br.refine_top_k(s, [init], seg_fn_builder(s), k=args.k_candidates,
                                       steps=args.refine_steps, device=device)
        tot_refine_s += time.perf_counter() - t0
        rec["refined_tre"] = br.tre(s["us_to_ct"], best["affine"], s["nx"], s["nz"],
                                    s["dx"], s["dz"])
        rec["sim"] = best["sim"]
        rec["loss"] = best["loss"]
        # confidence spread over the refinements of the candidates
        if len(allres) > 1:
            ts = np.array([np.linalg.norm(r["affine"][:3, 3]) for r in allres])
            rec["cand_spread"] = float(ts.std())
        else:
            rec["cand_spread"] = 0.0
        results.append(rec)

    # capture range (independent of coarse net): refine from deterministic perturbations
    for dt in args.capture_t:
        for dd in args.capture_deg:
            conv = []
            for s in data[: min(8, len(data))]:
                init = perturb(s["us_to_ct"], d_t=dt, d_deg=dd, rng=rng)
                best, _ = br.refine_top_k(s, [init], seg_fn_builder(s), k=1,
                                          steps=args.refine_steps, device=device)
                tre_r = br.tre(s["us_to_ct"], best["affine"], s["nx"], s["nz"],
                               s["dx"], s["dz"])
                conv.append(float(tre_r < 5.0))
            conv_matrix[f"t{dt:g}deg{dd:g}"] = float(np.mean(conv)) * 100.0

    refined_tre = [r["refined_tre"] for r in results]
    out = {
        "n": len(results),
        "coarse": summarize([r["coarse_tre"] for r in results if r.get("coarse_tre") is not None])
        if coarse_ok else None,
        "refined": summarize(refined_tre),
        "capture_rate_pct": conv_matrix,
        "latency_ms_per_frame": (tot_refine_s / max(1, len(results))) * 1000.0,
        "confidence": {
            "sim_mean": float(np.mean([r["sim"] for r in results])),
            "cand_spread_mean": float(np.mean([r["cand_spread"] for r in results])),
        },
        "samples": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    def _fmt(k):
        return "n/a" if out[k] is None else (f"{out[k]['mean']:.2f}mm "
                                             f"({out[k]['p5mm']:.0f}%<5mm "
                                             f"{out[k]['p10mm']:.0f}%<10mm)")

    print(f"samples={out['n']}")
    if out["coarse"] is not None:
        print(f"coarse  TRE = {out['coarse']['mean']:.2f}mm  "
              f"({out['coarse']['p5mm']:.0f}%<5mm {out['coarse']['p10mm']:.0f}%<10mm)")
    print(f"refined TRE = {out['refined']['mean']:.2f}mm  "
          f"median={out['refined']['median']:.2f}mm  "
          f"({out['refined']['p5mm']:.0f}%<5mm {out['refined']['p10mm']:.0f}%<10mm)")
    print("capture rate (%<5mm):")
    for k, v in conv_matrix.items():
        print(f"  {k}: {v:.0f}%")
    print(f"latency = {out['latency_ms_per_frame']:.1f} ms/frame")
    print(f"confidence sim={out['confidence']['sim_mean']:.3f} "
          f"spread={out['confidence']['cand_spread_mean']:.2f}")
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())