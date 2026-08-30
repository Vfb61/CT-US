#!/usr/bin/env python
"""Calibration experiment: compare the fast renderer against the physics-proxy
reference and (optionally) adjust render parameters to minimise the 6-axis
metric distance.

Example:
  python scripts/calibrate_quick.py --data dataset/Task08_HepaticVessel --out outputs/calib --iters 25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ct2us import calibrate, dataset, io_utils, physic_sim, render


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="task dir")
    ap.add_argument("--out", default="outputs/calib")
    ap.add_argument("--case", default="", help="specific case name")
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    name = args.case or io_utils.list_cases(Path(args.data), "imagesTr")[0]
    ctx = dataset.load_case(Path(args.data), "imagesTr", case_name=name)

    # fixed geometry for the comparison
    probe, pose = dataset.sample_probe_geometry(ctx, rng)
    planes = dataset.resample_planes(ctx, probe, pose)

    reference = physic_sim.PhysicsProxy(planes["ct"], planes["tissue"],
                                        planes["scatter"], probe).bmode()

    def render_fn(p):
        return render.render_bmode(planes["ct"], planes["tissue"],
                                  planes["scatter"], probe, p, rng,
                                  want_envelope=True)

    base = render_fn({})
    ref_metrics = calibrate.metrics(reference["envelope"], planes["tissue"],
                                    probe.depth_offsets(),
                                    base["specular_plane"])
    base_metrics = calibrate.metrics(base["envelope"], planes["tissue"],
                                     probe.depth_offsets(),
                                     base["specular_plane"])

    dist0 = calibrate.metric_distance(base_metrics, ref_metrics)
    result = calibrate.calibrate_renderer(render_fn, probe, planes, ref_metrics,
                                          iterations=args.iters, rng=rng)
    best = result["best"]

    print(f"case {name}  probe={probe.kind} freq={probe.freq:.1f}MHz depth={probe.depth_span:.0f}mm")
    print(f"base distance   = {dist0:.2f}")
    print(f"calibrated dist = {result['distance']:.2f}")
    print("metric comparison (render | physics | calibrated):")
    keys = calibrate.SCALAR_METRICS
    hv = {k: round(float(v), 3) for k, v in base_metrics.items()}
    rv = {k: round(float(v), 3) for k, v in ref_metrics.items()}
    cv = {k: round(float(v), 3) for k, v in best["metrics"].items()}
    for k in keys:
        a, b, c = hv.get(k, None), rv.get(k, None), cv.get(k, None)
        if a is None or b is None:
            continue
        print(f"  {k:<22} {a:>10} | {b:>10} | {c:>10}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best_params = {
        "seed": args.seed, "case": name,
        "base_distance": float(dist0), "calibrated_distance": float(result["distance"]),
        "best_params": {k: round(float(v), 4) for k, v in best["params"].items()},
        "reference_metrics": rv,
        "base_metrics": hv,
        "calibrated_metrics": cv,
    }
    io_utils.save_json(out / "calib_result.json", best_params)
    print(f"saved -> {out / 'calib_result.json'}")


if __name__ == "__main__":
    main()