# ======================================================================
# calib.py
# Role     : Measures the synthetic stars as the real ones are measured (ellipticity, width, density).
# Step     : 1. Training on synthetic data only (E2 / E3, 7-9 Nov) (see scripts/README.md)
# Inputs   : generator: seeds + catalogue.json (public instrument sheets); no image
# Outputs  : synthetic pairs (input/target/info.json + manifest.json), network E3
# Example  : python calib.py [--draws 60] [--workers 6]
# ======================================================================
"""The stars the generator draws, measured as our real images are (star_measure.measure): their ellipticity, FWHM and
densities against the real ranges (diagnose.py units), to calibrate instruments.py and stars.py on measures.

Each draw: an instrument and a field as make_dataset draws them (the background's kind with its weights), the stars
alone on a 1024 px canvas plus the noise our denoised stacks keep (0.02 sigma), measured.

    python calib.py [--draws 60] [--workers 6]
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import get_context

import numpy as np

REAL = "<ESSAIS>/starRemoval/diagnose/units.json"


def one(seed: int) -> dict | None:
    import cv2
    cv2.setNumThreads(1)
    from backgrounds import KINDS
    from star_measure import measure
    from stars import render
    rng = np.random.default_rng([31337, seed])
    kinds = list(KINDS)
    kind = kinds[int(rng.choice(len(kinds), p=np.array([KINDS[k] for k in kinds]) / sum(KINDS.values())))]
    info: dict = {}
    stars, _keep, inst = render(rng, 1024, info, kind=kind)
    img = stars + 0.02 * rng.standard_normal(stars.shape).astype(np.float32)
    m = measure("", img, None)
    return {"kind": kind, "night": inst.extra.get("night"), "density": m["density"], "fwhm": m.get("fwhm"),
            "ellipticity": m.get("ellipticity"), "isolated": m["isolated"], "track": round(inst.track_px, 2),
            "seeing_fwhm": round(inst.seeing_fwhm, 2)}


def pct(v):
    return np.percentile(v, [10, 50, 90]).round(3).tolist() if len(v) else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--draws", type=int, default=60)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    with get_context("spawn").Pool(a.workers) as pool:
        rows = [r for r in pool.map(one, range(a.draws)) if r]
    real = json.load(open(REAL))["details"]
    R = list(real["real_test"].values()) + list(real["real_train"].values())
    for name, rs in (("real", R), ("drawn", rows)):
        print(f"{name:6} ellipticity {pct([r['ellipticity'][1] for r in rs if r['ellipticity']])}"
              f"  fwhm {pct([r['fwhm'][1] for r in rs if r['fwhm']])}"
              f"  density>3 {pct([r['density']['3'] for r in rs])}  >10 {pct([r['density']['10'] for r in rs])}"
              f"  >100 {pct([r['density']['100'] for r in rs])}")
    json.dump(rows, open("<ESSAIS>/starRemoval/diagnose/calib_last.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
