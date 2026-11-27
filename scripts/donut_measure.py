# ======================================================================
# donut_measure.py
# Role     : Measurement of the defocused stars (rings) of a real stack.
# Step     : 1. Training on synthetic data only (E2 / E3, 7-9 Nov) (see scripts/README.md)
# Inputs   : generator: seeds + catalogue.json (public instrument sheets); no image
# Outputs  : synthetic pairs (input/target/info.json + manifest.json), network E3
# Example  : python donut_measure.py [--frames 6] [--out <ESSAIS>/starRemoval/measures/donuts_m31.json]
# ======================================================================
"""The defocused stars of a real stack: a real M31 (<DISQUE>/astro_gpu_m31, Newton 150/750, Canon 6000 x 4000, 3.72 um
pixels, RGGB), 30 x 60 s out of focus: each star is the image of the pupil, a ring with the secondary mirror's shadow
in it, a half-moon towards the edges (the defocused beam cut by the tube or the focuser: vignetting).

A few frames averaged minus the mean dark, read as 2 x 2 super-pixels (R, (G1 + G2) / 2, B: 3000 x 2000). On the
isolated rings: the outer diameter (where the radial profile falls under half the ring's peak), the hole (the centre
over the ring's peak), and the asymmetry (the light's centroid off the ring's centre, over its radius, and its
direction to the field's centre), per zone of a 3 x 3 grid.

    python donut_measure.py [--frames 6] [--out <ESSAIS>/starRemoval/measures/donuts_m31.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

RAW = Path("<DISQUE>/astro_gpu_m31/raw/M_31")


def superpixels(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    return np.stack([a[0::2, 0::2], (a[0::2, 1::2] + a[1::2, 0::2]) / 2, a[1::2, 1::2]])


def stack(n: int) -> np.ndarray:
    from astropy.io import fits
    darks = sorted((RAW / "Dark").glob("*.fits"))
    dark = np.mean([superpixels(fits.getdata(f)) for f in darks], 0)
    lights = sorted((RAW / "Light").glob("*.fits"))[:n]
    return np.mean([superpixels(fits.getdata(f)) for f in lights], 0) - dark


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--out", default="<ESSAIS>/starRemoval/measures/donuts_m31.json")
    a = ap.parse_args(argv)
    img = stack(a.frames)
    L = img.mean(0)
    H, W = L.shape
    b = 64
    bg = cv2.resize(np.median(L[:H // b * b, :W // b * b].reshape(H // b, b, W // b, b).transpose(0, 2, 1, 3)
                              .reshape(H // b, W // b, -1), -1).astype(np.float32), (W, H))
    D = L - bg
    noise = 1.4826 * float(np.median(np.abs(D - np.median(D))))
    sm = cv2.GaussianBlur(D, (0, 0), 4)                     # a ring blurred: one blob, its maximum near the centre
    mx = ndimage.maximum_filter(sm, 41)
    pk = (sm == mx) & (sm > 20 * noise / 4)
    pk[:40], pk[-40:], pk[:, :40], pk[:, -40:] = False, False, False, False
    ys, xs = np.nonzero(pk)
    rows = []
    for y, x in zip(ys, xs):
        st = D[y - 30:y + 31, x - 30:x + 31]
        if st.max() > 0.8 * float(np.percentile(D, 99.999)):          # saturated
            continue
        w = np.clip(st, 0, None)
        yy, xx = np.indices(st.shape) - 30
        r = np.hypot(xx, yy)
        prof = np.array([np.median(st[(r >= k) & (r < k + 1)]) for k in range(30)])
        peak = prof.max()
        if peak < 10 * noise:
            continue
        rpk = int(np.argmax(prof))
        outer = next((k for k in range(rpk, 30) if prof[k] < 0.5 * peak), None)
        if outer is None or outer < 4:
            continue
        if (np.abs(st[r > outer + 6]) > 0.3 * peak).any():             # a neighbour too close
            continue
        hole = float(prof[0] / peak)
        t = w.sum() + 1e-9
        cx, cy = (w * xx).sum() / t, (w * yy).sum() / t
        asym = float(np.hypot(cx, cy) / outer)
        to_centre = np.degrees(np.arctan2((H / 2 - y), (W / 2 - x)))
        dirn = float(abs(((np.degrees(np.arctan2(cy, cx)) - to_centre) + 180) % 360 - 180))
        rows.append((x / W, y / H, 2 * outer, hole, asym, dirn))
    R = np.array(rows)
    res = {"stars": len(R), "diameter_px_superpixel": [round(float(v), 1) for v in np.percentile(R[:, 2], [10, 50, 90])],
           "hole": [round(float(v), 2) for v in np.percentile(R[:, 3], [10, 50, 90])], "zones": {}}
    for zy in range(3):
        for zx in range(3):
            m = (R[:, 0] * 3 // 1 == zx) & (R[:, 1] * 3 // 1 == zy)
            if m.sum() < 3:
                continue
            res["zones"][f"{zy}{zx}"] = {"n": int(m.sum()), "diameter": round(float(np.median(R[m, 2])), 1),
                                        "hole": round(float(np.median(R[m, 3])), 2),
                                        "asymmetry": round(float(np.median(R[m, 4])), 3),
                                        "light_side_to_centre_deg": round(float(np.median(R[m, 5])), 0)}
    print(json.dumps(res, indent=1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    # a sheet of rings: centre and corners, at 400 %
    tiles = []
    for fy, fx in ((0.5, 0.5), (0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9)):
        d = np.hypot(R[:, 0] - fx, R[:, 1] - fy)
        for j in np.argsort(d)[:3]:
            x, y = int(R[j, 0] * W), int(R[j, 1] * H)
            c = img[:, y - 25:y + 26, x - 25:x + 26]
            c = c - np.median(c.reshape(3, -1), 1)[:, None, None]
            c = np.clip(c / max(float(c.max()), 1e-9), 0, 1) ** 0.6
            t = (c.transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)
            tiles.append(cv2.resize(t, (204, 204), interpolation=cv2.INTER_NEAREST))
    cv2.imwrite(str(Path(a.out).with_suffix(".jpg")), np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, 15, 3)]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
