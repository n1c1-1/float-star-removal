# ======================================================================
# star_measure.py
# Role     : Measures the stars of the real images to set the generator's ranges.
# Step     : 1. Training on synthetic data only (E2 / E3, 7-9 Nov) (see scripts/README.md)
# Inputs   : generator: seeds + catalogue.json (public instrument sheets); no image
# Outputs  : synthetic pairs (input/target/info.json + manifest.json), network E3
# Example  : python star_measure.py [--out <ESSAIS>/starRemoval/measures/stars_real.json]
# ======================================================================
"""The stars of our real images, measured: the ranges the synthetic stars are drawn in (stars.py).

On each training image with its stars (data/train_with_stars: linear, in noise sigmas, sky 0; the test images stay
out, for the validation):

- detection: local maxima of the luminance above its local background (a 64 px block median);
- isolated unsaturated stars (peak 15 sigma to half the saturation, nothing as bright within 15 px): the FWHM (from
  the area above half the peak), the Moffat beta (Moffat 1969, A&A 3, 455; fitted on their median radial profile
  scaled by their FWHM), the ellipticity and its direction from the second moments, per zone of a 3 x 3 grid of the
  field (coma and astigmatism grow towards the corners: the elongation's angle to the radial direction tells which);
  the FWHM per colour (blue larger: the chromatism of refractors and lenses);
- the brightest stars (the 8 largest saturated or brightest): the median radial profile in rings out to 800 px (the
  aureole: Racine 1996, PASP 108, 699), its log-log slope between 3 and 30 FWHM and between 30 and 300 px, its
  level at 10 and 100 px over the saturation level; the azimuthal profile between 4 and 20 FWHM: the diffraction
  spikes, their number and angles;
- the star density: maxima per megapixel above 1, 3, 10, 100 sigma (the counts' slope).

    python star_measure.py [--out <ESSAIS>/starRemoval/measures/stars_real.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

DATA = Path(__file__).resolve().parent / "data"
LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)


def images():
    """The training images only: the test images (data/test_with_stars) stay out of the calibration, for the final
    validation."""
    tr = json.loads((DATA / "train_with_stars" / "fields.json").read_text())
    for name in sorted(tr):
        f = DATA / "train_with_stars" / name / "input.npy"
        if f.exists():
            yield name, f, float(tr[name]["top"])


def load(path: Path) -> np.ndarray:
    return np.asarray(np.load(path), np.float32)


def local_background(L: np.ndarray, b: int = 64) -> np.ndarray:
    H, W = L.shape
    h, w = H // b, W // b
    m = np.median(L[:h * b, :w * b].reshape(h, b, w, b).transpose(0, 2, 1, 3).reshape(h, w, -1), -1)
    return cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)


def moments(stamp: np.ndarray) -> tuple[float, float]:
    """Ellipticity (a - b) / (a + b) and the major axis angle (degrees) from the second moments above 20 % of the
    peak."""
    s = np.clip(stamp, 0, None)
    s = np.where(s > 0.2 * s.max(), s, 0)
    yy, xx = np.indices(s.shape, dtype=np.float64)
    t = s.sum() + 1e-12
    cx, cy = (s * xx).sum() / t, (s * yy).sum() / t
    mxx = (s * (xx - cx) ** 2).sum() / t
    myy = (s * (yy - cy) ** 2).sum() / t
    mxy = (s * (xx - cx) * (yy - cy)).sum() / t
    tr, d = mxx + myy, np.sqrt(((mxx - myy) / 2) ** 2 + mxy ** 2)
    a, b = np.sqrt(tr / 2 + d), np.sqrt(max(tr / 2 - d, 1e-12))
    return float((a - b) / (a + b)), float(np.degrees(0.5 * np.arctan2(2 * mxy, mxx - myy)))


def radial_profile(img: np.ndarray, x: float, y: float, rmax: int, edges=None):
    H, W = img.shape
    x0, x1, y0, y1 = int(max(0, x - rmax)), int(min(W, x + rmax + 1)), int(max(0, y - rmax)), int(min(H, y + rmax + 1))
    sub = img[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    r = np.hypot(xx - x, yy - y)
    if edges is None:
        edges = np.unique(np.round(np.geomspace(1, rmax, 40)))
    prof = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (r >= a) & (r < b)
        prof.append(float(np.median(sub[m])) if m.sum() > 3 else np.nan)
    return np.sqrt(edges[:-1] * edges[1:]), np.array(prof)


def measure(name: str, img: np.ndarray, top: float | None) -> dict:
    _c, H, W = img.shape
    Lraw = np.tensordot(LUM, img, 1)                 # the sky is 0: the halos are read on this one
    L = Lraw - local_background(Lraw)
    if top is None:
        top = float(np.percentile(L, 99.999))
    mx = ndimage.maximum_filter(L, 7)
    pk = (L == mx) & (L > 1.0)
    pk[:20], pk[-20:], pk[:, :20], pk[:, -20:] = False, False, False, False
    ys, xs = np.nonzero(pk)
    vals = L[ys, xs]
    mp = H * W / 1e6
    out: dict = {"size": [W, H], "top": round(top, 1),
                 "density": {str(t): round(float((vals > t).sum() / mp), 1) for t in (1, 3, 10, 100)}}
    # isolated, unsaturated stars
    big = ndimage.maximum_filter(L, 31)
    iso = (vals > 15) & (vals < 0.5 * top)
    rows = []
    order = np.argsort(-vals)
    for i in order[:20000]:
        if not iso[i]:
            continue
        y, x = ys[i], xs[i]
        if y < 16 or x < 16 or y > H - 17 or x > W - 17:
            continue
        st = L[y - 15:y + 16, x - 15:x + 16]
        if (st > 0.3 * vals[i]).sum() > 4 * max((st > 0.5 * vals[i]).sum(), 1) or big[y, x] > vals[i]:
            continue
        fw = 2 * np.sqrt((st > 0.5 * vals[i]).sum() / np.pi)
        e, ang = moments(st)
        fwc = [2 * np.sqrt((img[c, y - 15:y + 16, x - 15:x + 16] > 0.5 * img[c, y, x]).sum() / np.pi) for c in range(3)]
        rows.append((x / W, y / H, fw, e, ang, *fwc))
        if len(rows) >= 1500:
            break
    if len(rows) < 20:
        out["isolated"] = len(rows)
        return out
    R = np.array(rows)
    out["isolated"] = len(R)
    out["fwhm"] = [round(float(v), 2) for v in np.percentile(R[:, 2], [10, 50, 90])]
    out["ellipticity"] = [round(float(v), 3) for v in np.percentile(R[:, 3], [10, 50, 90])]
    out["fwhm_rgb"] = [round(float(np.median(R[:, 5 + c])), 2) for c in range(3)]
    zones = {}
    for zy in range(3):
        for zx in range(3):
            m = (R[:, 0] * 3 // 1 == zx) & (R[:, 1] * 3 // 1 == zy)
            if m.sum() < 5:
                continue
            # the elongation's angle to the radial direction: 0 = radial (coma), 90 = tangential (astigmatism)
            radial = np.degrees(np.arctan2(R[m, 1] - 0.5, R[m, 0] - 0.5))
            d = np.abs(((R[m, 4] - radial) + 90) % 180 - 90)
            zones[f"{zy}{zx}"] = {"n": int(m.sum()), "fwhm": round(float(np.median(R[m, 2])), 2),
                                 "e": round(float(np.median(R[m, 3])), 3),
                                 "angle_to_radial": round(float(np.median(d)), 0)}
    out["zones"] = zones
    # Moffat beta: the median profile of the isolated stars in units of their FWHM
    prof = []
    for x, y, fw, *_ in R[:300]:
        px, py = int(round(x * W)), int(round(y * H))
        r, p = radial_profile(L, px, py, 15, edges=np.r_[0.0, np.arange(1.5, 16, 1.0)])
        r[0] = 0.0
        p[0] = float(L[py, px])
        if np.isfinite(p).all():
            prof.append(np.interp(np.arange(0.25, 3.01, 0.25), r / fw, p / max(p[0], 1e-9)))
    prof = np.nanmedian(np.array(prof), 0)
    best = min(((float(np.nansum((prof - (1 + (np.arange(0.25, 3.01, 0.25) * 2 * np.sqrt(2 ** (1 / b) - 1)) ** 2)
                                  ** -b) ** 2)), b) for b in np.arange(1.2, 8.0, 0.1)))
    out["moffat_beta"] = round(best[1], 1)
    # the brightest stars: halo profile and spikes
    fw = float(np.median(R[:, 2]))
    halos = []
    lab, n = ndimage.label(L > 0.9 * top)
    sizes = ndimage.sum(np.ones_like(L), lab, range(1, n + 1)) if n else np.array([])
    cand = []
    if n:
        for j in np.argsort(-sizes)[:8]:
            cy, cx = ndimage.center_of_mass(lab == j + 1)
            cand.append((cx, cy, float(sizes[j])))
    else:
        for i in order[:8]:
            cand.append((xs[i], ys[i], 0.0))
    for cx, cy, area in cand:
        r, p = radial_profile(Lraw, cx, cy, 800)
        p = p - float(np.nanpercentile(p[r > 500], 10)) if (r > 500).any() and np.isfinite(p[r > 500]).any() else p
        ok = np.isfinite(p) & (p > 0.05)
        sl_in = sl_out = None
        a, b = 3 * fw, 30 * fw
        m = ok & (r > a) & (r < b)
        if m.sum() > 4:
            sl_in = float(np.polyfit(np.log(r[m]), np.log(p[m]), 1)[0])
        m = ok & (r > 30) & (r < 300)
        if m.sum() > 4:
            sl_out = float(np.polyfit(np.log(r[m]), np.log(p[m]), 1)[0])
        lv = {str(k): round(float(np.interp(k, r, np.nan_to_num(p))) / top, 5) for k in (10, 100, 300)}
        # spikes: azimuthal profiles in two rings (4 - 10 and 10 - 20 FWHM, the radial fall flattened), a direction
        # counts when it stands out (2 x the median) in both rings and its opposite does too (spikes are symmetric)
        R2 = int(20 * fw)
        yy, xx = np.mgrid[-R2:R2 + 1, -R2:R2 + 1]
        rr, th = np.hypot(xx, yy), (np.degrees(np.arctan2(yy, xx)) + 360) % 360
        Y, X = (yy + int(cy)).clip(0, H - 1), (xx + int(cx)).clip(0, W - 1)
        v = L[Y, X] * rr ** 2
        hits = []
        for a0, a1 in ((4, 10), (10, 20)):
            sel = (rr > a0 * fw) & (rr < a1 * fw)
            az = np.array([np.median(v[sel & (th >= k) & (th < k + 3)]) for k in range(0, 360, 3)])
            hits.append(az > 2 * np.median(az))
        both = hits[0] & hits[1]
        both = both & np.roll(both, 60)                                  # its opposite, 180 degrees away
        peaks = [3 * k for k in range(120) if both[k] and not both[k - 1]]
        halos.append({"x": int(cx), "y": int(cy), "sat_area": int(area), "slope_core": sl_in and round(sl_in, 2),
                      "slope_far": sl_out and round(sl_out, 2), "level_over_top": lv, "spikes": peaks})
    out["bright"] = halos
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="<ESSAIS>/starRemoval/measures/stars_real.json")
    a = ap.parse_args(argv)
    res = {}
    for name, f, top in images():
        r = measure(name, load(f), top)
        res[name] = r
        b = r.get("bright", [])
        print(f"{name[:38]:38} fwhm {r.get('fwhm')} beta {r.get('moffat_beta')} e {r.get('ellipticity')} rgb "
              f"{r.get('fwhm_rgb')} dens {r['density']} far slopes {[h['slope_far'] for h in b[:4]]} spikes "
              f"{[len(h['spikes']) for h in b[:4]]}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
