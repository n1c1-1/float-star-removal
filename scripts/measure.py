# ======================================================================
# measure.py
# Role     : Statistics of the backgrounds, to compare the generated backgrounds with the real images.
# Step     : 1. Training on synthetic data only (E2 / E3, 7-9 Nov) (see scripts/README.md)
# Inputs   : generator: seeds + catalogue.json (public instrument sheets); no image
# Outputs  : synthetic pairs (input/target/info.json + manifest.json), network E3
# Example  : python measure.py real_input [--out <ESSAIS>/starRemoval/measures/real_input.json]
# ======================================================================
"""Statistics of backgrounds, to compare the generated ones (backgrounds.py, defects.py) with our real images, measured
with their stars masked.

The images are in the units the star removal network works in (astro_gpu/stages/star_removal.py): linear, each colour
minus its sky, over one noise sigma of the sky; 0 is the sky.

Per image (on a real image, off its stars: above their 16 px local background by 0.5 sigma, grown by 7 px):
- levels: percentiles of the luminance (1, 50, 90, 99, 99.9) and the share of the field above 3, 10 and 30 sigma;
- texture: the slope of the luminance's power spectrum, azimuthally averaged, fitted between 8 and 256 px (the
  interstellar medium's is about -2.7 to -3: Miville-Deschenes et al. 2010, A&A 518, L104); the noise left in a real
  image flattens it at the small scales, hence the 8 px floor;
- colour: the median and spread of (R - G) and (B - G) over the luminance, where the luminance is above 3 sigma;
- the sky's defects, on the sky only (the darkest 40 % of the 32 px blocks of the centre crop): the spread (std) of the
  luminance and of the colour (R - G, B - G) in three bands of scale, fine (< 2 px), middle (2 - 8 px), blotches
  (8 - 32 px), from differences of Gaussian blurs; the sky's large-scale spread (p95 - p5 of the 64 px block medians
  of the sky, over the whole image); the sky's tint (median R - G and B - G on the sky).

    python measure.py real_input [--out <ESSAIS>/starRemoval/measures/real_input.json]
    python measure.py FOLDER_OF_GENERATED_FITS [--out ...]

``real_input``: our training images with their stars (data/train_with_stars/<field>/input.npy, full resolution):
levels, colours and the sky's defects, off the stars.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REAL = Path(__file__).resolve().parent / "data" / "train_with_stars"
LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)
#: the crop the spectrum is measured on (px), the scales of the fit
CROP, FIT = 2048, (8.0, 256.0)


def load(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path, mmap_mode="r"), np.float32)
    from astropy.io import fits
    return np.asarray(fits.getdata(path), np.float32)


def spectrum_slope(L: np.ndarray, valid: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """The slope of the azimuthally averaged power spectrum of the centre crop, between FIT[0] and FIT[1] px."""
    H, W = L.shape
    c = min(CROP, H, W)
    y0, x0 = (H - c) // 2, (W - c) // 2
    a = L[y0:y0 + c, x0:x0 + c].astype(np.float64)
    v = valid[y0:y0 + c, x0:x0 + c]
    a = np.where(v, a - np.median(a[v]), 0.0)
    w = np.outer(np.hanning(c), np.hanning(c))
    P = np.abs(np.fft.fftshift(np.fft.fft2(a * w))) ** 2
    yy, xx = np.indices(P.shape) - c // 2
    r = np.hypot(yy, xx).astype(int)
    prof = np.bincount(r.ravel(), P.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    k = np.arange(len(prof)) / c                                 # cycles per pixel
    sel = (k >= 1 / FIT[1]) & (k <= 1 / FIT[0]) & (prof > 0)
    slope = float(np.polyfit(np.log(k[sel]), np.log(prof[sel]), 1)[0])
    return slope, k[1:c // 2], prof[1:c // 2]


def star_mask(L: np.ndarray) -> np.ndarray:
    """The stars of an image with stars, generously: above the 16 px local background by 0.5 sigma, grown by 7 px."""
    import cv2
    m = (L - cv2.GaussianBlur(L, (0, 0), 16)) > 0.5
    return cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0


def sky_defects(img: np.ndarray, valid: np.ndarray, stars: bool = False) -> dict:
    """The sky's defects (see the module): band spreads of luminance and colour, large-scale spread, tint. ``stars``:
    the image has its stars, they are masked out of the bands."""
    import cv2
    _c, H, W = img.shape
    c = min(CROP, H, W)
    y0, x0 = (H - c) // 2, (W - c) // 2
    a = np.asarray(img[:, y0:y0 + c, x0:x0 + c], np.float32)
    v = valid[y0:y0 + c, x0:x0 + c]
    L = np.tensordot(LUM, a, 1)
    b = 32
    nb = c // b
    blk = np.median(L[:nb * b, :nb * b].reshape(nb, b, nb, b).transpose(0, 2, 1, 3).reshape(nb, nb, -1), -1)
    vb = v[:nb * b, :nb * b].reshape(nb, b, nb, b).all(axis=(1, 3))
    if vb.sum() < 10:
        return {}
    sky_b = vb & (blk <= np.percentile(blk[vb], 40))
    sky = np.zeros_like(v)
    sky[:nb * b, :nb * b] = np.repeat(np.repeat(sky_b, b, 0), b, 1)
    sky = cv2.erode(sky.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0   # away from the blocks' edges
    if stars:
        sky &= ~star_mask(L)
    chans = {"lum": L, "rg": a[0] - a[1], "bg": a[2] - a[1]}
    out = {}
    for k, x in chans.items():
        g = [x] + [cv2.GaussianBlur(x, (0, 0), s) for s in (2, 8, 32)]
        out[k] = [round(float(np.std((g[i] - g[i + 1])[sky])), 3) for i in range(3)]   # fine, middle, blotches
    out["tint"] = [round(float(np.median(chans["rg"][sky])), 3), round(float(np.median(chans["bg"][sky])), 3)]
    # the large-scale spread, over the whole image: 64 px block medians of the sky blocks
    Lw = np.tensordot(LUM, np.asarray(img[:, ::4, ::4], np.float32), 1)
    vw = valid[::4, ::4]
    b2 = 16                                                     # 64 px at a quarter of the resolution
    h2, w2 = Lw.shape[0] // b2, Lw.shape[1] // b2
    bm = np.median(Lw[:h2 * b2, :w2 * b2].reshape(h2, b2, w2, b2).transpose(0, 2, 1, 3).reshape(h2, w2, -1), -1)
    bv = vw[:h2 * b2, :w2 * b2].reshape(h2, b2, w2, b2).all(axis=(1, 3))
    s = bm[bv & (bm <= np.percentile(bm[bv], 40))] if bv.sum() > 10 else bm.ravel()
    out["large"] = round(float(np.percentile(s, 95) - np.percentile(s, 5)), 3)
    return out


def stats(img: np.ndarray, stars: bool = False) -> dict:
    valid = np.abs(img).max(0) > 0
    if stars:     # an image with its stars: its levels and colours off the stars (masked), and its sky's defects
        import cv2
        L = np.tensordot(LUM, img, 1)
        ok = valid & ~star_mask(L)
        s = L[ok][::3]
        # the stars filled from around them (a normalised convolution, 8 px) for the spectrum: no holes in it
        w = ok.astype(np.float32)
        fill = cv2.GaussianBlur(np.where(ok, L, 0).astype(np.float32), (0, 0), 8) / \
            np.maximum(cv2.GaussianBlur(w, (0, 0), 8), 1e-3)
        out = {"p": [round(float(x), 2) for x in np.percentile(s, [1, 50, 90, 99, 99.9])],
               "slope": round(spectrum_slope(np.where(ok, L, fill), valid)[0], 2),
               "sky": sky_defects(img, valid, stars=True)}
        for lo, hi, k in ((0.5, 3, "colour_faint"), (3, 1e9, "colour_bright")):
            m = ok & (L > lo) & (L < hi)
            if m.sum() > 2000:
                r, g, b = (float(np.median(img[c][m][::3])) for c in range(3))
                out[k] = [round(r / g, 3), round(b / g, 3), round(float(m.mean()), 4)]
        return out
    L = np.tensordot(LUM, img, 1)
    s = L[valid][::3]
    p = np.percentile(s, [1, 50, 90, 99, 99.9])
    out = {"size": list(img.shape[1:]), "p": [round(float(x), 2) for x in p],
           "above": {str(t): round(float((s > t).mean()), 4) for t in (3, 10, 30)}}
    out["slope"], _k, _p = spectrum_slope(L, valid)
    out["slope"] = round(out["slope"], 2)
    bright = valid & (L > 3)
    if bright.sum() > 1000:
        Lb = L[bright][::3]
        rg = (img[0][bright][::3] - img[1][bright][::3]) / Lb
        bg = (img[2][bright][::3] - img[1][bright][::3]) / Lb
        out["colour"] = {"rg": [round(float(x), 3) for x in np.percentile(rg, [10, 50, 90])],
                         "bg": [round(float(x), 3) for x in np.percentile(bg, [10, 50, 90])]}
    out["sky"] = sky_defects(img, valid)
    return out


def real_images(which: str = "input"):
    """Our training images with their stars, full resolution."""
    fields = json.loads((REAL / "fields.json").read_text())
    for name in sorted(fields):
        f = REAL / name / f"{which}.npy"
        if f.exists():
            yield name, f


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", help="'real_input', or a folder of generated FITS")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    stars = a.what == "real_input"
    items = list(real_images()) if stars else [(f.stem, f) for f in sorted(Path(a.what).glob("*.fits"))]
    res = {}
    for name, f in items:
        res[name] = stats(load(f), stars)
        r = res[name]
        s = r.get("sky") or {}
        if stars:
            print(f"{name[:40]:40} p99 {r['p'][3]:7.1f} p99.9 {r['p'][4]:7.1f} slope {r['slope']:6.2f} colour "
                  f"{r.get('colour_faint')} "
                  f"{r.get('colour_bright')} | sky large {s.get('large')} tint {s.get('tint')}", flush=True)
            continue
        print(f"{name[:40]:40} p99 {r['p'][3]:7.1f} p99.9 {r['p'][4]:7.1f} slope {r['slope']:6.2f} | sky lum "
              f"{s.get('lum')} rg {s.get('rg')} bg {s.get('bg')} large {s.get('large')} tint {s.get('tint')}",
              flush=True)
    out = Path(a.out) if a.out else (Path(__import__("os").environ.get("ESSAIS", "essais")) / "starRemoval" / "measures") / f"{Path(a.what).name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
