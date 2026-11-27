# ======================================================================
# juger.py
# Role     : Check of a version: per-512 px-tile measurements and 100 % sheets of the suspect tiles.
# Step     : 2. Classic Gaia-guided subtraction chain (v33b) (see scripts/LISEZMOI.md)
# Inputs   : stacked image (noise units), astrometry, offline deep Gaia extract
# Outputs  : star-free image, class maps
# Example  : python juger.py <image> <tag> [--tuiles 8] [--xy x,y;x,y]
# ======================================================================
"""Check of a version: per-512 px-tile measurements and 100 % sheets of the suspect tiles.

    python juger.py <image> <tag> [--tuiles 8] [--xy x,y;x,y]

Measurements (on the star-free image, in linear):
- remnants: at Gaia positions, core (r <= 1 width) minus ring (2-4 widths), in noises; > 3 = star left;
- hollows (11 Nov method): median 0-4 px from the centre against the 8-15 px ring, in % of the ring level, for the
  bright stars on nebula; share of the stars more than 3 % below the ring, and the worst 10 %;
- changed far from the stars: mean |input - star-free| at more than 3 PSF radii from any Gaia star, in noises.
Sheets: input | star-free | star layer, same asinh stretch, 512 px at 100 %.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import images  # noqa: E402


def stretch(x, lo, hi, a=3.0):
    f = lambda v: np.arcsinh(v / a)
    return np.clip((f(x) - f(lo)) / (f(hi) - f(lo)), 0, 1)


def to_rgb8(a, lo, hi):
    if a.shape[0] == 1:
        a = np.repeat(a, 3, 0)
    v = stretch(a, lo, hi)
    return (np.transpose(v, (1, 2, 0)) * 255 + 0.5).astype(np.uint8)


def save_img(path: Path, rgb8: np.ndarray, quality=92):
    """Écriture sûre sur un chemin accentué (cv2.imwrite échoue en silence) ; vérifie que le fichier existe."""
    ext = path.suffix.lower()
    ok, buf = cv2.imencode(ext, cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else [])
    assert ok
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.tobytes())
    assert path.exists() and path.stat().st_size > 0, path


def read_stars(p):
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows


def hole_metrics(sl, xs, ys, Lin, fwhm, sel):
    """Creux : (médiane 0-4 px - médiane 8-15 px) / niveau de l'anneau, en % (linéaire, luminance)."""
    L = sl.mean(0)
    H, W = L.shape
    vals = []
    for x, y in zip(xs[sel], ys[sel]):
        xi, yi = int(round(x)), int(round(y))
        if xi < 16 or yi < 16 or xi >= W - 16 or yi >= H - 16:
            continue
        cut = L[yi - 15:yi + 16, xi - 15:xi + 16]
        gy, gx = np.mgrid[-15:16, -15:16]
        r = np.hypot(gx + xi - x, gy + yi - y)
        core = np.median(cut[r <= 4]); ring = np.median(cut[(r >= 8) & (r <= 15)])
        if ring <= 0.5:          # pas sur nébuleuse : le % n'a pas de sens
            continue
        vals.append(100 * (core - ring) / ring)
    v = np.array(vals)
    if not len(v):
        return dict(n=0)
    return dict(n=int(len(v)), median=round(float(np.median(v)), 2), share_below_3=round(float((v < -3).mean() * 100), 1),
                worst10=round(float(np.mean(np.sort(v)[:max(1, len(v) // 10)])), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("tag")
    ap.add_argument("--tuiles", type=int, default=8)
    ap.add_argument("--xy", default="")
    ap.add_argument("--size", type=int, default=512)
    a = ap.parse_args()
    d = HERE / "sorties" / a.name / a.tag
    img = images.load(a.name)
    sl = np.load(d / "starless.npy")
    summ = json.loads((d / "resume.json").read_text(encoding="utf-8"))
    fwhm = summ["fwhm"]; R = summ["R"]
    C, H, W = img.shape
    z = np.load(HERE / "sorties" / a.name / "astrometrie.npz", allow_pickle=True)
    cat = z["cat"].item()
    xg, yg, G = cat["x"], cat["y"], cat["g"]
    stars = read_stars(d / "etoiles.csv")
    Lin = img.mean(0); Ls = sl.mean(0)
    # bruit global (luminance) du ciel
    hp = Lin - cv2.GaussianBlur(Lin, (0, 0), 6)
    sig = float(np.median(np.abs(hp - np.median(hp))) * 1.4826)
    # restes aux positions Gaia (étoiles assez brillantes pour être vues : coeur > 5 bruits dans l'entrée)
    gy, gx = np.mgrid[-int(4 * fwhm) - 1:int(4 * fwhm) + 2, -int(4 * fwhm) - 1:int(4 * fwhm) + 2]
    left = []
    for x, y, g in zip(xg, yg, G):
        xi, yi = int(round(x)), int(round(y))
        rr = gy.shape[0] // 2
        if xi < rr or yi < rr or xi >= W - rr or yi >= H - rr:
            continue
        r = np.hypot(gx + xi - x, gy + yi - y)
        cin = Lin[yi - rr:yi + rr + 1, xi - rr:xi + rr + 1]
        cs = Ls[yi - rr:yi + rr + 1, xi - rr:xi + rr + 1]
        c0 = (r <= max(1.0, fwhm)); an = (r >= 2 * fwhm) & (r <= 4 * fwhm)
        vin = cin[c0].mean() - np.median(cin[an]); vs = cs[c0].mean() - np.median(cs[an])
        if vin < 5 * sig:
            continue
        left.append((x, y, g, vin / sig, vs / sig))
    left = np.array(left)
    res = dict(sig=sig, gaia_visible=int(len(left)))
    if len(left):
        res["left_gt3"] = int((left[:, 4] > 3).sum()); res["hole_lt_m3"] = int((left[:, 4] < -3).sum())
        res["left_share_pct"] = round(100 * float((left[:, 4] > 3).mean()), 2)
    # creux, étoiles brillantes (G < 11) sur nébuleuse
    res["creux"] = hole_metrics(sl, xg, yg, Lin, fwhm, G < 11)
    # changé loin des étoiles
    far = np.ones((H, W), np.uint8)
    for x, y, g in zip(xg, yg, G):
        rr = int(3 * R + max(0, 12 - g) * 10)
        cv2.circle(far, (int(round(x)), int(round(y))), rr, 0, -1)
    try:
        ex = [(float(s["x"]), float(s["y"])) for s in stars if s["drapeau"] == "8"]
        for x, y in ex:
            cv2.circle(far, (int(round(x)), int(round(y))), int(3 * R), 0, -1)
    except Exception:
        pass
    far = far.astype(bool) & np.any(img != 0, 0)
    diff = np.abs(img - sl).mean(0)
    res["far_pixels_pct"] = round(100 * float(far.mean()), 2)
    res["far_change"] = round(float(diff[far].mean() / sig), 4) if far.any() else None
    # tuiles
    S = a.size
    tiles = []
    for ty in range(0, H - S // 2, S):
        for tx in range(0, W - S // 2, S):
            y1, x1 = min(H, ty + S), min(W, tx + S)
            m = (left[:, 0] >= tx) & (left[:, 0] < x1) & (left[:, 1] >= ty) & (left[:, 1] < y1) if len(left) else np.zeros(0, bool)
            nl = int((left[m, 4] > 3).sum()) if len(left) else 0
            nh = int((left[m, 4] < -3).sum()) if len(left) else 0
            fc = float(diff[ty:y1, tx:x1][far[ty:y1, tx:x1]].mean() / sig) if far[ty:y1, tx:x1].any() else 0.0
            tiles.append(dict(x=tx, y=ty, left=nl, holes=nh, far=round(fc, 3), worst=float(left[m, 4].max()) if m.any() else 0))
    res["tiles"] = len(tiles)
    sc = sorted(tiles, key=lambda t: -(t["left"] + 2 * t["holes"] + 20 * t["far"]))
    pick = sc[:a.tuiles]
    if a.xy:
        for s in a.xy.split(";"):
            x, y = (int(v) for v in s.split(","))
            pick.append(dict(x=max(0, min(W - S, x - S // 2)), y=max(0, min(H - S, y - S // 2)), left=-1, holes=-1, far=-1))
    lo = -3 * sig
    hi = float(np.percentile(Lin, 99.9))
    sheets = d / "planches"
    for f in sheets.glob("*.jpg") if sheets.exists() else []:
        f.unlink()
    for t in pick:
        x0, y0 = t["x"], t["y"]
        a_ = img[:, y0:y0 + S, x0:x0 + S]; b_ = sl[:, y0:y0 + S, x0:x0 + S]
        row = np.concatenate([to_rgb8(a_, lo, hi), np.full((a_.shape[1], 6, 3), 255, np.uint8), to_rgb8(b_, lo, hi),
                              np.full((a_.shape[1], 6, 3), 255, np.uint8), to_rgb8(a_ - b_, lo, hi)], 1)
        save_img(sheets / f"tuile_x{x0}_y{y0}.jpg", row)
    # vue d'ensemble réduite
    k = max(1, int(np.ceil(max(H, W) / 1600)))
    ov = np.concatenate([to_rgb8(img[:, ::k, ::k], lo, hi), np.full((len(range(0, H, k)), 6, 3), 255, np.uint8), to_rgb8(sl[:, ::k, ::k], lo, hi)], 1)
    save_img(sheets / "ensemble.jpg", ov, 88)
    res["tiles_picked"] = pick
    (d / "mesures.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "tiles_picked"}))
    for t in pick:
        print(t)


if __name__ == "__main__":
    main()
