# ======================================================================
# defects.py
# Rôle     : Fonds du jeu 2 : le jeu 1 plus les défauts des vrais empilements (marbrures, vignetage, gradient, donuts).
# Étape    : 1. Entraînement sur données synthétiques seules (E2 / E3, 7-9 nov) (voir scripts/LISEZMOI.md)
# Entrées  : générateur : graines + catalogue.json (fiches d'instruments publiques) ; aucune image
# Sorties  : paires synthétiques (input/target/info.json + manifest.json), réseau E3
# Exemple  : python defects.py --n 48 [--seed 1] [--size 2048] [--out <ESSAIS>/starRemoval/backgrounds_defects]
# ======================================================================
"""The second set of backgrounds: the first one's (backgrounds.py) with the defects of real stacks on top.

Real stacks are far from clean: after the denoising, colour blotches instead of grain; flats that leave vignetting or
bands; what the Gradient block left; stacking edges; dust donuts; walking noise. These belong to the background: the
same draw goes into the network's input and into its target, so that it learns to take the stars off without
smoothing the sky.

The layer, each part drawn at random, in sigma units (measured on the training set's images with their stars, stars
masked, the sparse fields setting the scale: measure.py real_input):

- grain: white noise slightly blurred, 0.003 to 0.04;
- blotches: band-limited noise of scale 6 to 30 px, 0.015 to 0.12 in luminance, the colours partly independent
  (pinkish and greenish blotches); above about 0.15 they read as coloured confetti no real sky shows (the bands
  measured on the images with their stars are raised by the faint stars left unmasked: the scale is set by the sparse
  fields and by eye, sheets backgrounds_defects_sky.jpg vs real_input_sky.jpg);
- the large scale, its 64 px block spread drawn 0.02 to 0.35: a residual gradient (a plane and a curve), vignetting
  (centred off the middle, over- or under-corrected), a glow from a corner or an edge (amp glow, the Moon), bands along
  one axis (flats, sensor), slow blobs;
- sometimes dust donuts (rings 15 to 100 px, darker), walking noise (streaks along one direction), stacking edges (a
  band along one or two sides, another level and colour), a faint satellite trail left by the rejection;
- the sky's tint: R - G about -0.13, B - G about -0.05 (spread as measured).

    python defects.py --n 48 [--seed 1] [--size 2048] [--out <ESSAIS>/starRemoval/backgrounds_defects]

Image i of seed s is background i of seed s (the first set's, same draw) plus its defects: bgd_<s>_<i>.fits (3, H, W)
float32, bgd_<s>_<i>.json (what was drawn), previews/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from backgrounds import background, power_field, preview

OUT = (Path(__import__("os").environ.get("ESSAIS", "essais")) / "starRemoval" / "backgrounds_defects")


def loguniform(rng, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def band_noise(rng, n: int, scale: float) -> np.ndarray:
    """Noise of about ``scale`` px: white noise blurred at ``scale`` minus blurred at 4 x ``scale``, std 1."""
    w = rng.standard_normal((n, n)).astype(np.float32)
    f = cv2.GaussianBlur(w, (0, 0), scale) - cv2.GaussianBlur(w, (0, 0), 4 * scale)
    return f / (f.std() + 1e-12)


def block_spread(L: np.ndarray, b: int = 64) -> float:
    n = L.shape[0] // b
    m = np.median(L[:n * b, :n * b].reshape(n, b, n, b).transpose(0, 2, 1, 3).reshape(n, n, -1), -1)
    return float(np.percentile(m, 95) - np.percentile(m, 5))


def large_scale(rng, n: int, info: dict) -> np.ndarray:
    """The large-scale defects (luminance, 2D), mixed at random, before their overall level is set."""
    yy, xx = (np.mgrid[0:n, 0:n] / n - 0.5).astype(np.float32)
    parts = []
    a = rng.uniform(0, 2 * np.pi)                                  # residual gradient: a plane and a curve
    t = xx * np.cos(a) + yy * np.sin(a)
    parts.append(("gradient", t + float(rng.uniform(-1, 1)) * t ** 2))
    if rng.random() < 0.6:                                         # vignetting, off-centre, either sign
        cx, cy = rng.uniform(-0.15, 0.15, 2)
        r2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / 0.5
        parts.append(("vignetting", float(rng.choice([-1, 1])) * r2 ** float(rng.uniform(1.0, 2.5))))
    if rng.random() < 0.35:                                        # a glow from a corner or an edge
        px, py = rng.choice([-0.5, 0.5]), rng.choice([-0.5, 0.5])
        if rng.random() < 0.5:
            px = float(rng.uniform(-0.5, 0.5))
        d = np.hypot(xx - px, yy - py)
        parts.append(("glow", np.exp(-d / float(rng.uniform(0.08, 0.4))) * float(rng.uniform(1.0, 3.0))))
    if rng.random() < 0.3:                                         # bands along one axis (flats, the sensor)
        prof = power_field(rng, n, 3.0)[0] if rng.random() < 0.5 else power_field(rng, n, 3.0)[:, 0]
        prof = cv2.GaussianBlur(prof[None], (0, 0), float(rng.uniform(4, 40)))[0] if prof.ndim == 1 else prof
        prof = (prof - prof.mean()) / (prof.std() + 1e-12)
        parts.append(("bands", (prof[None, :] if rng.random() < 0.7 else prof[:, None]) * np.ones((n, n), np.float32)))
    if rng.random() < 0.5:                                         # slow blobs
        parts.append(("blobs", power_field(rng, n, 4.0, kmin=2.0)))
    out = np.zeros((n, n), np.float32)
    for name, p in parts:
        p = (p - np.median(p)) / (np.percentile(p, 95) - np.percentile(p, 5) + 1e-12)
        w = float(rng.uniform(0.3, 1.0))
        out += w * p
        info.setdefault("large_parts", []).append(name)
    return out


def defects(rng, n: int) -> tuple[np.ndarray, dict]:
    """The defects layer (3, n, n), added to a background."""
    info: dict = {}
    out = np.zeros((3, n, n), np.float32)
    # grain, per colour
    g = loguniform(rng, 0.003, 0.04)
    for c in range(3):
        out[c] += g * cv2.GaussianBlur(rng.standard_normal((n, n)).astype(np.float32), (0, 0), 0.7) / 0.4
    info["grain"] = round(g, 3)
    # colour blotches: a luminance part and a part of each colour of its own
    s = loguniform(rng, 6, 30)
    m = loguniform(rng, 0.015, 0.12)
    chroma = float(rng.uniform(0.3, 1.0))
    lum = band_noise(rng, n, s)
    for c in range(3):
        out[c] += m * (lum + chroma * band_noise(rng, n, s * float(rng.uniform(0.7, 1.4))))
    info.update(blotch_px=round(s, 1), blotch=round(m, 3), blotch_chroma=round(chroma, 2))
    # the large scale, its 64 px block spread drawn as measured, slightly coloured
    ls = large_scale(rng, n, info)
    target = loguniform(rng, 0.02, 0.35)
    ls *= target / max(block_spread(ls), 1e-9)
    tint_ls = np.array([1 + rng.uniform(-0.3, 0.3), 1.0, 1 + rng.uniform(-0.3, 0.3)], np.float32)
    out += tint_ls[:, None, None] * ls[None]
    info["large"] = round(target, 3)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    if rng.random() < 0.15:                                        # dust donuts: darker rings
        k = int(rng.integers(1, 6))
        for _ in range(k):
            cx, cy = rng.uniform(0, n, 2)
            r0 = loguniform(rng, 15, 100)
            r = np.hypot(xx - cx, yy - cy) / r0
            ring = np.exp(-((r - 0.8) / 0.25) ** 2) + 0.4 * (r < 0.8)
            out -= loguniform(rng, 0.05, 0.6) * ring[None]
        info["donuts"] = k
    if rng.random() < 0.10:                                        # walking noise: streaks along one direction
        length = int(loguniform(rng, 20, 120))
        ang = float(rng.uniform(0, 180))
        ker = np.zeros((length, length), np.float32)
        ker[length // 2, :] = 1
        ker = cv2.warpAffine(ker, cv2.getRotationMatrix2D((length / 2, length / 2), ang, 1.0), (length, length))
        ker /= ker.sum()
        amp = loguniform(rng, 0.03, 0.2)
        for c in range(3):
            st = cv2.filter2D(rng.standard_normal((n, n)).astype(np.float32), -1, ker)
            out[c] += amp * st / (st.std() + 1e-12)
        info["walking"] = round(amp, 3)
    if rng.random() < 0.30:                                        # stacking edges: a band along one or two sides
        sides = rng.choice(4, int(rng.integers(1, 3)), replace=False)
        for side in sides:
            w = int(loguniform(rng, 5, 80))
            band = np.zeros((n, n), np.float32)
            prof = np.clip(1 - np.arange(w + 8) / w, 0, 1) ** float(rng.uniform(0.3, 3.0))
            if side == 0:
                band[:w + 8, :] = prof[:, None]
            elif side == 1:
                band[-(w + 8):, :] = prof[::-1, None]
            elif side == 2:
                band[:, :w + 8] = prof[None, :]
            else:
                band[:, -(w + 8):] = prof[None, ::-1]
            col = rng.normal(0, 1, 3).astype(np.float32) * loguniform(rng, 0.1, 1.5)
            out += col[:, None, None] * band[None]
        info["edges"] = len(sides)
    if rng.random() < 0.05:                                        # a satellite trail the rejection left
        a = rng.uniform(0, np.pi)
        cx, cy = rng.uniform(0, n, 2)
        d = np.abs((xx - cx) * np.sin(a) - (yy - cy) * np.cos(a))
        out += loguniform(rng, 0.1, 1.0) * np.exp(-(d / float(rng.uniform(0.8, 2.5))) ** 2)[None]
        info["trail"] = True
    tint = np.array([rng.normal(-0.13, 0.12), 0.0, rng.normal(-0.05, 0.07)], np.float32)
    out += tint[:, None, None]
    info["tint"] = [round(float(tint[0]), 3), round(float(tint[2]), 3)]
    return out, info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    from astropy.io import fits
    out = Path(a.out)
    (out / "previews").mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("bgd_*.fits")) + list(out.glob("bgd_*.json")) + list((out / "previews").glob("bgd_*")):
        old.unlink()
    for i in range(a.n):
        t = time.time()
        bg, info = background(np.random.default_rng([a.seed, i]), a.size)      # the first set's image i
        d, dinfo = defects(np.random.default_rng([a.seed, i, 1]), a.size)
        rgb = bg + d
        name = f"bgd_{a.seed}_{i:04d}"
        fits.writeto(out / f"{name}.fits", rgb, overwrite=True)
        (out / f"{name}.json").write_text(json.dumps({**info, "defects": dinfo}))
        cv2.imwrite(str(out / "previews" / f"{name}.jpg"), preview(rgb), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"{name} {time.time() - t:4.1f} s {dinfo}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
