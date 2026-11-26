# ======================================================================
# evaluate.py
# Rôle     : Banc de mesure des réseaux : mêmes mesures pour chacun, sur ce qu'aucun n'a vu à l'entraînement.
# Étape    : 1. Entraînement sur données synthétiques seules (E2 / E3, 7-9 nov) (voir scripts/LISEZMOI.md)
# Entrées  : générateur : graines + catalogue.json (fiches d'instruments publiques) ; aucune image
# Sorties  : paires synthétiques (input/target/info.json + manifest.json), réseau E3
# Exemple  : python evaluate.py A=<ESSAIS>/starRemoval/train/A/best.pt B=... C=... current [--device cuda:0]
# ======================================================================
"""The bench of the star removal networks: the same measures for each, on what none of them was trained on.

1. The held-out generated pairs (the last ones of each set, both versions), where the truth is exact; in noise sigmas:
   - under the stars (where they bring more than half a sigma): the mean error, and the share of their light left;
   - off the stars: the mean error (in the noisy version, whether the noise comes back unchanged);
   - the stars left: maxima of (output - truth) above 1 sigma (noisy: 3 sigma), per megapixel;
   - the halos left: the mean of (output - truth) from 50 to 500 px around the three brightest stars;
   - the grid: the spectrum of (output - truth) at the multiples of 1/16 px^-1 over its neighbours (a tiling pattern).
2. Our 6 test images (denoised stacks): synthetic stars added to each (stars.py, an instrument drawn), the network run
   with and without them; what it leaves of the added stars is (output with) - (output without) - (the light of the
   unseen ones the target keeps): their light left, and the maxima left. Plus a sheet to look at.
3. The speed: seconds per megapixel on the device and on the processor (all its threads).

Models: checkpoints of train.py (name=path), and ``current`` for the network the chain ships now.

    python evaluate.py A=<ESSAIS>/starRemoval/train/A/best.pt B=... C=... current [--device cuda:0]
                       [--held 6] [--speed] [--out <ESSAIS>/starRemoval/eval/<name>]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from models import INFER, apply, load
from train import PAIRS, STAR_PX, pair_dirs, saturation_limit

DATA = Path(__file__).resolve().parent / "data" / "test_with_stars"
LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)


def get_models(specs, device):
    out = []
    for s in specs:
        if s == "current":
            sys.path.insert(0, "<FLOAT>")
            from astro_gpu.stages.star_removal import network
            out.append(("current", network(torch.device(device)), (256, 64)))
        else:
            name, path = s.split("=", 1)
            net, cfg = load(path, device)
            out.append((name, net, INFER[cfg["model"]]))
    return out


def peaks(L: np.ndarray, thr: float) -> int:
    from scipy import ndimage
    mx = ndimage.maximum_filter(L, 7)
    return int(((L == mx) & (L > thr)).sum())


def grid_ratio(r: np.ndarray) -> float:
    """Power of a residual at the multiples of 1/16 px^-1 (a 16 px tiling's pattern) over the power around them."""
    L = r.mean(0)
    n = min(L.shape) // 64 * 64
    P = np.abs(np.fft.fft2(L[:n, :n] - L[:n, :n].mean())) ** 2
    k = n // 16
    on, near = [], []
    for i in range(0, n, k):
        for j in range(0, n, k):
            if i == 0 and j == 0:
                continue
            on.append(P[i, j])
            near.append(np.median(P[max(i - 3, 0):i + 4, max(j - 3, 0):j + 4]))
    return float(np.median(np.array(on) / (np.array(near) + 1e-12)))


def held_measures(out, inp, tgt, noisy: bool) -> dict:
    star = inp - tgt
    under = star.mean(0) > STAR_PX
    off = star.mean(0) < 0.02
    r = out - tgt
    Lr = (r * LUM[:, None, None]).sum(0)
    mp = r.shape[1] * r.shape[2] / 1e6
    left = float(np.clip(r, 0, None)[:, under].sum() / max(star[:, under].sum(), 1e-9))
    # the halos: around the three brightest stars (the largest star light), 50 - 500 px
    S = star.mean(0)
    idx = np.argsort(S[::8, ::8].ravel())[-3:]
    yy, xx = np.indices(S.shape)
    halo = []
    for k in idx:
        y, x = divmod(int(k), S[::8, ::8].shape[1])
        y, x = y * 8, x * 8
        d = np.hypot(yy - y, xx - x)
        ring = (d > 50) & (d < 500)
        halo.append(float(Lr[ring].mean()))
    return {"err_under": float(np.abs(r)[:, under].mean()) if under.any() else 0.0, "light_left": left,
            "err_off": float(np.abs(r)[:, off].mean()) if off.any() else 0.0,
            "stars_left_per_mpx": peaks(Lr, 3.0 if noisy else 1.0) / mp, "halo_left": float(np.mean(halo)),
            "grid": grid_ratio(r)}


def real_case(case: str) -> np.ndarray:
    from astropy.io import fits
    m = json.loads((DATA / case / "meta.json").read_text())
    a = np.asarray(fits.getdata(DATA / case / "input.fits"), np.float32)
    if a.shape[-1] == 3 and a.shape[0] != 3:
        a = a.transpose(2, 0, 1)
    return (a * m["scale"] - np.asarray(m["background"], np.float32)[:, None, None]) / m["unit"]


def inject(img: np.ndarray, seed: int, size: int = 2048):
    """Synthetic stars added to a central crop of a real image: (crop, crop + stars, the light the target keeps, the
    stars)."""
    from stars import render
    _c, H, W = img.shape
    n = min(size, H, W) // 8 * 8
    y0, x0 = (H - n) // 2, (W - n) // 2
    crop = np.ascontiguousarray(img[:, y0:y0 + n, x0:x0 + n])
    info: dict = {}
    stars, keep, _ = render(np.random.default_rng([777, seed]), n, info)
    return crop, crop + stars, keep, stars


def stretch(a):
    x = np.arcsinh(np.clip(a + 1.0, 0, None) / 0.5) / np.arcsinh(3000 / 0.5)
    return np.ascontiguousarray((np.clip(x, 0, 1).transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("models", nargs="+")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--held", type=int, default=6, help="held-out pairs per set measured")
    ap.add_argument("--speed", action="store_true")
    ap.add_argument("--pairs", default=str(PAIRS))
    ap.add_argument("--out", default="<ESSAIS>/starRemoval/eval/bench")
    ap.add_argument("--crop", type=int, default=0, help="work on this many px only (a quick check of the code)")
    ap.add_argument("--cases", default="heart,iris,m51,m8,rosette,seestar")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    models = get_models(a.models, a.device)
    res: dict = {name: {"held": [], "real": {}} for name, _, _ in models}
    # 1. held-out pairs
    _train, val = pair_dirs(Path(a.pairs), 25)
    picked = [d for s in ("clean", "defects") for d in [v for v in val if f"\\{s}\\" in str(v) or f"/{s}/" in str(v)]
              [:a.held]]
    for d in picked:
        for v in ("noisy", "denoised"):
            c = a.crop or None
            inp = np.load(d / f"input_{v}.npy", mmap_mode="r")[:, :c, :c].astype(np.float32)
            tgt = np.minimum(np.load(d / f"target_{v}.npy", mmap_mode="r")[:, :c, :c].astype(np.float32),
                             saturation_limit(d))
            for name, net, (tile, ov) in models:
                o = apply(net, inp, a.device, tile, ov)
                m = held_measures(o, inp, tgt, v == "noisy")
                res[name]["held"].append({"pair": d.parent.name + "/" + d.name, "version": v, **m})
                print(f"{name:8} {d.parent.name}/{d.name} {v:8} " + " ".join(f"{k} {x:.3f}" for k, x in m.items()),
                      flush=True)
    # 2. the test images: injected stars
    sheet = []
    for ci, case in enumerate(a.cases.split(",")):
        crop, with_stars, keep, stars = inject(real_case(case), ci, a.crop or 2048)
        under = stars.mean(0) > STAR_PX
        cs = min(512, crop.shape[1])
        row = [stretch(crop[:, :cs, :cs]), stretch(with_stars[:, :cs, :cs])]
        for name, net, (tile, ov) in models:
            o0 = apply(net, crop, a.device, tile, ov)
            o1 = apply(net, with_stars, a.device, tile, ov)
            r = o1 - o0 - keep
            Lr = (r * LUM[:, None, None]).sum(0)
            res[name]["real"][case] = {
                "light_left": float(np.clip(r, 0, None)[:, under].sum() / max(stars[:, under].sum(), 1e-9)),
                "err_under": float(np.abs(r)[:, under].mean()), "stars_left_per_mpx": peaks(Lr, 1.0) / (r[0].size / 1e6),
                "changed_off_stars": float(np.abs(o0 - crop)[:, stars.mean(0) < 0.02].mean())}
            print(f"{name:8} {case:8} " + " ".join(f"{k} {x:.4f}" for k, x in res[name]["real"][case].items()),
                  flush=True)
            row.append(stretch(o1[:, :cs, :cs]))
            cv2.putText(row[-1], name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(row[0], case, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(row[1], "+ stars", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        sheet.append(np.hstack(row))
    cv2.imwrite(str(out / "real_injected.jpg"), np.vstack(sheet), [cv2.IMWRITE_JPEG_QUALITY, 88])
    # 3. the speed
    if a.speed:
        x = np.random.default_rng(0).standard_normal((3, 1024, 1024)).astype(np.float32)
        for name, net, (tile, ov) in models:
            sp = {}
            for dev in ([a.device] if a.device != "cpu" else []) + ["cpu"]:
                n2 = net.to(dev)
                if dev == "cpu":
                    torch.set_num_threads(torch.get_num_threads())
                apply(n2, x[:, :256, :256], dev, tile, ov)
                t = time.time()
                apply(n2, x, dev, tile, ov)
                sp[dev] = round((time.time() - t) / 1.048576, 2)
            net.to(a.device)
            res[name]["seconds_per_mpx"] = sp
            print(f"{name:8} seconds per Mpx {sp}", flush=True)
    # the summary
    summ = {}
    for name in res:
        H = res[name]["held"]
        summ[name] = {f"held_{v}_{k}": round(float(np.mean([h[k] for h in H if h["version"] == v])), 4)
                      for v in ("noisy", "denoised") for k in ("err_under", "light_left", "err_off",
                                                                "stars_left_per_mpx", "halo_left", "grid")}
        R = res[name]["real"].values()
        # nanmean: a field whose injected stars leave no pixel under 0.02 has no "off the stars" measure
        summ[name].update({f"real_{k}": round(float(np.nanmean([r[k] for r in R])), 4)
                           for k in ("light_left", "err_under", "stars_left_per_mpx", "changed_off_stars")})
        if "seconds_per_mpx" in res[name]:
            summ[name]["seconds_per_mpx"] = res[name]["seconds_per_mpx"]
    (out / "results.json").write_text(json.dumps({"summary": summ, "details": res}, indent=1))
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
