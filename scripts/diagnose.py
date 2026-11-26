# ======================================================================
# diagnose.py
# Rôle     : Diagnostic de l'écart entre un réseau et le réseau de la chaîne, avant tout changement.
# Étape    : 1. Entraînement sur données synthétiques seules (E2 / E3, 7-9 nov) (voir scripts/LISEZMOI.md)
# Entrées  : générateur : graines + catalogue.json (fiches d'instruments publiques) ; aucune image
# Sorties  : paires synthétiques (input/target/info.json + manifest.json), réseau E3
# Exemple  : python diagnose.py units|gap|valset [--models C_long=...best.pt current] [--device cuda:0]
# ======================================================================
"""Where the gap to the chain's current network comes from, before changing anything.

1. ``units``: the same measures on our real images and on our generated pairs (both versions): the noise left (a
   robust sigma of neighbour differences in the darker half), the star densities above 3, 10, 100 sigmas per megapixel,
   their FWHM and ellipticity (star_measure.measure). If our pairs do not speak the real images' units, the network
   meets on them something it never saw.
2. ``gap``: four ways to add stars, and what a network leaves of them, star by star:
   - SS our stars (stars.render) on our backgrounds (held-out targets, denoised version);
   - RS real stars on our backgrounds: stamps cut around the stars of our 23 training images (their local background
     taken off, a soft round edge), pasted at random places;
   - SR our stars on our 6 real test images (denoised stacks);
   - RR real stars on the real test images.
   The residual is (output with the added stars) - (output without them) - (the faint light the target keeps); per
   added star brighter than 15 sigmas: |residual| / its light in a window that grows with its brightness, and the dark
   part (residual below 0) alone. The models' ratios per way say which one, the stars or the backgrounds, makes the gap.
3. ``valset``: a fixed validation set, the same for every run: tiles of 384 px of the held-out pairs, both versions,
   half at random places and half on the brightest stars; its spread (bootstrap) per model.

    python diagnose.py units|gap|valset [--models C_long=...best.pt current] [--device cuda:0]

Outputs <ESSAIS>/starRemoval/diagnose/: units.json, gap.json, gap.jpg, valset.npz, valset.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from evaluate import get_models, real_case, stretch
from models import apply, fwd
from star_measure import measure
from stars import render
from train import PAIRS, pair_dirs, saturation_limit

HERE = Path(__file__).resolve().parent
TRAIN_REAL = HERE / "data" / "train_with_stars"
CASES = ("heart", "iris", "m51", "m8", "rosette", "seestar")
OUT = (Path(__import__("os").environ.get("ESSAIS", "essais")) / "starRemoval" / "diagnose")
LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)


def lum(a):
    return np.tensordot(LUM, a, 1)


def noise_sigma(img: np.ndarray) -> float:
    """The noise left: a robust sigma of horizontal neighbour differences, in the darker half of the image."""
    L = lum(img)
    d = L[:, 1:] - L[:, :-1]
    dark = L[:, :-1] < np.median(L)
    v = d[dark]
    return float(1.4826 * np.median(np.abs(v - np.median(v))) / np.sqrt(2))


def real_train():
    for d in sorted(p for p in TRAIN_REAL.iterdir() if (p / "input.npy").exists()):
        yield d.name, np.asarray(np.load(d / "input.npy"), np.float32)


# ---------------------------------------------------------------------------------------------------- 1. the units
def units(a) -> dict:
    res = {"real_test": {}, "real_train": {}, "pairs_noisy": {}, "pairs_denoised": {}}

    def row(img):
        m = measure("", img, None)
        return {"noise": round(noise_sigma(img), 3), "density": m["density"], "fwhm": m.get("fwhm"),
                "ellipticity": m.get("ellipticity"), "isolated": m["isolated"], "top": m["top"]}

    for c in CASES:
        res["real_test"][c] = row(real_case(c))
        print("test", c, res["real_test"][c], flush=True)
    for name, img in real_train():
        res["real_train"][name] = row(img)
        print("train", name, res["real_train"][name], flush=True)
    _tr, held = pair_dirs(Path(a.pairs), 25)
    for d in held[::4]:
        for v in ("noisy", "denoised"):
            img = np.asarray(np.load(d / f"input_{v}.npy"), np.float32)
            res[f"pairs_{v}"][f"{d.parent.name}/{d.name}"] = row(img)
        print("pair", d.name, res["pairs_denoised"][f"{d.parent.name}/{d.name}"], flush=True)
    summ = {}
    for k, rows in res.items():
        R = list(rows.values())
        summ[k] = {"noise": np.percentile([r["noise"] for r in R], [10, 50, 90]).round(3).tolist(),
                   "density_3": np.percentile([r["density"]["3"] for r in R], [10, 50, 90]).round(0).tolist(),
                   "density_100": np.percentile([r["density"]["100"] for r in R], [10, 50, 90]).round(0).tolist(),
                   "fwhm_median": np.percentile([r["fwhm"][1] for r in R if r["fwhm"]], [10, 50, 90]).round(2).tolist(),
                   "ellipticity_median": np.percentile([r["ellipticity"][1] for r in R if r["ellipticity"]],
                                                       [10, 50, 90]).round(3).tolist()}
    return {"summary": summ, "details": res}


# ------------------------------------------------------------------------------------------------------- 2. the gap
def stamps_library(per_image: int = 400, bright: int = 15):
    """Real stars cut from our 23 training images: (stamp (3, s, s), peak) with the local background off and a soft
    round edge; ordinary stars in 65 px, the brightest of each image in 193 px. Plus their density above 15 sigmas
    per megapixel in each image."""
    from star_measure import local_background
    lib, dens = [], []
    for _name, img in real_train():
        # the image's background (64 px blocks, stars out by the median, smoothly enlarged) off each colour
        img = img - np.stack([local_background(img[c]) for c in range(3)])
        L = lum(img)
        mx = ndimage.maximum_filter(L, 7)
        pk = (L == mx) & (L > 15)
        H, W = L.shape
        ys, xs = np.nonzero(pk)
        dens.append(len(ys) / (H * W / 1e6))
        order = np.argsort(-L[ys, xs])
        rng = np.random.default_rng(len(lib))
        pick = list(order[:bright]) + list(rng.choice(order[bright:], min(per_image, max(len(order) - bright, 0)),
                                                      replace=False))
        for j, i in enumerate(pick):
            r = 96 if j < bright else 32
            y, x = ys[i], xs[i]
            if y < r or x < r or y >= H - r or x >= W - r:
                continue
            st = img[:, y - r:y + r + 1, x - r:x + r + 1].copy()
            # the star's own footprint: what stands above 1 sigma connected to its centre, widened by 3 px, a soft
            # edge; the nebula and the other stars of the stamp stay out
            lab, _n = ndimage.label(lum(st) > 1.0)
            foot = lab == lab[r, r]
            if foot[0].any() or foot[-1].any() or foot[:, 0].any() or foot[:, -1].any():
                continue                                                  # larger than its stamp: a nebula's part
            soft = cv2.GaussianBlur(ndimage.binary_dilation(foot, iterations=3).astype(np.float32), (0, 0), 1.5)
            lib.append(((st * soft[None]).astype(np.float32), float(L[y, x])))
    return lib, dens


def paste_real(rng, n: int, lib, dens) -> np.ndarray:
    out = np.zeros((3, n, n), np.float32)
    k = int(rng.choice(dens) * n * n / 1e6)
    for _ in range(k):
        st, _p = lib[int(rng.integers(len(lib)))]
        s = st.shape[1] // 2
        y, x = int(rng.integers(n)), int(rng.integers(n))
        y0, y1, x0, x1 = max(0, y - s), min(n, y + s + 1), max(0, x - s), min(n, x + s + 1)
        out[:, y0:y1, x0:x1] += st[:, y0 - (y - s):y1 - (y - s), x0 - (x - s):x1 - (x - s)]
    return out


def per_star(added: np.ndarray, res: np.ndarray) -> list[tuple[float, float, float]]:
    """(peak, |residual| / light, dark part / light) for each added star above 15 sigmas, alone in its window."""
    A = lum(added)
    R = lum(res)
    mx = ndimage.maximum_filter(A, 9)
    ys, xs = np.nonzero((A == mx) & (A > 15))
    H, W = A.shape
    rows = []
    for y, x in zip(ys, xs):
        p = float(A[y, x])
        r = 6 if p < 100 else 14 if p < 1000 else 30
        if y < r or x < r or y >= H - r or x >= W - r:
            continue
        a = A[y - r:y + r + 1, x - r:x + r + 1]
        if (a.max() > p) or ((mx[y - r:y + r + 1, x - r:x + r + 1] == a) & (a > 0.3 * p)).sum() > 1:
            continue                                                      # not alone in its window
        q = R[y - r:y + r + 1, x - r:x + r + 1]
        light = float(np.clip(a, 0, None).sum())
        rows.append((p, float(np.abs(q).sum() / light), float(np.clip(-q, 0, None).sum() / light)))
    return rows


def gap(a) -> dict:
    models = get_models(a.models, a.device)
    lib, dens = stamps_library()
    print(f"{len(lib)} real stars cut, density above 15 sigmas: median {np.median(dens):.0f} per Mpx", flush=True)
    _tr, held = pair_dirs(PAIRS, 25)
    n = 1024
    backs = []
    for d in held[::8][:6]:
        t = np.minimum(np.asarray(np.load(d / "target_denoised.npy", mmap_mode="r")[:, :n, :n], np.float32),
                       saturation_limit(d))
        backs.append(("ours", d.name, t))
    for c in CASES:
        img = real_case(c)
        _c, H, W = img.shape
        y0, x0 = (H - n) // 2, (W - n) // 2
        backs.append(("real", c, np.ascontiguousarray(img[:, y0:y0 + n, x0:x0 + n])))
    rows = {m[0]: {w: [] for w in ("SS", "RS", "SR", "RR")} for m in models}
    sheet = []
    for bi, (kind, name, bg) in enumerate(backs):
        rng = np.random.default_rng([4242, bi])
        info: dict = {}
        syn, keep, _ = render(rng, n, info)
        real = paste_real(rng, n, lib, dens)
        base = {m[0]: apply(m[1], bg, a.device, *m[2]) for m in models}
        for stars_kind, added, kp in (("S", syn, keep), ("R", real, np.zeros_like(real))):
            way = stars_kind + ("S" if kind == "ours" else "R")
            line = [stretch(bg[:, :384, :384]), stretch((bg + added)[:, :384, :384])]
            for mname, net, (tile, ov) in models:
                o = apply(net, bg + added, a.device, tile, ov)
                res = o - base[mname] - kp
                rows[mname][way] += per_star(added, res)
                line.append(stretch(o[:, :384, :384]))
            if bi % 3 == 0:
                sheet.append(np.hstack(line))
        print(kind, name, {m: {w: len(v) for w, v in r.items()} for m, r in rows.items()}, flush=True)
    summ = {}
    for mname, r in rows.items():
        summ[mname] = {}
        for way, v in r.items():
            if not v:
                continue
            V = np.array(v)
            summ[mname][way] = {"stars": len(V),
                                "residual_median": round(float(np.median(V[:, 1])), 4),
                                "residual_p90": round(float(np.percentile(V[:, 1], 90)), 4),
                                "dark_median": round(float(np.median(V[:, 2])), 4),
                                "residual_bright_median": round(float(np.median(V[V[:, 0] > 300, 1])), 4)
                                if (V[:, 0] > 300).any() else None}
    cv2.imwrite(str(OUT / "gap.jpg"), np.vstack(sheet), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return {"summary": summ, "models": [m[0] for m in models], "real_stars_cut": len(lib)}


# -------------------------------------------------------------------------------------------- 3. the validation set
def valset(a) -> dict:
    _tr, held = pair_dirs(Path(a.pairs), 25)
    P, per = 384, 8
    X, Y, V = [], [], []
    for i, d in enumerate(held):
        lim = saturation_limit(d)
        for vi, v in enumerate(("noisy", "denoised")):
            x = np.load(d / f"input_{v}.npy", mmap_mode="r")
            t = np.load(d / f"target_{v}.npy", mmap_mode="r")
            H, W = x.shape[1:]
            rng = np.random.default_rng([99, i, vi])
            S = lum(np.asarray(x[:, ::8, ::8], np.float32) - np.minimum(np.asarray(t[:, ::8, ::8], np.float32),
                                                                         lim[:, :1, :1]))
            top = np.argsort(S.ravel())[::-1]
            spots = [(int(rng.integers(0, H - P)), int(rng.integers(0, W - P))) for _ in range(per // 2)]
            for k in top[:200:50][:per // 2]:
                cy, cx = divmod(int(k), S.shape[1])
                spots.append((int(np.clip(cy * 8 - P // 2, 0, H - P)), int(np.clip(cx * 8 - P // 2, 0, W - P))))
            for y, x0 in spots:
                xi = np.asarray(x[:, y:y + P, x0:x0 + P], np.float32)
                yi = np.minimum(np.asarray(t[:, y:y + P, x0:x0 + P], np.float32), lim)
                X.append(fwd(xi).astype(np.float16))
                Y.append(fwd(yi).astype(np.float16))
                V.append(vi)
    a.val_file = str(OUT / f"valset_{Path(a.pairs).name}.npz")
    np.savez(a.val_file, x=np.stack(X), y=np.stack(Y), version=np.array(V, np.int8))
    print(f"{len(X)} tiles of {P} px -> {a.val_file}", flush=True)
    return spread(a)


def spread(a) -> dict:
    """Each model on the fixed set: the l1 (as train.validate, x1000) per tile; its mean and the bootstrap spread of
    the mean, overall and per version; and the same on 96 tiles drawn at random (the size of the old set)."""
    import torch
    z = np.load(a.val_file or OUT / "valset.npz")
    X, Yt, V = z["x"], z["y"], z["version"]
    models = get_models([m for m in a.models if m != "current"], a.device)
    out = {}
    for name, net, _ in models:
        e = []
        with torch.no_grad():
            for i in range(0, len(X), 8):
                x = torch.from_numpy(X[i:i + 8].astype(np.float32)).to(a.device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = net(x).float()
                e += (o - torch.from_numpy(Yt[i:i + 8].astype(np.float32)).to(a.device)).abs().mean((1, 2, 3)).tolist()
        e = np.array(e) * 1e3
        rng = np.random.default_rng(0)
        boot = [e[rng.integers(0, len(e), len(e))].mean() for _ in range(500)]
        small = [e[rng.choice(len(e), 96, replace=False)].mean() for _ in range(500)]
        out[name] = {"tiles": len(e), "l1": round(float(e.mean()), 3),
                     "l1_noisy": round(float(e[V == 0].mean()), 3), "l1_denoised": round(float(e[V == 1].mean()), 3),
                     "spread_full_set_p5_p95": np.percentile(boot, [5, 95]).round(3).tolist(),
                     "spread_96_tiles_p5_p95": np.percentile(small, [5, 95]).round(3).tolist()}
        print(name, out[name], flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=["units", "gap", "valset", "spread"])
    ap.add_argument("--models", nargs="+", default=["C_long=<ESSAIS>/starRemoval/train/C_long/best.pt",
                                                     "C=<ESSAIS>/starRemoval/train/C/best.pt", "current"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pairs", default=str(PAIRS), help="the pairs the validation set is cut from")
    ap.add_argument("--val-file", default="", help="spread: the validation set to measure on")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"units": units, "gap": gap, "valset": valset, "spread": spread}[a.what](a)
    (OUT / f"{a.what}.json").write_text(json.dumps(res, indent=1))
    print(f"-> {OUT / (a.what + '.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
