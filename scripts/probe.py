# ======================================================================
# probe.py
# Rôle     : Regarde les instantanés d'un entraînement en cours sur de vrais fonds.
# Étape    : 1. Entraînement sur données synthétiques seules (E2 / E3, 7-9 nov) (voir scripts/LISEZMOI.md)
# Entrées  : générateur : graines + catalogue.json (fiches d'instruments publiques) ; aucune image
# Sorties  : paires synthétiques (input/target/info.json + manifest.json), réseau E3
# Exemple  : python probe.py --run <ESSAIS>/starRemoval/train/D [--every 45]
# ======================================================================
"""A running training's checkpoints on real backgrounds, while it trains (on the GPU: a few seconds each; the processor
feeds the training).

Every ``--every`` minutes: the checkpoints copied to <run>/snap/step_<n>.pt not yet measured, then what each leaves of
stars added to three of our real training images (not the test images, kept for the final judgement): a dense Milky Way field, a
nebula, a galaxy field. Two ways (diagnose.py gap): our stars (stars.render) and real stars cut from our images, the
residual per added star above 15 sigmas, its median and its dark part. The current network and the starting point are
measured once, as references. Stops when the run's log says done.

    python probe.py --run <ESSAIS>/starRemoval/train/D [--every 45]

Log: <run>/probe.txt, <run>/probe.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from diagnose import TRAIN_REAL, paste_real, per_star, stamps_library
from evaluate import get_models
from models import apply
from stars import render

FIELDS = ("set_22_dslr_lens_canonra_ngc7000_135mm", "set_44_dslr_refractor_canon40da_horsehead",
          "set_31_smart_seestar_s30_m51")


def backgrounds(n: int = 1024):
    out = []
    for f in FIELDS:
        img = np.asarray(np.load(TRAIN_REAL / f / "input.npy"), np.float32)
        _c, H, W = img.shape
        y0, x0 = (H - n) // 2, (W - n) // 2
        out.append((f, np.ascontiguousarray(img[:, y0:y0 + n, x0:x0 + n])))
    return out


def ours_backgrounds(n: int = 1024):
    """diagnose.py gap's own: our backgrounds (held-out targets, denoised version) and our stars, the same draws, so
    that the numbers compare with its SS way."""
    from train import PAIRS, pair_dirs, saturation_limit
    _tr, held = pair_dirs(PAIRS, 25)
    out = []
    for bi, d in enumerate(held[::8][:6]):
        t = np.minimum(np.asarray(np.load(d / "target_denoised.npy", mmap_mode="r")[:, :n, :n], np.float32),
                       saturation_limit(d))
        syn, keep, _ = render(np.random.default_rng([4242, bi]), n, {})
        out.append((t, syn, keep))
    return out


def measure(model, backs, adds, dev, ours=()) -> dict:
    name, net, (tile, ov) = model
    res = {"S": [], "R": [], "SS": []}
    for (_f, bg), (syn, keep, real) in zip(backs, adds):
        base = apply(net, bg, dev, tile, ov)
        for k, added, kp in (("S", syn, keep), ("R", real, 0.0)):
            res[k] += per_star(added, apply(net, bg + added, dev, tile, ov) - base - kp)
    for bg, syn, keep in ours:
        res["SS"] += per_star(syn, apply(net, bg + syn, dev, tile, ov) - apply(net, bg, dev, tile, ov) - keep)
    res = {k: v for k, v in res.items() if v}
    out = {}
    for k, v in res.items():
        V = np.array(v)
        out[k] = {"residual_median": round(float(np.median(V[:, 1])), 4), "dark_median": round(float(np.median(V[:, 2])), 4),
                  "stars": len(V)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--every", type=float, default=45.0)
    ap.add_argument("--start", default="<ESSAIS>/starRemoval/train/C_long/best.pt")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(argv)
    run = Path(a.run)
    snap = run / "snap"
    snap.mkdir(exist_ok=True)
    logf = run / "probe.txt"

    def log(msg):
        line = f"{datetime.now():%H:%M:%S} {msg}"
        print(line, flush=True)
        with open(logf, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    lib, dens = stamps_library()
    backs = backgrounds()
    adds = []
    for i, (_f, bg) in enumerate(backs):
        rng = np.random.default_rng([555, i])
        syn, keep, _ = render(rng, bg.shape[1], {})
        adds.append((syn, keep, paste_real(rng, bg.shape[1], lib, dens)))
    ours = ours_backgrounds()
    hist = {}
    for spec in ("current", f"start={a.start}"):
        m = get_models([spec], a.device)[0]
        hist[m[0]] = measure(m, backs, adds, a.device, ours)
        log(f"{m[0]:>14}: ours {hist[m[0]]['S']} | real {hist[m[0]]['R']} | ours on ours {hist[m[0]]['SS']}")
        del m
    sheet(get_models(["current"], a.device)[0], backs, adds, a.device, run / "probe_current.jpg")
    while True:
        done = "done:" in (run / "log.txt").read_text(encoding="utf-8")
        for p in sorted(snap.glob("step_*.pt"), key=lambda q: int(q.stem.split("_")[1])):
            if p.stem in hist:
                continue
            m = get_models([f"{p.stem}={p}"], a.device)[0]
            hist[p.stem] = measure(m, backs, adds, a.device, ours)
            sheet(m, backs, adds, a.device, run / f"probe_{p.stem}.jpg")
            log(f"{p.stem:>14}: ours {hist[p.stem]['S']} | real {hist[p.stem]['R']} | ours on ours {hist[p.stem]['SS']}")
            del m
            (run / "probe.json").write_text(json.dumps(hist, indent=1))
        if done:
            return 0
        time.sleep(a.every * 60)


def sheet(model, backs, adds, dev, path):
    """At 100 %: for each background, a 384 px crop with the added stars (ours, then real ones) and the model's
    output, one row per background."""
    import cv2
    from evaluate import stretch
    _name, net, (tile, ov) = model
    rows = []
    for (_f, bg), (syn, _keep, real) in zip(backs, adds):
        y0 = x0 = bg.shape[1] // 2 - 192
        row = []
        for added in (syn, real):
            o = apply(net, bg + added, dev, tile, ov)
            for img in (bg + added, o):
                row.append(stretch(img[:, y0:y0 + 384, x0:x0 + 384]))
        rows.append(np.hstack(row))
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    sys.exit(main())
