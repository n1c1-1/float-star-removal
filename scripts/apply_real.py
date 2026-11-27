# ======================================================================
# apply_real.py
# Role     : Applies a chosen network to real images (final comparison images).
# Step     : 1. Training on synthetic data only (E2 / E3, 7-9 Nov) (see scripts/README.md)
# Inputs   : generator: seeds + catalogue.json (public instrument sheets); no image
# Outputs  : synthetic pairs (input/target/info.json + manifest.json), network E3
# Example  : python apply_real.py --model <ESSAIS>/starRemoval/train/<tag>/best.pt [--device cuda:0]
# ======================================================================
"""The final pictures: the chosen network on our real images (the 6 test images, and our 23 training images, used
only for the measures), next to the network the chain ships now; stretched sheets to look at.

Per image (<out>/<name>.jpg), one stretch for all its panels (asinh from the sky's median, set on the original):
- the whole image reduced: the original, without its stars (the chosen network), its stars alone, and without its
  stars by the current network;
- 100 % crops of 512 px around the brightest star and in the densest part: the original, the chosen network, the
  current network.
And <out>/index.jpg: the whole-image rows of all the images.

    python apply_real.py --model <ESSAIS>/starRemoval/train/<tag>/best.pt [--device cuda:0]
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

from evaluate import real_case
from models import INFER, apply, load

DATA = Path(__file__).resolve().parent / "data"
LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)


def images():
    for case in ("heart", "iris", "m51", "m8", "rosette", "seestar"):
        yield f"test_{case}", lambda c=case: real_case(c)
    fields = json.loads((DATA / "train_with_stars" / "fields.json").read_text())
    for name in sorted(fields):
        f = DATA / "train_with_stars" / name / "input.npy"
        if f.exists():
            yield name, lambda p=f: np.asarray(np.load(p), np.float32)


class Stretch:
    def __init__(self, img: np.ndarray):
        L = (img[:, ::8, ::8] * LUM[:, None, None]).sum(0)
        self.m = float(np.median(L))
        d = L - self.m
        self.s = max(1.4826 * float(np.median(np.abs(d))), 0.05)
        self.top = float(np.arcsinh((np.percentile(L, 99.95) - self.m) / self.s))

    def __call__(self, a: np.ndarray, zero: bool = False) -> np.ndarray:
        base = 0.0 if zero else self.m
        x = np.arcsinh(np.clip(a - base + 2 * self.s, 0, None) / self.s) / max(self.top, 1e-6)
        return np.ascontiguousarray((np.clip(x, 0, 1).transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8))


def label(t, text):
    cv2.putText(t, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return t


def reduced(a: np.ndarray, width: int = 640) -> np.ndarray:
    _c, H, W = a.shape
    k = max(1, W // width)
    h, w = H // k * k, W // k * k
    return a[:, :h, :w].reshape(3, h // k, k, w // k, k).mean(axis=(2, 4))


def places(img: np.ndarray) -> list[tuple[str, int, int]]:
    L = (img * LUM[:, None, None]).sum(0)
    H, W = L.shape
    k = 128
    b = L[:H // k * k, :W // k * k].reshape(H // k, k, W // k, k)
    by, bx = np.unravel_index(np.argmax(b.max(axis=(1, 3))), b.shape[::2])
    cnt = (b > np.percentile(L, 99)).mean(axis=(1, 3))
    dy, dx = np.unravel_index(np.argmax(cnt), cnt.shape)
    return [("brightest star", bx * k + k // 2, by * k + k // 2), ("densest", dx * k + k // 2, dy * k + k // 2)]


def crop(a, x, y, s=512):
    _c, H, W = a.shape
    s = min(s, H, W)
    x0, y0 = int(np.clip(x - s // 2, 0, W - s)), int(np.clip(y - s // 2, 0, H - s))
    return a[:, y0:y0 + s, x0:x0 + s]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="<ESSAIS>/starRemoval/final")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    net, cfg = load(a.model, a.device)
    tile, ov = INFER[cfg["model"]]
    sys.path.insert(0, "<FLOAT>")
    from astro_gpu.stages.star_removal import network
    cur = network(torch.device(a.device))
    rows = []
    for name, get in images():
        t = time.time()
        img = get()
        s1 = apply(net, img, a.device, tile, ov)
        s0 = apply(cur, img, a.device, 256, 64)
        st = Stretch(img)
        whole = [label(st(reduced(img)), f"{name}: original"), label(st(reduced(s1)), "without stars (new)"),
                 label(st(reduced(img - s1), zero=True), "stars (new)"), label(st(reduced(s0)), "without stars (current)")]
        row = np.hstack(whole)
        parts = []
        for pname, x, y in places(img):
            parts += [label(st(crop(img, x, y)), f"{pname}: original"), label(st(crop(s1, x, y)), "new"),
                      label(st(crop(s0, x, y)), "current")]
        hh = max(p.shape[0] for p in parts)
        parts = [cv2.copyMakeBorder(p, 0, hh - p.shape[0], 0, 0, cv2.BORDER_CONSTANT) for p in parts]
        detail = np.hstack(parts)
        wmax = max(row.shape[1], detail.shape[1])
        pad = lambda m: cv2.copyMakeBorder(m, 0, 0, 0, wmax - m.shape[1], cv2.BORDER_CONSTANT)  # noqa: E731
        cv2.imwrite(str(out / f"{name}.jpg"), np.vstack([pad(row), pad(detail)]), [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append(row)
        print(f"{name}: {img.shape[2]} x {img.shape[1]} in {time.time() - t:.0f} s -> {out / (name + '.jpg')}",
              flush=True)
    wmax = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, wmax - r.shape[1], cv2.BORDER_CONSTANT) for r in rows]
    for i in range(0, len(rows), 8):
        cv2.imwrite(str(out / f"index_{i // 8 + 1}.jpg"), np.vstack(rows[i:i + 8]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return 0


if __name__ == "__main__":
    sys.exit(main())
