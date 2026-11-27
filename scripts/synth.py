# ======================================================================
# synth.py
# Role     : Stars added on the fly to the training tiles (first version, trial).
# Step     : 1. Training on synthetic data only (E2 / E3, 7-9 Nov) (see scripts/README.md)
# Inputs   : generator: seeds + catalogue.json (public instrument sheets); no image
# Outputs  : synthetic pairs (input/target/info.json + manifest.json), network E3
# Example  : python synth.py build [--instruments 500 --places 3 --workers 2]     # -> <ESSAIS>/starRemoval/psf_bank.npz
# ======================================================================
"""Stars added on the fly to the training tiles, on the GPU: every batch meets stars it never saw (domain
randomisation: Tobin et al. 2017, arXiv:1703.06907), on our backgrounds or over the tile's own stars, whose truth (the
pair's target) stays exact.

- the shapes: a bank of PSFs computed once by our optical model (psf.psf_at) for instruments drawn as the pairs draw
  them (instruments.draw, calibrated on our images' stars: calib.py), each at a random place of its field: ``build``;
- per tile: one shape of the bank (a tile is small next to the field's variation), and its far aureole (Racine 1996,
  PASP 108, 699: (1 + (r / r0)^2)^(-gamma / 2), 0.3 - 5 % of the light, as stars.far_kernel);
- the stars: a power law of the counts, N(> peak) ~ peak^-alpha (alpha 0.55 - 0.9), from 1.5 sigmas at the peak (the
  fainter ones are the stored pairs' business: their target keeps the light of the unseen ones) to far past
  saturation; their number drawn over our images' range (diagnose.py units: 100 - 3000 per megapixel above 3 sigmas);
  colours spread as field stars around the balance's white;
- the camera: the tile's own saturation per colour (the pair's, train.saturation_limit) times 0.5 - 1, the input
  clipped there; the photon noise of the added stars in the noisy version (the pair's own photon_k), a little in the
  denoised one (what the denoiser leaves of it).

    python synth.py build [--instruments 500 --places 3 --workers 2]     # -> <ESSAIS>/starRemoval/psf_bank.npz
"""
from __future__ import annotations

import argparse
import math
import sys
from multiprocessing import get_context

import numpy as np

BANK = "<ESSAIS>/starRemoval/psf_bank.npz"
PSF = 129


def _shapes(i: int, places: int):
    import cv2
    cv2.setNumThreads(1)
    from instruments import draw
    from psf import psf_at
    rng = np.random.default_rng([2718, i])
    inst, _ = draw(rng, n=2048, kind="emission")
    out = []
    for _ in range(places):
        u, v = float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1))
        out.append(psf_at(inst, u - inst.centre[0], v - inst.centre[1], PSF).astype(np.float16))
    return out


def build(instruments: int, places: int, workers: int, out: str = BANK) -> None:
    with get_context("spawn").Pool(workers) as pool:
        res = pool.starmap(_shapes, [(i, places) for i in range(instruments)])
    psf = np.stack([p for r in res for p in r])
    np.savez(out, psf=psf)
    print(f"{len(psf)} shapes -> {out}", flush=True)


class Synth:
    """Adds stars to a batch of linear tiles on the GPU."""

    def __init__(self, device, bank: str = BANK, seed: int = 0):
        import torch
        self.t = torch
        self.dev = device
        psf = np.load(bank)["psf"].astype(np.float32)
        self.psf = torch.from_numpy(psf).to(device)                       # (N, 3, 129, 129), each colour sums to 1
        self.peak = self.psf[:, 1].amax((1, 2))                           # the peak of a star of flux 1
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)

    def U(self, a, b, size=()):
        t = self.t
        return a + (b - a) * t.rand(size, generator=self.gen, device=self.dev)

    def stars(self, n: int, P: int, sat):
        """n tiles of stars only (n, 3, P, P), drawn on a canvas with a margin (the halos of the stars outside)."""
        t = self.t
        M = PSF // 2 + 64
        G = P + 2 * M
        out = t.zeros(n, 3, G, G, device=self.dev)
        ker = t.zeros(n, 3, G, G, device=self.dev)
        yy, xx = t.meshgrid(t.arange(G, device=self.dev, dtype=t.float32) - G // 2,
                            t.arange(G, device=self.dev, dtype=t.float32) - G // 2, indexing="ij")
        rr2 = xx ** 2 + yy ** 2
        idx = t.randint(len(self.psf), (n,), generator=self.gen, device=self.dev)
        h = PSF // 2
        for i in range(n):
            p = self.psf[idx[i]]
            # the far aureole: a share of the light in (1 + (r / r0)^2)^(-gamma / 2), r0 a few FWHM
            share = float(self.U(0.003, 0.05)) if float(self.U(0, 1)) < 0.6 else 0.0
            k = t.zeros(3, G, G, device=self.dev)
            k[:, G // 2 - h:G // 2 + h + 1, G // 2 - h:G // 2 + h + 1] = p * (1 - share)
            if share:
                fw = 2 * math.sqrt(float((p[1] > 0.5 * p[1].max()).sum()) / math.pi)
                r0, gam = fw * float(self.U(1.0, 3.0)), float(self.U(1.5, 2.6))
                a = (1 + rr2 / r0 ** 2) ** (-gam / 2)
                k = k + share * (a / a.sum())[None]
            ker[i] = t.fft.ifftshift(k, dim=(-2, -1))
            # the counts: a power law from 1.5 sigmas at the peak to far past the saturation
            dens = math.exp(float(self.U(math.log(100.0), math.log(3000.0))))       # per Mpx above 3 sigmas
            alpha = float(self.U(0.55, 0.9))
            n3 = dens * G * G / 1e6
            nstar = int(n3 * (3.0 / 1.5) ** alpha)
            if nstar == 0:
                continue
            pk1 = float(self.peak[idx[i]]) * (1 - share)
            pmin = 1.5
            pmax = 50.0 * float(sat[i].max())
            u = self.U(0, 1, (nstar,)).clamp(min=(pmin / pmax) ** alpha)
            peaks = pmin * u ** (-1.0 / alpha)
            flux = peaks / pk1
            bv = (0.7 + 0.35 * t.randn(nstar, generator=self.gen, device=self.dev)).clamp(-0.3, 2.0)
            col = t.stack([10 ** (0.2 * (bv - 0.65)), t.ones_like(bv), 10 ** (-0.36 * (bv - 0.65))], 0)
            col = col / col.mean(0, keepdim=True)
            px, py = self.U(0, G - 1.001, (nstar,)), self.U(0, G - 1.001, (nstar,))
            x0, y0 = px.floor().long(), py.floor().long()
            fx, fy = px - x0, py - y0
            img = out[i].view(3, -1)
            for dy, dx, w in ((0, 0, (1 - fx) * (1 - fy)), (0, 1, fx * (1 - fy)), (1, 0, (1 - fx) * fy),
                              (1, 1, fx * fy)):
                j = ((y0 + dy) * G + (x0 + dx)).clamp(max=G * G - 1)
                for ch in range(3):
                    img[ch].index_add_(0, j, flux * col[ch] * w)
        s = t.fft.irfft2(t.fft.rfft2(out) * t.fft.rfft2(ker), s=(G, G))
        return s[:, :, M:M + P, M:M + P].clamp(min=0)

    def add(self, x, y, version, sat, photon_k, share: float):
        """x, y (B, 3, P, P) linear; version (B,) 0 noisy 1 denoised; sat (B, 3); photon_k (B,). A share of the tiles
        gets added stars, over the starless target or over the tile's own stars (half each); the target is unchanged.
        Returns the new x."""
        t = self.t
        B, _c, P, _ = x.shape
        pick = t.nonzero(t.rand(B, generator=self.gen, device=self.dev) < share).flatten()
        if len(pick) == 0:
            return x
        k = len(pick)
        top = sat[pick] * self.U(0.5, 1.0, (k, 1))
        s = self.stars(k, P, top)
        kn = t.where(version[pick] == 0, photon_k[pick], self.U(0.0, 0.02, (k,)))
        s = s + t.sqrt(s * kn[:, None, None, None]) * t.randn(s.shape, generator=self.gen, device=self.dev)
        on_target = self.U(0, 1, (k, 1, 1, 1)) < 0.5
        base = t.where(on_target, y[pick], x[pick])
        x = x.clone()
        x[pick] = t.minimum(base + s, t.maximum(base, top[:, :, None, None]))
        return x


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=["build"])
    ap.add_argument("--instruments", type=int, default=500)
    ap.add_argument("--places", type=int, default=3)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    build(a.instruments, a.places, a.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
