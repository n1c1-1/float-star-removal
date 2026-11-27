# ======================================================================
# train_e.py
# Role     : Training of the chain's recipe (NAFNet, 256 tiles, L1 loss on the fixed stretch) on the generated pairs: produces E2/E3.
# Step     : 1. Training on synthetic data only (E2 / E3, 7-9 Nov) (see scripts/README.md)
# Inputs   : generator: seeds + catalogue.json (public instrument sheets); no image
# Outputs  : synthetic pairs (input/target/info.json + manifest.json), network E3
# Example  : python train_e.py --tag E --hours 8.5 --device cuda:0
# ======================================================================
"""Training of the star network (NAFNet) on the generated pairs only: recipes E, E2, E3.

Recipe (everything is written here; no pre-existing weights are read):
- the network: NAFNet (arXiv:2204.04676), width 32, 256 px tiles in training as in use, random weights at the start
  (or those of a previous run with --init: E2 for E3);
- the examples: our generated pairs (make_dataset.py), their denoised version: input_denoised -> target_denoised;
  60 % of the tiles also receive stars drawn on the fly (``synthetic_stars``: a Moffat core, an r^-3 aureole,
  sometimes four spikes, a power law of the brightnesses, field-star colours, a little photon noise; nothing beyond
  the reach that the generator leaves whole, stars.py), either on the target (synthetic stars alone) or on the input
  (its own stars and more), half each, each colour clipped at the pair's own saturation level for that colour,
  multiplied by 0.5 to 1;
- the drawing's starting star width: the core width of each pair (seeing and Airy disc of its instrument, info.json);
- the loss and the optimisation: L1 on the fixed asinh stretch, AdamW 5e-4 (0.9; 0.9), weight decay 1e-4, 1000
  warm-up steps then a cosine down to 0 over the given hours, gradient clipped at 0.1, batch of 8, bf16;
- the validation, every 2000 steps: the fixed set of diagnose.py valset (its denoised tiles, the 256 centre px): the
  error under the stars; best.pt is its minimum; the weights of each validation are kept in snap/ (probe.py follows
  them);
- with --real, a share of the tiles comes from our real images and their corrected targets (real_targets.py), outside
  the zones left out. This option was not used for E, E2 or E3.

    python train_e.py --tag E --hours 8.5 --device cuda:0

Outputs <ESSAIS>/starRemoval/train/<tag>/: best.pt, last.pt, snap/step_<n>.pt, history.json, log.txt.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from models import chain_nafnet, fwd, inv
from stars import REACH0, REACH_LEVEL
from train import STAR_PX, pair_dirs, ram_free, saturation_limit

PAIRS = (Path(__import__("os").environ.get("ESSAIS", "essais")) / "starRemoval" / "pairs_v2")
OUT = (Path(__import__("os").environ.get("ESSAIS", "essais")) / "starRemoval" / "train")
VAL = (Path(__import__("os").environ.get("ESSAIS", "essais")) / "starRemoval" / "diagnose" / "valset_pairs_v2.npz")
PATCH, MARGIN = 256, 128


def synthetic_stars(n: int, fwhm_field, top, gen, dev):
    """The chain's run's star drawing (tools/bench/stars_train.py, kept as it is): n tiles (n, 3, PATCH, PATCH) of
    stars only, in sigmas, drawn on a wider area (the margin holds the stars whose halo reaches the tile), one profile
    per tile, made by a product in Fourier space."""
    G = PATCH + 2 * MARGIN
    yy, xx = torch.meshgrid(torch.arange(G, device=dev, dtype=torch.float32) - G // 2,
                            torch.arange(G, device=dev, dtype=torch.float32) - G // 2, indexing="ij")
    out = torch.zeros(n, 3, G, G, device=dev)
    psfs = torch.zeros(n, 3, G, G, device=dev)

    def U(a, b, size=()):
        return a + (b - a) * torch.rand(size, generator=gen, device=dev)

    for i in range(n):
        fw = float(fwhm_field[i]) * float(U(0.8, 1.25)) if float(U(0, 1)) < 0.7 else float(torch.exp(U(np.log(1.3),
                                                                                                        np.log(8.0))))
        beta = float(U(1.8, 4.5))
        alpha = fw / (2 * np.sqrt(2 ** (1 / beta) - 1))
        e, th = float(U(0, 0.25)), float(U(0, np.pi))
        c, s = np.cos(th), np.sin(th)
        u, v = c * xx + s * yy, -s * xx + c * yy
        r2 = (u / (1 - e / 2)) ** 2 + (v / (1 + e / 2)) ** 2
        core = (1 + r2 / alpha ** 2) ** (-beta)
        core = core / core.sum()
        fh = float(U(0, 0.08)) if float(U(0, 1)) < 0.6 else 0.0
        rh = fw * float(torch.exp(U(np.log(3.0), np.log(25.0))))
        fs = float(U(0.002, 0.02)) if float(U(0, 1)) < 0.2 else 0.0
        if fs:
            ts = float(U(0, np.pi / 2))
            spikes = torch.zeros_like(core)
            for k in range(2):
                a = ts + k * np.pi / 2
                along = np.cos(a) * xx + np.sin(a) * yy
                perp = -np.sin(a) * xx + np.cos(a) * yy
                spikes += torch.exp(-perp ** 2 / (2 * (0.35 * fw) ** 2)) / (1 + (along / fw) ** 2)
            spikes = spikes / spikes.sum()
        for ch in range(3):
            p = (1 - fh - fs) * core
            if fh:
                rc = rh * (1 + float(U(-0.25, 0.25)))           # a halo a little larger in one colour (refractors)
                halo = (1 + (xx ** 2 + yy ** 2) / rc ** 2) ** (-1.5)
                p = p + fh * halo / halo.sum()
            if fs:
                p = p + fs * spikes
            psfs[i, ch] = torch.fft.ifftshift(p)
        # none is a giant: the stars taken off stay under the reach the generator leaves whole (stars.py)
        at_edge = float(p[G // 2, G // 2 + int(REACH0) // 2])
        # the stars: a power law of brightness, from about a third of a sigma at the peak up to far past saturation
        nstar = int(float(torch.exp(U(0.0, np.log(400.0)))) * (G / PATCH) ** 2)
        if nstar == 0:
            continue
        a_pow = float(U(0.7, 1.2))
        peak1 = (beta - 1) / (np.pi * alpha ** 2)                   # the peak of a star of flux 1
        fmin = float(U(0.3, 1.0)) / peak1
        flux = fmin * U(0, 1, (nstar,)).clamp(min=1e-9) ** (-1 / a_pow)
        flux = flux.clamp(max=min(50 * float(top[i]) / peak1, REACH_LEVEL / at_edge))
        # colours: B-V spread as field stars (about the Sun's, redder in the dust), the balance's white at the Sun's
        # (0.65), turned into multipliers
        bv = (0.7 + 0.35 * torch.randn(nstar, generator=gen, device=dev)).clamp(-0.3, 2.0)
        col = torch.stack([10 ** (0.2 * (bv - 0.65)), torch.ones_like(bv), 10 ** (-0.36 * (bv - 0.65))], 0)
        col = col / col.mean(0, keepdim=True)
        px, py = U(0, G - 1.001, (nstar,)), U(0, G - 1.001, (nstar,))
        x0, y0 = px.floor().long(), py.floor().long()
        fx, fy = px - x0, py - y0
        img = out[i].view(3, -1)
        for dy, dx, w in ((0, 0, (1 - fx) * (1 - fy)), (0, 1, fx * (1 - fy)), (1, 0, (1 - fx) * fy), (1, 1, fx * fy)):
            idx = ((y0 + dy) * G + (x0 + dx)).clamp(max=G * G - 1)
            for ch in range(3):
                img[ch].index_add_(0, idx, flux * col[ch] * w)
    stars = torch.fft.irfft2(torch.fft.rfft2(out) * torch.fft.rfft2(psfs), s=(G, G))
    stars = stars[:, :, MARGIN:MARGIN + PATCH, MARGIN:MARGIN + PATCH].clamp(min=0)
    k = U(0, 0.15, (n, 1, 1, 1))
    return stars + k * stars.sqrt() * torch.randn(stars.shape, generator=gen, device=dev)


class Tiles:
    """Random tiles of our pairs, denoised version (memory-mapped, read in the training's own process as the chain's
    run did): (input, target, the pair's star core width, its saturation)."""

    def __init__(self, dirs, seed: int):
        self.dirs = list(dirs)
        self.rng = np.random.default_rng(seed)
        self.meta = []
        for d in self.dirs:
            inst = json.loads((d / "info.json").read_text())["stars"]["instrument"]
            fw = math.sqrt(inst["seeing_fwhm"] ** 2 + (1.03 * inst["airy_px"]) ** 2)
            self.meta.append((fw, float(saturation_limit(d).max()), saturation_limit(d)))
        self.cache: dict = {}

    def arrays(self, i):
        if i not in self.cache:
            if len(self.cache) > 64:
                self.cache.clear()
            d = self.dirs[i]
            self.cache[i] = (np.load(d / "input_denoised.npy", mmap_mode="r"), np.load(d / "target_denoised.npy",
                                                                                      mmap_mode="r"))
        return self.cache[i]

    def get(self, n):
        P = PATCH
        xin, tgt, fw, top, lims = [], [], [], [], []
        while len(xin) < n:
            i = int(self.rng.integers(len(self.dirs)))
            A, T = self.arrays(i)
            H, W = A.shape[1:]
            y, x = int(self.rng.integers(0, H - P)), int(self.rng.integers(0, W - P))
            a = np.asarray(A[:, y:y + P, x:x + P], np.float32)
            t = np.minimum(np.asarray(T[:, y:y + P, x:x + P], np.float32), self.meta[i][2])
            k = int(self.rng.integers(4))
            a, t = np.rot90(a, k, (1, 2)), np.rot90(t, k, (1, 2))
            if self.rng.random() < 0.5:
                a, t = a[:, :, ::-1], t[:, :, ::-1]
            xin.append(np.ascontiguousarray(a))
            tgt.append(np.ascontiguousarray(t))
            fw.append(self.meta[i][0])
            top.append(self.meta[i][1])
            lims.append(self.meta[i][2].ravel())
        return np.stack(xin), np.stack(tgt), np.array(fw), np.array(top), np.stack(lims).astype(np.float32)


class RealTiles:
    """Random tiles of our real images and their corrected targets (real_targets.py): the same as Tiles gives. A tile never touches a zone left out of the training (the discs and the frame's margin of ignored.json) nor an image it drops."""

    def __init__(self, root: Path, seed: int):
        data = Path(__file__).resolve().parent / "data" / "train_with_stars"
        ign = json.loads((root / "ignored.json").read_text())
        fields = json.loads((root / "fields.json").read_text())
        self.rng = np.random.default_rng(seed)
        self.items = []
        for name, f in sorted(fields.items()):
            A = np.load(data / name / "input.npy", mmap_mode="r")
            T = np.load(root / name / f["target"], mmap_mode="r")
            H, W = A.shape[1:]
            b = ign["margin"] // 8 + 1
            blocked = np.zeros((H // 8 + 1, W // 8 + 1), bool)
            blocked[:b], blocked[-b:], blocked[:, :b], blocked[:, -b:] = True, True, True, True
            for t in ign["discs"].get(name, []):
                yy, xx = np.ogrid[:H // 8 + 1, :W // 8 + 1]
                blocked |= np.hypot(yy * 8 - t["y"], xx * 8 - t["x"]) <= t["r"] + 8
            self.items.append((A, T, f["fwhm"], np.asarray(f["saturation"], np.float32)[:, None, None], blocked))

    def get(self, n):
        P = PATCH
        xin, tgt, fw, top, lims = [], [], [], [], []
        while len(xin) < n:
            A, T, width, lim, blocked = self.items[int(self.rng.integers(len(self.items)))]
            H, W = A.shape[1:]
            y, x = int(self.rng.integers(0, H - P)), int(self.rng.integers(0, W - P))
            if blocked[y // 8:(y + P) // 8 + 1, x // 8:(x + P) // 8 + 1].any():
                continue
            a = np.asarray(A[:, y:y + P, x:x + P], np.float32)
            t = np.minimum(np.asarray(T[:, y:y + P, x:x + P], np.float32), lim)
            k = int(self.rng.integers(4))
            a, t = np.rot90(a, k, (1, 2)), np.rot90(t, k, (1, 2))
            if self.rng.random() < 0.5:
                a, t = a[:, :, ::-1], t[:, :, ::-1]
            xin.append(np.ascontiguousarray(a))
            tgt.append(np.ascontiguousarray(t))
            fw.append(width)
            top.append(float(lim.max()))
            lims.append(lim.ravel())
        return np.stack(xin), np.stack(tgt), np.array(fw), np.array(top), np.stack(lims).astype(np.float32)


class Mixed:
    """Each tile from the real pairs with the probability given, from the generated ones otherwise."""

    def __init__(self, generated, real, share: float, seed: int):
        self.generated, self.real, self.share = generated, real, share
        self.rng = np.random.default_rng(seed)

    def get(self, n):
        k = int((self.rng.random(n) < self.share).sum())
        parts = [p for p in ((self.real.get(k) if k else None), (self.generated.get(n - k) if n - k else None)) if p is not None]
        return tuple(np.concatenate([p[i] for p in parts]) for i in range(5))


def val_set(dev, path=VAL):
    """The fixed validation set's denoised tiles, their centre 256 px (the network's tile), in the stretch."""
    z = np.load(path)
    keep = z["version"] == 1
    c0 = (z["x"].shape[-1] - PATCH) // 2
    X = torch.from_numpy(z["x"][keep][:, :, c0:c0 + PATCH, c0:c0 + PATCH].astype(np.float32))
    Y = torch.from_numpy(z["y"][keep][:, :, c0:c0 + PATCH, c0:c0 + PATCH].astype(np.float32))
    M = ((inv(X) - inv(Y)).mean(1, keepdim=True) > STAR_PX).float()
    return X, Y, M


def validate(net, V, dev, bs: int = 16) -> dict:
    net.eval()
    X, Y, M = V
    tot = under = 0.0
    nu = 0.0
    for i in range(0, len(X), bs):
        x, y, m = X[i:i + bs].to(dev), Y[i:i + bs].to(dev), M[i:i + bs].to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            o = net(x).float()
        e = (o - y).abs().mean(1, keepdim=True)
        tot += float(e.sum())
        under += float((e * m).sum())
        nu += float(m.sum())
    net.train()
    return {"val": tot / (X.shape[0] * X.shape[2] * X.shape[3]), "val_stars": under / max(nu, 1)}
validate = torch.no_grad()(validate)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", default="E")
    ap.add_argument("--hours", type=float, default=8.5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--synthetic", type=float, default=0.6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--held", type=int, default=25)
    ap.add_argument("--pairs", default=str(PAIRS))
    ap.add_argument("--val", default=str(VAL), help="the fixed validation set (diagnose.py valset)")
    ap.add_argument("--init", default="", help="start from this checkpoint's weights (a train_e run)")
    ap.add_argument("--real", default="", help="the corrected real pairs (real_targets.py), mixed with the generated ones")
    ap.add_argument("--real-share", type=float, default=0.5, help="the share of the tiles taken from the real pairs")
    a = ap.parse_args(argv)
    dev = torch.device(a.device)
    torch.backends.cudnn.benchmark = True
    out = OUT / a.tag
    (out / "snap").mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a", encoding="utf-8")

    def log(msg):
        line = f"{datetime.now():%H:%M:%S} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    train_dirs, _held = pair_dirs(Path(a.pairs), a.held)
    cfg = {**vars(a), "model": "chain_nafnet", "patch": PATCH, "train_pairs": len(train_dirs),
           "version": "denoised", "recipe": "the chain's reseau etoiles run, on our pairs"}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    data = Tiles(train_dirs, seed=0)
    if a.real:
        real = RealTiles(Path(a.real), seed=1)
        data = Mixed(data, real, a.real_share, seed=2)
    V = val_set(dev, Path(a.val))
    gen = torch.Generator(device=dev)
    gen.manual_seed(0)
    torch.manual_seed(0)
    net = chain_nafnet().to(dev)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location=dev, weights_only=False)["net"])
    log(f"{a.tag}: the chain's NAFNet ({sum(p.numel() for p in net.parameters()) / 1e6:.1f} M parameters, "
        f"{('from ' + a.init) if a.init else 'random weights'}), {len(train_dirs)} pairs (denoised){(f' and {len(real.items)} real images for {a.real_share:.0%} of the tiles') if a.real else ''}, stars drawn on {a.synthetic:.0%} of the tiles, batch {a.batch}, "
        f"lr {a.lr}, {a.hours} h; validation {len(V[0])} tiles")
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4, betas=(0.9, 0.9))
    hist, best = [], None
    t0, it, run, warm = time.time(), 0, None, 1000
    while True:
        frac = (time.time() - t0) / (a.hours * 3600)
        if frac >= 1.0:
            break
        for g in opt.param_groups:                       # warm-up, then a cosine to zero at the end of the time given
            g["lr"] = a.lr * min(1.0, (it + 1) / warm) * 0.5 * (1 + np.cos(np.pi * frac))
        x, t, fw, top, lims = data.get(a.batch)
        x, t = torch.from_numpy(x).to(dev), torch.from_numpy(t).to(dev)
        pick = torch.rand(a.batch, generator=gen, device=dev) < a.synthetic
        if pick.any():
            k = int(pick.sum())
            sel = pick.cpu().numpy()
            stars = synthetic_stars(k, fw[sel], top[sel], gen, dev)
            # on the starless target (stars of synthesis only) or on the input (its own stars and more), half each
            base = torch.where(torch.rand(k, 1, 1, 1, generator=gen, device=dev) < 0.5, t[pick], x[pick])
            # each colour clips at its own level (the pair's: the white balance multiplies the raw's clip), not one
            # level for the three as the chain's run did: the saturated cores keep their real tint
            sat = torch.from_numpy(lims[sel]).to(dev)[:, :, None, None] \
                * (0.5 + 0.5 * torch.rand(k, 1, 1, 1, generator=gen, device=dev))
            x[pick] = torch.minimum(base + stars, torch.maximum(base, sat))
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            pred = net(fwd(x)).float()
        loss = (pred - fwd(t)).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
        opt.step()
        run = loss.item() if run is None else 0.99 * run + 0.01 * loss.item()
        it += 1
        if it % 2000 == 0:
            v = validate(net, V, dev)
            hist.append({"step": it, "minutes": round((time.time() - t0) / 60, 1), "train": round(run, 5),
                         "val": round(v["val"], 5), "val_stars": round(v["val_stars"], 5),
                         "lr": opt.param_groups[0]["lr"]})
            (out / "history.json").write_text(json.dumps(hist, indent=1))
            ck = {"model": "chain_nafnet", "net": net.state_dict(), "config": cfg, "step": it, "val": v}
            torch.save(ck, out / "last.pt")
            torch.save(ck, out / "snap" / f"step_{it}.pt")
            if best is None or v["val_stars"] < best:
                best = v["val_stars"]
                torch.save(ck, out / "best.pt")
            log(f"step {it}: train {run:.5f}, validation {v['val']:.5f}, on the stars {v['val_stars']:.5f}"
                f"{' best' if v['val_stars'] <= best else ''}, lr {opt.param_groups[0]['lr']:.1e}, "
                f"{(time.time() - t0) / it:.2f} s/step, RAM free {ram_free():.1f} GB")
    ck = {"model": "chain_nafnet", "net": net.state_dict(), "config": cfg, "step": it}
    torch.save(ck, out / "last.pt")
    log(f"done: {it} steps in {(time.time() - t0) / 60:.0f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
