# ======================================================================
# train.py
# Role     : Trains a network from random weights on the generated pairs only.
# Step     : 1. Training on synthetic data only (E2 / E3, 7-9 Nov) (see scripts/README.md)
# Inputs   : generator: seeds + catalogue.json (public instrument sheets); no image
# Outputs  : synthetic pairs (input/target/info.json + manifest.json), network E3
# Example  : python train.py --model nafnet --hours 1.5 --device cuda:0 --tag A
# ======================================================================
"""Train a star removal network of the bench (models.py) from random weights, on the generated pairs only
(make_dataset.py), for a budget of time.

- the pairs: both sets (clean backgrounds, backgrounds with defects); the last ``--held`` pairs of each set are kept
  out (the validation here, evaluate.py's measures); each tile is drawn from a random pair, in its noisy or its
  denoised version half and half (the user chooses in the chain whether to denoise first), at a random place, turned
  and flipped at random;
- the loss: L1 on the network's stretch (models.fwd); for ``nafnet_mask`` plus a small term on its mask (the pixels
  where the stars bring more than half a sigma);
- AdamW, a cosine learning rate over the time budget after a short warm-up, bf16 on a GPU;
- the validation: fixed tiles of the held-out pairs (both versions), every few minutes: the error over all, under
  the stars and off them; best.pt is the checkpoint of the lowest validation error.

    python train.py --model nafnet --hours 1.5 --device cuda:0 --tag A
    python train.py --model nafnet_mask --hours 1.5 --device cuda:1 --tag B
    python train.py --model lama --hours 1.5 --device cuda:0 --tag C --patch 384 --batch 6

Outputs <ESSAIS>/starRemoval/train/<tag>/: best.pt, last.pt, config.json, history.json, log.txt.
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
import torch.nn.functional as F

from models import build, fwd

PAIRS = (Path(__import__("os").environ.get("ESSAIS", "essais")) / "starRemoval" / "pairs_1000")
OUT = (Path(__import__("os").environ.get("ESSAIS", "essais")) / "starRemoval" / "train")
#: the stars' share that makes a pixel "a star's" for the mask and the measures (noise sigmas, linear)
STAR_PX = 0.5


def saturation_limit(d) -> np.ndarray:
    """The pair's saturation per colour in its unit, from its info.json (make_dataset.make_pair: the camera's clip
    times the white balance, less the sky taken). The pairs written before the target was clipped there (a compact
    object's core above saturation) get the same clip when read."""
    info = json.loads((Path(d) / "info.json").read_text())
    wb = np.asarray(info["white_balance"], np.float32)
    return (info["saturation"] * wb / wb.mean() - np.asarray(info["sky_taken"], np.float32))[:, None, None]


def ram_free() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except ImportError:
        return float("nan")


def pair_dirs(root: Path, held: int) -> tuple[list[Path], list[Path]]:
    train, val = [], []
    for s in ("clean", "defects"):
        ds = sorted((root / s).glob("pair_*"))
        train += ds[:-held] if held else ds
        val += ds[-held:] if held else []
    return train, val


class Tiles(torch.utils.data.IterableDataset):
    """Endless random tiles from memory-mapped pairs, linear (the stretch is taken on the GPU, after the stars added on
    the fly): (x, y, version 0 noisy / 1 denoised, the saturation per colour, the pair's photon noise factor)."""

    def __init__(self, dirs, patch: int, seed: int, denoised_share: float = 0.5):
        self.dirs, self.patch, self.seed, self.dshare = [str(d) for d in dirs], patch, seed, denoised_share

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, info.id if info else 0])
        P = self.patch
        cache = {}
        while True:
            d = self.dirs[int(rng.integers(len(self.dirs)))]
            vi = int(rng.random() < self.dshare)
            v = ("noisy", "denoised")[vi]
            key = (d, v)
            if key not in cache:
                if len(cache) > 16:                         # each worker's open pairs (the RAM of a long run)
                    cache.clear()
                pk = json.loads((Path(d) / "info.json").read_text()).get("photon_k", 0.05)
                cache[key] = (np.load(f"{d}/input_{v}.npy", mmap_mode="r"), np.load(f"{d}/target_{v}.npy", mmap_mode="r"),
                              saturation_limit(d), float(pk))
            a, t, lim, pk = cache[key]
            H, W = a.shape[1:]
            y, x = int(rng.integers(0, H - P)), int(rng.integers(0, W - P))
            xi = np.asarray(a[:, y:y + P, x:x + P], np.float32)
            yi = np.minimum(np.asarray(t[:, y:y + P, x:x + P], np.float32), lim)
            k = int(rng.integers(4))
            xi, yi = np.rot90(xi, k, (1, 2)), np.rot90(yi, k, (1, 2))
            if rng.random() < 0.5:
                xi, yi = xi[:, :, ::-1], yi[:, :, ::-1]
            yield (torch.from_numpy(np.ascontiguousarray(xi)), torch.from_numpy(np.ascontiguousarray(yi)), vi,
                   torch.from_numpy(lim.ravel().astype(np.float32)), pk)


def val_tiles(dirs, patch: int, n: int = 96):
    """Fixed validation tiles (half noisy, half denoised), the same for every run, in the stretch."""
    it = iter(Tiles(dirs, patch, seed=12345))
    xs, ys, ms = [], [], []
    for _ in range(n):
        xi, yi, _v, _s, _k = next(it)
        ms.append(((xi - yi).mean(0, keepdim=True) > STAR_PX).float())
        xs.append(fwd(xi))
        ys.append(fwd(yi))
    return torch.stack(xs), torch.stack(ys), torch.stack(ms)


def val_file(path: str):
    """The fixed validation set of diagnose.py valset (tiles already in the stretch), its star mask."""
    z = np.load(path)
    X, Y = torch.from_numpy(z["x"].astype(np.float32)), torch.from_numpy(z["y"].astype(np.float32))
    M = ((models_inv(X) - models_inv(Y)).mean(1, keepdim=True) > STAR_PX).float()
    return X, Y, M


def models_inv(v):
    from models import inv
    return inv(v)


def validate(net, V, dev, bs: int = 8) -> dict:
    net.eval()
    tot = under = off = 0.0
    nu = no = 0.0
    X, Y, M = V
    for i in range(0, len(X), bs):
        x, y, m = X[i:i + bs].to(dev), Y[i:i + bs].to(dev), M[i:i + bs].to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            o = net(x).float()
        e = (o - y).abs().mean(1, keepdim=True)
        tot += float(e.sum())
        under += float((e * m).sum())
        off += float((e * (1 - m)).sum())
        nu += float(m.sum())
        no += float((1 - m).sum())
    n = X.shape[0] * X.shape[2] * X.shape[3]
    net.train()
    return {"l1": tot / n * 1e3, "under_stars": under / max(nu, 1) * 1e3, "off_stars": off / max(no, 1) * 1e3}
validate = torch.no_grad()(validate)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, choices=["nafnet", "nafnet_mask", "lama"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--hours", type=float, default=1.5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--held", type=int, default=25, help="pairs of each set kept out")
    ap.add_argument("--pairs", default=str(PAIRS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-every", type=float, default=5.0, help="minutes")
    ap.add_argument("--init", default="", help="start from this checkpoint's weights (same model)")
    ap.add_argument("--schedule", default="cosine", choices=["cosine", "plateau"],
                    help="plateau: the rate halves after --patience validations without gain; it stops after "
                         "--reductions halvings without gain")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--reductions", type=int, default=3)
    ap.add_argument("--deadline", default="", help="HH:MM: stop then at the latest (the next such time)")
    ap.add_argument("--synthetic", type=float, default=0.0, help="share of the tiles given stars on the fly (synth.py)")
    ap.add_argument("--bank", default="<ESSAIS>/starRemoval/psf_bank.npz")
    ap.add_argument("--denoised-share", type=float, default=0.5, help="share of the tiles in the denoised version")
    ap.add_argument("--val-file", default="", help="a fixed validation set (diagnose.py valset) instead of 96 tiles")
    a = ap.parse_args(argv)
    out = OUT / a.tag
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a", encoding="utf-8")

    def log(msg):
        line = f"{datetime.now():%H:%M:%S} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    train_dirs, val_dirs = pair_dirs(Path(a.pairs), a.held)
    cfg = {**vars(a), "train_pairs": len(train_dirs), "val_pairs": len(val_dirs), "from": "random weights"}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    net = build(a.model).to(dev)
    if a.init:
        ck0 = torch.load(a.init, map_location=dev, weights_only=False)
        assert ck0["model"] == a.model, f"{a.init} is a {ck0['model']}"
        net.load_state_dict(ck0["net"])
        cfg["from"] = f"{a.init} (itself from random weights)"
        (out / "config.json").write_text(json.dumps(cfg, indent=1))
    nparam = sum(p.numel() for p in net.parameters()) / 1e6
    log(f"{a.tag}: {a.model} ({nparam:.1f} M parameters, {cfg['from']}), {len(train_dirs)} training pairs, "
        f"{len(val_dirs)} held out, patch {a.patch}, batch {a.batch}, {a.hours} h on {dev}, schedule {a.schedule}")
    if a.deadline:                                         # the budget ends at the deadline at the latest
        hh, mm = (int(v) for v in a.deadline.split(":"))
        now = datetime.now()
        end = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if end <= now:
            from datetime import timedelta
            end += timedelta(days=1)
        a.hours = min(a.hours, (end - now).total_seconds() / 3600)
        log(f"deadline {a.deadline}: at most {a.hours:.2f} h")
    V = val_file(a.val_file) if a.val_file else val_tiles(val_dirs, a.patch)
    log(f"validation: {len(V[0])} tiles{' (' + a.val_file + ')' if a.val_file else ''}; denoised share "
        f"{a.denoised_share}, stars on the fly {a.synthetic}")
    synth = None
    if a.synthetic > 0:
        from synth import Synth
        synth = Synth(dev, a.bank, seed=a.seed)
    loader = torch.utils.data.DataLoader(Tiles(train_dirs, a.patch, a.seed, a.denoised_share), batch_size=a.batch,
                                         num_workers=a.workers, persistent_workers=a.workers > 0, pin_memory=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4, betas=(0.9, 0.9))
    T = a.hours * 3600
    t0 = time.time()
    last_val = t0
    best = math.inf
    hist = []
    step = 0
    run = None
    skipped = 0
    plateau_lr, waited, halvings, stop = a.lr, 0, 0, False
    for x, y, ver, sat, pk in loader:
        el = time.time() - t0
        if el > T or stop:
            break
        f = el / T
        if a.schedule == "cosine":
            lr = a.lr * (f / 0.02 if f < 0.02 else 0.5 * (1 + math.cos(math.pi * (f - 0.02) / 0.98)))
        else:
            lr = plateau_lr * min(1.0, el / 120.0 + 0.05)          # a two-minute warm-up, then the plateau rate
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        if synth is not None:
            x = synth.add(x, y, ver.to(dev), sat.to(dev), pk.to(dev).float(), a.synthetic)
        m = ((x - y).mean(1, keepdim=True) > STAR_PX).float()
        x, y = fwd(x), fwd(y)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            if a.model == "nafnet_mask":
                o, mk = net(x, with_mask=True)
                loss = F.l1_loss(o.float(), y) + 0.1 * F.binary_cross_entropy_with_logits(mk.float(), m)
            else:
                loss = F.l1_loss(net(x).float(), y)
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):                       # a blown-up batch: it teaches nothing, it is skipped
            skipped += 1
            continue
        loss.backward()
        if not torch.isfinite(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)):
            opt.zero_grad(set_to_none=True)
            skipped += 1
            continue
        opt.step()
        step += 1
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if time.time() - last_val > a.val_every * 60 or time.time() - t0 > T:
            last_val = time.time()
            v = validate(net, V, dev)
            hist.append({"minutes": round((time.time() - t0) / 60, 1), "step": step, "loss": run * 1e3, **v})
            (out / "history.json").write_text(json.dumps(hist, indent=1))
            ck = {"model": a.model, "net": net.state_dict(), "config": cfg, "step": step, "val": v}
            torch.save(ck, out / "last.pt")
            gained = v["l1"] < best * 0.998                     # a gain worth the name: 0.2 %
            if v["l1"] < best:
                best = v["l1"]
                torch.save(ck, out / "best.pt")
            if a.schedule == "plateau":
                waited = 0 if gained else waited + 1
                if waited >= a.patience:
                    if halvings >= a.reductions:
                        stop = True
                        log(f"no gain after {halvings} halvings of the rate: it has stopped improving")
                    else:
                        halvings += 1
                        plateau_lr *= 0.5
                        waited = 0
                        log(f"no gain in {a.patience} validations: the rate halves to {plateau_lr:.2e}")
            log(f"step {step} ({(time.time() - t0) / 60:.0f} min) loss {run * 1e3:.3f} lr {lr:.2e} | val l1 "
                f"{v['l1']:.3f} under {v['under_stars']:.3f} off {v['off_stars']:.3f}"
                f"{' best' if v['l1'] <= best else ''}{f' | {skipped} batches skipped (not finite)' if skipped else ''}"
                f" | RAM free {ram_free():.1f} GB")
    v = validate(net, V, dev)                          # the end of the budget: a last validation
    hist.append({"minutes": round((time.time() - t0) / 60, 1), "step": step, "loss": (run or 0) * 1e3, **v})
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    ck = {"model": a.model, "net": net.state_dict(), "config": cfg, "step": step, "val": v}
    torch.save(ck, out / "last.pt")
    if v["l1"] < best:
        best = v["l1"]
        torch.save(ck, out / "best.pt")
    log(f"done: {step} steps ({skipped} batches skipped, not finite), last val l1 {v['l1']:.3f}, best {best:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
