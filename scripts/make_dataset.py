# ======================================================================
# make_dataset.py
# Role     : Builds the whole synthetic set from the code and the seeds; writes the manifest of fingerprints; --verify rebuilds and compares.
# Step     : 1. Training on synthetic data only (E2 / E3, 7-9 Nov) (see scripts/README.md)
# Inputs   : generator: seeds + catalogue.json (public instrument sheets); no image
# Outputs  : synthetic pairs (input/target/info.json + manifest.json), network E3
# Example  : python make_dataset.py --n 48 [--seed 1] [--sets clean,defects] [--workers 14] [--device cpu] [--out ...]
# ======================================================================
"""The whole training set of the star removal, regenerated from the code and its seeds alone: the proof that it owes
nothing to any outside image nor to any other tool.

No image is read (the files read are catalogue.json, the published specifications of the instruments, and our own
denoising network, both in the manifest's hashes). Each pair is drawn from its seed:

- the background (backgrounds.py: nebulae, galaxies, dust, the sky), set "clean" as it is, set "defects" with the
  defects of real stacks on top (defects.py: blotches, vignetting, bands, donuts, walking noise, edges, tint);
- the stars (stars.py, psf.py): an instrument (instruments.py: a real telescope or lens and camera from their
  published specifications, catalogue.json, each with its source), its PSF over the field, the field of stars, their
  halos and spikes;
- the noise of a stack as it comes out of the stacking: the sky's (1 / sqrt 2 of the network's unit, the full stack of
  two half-stacks), correlated between neighbouring pixels as the drizzle and the registration leave it (the
  correlations our stacks show: 0.09 - 0.42), and the photon noise of the light, the background's and the stars';
- the saturation, per colour: each channel clips at its own level after the white balance (the coloured cores);
- the sky's level taken off as the chain does (with the stars' diffuse light in it).

Two versions of each pair, for a network that works on the stack as it is or after the denoising (the user's choice
in the chain), to be mixed half and half in the training:

- noisy: input = background + stars + all the noise, clipped; target = background + the stars too faint to be seen
  (their light smoothed) + the giants, left whole (stars.py) + the same noise of the sky and of the background's light
  (the network keeps the noise: it does not denoise in the user's place); only the stars and their own photon noise
  differ;
- denoised: both through our own denoising network (astro_gpu/stages/denoise.py, pretrained), as the chain does.

The real images were only measured, to set the ranges the draws are made in (constants in the code, each with the
measure it comes from: measure.py, star_measure.py, donut_measure.py), and are kept for the validation.

Outputs (<out>/<set>/pair_<seed>_<i>/): input_noisy.npy, target_noisy.npy, input_denoised.npy, target_denoised.npy
(3, n, n) float16 in the network's unit, info.json (all that was drawn); <out>/manifest.json: the SHA-256 of every
file, of the code and of the denoising network, the library versions.

    python make_dataset.py --n 48 [--seed 1] [--sets clean,defects] [--workers 14] [--device cpu] [--out ...]
    python make_dataset.py --verify 3 [--out ...]

The verification regenerates the first pairs of each set and compares them with the manifest: the noisy files to the
bit; the denoised ones to the bit on the processor (one thread), within a tolerance when the denoising ran on a GPU.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")                # one thread per worker: the workers share the cores

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from backgrounds import background  # noqa: E402
from defects import defects  # noqa: E402
from psf import loguniform  # noqa: E402
from stars import render  # noqa: E402

OUT = (Path(__import__("os").environ.get("ESSAIS", "essais")) / "starRemoval" / "pairs")
HERE = Path(__file__).resolve().parent
CODE = ("backgrounds.py", "defects.py", "psf.py", "instruments.py", "catalogue.json", "stars.py", "make_dataset.py")
#: our denoising network (the chain's Denoise block)
ASTRO = Path("<FLOAT>")
DENOISER = ASTRO / "astro_gpu" / "resources" / "denoise" / "pretrained_unet.pt"
FILES = ("input_noisy", "target_noisy", "input_denoised", "target_denoised")
#: the denoised files compared within this (network units) when the denoising ran on a GPU
TOLERANCE = 0.05

_NET = None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def correlated_noise(rng, n: int, corr: float) -> np.ndarray:
    """(3, n, n) noise of std 1 whose correlation between neighbouring pixels is about ``corr`` (a gaussian blur of
    white noise: sigma 0 - 0.8 px)."""
    import cv2
    s = float(np.interp(corr, [0.0, 0.09, 0.2, 0.3, 0.42], [0.0, 0.42, 0.52, 0.6, 0.7]))
    out = np.empty((3, n, n), np.float32)
    for c in range(3):
        w = rng.standard_normal((n, n)).astype(np.float32)
        if s > 0.05:
            w = cv2.GaussianBlur(w, (0, 0), s)
        out[c] = w / (w.std() + 1e-12)
    return out


def make_pair(seed: int, i: int, which: str, n: int) -> tuple[dict, dict]:
    bg, info = background(np.random.default_rng([seed, i]), n)
    if which == "defects":
        d, dinfo = defects(np.random.default_rng([seed, i, 1]), n)
        bg = bg + d
        info["defects"] = dinfo
    rng = np.random.default_rng([seed, i, 2])
    sinfo: dict = {}
    stars, keep, _inst = render(rng, n, sinfo, kind=info["kind"])
    info["stars"] = sinfo
    # the noise: the sky's (the full stack: 1 / sqrt 2 of the half-stacks' unit), correlated as our stacks are, and
    # the photon noise of the light (k: the variance per unit of light), the background's shared with the target
    rng_n = np.random.default_rng([seed, i, 3])
    corr = float(rng_n.uniform(0.09, 0.42))
    k = loguniform(rng_n, 0.01, 0.3)
    sky_noise = correlated_noise(rng_n, n, corr) / np.sqrt(2)
    light = np.clip(bg + keep, 0, None)
    bg_noise = np.sqrt(light * k) * correlated_noise(rng_n, n, corr)
    star_noise = np.sqrt(np.clip(stars - keep, 0, None) * k) * correlated_noise(rng_n, n, corr)
    # the saturation per colour: the raw clips, then the white balance multiplies each channel (coloured cores)
    # (the top of our 29 images: 74 - 10 300 sigmas, median 1900)
    top = loguniform(rng_n, 100, 15000)
    # (the shares calibrated on our 23 images' saturated cores: red 1.3 x green, blue 1.4 x at the median, diagnose)
    wb = np.array([rng_n.uniform(1.05, 1.6), 1.0, rng_n.uniform(1.15, 1.75)], np.float32)
    top_c = (top * wb / wb.mean())[:, None, None]
    inp = bg + stars + sky_noise + bg_noise + star_noise
    tgt = bg + keep + sky_noise + bg_noise
    # the chain takes the sky's level off the image with the stars' diffuse light in it (the sum of their halos, the
    # unresolved stars): the same level comes off the input and the target (the target plus the stars stays the
    # image). The chain's sky estimate is robust to the stars: a low quantile here (a median would land on the stars
    # of a dense field)
    sky = np.percentile(stars.reshape(3, -1)[:, ::7], 20, axis=1)[:, None, None].astype(np.float32)
    # the camera saturates the background too (a compact object's core): the target clips where the input does
    inp = np.minimum(inp - sky, top_c - sky)
    tgt = np.minimum(tgt - sky, top_c - sky)
    info.update(noise_correlation=round(corr, 3), photon_k=round(k, 4), saturation=round(top, 1),
                white_balance=[round(float(v), 3) for v in wb], sky_taken=[round(float(v), 3) for v in sky.ravel()])
    return {"input_noisy": inp.astype(np.float32), "target_noisy": tgt.astype(np.float32)}, info


def denoiser(device: str):
    """Our denoising network (the chain's), loaded once per worker."""
    global _NET
    if _NET is None:
        import torch
        sys.path.insert(0, str(ASTRO))
        from astro_gpu.stages.denoise import pretrained
        if device == "cpu":
            torch.set_num_threads(1)
        _NET = pretrained(torch.device(device)).eval()
    return _NET


def denoise(x: np.ndarray, device: str) -> np.ndarray:
    """The chain's denoising of a full stack in the network's unit (the half-stacks' noise 1: the full stack's is
    1 / sqrt 2, as the Denoise block applies it to the full stack)."""
    import torch
    sys.path.insert(0, str(ASTRO))
    from astro_gpu.stages.denoise import apply_net
    n = x.shape[1]
    nm = np.full((n, x.shape[2]), 1 / np.sqrt(2), np.float32)
    return apply_net(denoiser(device), np.ascontiguousarray(x), nm, np.zeros(3, np.float32),
                     np.ones(3, np.float32), torch.device(device))


def write(job) -> dict:
    out, which, seed, i, n, device = job[:6]
    resume = len(job) > 6 and job[6]
    d = Path(out) / which / f"pair_{seed}_{i:04d}"
    if resume and all((d / f"{f}.npy").exists() for f in FILES) and (d / "info.json").exists():
        return {str((d / f"{f}.npy").relative_to(out)).replace("\\", "/"): sha256(d / f"{f}.npy") for f in FILES}
    t = time.time()
    arrs, info = make_pair(seed, i, which, n)
    arrs["input_denoised"] = denoise(arrs["input_noisy"], device)
    arrs["target_denoised"] = denoise(arrs["target_noisy"], device)
    info["denoise_device"] = "cpu" if device == "cpu" else "gpu"
    d.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        np.save(d / f"{f}.npy", arrs[f].astype(np.float16))
    (d / "info.json").write_text(json.dumps(info, indent=1, default=float))
    print(f"{which} {d.name} {time.time() - t:5.1f} s  kind {info['kind']}  stars {info['stars']['stars']}  "
          f"{info['stars']['instrument']['kind']}", flush=True)
    return {str((d / f"{f}.npy").relative_to(out)).replace("\\", "/"): sha256(d / f"{f}.npy") for f in FILES}


def run(jobs, workers: int) -> dict:
    files: dict = {}
    if workers <= 1:
        for j in jobs:
            files.update(write(j))
        return files
    from multiprocessing import get_context
    with get_context("spawn").Pool(workers) as pool:
        for r in pool.imap_unordered(write, jobs):
            files.update(r)
    return files


def versions() -> dict:
    import cv2
    import scipy
    import torch
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "opencv": cv2.__version__, "torch": torch.__version__, "machine": platform.machine(),
            "system": platform.system()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sets", default="clean,defects")
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--device", default="cpu", help="the denoising's device: cpu (exact) or cuda:N")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--verify", type=int, default=0, help="regenerate this many pairs per set, compare")
    ap.add_argument("--resume", action="store_true", help="keep the pairs already complete in --out")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.verify:
        man = json.loads((out / "manifest.json").read_text())
        code_now = {f: sha256(HERE / f) for f in CODE}
        if code_now != man["code"]:
            print("the code changed since the manifest:", [f for f in CODE if code_now[f] != man["code"].get(f)])
        bad = 0
        gpu = man.get("device", "cpu") != "cpu"
        with tempfile.TemporaryDirectory(prefix="verify_") as td:
            jobs = [(td, w, man["seed"], i, man["size"], man.get("device", "cpu"))
                    for w in man["sets"] for i in range(a.verify)]
            got = run(jobs, min(a.workers, len(jobs)))
            for key, h in sorted(got.items()):
                if man["files"].get(key) == h:
                    print(f"  {key}: identical")
                    continue
                diff = float(np.abs(np.load(Path(td) / key).astype(np.float32)
                                    - np.load(out / key).astype(np.float32)).max())
                ok = gpu and "denoised" in key and diff <= TOLERANCE
                bad += not ok
                print(f"  {key}: {'within tolerance' if ok else 'DIFFERENT'} (largest difference {diff:.4f})")
        print("verified" if not bad else f"{bad} files differ")
        return 1 if bad else 0
    jobs = [(str(out), w, a.seed, i, a.size, a.device, a.resume) for w in a.sets.split(",") for i in range(a.n)]
    t = time.time()
    files = run(jobs, a.workers)
    man = {"seed": a.seed, "n": a.n, "size": a.size, "sets": a.sets.split(","), "device": a.device, "files": files,
           "code": {f: sha256(HERE / f) for f in CODE}, "denoiser": sha256(DENOISER), "versions": versions(),
           "made": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t), "reads_no_image": True}
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"-> {out / 'manifest.json'} ({len(files)} files, {time.time() - t:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
