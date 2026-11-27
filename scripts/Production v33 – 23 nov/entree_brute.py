# ======================================================================
# entree_brute.py
# Role     : NON-denoised input of an image, same geometry, same units and same gain as the denoised input already made (meta.json gain): raw = mean of the normalised half-stacks (gradient removed, background and noise of half-stack A) x gain,
# Stage    : 2. Production of the v33 pairs (23 Nov) (see scripts/LISEZMOI.md)
# Inputs   : chosen tiles, v33/v33b chain
# Outputs  : _debruitee/_brute .npz pairs
# Example  : usage : entree_brute.py <nom> ; output <DONNEES>/Paires étoiles v33 – 23 nov/entrees_brutes/<nom>_brut.npy
# ======================================================================
"""NON-denoised input of an image, same geometry, same units and same gain as the denoised input already made (meta.json gain):
raw = mean of the normalised half-stacks (gradient removed, background and noise of half-stack A) x gain, outside the field at 0. No network.
Sources: set 2 modes prep/fits (selection_jeu2.csv); set 1 ic5070, ngc7000_tiers1, ngc7000_gfx (meta.json source). The other images
("ready" inputs: data sets whose raw data is not kept) have no raw version.
usage : entree_brute.py <nom> ; output <DONNEES>/Paires étoiles v33 – 23 nov/entrees_brutes/<nom>_brut.npy"""
import os, sys, json, csv, time
N = os.environ.get("BRUT_FILS", "3")
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = N
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np
ROOT = Path(r"<FLOAT>")
for p in (ROOT, ROOT / "tools", ROOT / "tools" / "bench"):
    sys.path.insert(0, str(p))
import stars_dataset as SD
from astro_gpu.stages.denoise import norm_params
nom = sys.argv[1]
J2 = (Path(__import__("os").environ.get("RACINE_SCRIPTS") or Path(__file__).resolve().parents[1]) / "Paires jeu 2 – 23 nov")
OUT = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Paires étoiles v33 – 23 nov" / "entrees_brutes"); OUT.mkdir(parents=True, exist_ok=True)
rows = {r["nom"]: r for r in csv.DictReader(open(J2 / "selection_jeu2.csv", encoding="utf-8-sig"), delimiter=";")}
t0 = time.time()
if nom in rows:
    r = rows[nom]; mode, src, REC = r["mode"], Path(r["entree"]), r["recadre"]
    den = Path(r["entree_finale"])
else:
    den = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Images jeu 1 – 22 nov") / nom / "input.npy"
    src, REC, mode = Path(json.loads((den.parent / "meta.json").read_text(encoding="utf-8"))["source"]), "", "prep"
gain = np.asarray(json.loads((den.parent / "meta.json").read_text(encoding="utf-8"))["gain"], np.float32)


def rec(a):
    if not REC:
        return a
    w, h = map(int, REC.split("x"))
    H, W = a.shape[-2:]
    y0, x0 = (H - min(h, H)) // 2, (W - min(w, W)) // 2
    return np.ascontiguousarray(a[..., y0:y0 + min(h, H), x0:x0 + min(w, W)])


if mode == "prep":
    wt = rec(np.load(src / "drz_weight.npy", mmap_mode="r")).astype(np.float32)
    full = rec(np.load(src / "drz_full.npy", mmap_mode="r")).astype(np.float32)
    valid = SD.valid_of(wt); var = SD.sky_variation_of(full, valid)
    halves = [rec(np.load(src / f"drz_{h}.npy", mmap_mode="r")).astype(np.float32) - var for h in "AB"]
    del var
elif mode == "fits":
    from astropy.io import fits
    full = rec(fits.getdata(src).astype(np.float32))
    if full.ndim == 2:
        full = np.stack([full] * 3)
    full = np.ascontiguousarray(full[:3]); full[~np.isfinite(full)] = 0
    wt = (np.abs(full) > 0).astype(np.float32)
    valid = SD.valid_of(wt); var = SD.sky_variation_of(full, valid)
    halves = [full - var]
else:
    sys.exit("pas de brut pour ce mode")
bk, sg, nm = norm_params(halves[0], wt)
img = sum((h - bk[:, None, None]) / sg[:, None, None] for h in halves) / len(halves)
img *= gain[:, None, None]
img[:, nm <= 0] = 0.0
d = np.load(den, mmap_mode="r")
assert d.shape == img.shape, (d.shape, img.shape)
# controle : le brut et le debruite ne different que par le bruit (moyennes par blocs de 16 px)
def blocs(a):
    c, h, w = a.shape; h16, w16 = h // 16 * 16, w // 16 * 16
    return a[:, :h16, :w16].reshape(c, h16 // 16, 16, w16 // 16, 16).mean((2, 4))
bb, bd = blocs(img), blocs(np.asarray(d))
ok = (bd != 0)
corr = float(np.corrcoef(bb[ok][::7], bd[ok][::7])[0, 1])
dec = float(np.median(np.abs(bb[ok] - bd[ok])))
np.save(OUT / f"{nom}_brut.npy", img.astype(np.float32))
(OUT / f"{nom}_brut.json").write_text(json.dumps(dict(nom=nom, source=str(src), forme=list(img.shape), corr_blocs=round(corr, 5), ecart_median_blocs=round(dec, 4), s=round(time.time() - t0)), ensure_ascii=False), encoding="utf-8")
print(nom, "corr blocs", round(corr, 5), "ecart median", round(dec, 4), f"{time.time() - t0:.0f} s", flush=True)
