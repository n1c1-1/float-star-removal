# ======================================================================
# mesures.py
# Role     : Per-tile measurements (planches_choix grid): one image at a time, 2 threads, no graphics card. -> mesures/<image>.csv
# Step     : 2. Tile selection (23 Nov) (see scripts/LISEZMOI.md)
# Inputs   : images, mosaics
# Outputs  : list of tiles
# Example  : python mesures.py [arguments]  (see the docstring; paths via the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Per-tile measurements (planches_choix grid): one image at a time, 2 threads, no graphics card. -> mesures/<image>.csv"""
import os, sys, csv
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[k] = "2"
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np, cv2
cv2.setNumThreads(2)
ICI = Path(__file__).resolve().parent
JP2 = ICI.parent / "Paires jeu 2 – 23 nov"
sys.path.insert(0, str(JP2))
import planches_choix as pc
SRC = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Choix des tuiles – 23 nov")
(ICI / "mesures").mkdir(exist_ok=True)
ECART = {r["nom"] for r in csv.DictReader(open(JP2 / "ecartees.csv", encoding="utf-8-sig"), delimiter=";")}


def mad(a):
    return 1.4826 * float(np.median(np.abs(a - np.median(a)))) + 1e-12


def faire(nom):
    a = pc.charger(nom)
    a = a[None] if a.ndim == 2 else a
    C, H, W = a.shape
    lum = np.zeros((H, W), np.float32); inval = np.zeros((H, W), bool)
    for c in range(min(C, 3)):
        p = np.asarray(a[c], np.float32)
        inval |= ~np.isfinite(p) | (p == 0)
        lum += np.nan_to_num(p)
    lum /= min(C, 3)
    # binning 8 px
    h8, w8 = H // 8, W // 8
    b = cv2.resize(lum[:h8 * 8, :w8 * 8], (w8, h8), interpolation=cv2.INTER_AREA)
    iv = cv2.resize(inval[:h8 * 8, :w8 * 8].astype(np.float32), (w8, h8), interpolation=cv2.INTER_AREA)
    bg = float(np.median(b)); sg = mad(b)
    # bruit propre : MAD de la difference a la mediane locale (hors structure large)
    sg = max(mad(b - cv2.medianBlur(b, 5)), 1e-9) * 1.0
    diff = cv2.medianBlur(b, 5)            # etoiles supprimees (40 px)
    bg = float(np.percentile(diff, 5))      # plancher du ciel (pas la mediane : champs remplis de nebuleuse)
    # etoiles : passe-haut plein champ
    hp = lum - cv2.GaussianBlur(lum, (0, 0), 6)
    shp = mad(hp[::2, ::2])
    mx = cv2.dilate(hp, np.ones((7, 7), np.uint8))
    pk = (hp == mx) & (hp > 8 * shp)
    ys, xs = np.nonzero(pk); amp = hp[ys, xs] / shp
    lvl = float(np.percentile(b, 99.95))
    gros = (b > bg + 0.5 * (lvl - bg)).astype(np.float32)
    lf = cv2.GaussianBlur(b, (0, 0), 8)
    rows = []
    for r in csv.DictReader(open(SRC / f"tuiles_{nom}.csv", encoding="utf-8-sig"), delimiter=";"):
        x, y = int(r["x"]), int(r["y"])
        sl = (slice(y // 8, (y + 512) // 8), slice(x // 8, (x + 512) // 8))
        bt = b[sl]; dt = diff[sl]
        m = (xs >= x) & (xs < x + 512) & (ys >= y) & (ys < y + 512)
        am = amp[m]
        rows.append(dict(image=nom, numero=r["numero"], x=x, y=y, jeu1=r["jeu1"],
            diffus=round(float((dt > bg + 5 * sg).mean()), 4),
            tout=round(float((bt > bg + 5 * sg).mean()), 4),
            nb_etoiles=int(m.sum()), petites=int((am < 30).sum()), moyennes=int(((am >= 30) & (am < 200)).sum()),
            brillantes=int((am >= 200).sum()), amp_max=round(float(am.max()) if len(am) else 0, 1),
            gros=round(float(gros[sl].mean()), 4), invalide=round(float(iv[sl].mean()), 4),
            fond=round(float((np.percentile(bt, 5) - bg) / sg), 1),
            degrade=round(float((lf[sl].max() - lf[sl].min()) / sg), 1)))
    with open(ICI / "mesures" / f"{nom}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=";"); w.writeheader(); w.writerows(rows)
    print(nom, (H, W), len(rows), "tuiles", len(xs), "etoiles", flush=True)


noms = sys.argv[1:] or sorted(p.stem for p in SRC.glob("*.jpg") if p.stem not in ECART)
for nom in noms:
    if nom in ECART:
        continue
    try:
        faire(nom)
    except Exception as e:
        print(nom, "ECHEC", repr(e)[:300], flush=True)
