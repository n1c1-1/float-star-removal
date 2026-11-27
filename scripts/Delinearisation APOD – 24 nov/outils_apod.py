# ======================================================================
# outils_apod.py
# Role     : Common tools: paths, reading of the choices, measurements on an image (stretched or linear).
# Step     : 3. De-linearisation and astrometry of the APOD web images (24 Nov) (see scripts/LISEZMOI.md)
# Inputs   : downloaded APOD web images
# Outputs  : linear images, astrometry, tiles
# Example  : python outils_apod.py [arguments]  (see the docstring; paths via the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Common tools: paths, reading of the choices, measurements on an image (stretched or linear). Reuses, without modifying it,
the etoiles_auto method of “Delinearisation – 23 nov” (commit 0dfc2c2)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ICI = Path(__file__).resolve().parent
M23 = ICI.parent / "Delinearisation – 23 nov"
sys.path.insert(0, str(M23))
SRC = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Images web APOD – 24 nov")
OUT = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Delinearisation APOD – 24 nov")
LIN, CTRL = OUT / "lineaire", OUT / "controle"


def choix():
    """-> liste de (numéro, verdict, fichier, titre) pour les retenues et douteuses."""
    idx = {}
    with open(SRC / "index.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f, delimiter=";"):
            idx[int(r["numero"])] = r
    out = []
    for ligne in open(SRC / "choix_orchestrateur.csv", encoding="utf-8-sig"):
        if ligne.startswith("#") or ligne.startswith("numero"):
            continue
        n, v = ligne.strip().split(";")[:2]
        if v in ("retenue", "douteuse"):
            r = idx[int(n)]
            out.append((int(n), v, r["fichier"], r["titre"]))
    return out


def lire(fichier):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    a = np.asarray(Image.open(SRC / "brut" / fichier).convert("RGB")).astype(np.float32) / 255.0
    return np.moveaxis(a, -1, 0)


def prestats(y):
    """Fond écrasé et zones écrêtées d'une image étirée (3, H, W) 0-1."""
    lum = y.mean(0)
    s = {"noir_zero": float((y.max(0) <= 0.5 / 255).mean()), "noir_2": float((lum <= 2.0 / 255).mean())}
    m = y.max(0) >= 253.5 / 255
    lab, n = ndi.label(m)
    if n:
        aire = np.bincount(lab.ravel())[1:]
        s["ecrete_total"] = float(m.mean())
        s["ecrete_larges"] = float(aire[aire >= 400].sum() / m.size)      # hors étoiles (>= 400 px)
        s["ecrete_plus_grand_px"] = int(aire.max())
    else:
        s["ecrete_total"] = s["ecrete_larges"] = 0.0; s["ecrete_plus_grand_px"] = 0
    return s


def mesures_etoiles(lum_orig, lum_x, sig_orig):
    """Étoiles détectées sur l'image d'origine (non saturées) ; flux et taille (r50) mesurés sur lum_x (même grille).
    -> dict + positions. Rapport de taille = r50 des 10 % les plus brillantes / r50 du tiers le plus faible."""
    from etoiles_outils import detecter, fwhm_median, photometrie
    ys, xs = detecter(lum_orig, sig_orig, k=15.0, isol=14, bord=20, max_n=3000, plafond=0.93)
    fw = fwhm_median(lum_orig, ys, xs) if len(ys) else 3.0
    ap = (max(3.0, 2.5 * fw), max(4.0, 3.5 * fw), max(6.0, 5.0 * fw))
    F, r50, _ = photometrie(lum_x, ys, xs, *ap)
    ok = np.isfinite(F) & (F > 0) & np.isfinite(r50)
    out = {"n_etoiles": int(ok.sum()), "fwhm_px": round(float(fw), 2)}
    if ok.sum() < 25:
        out.update(r50_rapport=np.nan, pente_taille=np.nan, decades_flux=np.nan)
        return out, (ys, xs, F, r50, ok)
    lf, lr = np.log10(F[ok]), np.log10(r50[ok])
    q33, q90 = np.percentile(lf, [33, 90])
    out["r50_rapport"] = float(10 ** (np.median(lr[lf >= q90]) - np.median(lr[lf <= q33])))
    k = np.polyfit(lf, lr, 1)[0]
    out["pente_taille"] = float(k)
    out["decades_flux"] = float(q90 - q33)
    return out, (ys, xs, F, r50, ok)
