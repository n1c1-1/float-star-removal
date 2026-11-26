# ======================================================================
# prep.py
# Rôle     : Étape 1 du banc : recadrer les 6 piles linéaires, fond neutre, balance des blancs sur les étoiles (piles couleur), piédestal, blanc = 1.
# Étape    : 3. Délinéarisation des images web (23 nov) (voir scripts/LISEZMOI.md)
# Entrées  : image étirée
# Sorties  : image rendue linéaire
# Exemple  : python prep.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Étape 1 du banc : recadrer les 6 piles linéaires, fond neutre, balance des blancs sur les étoiles (piles couleur),
piédestal, blanc = 1. Écrit banc/<nom>_lin.npy (3, H, W) float32 et banc/<nom>_lin.json."""
from __future__ import annotations

import sys
import time

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, __import__("os").path.dirname(__file__))
from commun import BANC, PILES, bruit_laplacien, ecrire_json, mad_sigma, vers_chw  # noqa: E402
from etoiles_outils import detecter, fwhm_median, photometrie  # noqa: E402

COTE = 1600


def main(noms):
    for nom in noms:
        t0 = time.time()
        d = PILES[nom]
        a = np.load(d["chemin"], mmap_mode="r")
        a = vers_chw(a)
        C, H, W = a.shape
        c = min(COTE, H - 100, W - 100)
        y0, x0 = (H - c) // 2, (W - c) // 2
        x = np.array(a[:, y0:y0 + c, x0:x0 + c], dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if C == 1:
            x = np.repeat(x, 3, axis=0)
        sky = np.array([np.median(ch) for ch in x])
        sig = np.array([mad_sigma(bruit_laplacien(ch)[::2, ::2]) for ch in x])
        x = x - sky[:, None, None]
        L = x.mean(0)
        sL = mad_sigma(bruit_laplacien(L)[::2, ::2])
        ys, xs = detecter(L, sL, k=30.0)
        fw = fwhm_median(L, ys, xs)
        ra, rin, rout = max(3.0, 2.5 * fw), max(4.0, 3.5 * fw), max(6.0, 5.0 * fw)
        gains = np.ones(3)
        if d["genre"] == "rgb" and len(ys) > 20:
            fl = np.array([photometrie(ch, ys, xs, ra, rin, rout)[0] for ch in x])
            ok = np.all(fl > 0, axis=0)
            ratio = np.median(fl[:, ok] / fl[1, ok], axis=1)
            gains = 1.0 / ratio
        x = x * gains[:, None, None]
        sig_b = sig * gains
        ped = float(np.mean(sky * gains)) if d["genre"] == "rgb" else 0.0
        ped = max(ped, 30.0 * float(np.mean(sig_b)))
        x = x + ped
        top = float(max(ndi.median_filter(ch, size=3).max() for ch in x))
        x = (x / top).astype(np.float32)
        np.save(BANC / f"{nom}_lin.npy", x)
        meta = dict(nom=nom, source=d["chemin"], quoi=d["quoi"], genre=d["genre"], crop=[int(y0), int(x0), int(c)],
                    ciel_brut=sky.tolist(), gains=gains.tolist(), piedestal=ped / top,
                    sigma=(sig_b / top).tolist(), blanc_brut=top, etoiles_detectees=int(len(ys)), fwhm=fw,
                    ciel_sur_bruit=float(ped / np.mean(sig_b)))
        ecrire_json(BANC / f"{nom}_lin.json", meta)
        print(nom, f"{time.time() - t0:.1f}s", "côté", c, "étoiles", len(ys), "fwhm %.2f" % fw, "gains", np.round(gains, 3),
              "ciel/bruit %.1f" % meta["ciel_sur_bruit"], "sigma/blanc %.2e" % np.mean(sig_b / top), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(PILES))
