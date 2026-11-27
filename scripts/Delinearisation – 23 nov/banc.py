# ======================================================================
# banc.py
# Role     : Bench: de-linearises each stretched image with each method and compares it with its linear stack.
# Step     : 3. De-linearisation of web images (23 Nov) (see scripts/LISEZMOI.md)
# Inputs   : stretched image
# Outputs  : image made linear
# Example  : Usage: banc.py <method>[,<method>...] [stacks...]   -> resultats/mesures_<methods>.csv + summary on screen.
# ======================================================================
"""Bench: de-linearises each stretched image with each method and compares it with its linear stack.
Usage: banc.py <method>[,<method>...] [stacks...]   -> resultats/mesures_<methods>.csv + summary on screen."""
from __future__ import annotations

import csv
import sys
import time

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__file__))
from commun import BANC, PILES, RES, lire_etiree  # noqa: E402
from etirer import REC  # noqa: E402
from mesurer import Reference, mesurer  # noqa: E402
import methodes  # noqa: E402

CLES = ("pente_flux", "compression", "flux_faibles", "flux_moyennes", "flux_brillantes", "r50_faibles", "r50_moyennes",
        "r50_brillantes", "bruit_ciel_rapport", "bruit_fort_rapport", "bruit_dependance", "fond_ecart_sigma",
        "pente_rapport", "couleur_etoiles_pct", "couleur_neb_pct", "n_etoiles", "signal_fort_sigma", "echelle_a")


def main():
    meths = sys.argv[1].split(",")
    piles = sys.argv[2:] or list(PILES)
    rows = []
    for p in piles:
        L = np.load(BANC / f"{p}_lin.npy").astype(np.float64)
        ref = Reference(L)
        for r in REC:
            f = next(BANC.glob(f"{p}__{r}.*[fg]"))
            y = lire_etiree(f)
            for m in meths:
                t0 = time.time()
                D, info = methodes.METHODES[m](y)
                dt = time.time() - t0
                if D is None:
                    rows.append(dict({k: np.nan for k in CLES}, pile=p, recette=r, methode=m, duree_s=round(dt, 1),
                                     info=str(info)))
                    print(p, r, m, "REFUS", info, flush=True)
                    continue
                np.save(RES / f"{p}__{r}__{m}.npy", D.astype(np.float32))
                o = mesurer(ref, D)
                o.update(pile=p, recette=r, methode=m, duree_s=round(dt, 1), info=str(info)[:300])
                rows.append(o)
                print(p, r, m, f"{dt:.1f}s", "pente %.2f compr %.2f r50b %.2f bruitdep %.2f coulE %.0f%% coulN %.0f%%" % (
                    o["pente_flux"], o["compression"], o["r50_brillantes"], o["bruit_dependance"], o["couleur_etoiles_pct"],
                    o["couleur_neb_pct"]), flush=True)
    out = RES / f"mesures_{'_'.join(meths)}{'' if len(sys.argv) < 3 else '__' + '_'.join(piles)}.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["pile", "recette", "methode", *CLES, "duree_s", "info"], delimiter=";")
        w.writeheader()
        for o in rows:
            w.writerow({k: (f"{o[k]:.4g}" if isinstance(o[k], float) else o[k]) for k in w.fieldnames})
    print("écrit", out)


if __name__ == "__main__":
    main()
