# ======================================================================
# galerie.py
# Role     : Gallery of 24 free draws of the v2 generator (no imposed parameter), each on a real star-free 512 px tile (outputs of the last t3 pass of methods v21-v24, read in place).
# Step     : 4. Realistic synthetic stars (22 Nov) (see scripts/LISEZMOI.md)
# Inputs   : star-free backgrounds (pair targets), star shape banks
# Outputs  : pairs (input, target), sheets, blind test
# Example  : Usage: python galerie.py [graine]
# ======================================================================
"""Gallery of 24 free draws of the v2 generator (no imposed parameter), each on a real star-free 512 px tile
(outputs of the last t3 pass of methods v21-v24, read in place).
Outputs: A_REGARDER – 21 nov soir\\Synth_11_galerie_post.png (arcsinh 60) and Synth_11_galerie_fort.png ; galerie_fiches.json.
Usage: python galerie.py [graine]"""
from __future__ import annotations

import os as _os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    _os.environ[_k] = "2"                                  # au plus 2 fils (consigne du 22 nov après la saturation)

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_etoiles as A                                            # noqa: E402
import etoiles2 as E                                                 # noqa: E402

# tour 2 : seulement de bonnes tuiles sans étoiles (consigne du coordinateur) : écartées les sorties abîmées autour de
# sigma Ori et d'Alnitak (v24), M8 c et d (texture résiduelle forte à l'étirement) et NGC 2023 (disque pâle recréé)
TUILES = [
    ("Methode v21 SHO – 22 nov", "decoupe6"), ("Methode v21 SHO – 22 nov", "coeur_est"),
    ("Methode v21 SHO – 22 nov", "a_coeur"), ("Methode v21 SHO – 22 nov", "b_filaments"),
    ("Methode v22 M57 – 22 nov", "a_m57"), ("Methode v22 M57 – 22 nov", "b_brillante"),
    ("Methode v22 M57 – 22 nov", "c_champ_dense"), ("Methode v22 M57 – 22 nov", "d_fond_calme"),
    ("Methode v23 Horsehead – 22 nov", "a_tete_de_cheval"),
]


def main():
    g0 = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    tiles = []
    for d, t in TUILES:
        p = A.SR / d / "sorties" / t / "B" / "t3" / "sans_etoiles_tuile.npy"
        if p.exists():
            tiles.append((t, np.load(p).astype(np.float32)))
    print("tuiles", len(tiles), flush=True)
    post, fort, fiches = [], [], []
    for k in range(24):
        t0 = time.time()
        name, bg = tiles[k % len(tiles)]
        o = E.ajouter_etoiles(bg, [g0, k])
        f = o["fiche"]
        im = f["image"]
        inp = o["entree"]
        bk = float(np.median(bg[1])); wt = float(np.percentile(inp[1], 99.8))
        sig = np.array([1.4826 * np.median(np.abs(b - np.median(b))) for b in bg]) + 1e-3
        p1 = (A.stretch60(inp, bk, wt) * 255).astype(np.uint8)
        p2 = (A.stretch_strong(inp, sig, np.asarray(im["sat"])) * 255).astype(np.uint8)
        ag = im["aigrettes"]
        nsp = len(ag["angles"]) * (2 if ag["dedoublement"] else 1)
        lab = (f"#{k + 1} {im['instrument']['kind']} {nsp} aigr{' doubles' if ag['dedoublement'] else ''} "
               f"fw {im['fwhm']:.1f} a50 {im['a50']:.1e} plat {im['plateau'][0]:.2f}-{im['plateau'][1]:.2f} "
               f"{im['mode']} {'pointilles' if im['pointilles'] else ''} geantes {len(f['geantes'])} n {f['etoiles']}")
        for lst, p in ((post, p1), (fort, p2)):
            band = np.full((26, p.shape[1], 3), 255, np.uint8)
            cv2.putText(band, lab, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
            lst.append(np.concatenate([band, p], 0))
        fiches.append(dict(numero=k + 1, tuile=name, etoiles_ecretees_px=int(o["ecretes"].sum()), **f))
        print(k + 1, lab, f"{time.time() - t0:.1f} s", flush=True)
    for lst, nom in ((post, "post"), (fort, "fort")):
        rows = [np.concatenate(sum([[im_, np.full((im_.shape[0], 8, 3), 255, np.uint8)] for im_ in lst[r:r + 6]], [])[:-1], 1)
                for r in range(0, 24, 6)]
        sheet = np.concatenate(sum([[r_, np.full((8, r_.shape[1], 3), 255, np.uint8)] for r_ in rows], [])[:-1], 0)
        ok, buf = cv2.imencode(".png", cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)); assert ok
        buf.tofile(str(A.OUT / f"Synth_11_galerie_{nom}.png"))
    (HERE / "galerie_fiches.json").write_text(json.dumps(fiches, ensure_ascii=False, indent=1, default=str),
                                              encoding="utf-8")


if __name__ == "__main__":
    main()
