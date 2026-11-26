# ======================================================================
# rejuger.py
# Rôle     : Réapplique aux paires déjà faites les deux tests ajoutés au juge le 22 nov soir (anneaux, niveau ; construire_paire.py).
# Étape    : 2. Juge des zones : poids 0 où la chaîne échoue (v33) (voir scripts/LISEZMOI.md)
# Entrées  : entrée, cible de la chaîne, zones
# Sorties  : carte de poids
# Exemple  : usage : rejuger.py [noms...]   (sans nom : toutes les paires)
# ======================================================================
"""Réapplique aux paires déjà faites les deux tests ajoutés au juge le 22 nov soir (anneaux, niveau ; construire_paire.py).

Ils ne font qu'ajouter des conditions : une grande zone refaite (classe 2 ou 4-8, >= 200 px) de poids > 0 qui échoue
passe à 0. Le reste du poids ne change pas, la chaîne n'est pas refaite. Réécrit le .npz, le panneau poids du .jpg et
ajoute une ligne « rejuge » à paires.jsonl. Sans effet sur une paire faite avec le nouveau juge (idempotent).

usage : rejuger.py [noms...]   (sans nom : toutes les paires)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from construire_paire import OUT, ilots_dans_rejet, restes_etoiles, score_anneaux, score_niveau  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    noms = sys.argv[1:] or [p.stem for p in sorted(OUT.glob("*.npz"))]
    tot = 0
    for nom in noms:
        z = dict(np.load(OUT / f"{nom}.npz"))
        T, cl, W = z["target"], z["classes"], z["weight"].copy()
        L = T.mean(0)
        inv = (cl == 2) | (cl >= 4)
        lab, _ = ndi.label(inv)
        chang = []
        for k, sl in enumerate(ndi.find_objects(lab), 1):
            comp = lab == k
            if comp.sum() < 200 or W[comp].max() == 0:
                continue
            ring = ndi.binary_dilation(comp, iterations=16) & ~ndi.binary_dilation(comp, iterations=4) & ~(inv & ~comp)
            if ring.sum() < 40:
                ring = ndi.binary_dilation(comp, iterations=16) & ~ndi.binary_dilation(comp, iterations=4)
            an, nv = score_anneaux(L, comp), score_niveau(L, comp, ring)
            if an > 0.6 or abs(nv) > 3.0:
                W[comp] = 0.0
                cy, cx = ndi.center_of_mass(comp)
                chang.append(dict(px=int(comp.sum()), x=int(cx), y=int(cy), anneau=round(an, 2), niveau=round(nv, 1)))
        n_re = restes_etoiles(T, cl, W, 3.0)          # FWHM non gardée dans la paire : 3 px (valeur de construire_paire)
        if n_re:
            chang.append(dict(restes_etoiles=n_re))
        n_il = ilots_dans_rejet(W, inv)
        if n_il:
            chang.append(dict(ilots_mis_a_0=n_il))
        if not chang:
            continue
        tot += len(chang)
        z["weight"] = W
        np.savez_compressed(OUT / f"{nom}.npz", **z)
        jp = OUT / f"{nom}.jpg"
        if jp.exists():
            im = cv2.imdecode(np.fromfile(str(jp), np.uint8), cv2.IMREAD_COLOR)
            wv = np.stack([255 * (W == 0), 255 * (W == 0.5), 255 * (W == 1.0)], -1).astype(np.uint8)
            wv[(W == 1.0) & (cl == 0)] = 40
            im[:, -512:] = wv[:, :, ::-1]
            cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(jp))
        r = dict(paire=nom, rejuge="22 nov soir : anneaux, niveau", zones_mises_a_0=chang,
                 poids_1=round(float((W == 1).mean()), 4), poids_05=round(float((W == 0.5).mean()), 4),
                 poids_0=round(float((W == 0).mean()), 4))
        with open(OUT / "paires.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(nom, len(chang), "zone(s) à 0")
    print("total", tot, "zones mises à 0 sur", len(noms), "paires")


if __name__ == "__main__":
    main()
