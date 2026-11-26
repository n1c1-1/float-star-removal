# ======================================================================
# delineariser.py
# Rôle     : Délinéarise les images web APOD choisies (retenues + douteuses) avec etoiles_auto (copie de rien : appelée telle quelle depuis le dossier de la délinéarisation du 23 nov).
# Étape    : 3. Délinéarisation et astrométrie des images web APOD (24 nov) (voir scripts/LISEZMOI.md)
# Entrées  : images web APOD téléchargées
# Sorties  : images linéaires, astrométrie, tuiles
# Exemple  : python delineariser.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Délinéarise les images web APOD choisies (retenues + douteuses) avec etoiles_auto (copie de rien : appelée telle quelle
depuis le dossier de la délinéarisation du 23 nov). Une image à la fois, reprise possible. Sorties : lineaire/NNN.npy (float32, 3 x H x W),
controle\mesures.jsonl.  Usage : delineariser.py [numéros...]"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from outils_apod import CTRL, LIN, choix, lire, mesures_etoiles, prestats  # noqa: E402
import methodes  # noqa: E402
from commun import bruit_laplacien, mad_sigma  # noqa: E402
import courbe_etoiles2 as ce  # noqa: E402


def main():
    sel = [int(a) for a in sys.argv[1:]]
    fait = set()
    mj = CTRL / "mesures.jsonl"
    if mj.exists():
        fait = {json.loads(l)["numero"] for l in open(mj, encoding="utf-8")}
    for n, v, fic, titre in choix():
        if (sel and n not in sel) or (not sel and n in fait):
            continue
        t0 = time.time()
        y = lire(fic)
        row = {"numero": n, "verdict_orch": v, "fichier": fic, "titre": titre, "taille": list(y.shape[1:])}
        row.update(prestats(y))
        try:
            D, info = methodes.etoiles_auto(y)
        except Exception as e:  # noqa: BLE001
            row.update(echec=repr(e)[:200], duree_s=round(time.time() - t0, 1))
            open(mj, "a", encoding="utf-8").write(json.dumps(row, ensure_ascii=False) + "\n")
            print(n, "ECHEC", e, flush=True)
            continue
        D = np.asarray(D, np.float32)
        np.save(LIN / f"{n:03d}.npy", D)
        info = dict(info) if isinstance(info, dict) else {"info": str(info)}
        for k in ("noeuds", "P"):
            info.pop(k, None)
        row["info"] = info
        lum, Dl = y.mean(0), D.mean(0)
        sig = max(mad_sigma(bruit_laplacien(lum)[::2, ::2]), 1.0 / 255 / np.sqrt(12))
        m_orig, _ = mesures_etoiles(lum, lum, sig)
        m_lin, (ys, xs, F, r50, ok) = mesures_etoiles(lum, Dl, sig)
        row["etoiles_orig"] = m_orig
        row["etoiles_lin"] = m_lin
        # courbe : étendue dynamique entre le ciel et 0,98, et monotonie
        ciel = float(np.median(lum))
        row["ciel_etire"] = round(ciel, 4)
        row["lin_ciel_med"] = float(np.median(Dl)); row["lin_p999"] = float(np.percentile(Dl, 99.9)); row["lin_max"] = float(Dl.max())
        row["dyn_decades"] = float(np.log10(max(np.percentile(Dl, 99.9), 1e-12) / max(np.median(Dl), 1e-12))) if np.median(Dl) > 0 else None
        # étoiles pour les gros plans : brillante = plus brillante de la tranche non saturée, faible = vers le 25e centile
        idx = np.flatnonzero(ok)
        if len(idx) >= 25:
            o = idx[np.argsort(F[idx])]
            b, f = o[int(0.97 * len(o))], o[int(0.25 * len(o))]
            row["zoom_etoiles"] = {"brillante": [float(ys[b]), float(xs[b]), float(F[b]), float(r50[b])],
                                   "faible": [float(ys[f]), float(xs[f]), float(F[f]), float(r50[f])]}
        row["duree_s"] = round(time.time() - t0, 1)
        open(mj, "a", encoding="utf-8").write(json.dumps(row, ensure_ascii=False) + "\n")
        print(n, fic, row["duree_s"], "s", info.get("couleur"), "cout", info.get("cout_depart"), "->", info.get("cout_fin"),
              "etoiles", info.get("etoiles_utilisees"), "r50rap orig/lin %.2f/%.2f" % (m_orig.get("r50_rapport", np.nan),
              m_lin.get("r50_rapport", np.nan)), flush=True)


if __name__ == "__main__":
    main()
