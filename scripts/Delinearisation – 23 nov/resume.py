# ======================================================================
# resume.py
# Rôle     : Tableau court du banc : resume.py <nom du csv dans resultats> -> tableau.txt à côté de la planche.
# Étape    : 3. Délinéarisation des images web (23 nov) (voir scripts/LISEZMOI.md)
# Entrées  : image étirée
# Sorties  : image rendue linéaire
# Exemple  : python resume.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Tableau court du banc : resume.py <nom du csv dans resultats> -> tableau.txt à côté de la planche."""
from __future__ import annotations

import csv
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__file__))
from commun import RES, SORTIE  # noqa: E402


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def main():
    rows = list(csv.DictReader(open(RES / sys.argv[1], encoding="utf-8-sig"), delimiter=";"))
    meths = list(dict.fromkeys(r["methode"] for r in rows))
    recs = list(dict.fromkeys(r["recette"] for r in rows))
    L = []
    L.append("Banc de délinéarisation, 23 nov : 6 piles x 6 traitements = 36 images étirées. Idéal = 1 (0 % pour les couleurs).")
    L.append("Médiane sur les images traitées [écart entre les 25e et 75e centiles] ; « refus » = la méthode refuse l'image.")
    L.append("")
    cols = [("pente_flux", "flux étoiles (pente log-log)"), ("r50_brillantes", "taille étoiles brillantes"),
            ("bruit_dependance", "bruit fort/ciel"), ("pente_rapport", "pente du fond"),
            ("couleur_etoiles_pct", "couleur étoiles %"), ("couleur_neb_pct", "couleur nébuleuse %")]
    L.append(f"{'méthode':<10} {'refus':>6} " + " ".join(f"{c[1]:>28}" for c in cols))
    for m in meths:
        rs = [r for r in rows if r["methode"] == m]
        ok = [r for r in rs if np.isfinite(f(r["pente_flux"]))]
        cells = []
        for k, _ in cols:
            v = np.array([f(r[k]) for r in ok])
            v = v[np.isfinite(v)]
            cells.append(f"{np.median(v):.2f} [{np.percentile(v, 25):.2f}-{np.percentile(v, 75):.2f}]" if v.size else "-")
        L.append(f"{m:<10} {len(rs) - len(ok):>3}/{len(rs):<2} " + " ".join(f"{c:>28}" for c in cells))
    L.append("")
    L.append("Flux des étoiles (pente log-log, idéal 1) par traitement, médiane des 6 piles :")
    L.append(f"{'traitement':<24} " + " ".join(f"{m:>10}" for m in meths))
    for rc in recs:
        cells = []
        for m in meths:
            v = np.array([f(r["pente_flux"]) for r in rows if r["methode"] == m and r["recette"] == rc])
            n_ref = int(np.sum(~np.isfinite(v)))
            v = v[np.isfinite(v)]
            cells.append((f"{np.median(v):.2f}" if v.size else "-") + (f" ({n_ref}r)" if n_ref else ""))
        L.append(f"{rc:<24} " + " ".join(f"{c:>10}" for c in cells))
    L.append("")
    L.append("Couleur des étoiles (écart %, idéal 0) par traitement, médiane des 6 piles :")
    for rc in recs:
        cells = []
        for m in meths:
            v = np.array([f(r["couleur_etoiles_pct"]) for r in rows if r["methode"] == m and r["recette"] == rc])
            v = v[np.isfinite(v)]
            cells.append(f"{np.median(v):.0f}" if v.size else "-")
        L.append(f"{rc:<24} " + " ".join(f"{c:>10}" for c in cells))
    txt = "\n".join(L)
    (SORTIE / "tableau.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
