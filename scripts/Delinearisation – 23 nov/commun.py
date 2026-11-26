# ======================================================================
# commun.py
# Rôle     : Chemins et outils communs du banc de délinéarisation (23 nov).
# Étape    : 3. Délinéarisation des images web (23 nov) (voir scripts/LISEZMOI.md)
# Entrées  : image étirée
# Sorties  : image rendue linéaire
# Exemple  : python commun.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Chemins et outils communs du banc de délinéarisation (23 nov)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))
SORTIE = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Delinearisation – 23 nov")
BANC = SORTIE / "banc"            # piles linéaires recadrées + images étirées
RES = SORTIE / "resultats"        # délinéarisées + mesures
for d in (SORTIE, BANC, RES):
    d.mkdir(parents=True, exist_ok=True)

# Six piles linéaires variées (inventaire Bibliotheque images – 22 nov)
PILES = {
    "rosette":  dict(chemin="<DISQUE>/astro_float/Rosette/store/stack/4297845b9624d4a1/drz_full.npy", genre="rgb",
                     quoi="nébuleuse, couleur A7S + Newton"),
    "m51":      dict(chemin="<ESSAIS>/projets/canon6d_m51/work/stack/drz_full.npy", genre="rgb",
                     quoi="galaxie, couleur Canon 6D"),
    "ngc869":   dict(chemin="<BANQUE>/v2/38_smart_dwarfmini_ngc869/store/stack/3aeb8000a5cb6057/drz_full.npy",
                     genre="rgb", quoi="amas, télescope automatique"),
    "m8_sho":   dict(chemin="<ESSAIS>/stars_dataset/_revue_nb/57_comb/input.npy", genre="nb",
                     quoi="nébuleuse, filtres SII Ha OIII (mono)"),
    "ic443_hoo": dict(chemin="<ESSAIS>/stars_dataset/_revue_nb/58_comb/input.npy", genre="nb",
                      quoi="rémanent, filtres Ha OIII (mono)"),
    "pleiades": dict(chemin="<DISQUE>/astro_float/Pleiades/store/stack/233f004e6eaca36b/drz_full.npy", genre="rgb",
                     quoi="amas + nébuleuse par réflexion, couleur A7S + Newton"),
}


def vers_chw(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 2:
        a = a[None]
    if a.shape[-1] == 3 and a.shape[0] != 3:
        a = np.moveaxis(a, -1, 0)
    return a


def lum(a: np.ndarray) -> np.ndarray:
    return a.mean(axis=0)


def ecrire_json(p: Path, d) -> None:
    Path(p).write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")


def lire_json(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def mad_sigma(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    return 1.4826 * float(np.median(np.abs(v - np.median(v))))


def bruit_laplacien(x: np.ndarray) -> np.ndarray:
    """Résidu haute fréquence (pixel moins la moyenne de ses 4 voisins), écart type ramené à celui du pixel pour un
    bruit blanc (facteur sqrt(1 + 1/4))."""
    r = x[1:-1, 1:-1] - 0.25 * (x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:])
    return np.pad(r / np.sqrt(1.25), 1, mode="edge")


def mtf(x, m, lo=0.0, hi=1.0):
    xp = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = ((m - 1.0) * xp) / ((2.0 * m - 1.0) * xp - m)
    return np.where(xp <= 0, 0.0, np.where(xp >= 1, 1.0, y))


def m_pour(x, cible):
    """Le m de la MTF qui envoie x sur cible (définition de l'étirement automatique)."""
    return ((cible - 1.0) * x) / ((2.0 * cible - 1.0) * x - cible)


def lire_etiree(p: Path) -> np.ndarray:
    """Une image étirée relue comme un utilisateur la donnerait : (3, H, W) en 0-1."""
    p = Path(p)
    if p.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        a = tifffile.imread(str(p)).astype(np.float64) / 65535.0
    else:
        from PIL import Image
        a = np.asarray(Image.open(p).convert("RGB")).astype(np.float64) / 255.0
    return vers_chw(a)
