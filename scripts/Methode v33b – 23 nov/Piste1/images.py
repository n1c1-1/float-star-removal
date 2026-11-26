# ======================================================================
# images.py
# Rôle     : Les 6 images du travail : chargement en linéaire, unités de bruit, fond 0 (C, H, W) float32.
# Étape    : 2. Chaîne classique de soustraction guidée par Gaia (v33b) (voir scripts/LISEZMOI.md)
# Entrées  : image empilée (unités de bruit), astrométrie, extrait Gaia profond hors ligne
# Sorties  : image sans étoiles, cartes de classes
# Exemple  : python images.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Les 6 images du travail : chargement en linéaire, unités de bruit, fond 0 (C, H, W) float32."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SR = (Path(__import__("os").environ.get("RACINE_SCRIPTS") or Path(__file__).resolve().parents[2]) / "data")
NB = (Path(__import__("os").environ.get("ESSAIS", "essais")) / "stars_dataset" / "paires_nb")
LIB = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Images jeu 1 – 22 nov")

IMAGES = {
    # nom: (fichier, objet pour le centre approché, type)
    "rosette": (SR / "test_with_stars" / "rosette" / "input.fits", "NGC 2244", "couleur"),
    "ngc6543": (NB / "ngc6543" / "input.npy", "NGC 6543", "filtres"),
    "m57": (SR / "train_with_stars" / "set_24_dslr_astrograph_d810a_eps180_m57" / "input.npy", "M57", "couleur"),
    "horsehead": (SR / "train_with_stars" / "user_Horsehead" / "input.npy", "IC 434", "couleur"),
    "ngc869": (SR / "train_with_stars" / "set_38_smart_dwarfmini_ngc869" / "input.npy", "NGC 869", "couleur"),
    "58_hoo": ((Path(__import__("os").environ.get("ESSAIS", "essais")) / "stars_dataset" / "_revue_combi" / "58_hoo" / "original.npy"), "IC 443", "filtres"),
    # 21 nov soir : essai de la v13 sur une image SHO (palette Hubble, mono bandes étroites SII/Ha/OIII, lu en place)
    "m8_sho": ((Path(__import__("os").environ.get("ESSAIS", "essais")) / "stars_dataset" / "_revue_combi" / "57_sho" / "original.npy"), "M8", "filtres"),
    # 22 nov soir : jeu 1, images de la bibliotheque (peu de carres par image, beaucoup d'instruments). Entrees des piles LCO deja
    # faites (paires_nb/<x>/input.npy : empilement FLOAT, gradient retire, debruite par le reseau livre, unites de bruit, fond 0).
    "m97": (NB / "m97" / "input.npy", "M97", "filtres"),
    "m1": (NB / "m1" / "input.npy", "M1", "filtres"),
    "m82": (NB / "m82" / "input.npy", "M82", "filtres"),
    "m13": (NB / "m13" / "input.npy", "M13", "filtres"),
    "ngc7662": (NB / "ngc7662" / "input.npy", "NGC 7662", "filtres"),
    "ngc6888": (NB / "ngc6888" / "input.npy", "NGC 6888", "filtres"),
    # 22 nov soir : piles de la bibliotheque, entrees faites par Images jeu 1/preparer_entree.py (comme stars_dataset.generate, sans l'outil de reference)
    "ic5070": (LIB / "ic5070" / "input.npy", "IC 5070", "couleur"),
    "wasp74": (LIB / "wasp74" / "input.npy", "WASP-74", "filtres"),
    "ngc7000_tiers1": (LIB / "ngc7000_tiers1" / "input.npy", "NGC 7000", "couleur"),
    "ngc7000_gfx": (LIB / "ngc7000_gfx" / "input.npy", "NGC 7000", "couleur"),
}


def meta(name: str) -> dict:
    p = IMAGES[name][0].parent / "meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def load(name: str) -> np.ndarray:
    """L'image en unités de bruit du ciel, fond 0, (C, H, W) float32."""
    path = IMAGES[name][0]
    m = meta(name)
    if path.suffix == ".fits":
        from astropy.io import fits
        a = fits.getdata(path).astype(np.float32)
        # unités du réseau = (image x scale - background) / unit (meta.json)
        a = (a * np.float32(m["scale"]) - np.asarray(m["background"], np.float32)[:, None, None]) / np.float32(m["unit"])
    else:
        a = np.load(path, mmap_mode="r")
        a = np.array(a, np.float32)
    if a.ndim == 2:
        a = a[None]
    a = np.ascontiguousarray(a)
    a[~np.isfinite(a)] = 0
    if name == "m8_sho":
        # SHO 21 nov (lecture de l'entrée, pas de changement de méthode) : hors de la zone commune des trois filtres, la
        # fusion écrit 0 AVANT la balance ; après décalage et fond retiré ces pixels valent une constante par couleur
        # (-0,950 / -0,849 / -0,840 bruits, 5,1 % de l'image : bord gauche, 20 lignes du bas, coins). Le code attend 0
        # hors champ (valid = img != 0) : ces pixels sont remis à 0. Sans cela, bruit du ciel nul par blocs -> division
        # par zéro dans le modèle d'étoile de l'image entière.
        const = a.reshape(a.shape[0], -1)[:, 0].copy()      # pixel (0, 0) : dans la zone hors champ
        hors = np.all(a == const[:, None, None], axis=0)
        if hors.mean() > 0.01:
            a[:, hors] = 0
    return a


# 23 nov (production v33) : images du jeu 2 (entrees faites : selection_jeu2.csv du dossier Paires jeu 2 ; seules les images retenues par
# les tuiles finales ont leur astrometrie et leur Gaia profond copies ici). Aucune autre ligne de ce fichier ne change.
def _jeu2():
    import csv
    f = (Path(__import__("os").environ.get("RACINE_SCRIPTS") or Path(__file__).resolve().parents[2]) / "Paires jeu 2 – 23 nov" / "selection_jeu2.csv")
    if not f.exists():
        return
    for r in csv.DictReader(open(f, encoding="utf-8-sig"), delimiter=";"):
        IMAGES[r["nom"]] = (Path(r["entree_finale"]), r["objet"], "filtres" if r["genre"] in ("mono", "filtres") else "couleur")


_jeu2()


# 24 nov (images web APOD) : entrees faites par Production APOD – 24 nov\preparer_apod.py (unites de bruit, fond 0), une par dossier.
def _apod():
    for d in sorted((Path(__import__("os").environ.get("DONNEES", "donnees")) / "Paires étoiles APOD – 24 nov" / "entrees").glob("apod_*")):
        IMAGES[d.name] = (d / "input.npy", "APOD", "couleur")


_apod()
