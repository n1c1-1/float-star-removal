# ======================================================================
# inferer_une_image.py
# Rôle     : applique un réseau publié (E3, N5, N7 ou le mélange N7/N5) à UNE image à soi, sans l'application FLOAT.
# Étape    : 14. Utiliser un réseau (voir scripts/LISEZMOI.md)
# Entrées  : une image linéaire couleur (3, H, W) en .npy, ou en .fits / .tif / .tiff (empilement calibré, ciel non étiré),
#            et un fichier de poids .pt (voir modeles/LISEZMOI.md pour le téléchargement).
# Sorties  : <sortie>/starless.npy et stars.npy (unités de bruit), starless.tif et stars.tif (linéaires, 32 bits),
#            apercu.jpg (entrée | sans étoiles | étoiles, même étirement asinh).
# Exemple  : python scripts/inferer_une_image.py mon_empilement.fits --poids modeles/N7N5_075.pt --sortie sortie/
# ======================================================================
"""Applique un réseau de retrait d'étoiles à une image linéaire.

Ce que fait le script, dans l'ordre :
1. lit l'image, la met en (3, H, W) float32 ;
2. retire le ciel de chaque couleur (médiane robuste) et divise par une estimation du bruit du ciel : ce sont les
   « unités de bruit » dans lesquelles le réseau a été entraîné (le réseau voit asinh(v / 3) / 10) ;
3. applique le réseau par tuiles de 1024 px recouvertes (256 px), fondues par une fenêtre en cosinus ;
4. remet les unités d'origine : image sans étoiles = (résultat * bruit) + ciel ; étoiles = image - image sans étoiles.

Limites de cette version simple (voir docs/AMELIORER.md) :
- une seule échelle : les très grosses étoiles à large halo (G < 6,5) demandent l'inférence à deux échelles
  (scripts/Portée M27 – 25 nov/deux_echelles_v1v2.py), qui a besoin d'un extrait du catalogue Gaia de l'image ;
- l'estimation du bruit est celle d'une image sans demi-piles (écart robuste des différences de pixels voisins) : un
  empilement qui a un bruit très corrélé (drizzle) sera sous-estimé, et le résultat moins bon. Mieux : fournir
  --bruit avec la valeur mesurée sur le ciel, en unités de l'image.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))


def lire(chemin):
    chemin = Path(chemin)
    suffixe = chemin.suffix.lower()
    if suffixe == ".npy":
        a = np.load(chemin)
    elif suffixe in (".fits", ".fit", ".fts"):
        from astropy.io import fits
        a = fits.getdata(chemin)
    elif suffixe in (".tif", ".tiff"):
        import tifffile
        a = tifffile.imread(chemin)
    else:
        raise SystemExit(f"format non pris en charge : {suffixe}")
    a = np.asarray(a, np.float32)
    if a.ndim == 2:
        raise SystemExit("image monochrome : mettre trois fois le même canal, ou lancer le script canal par canal")
    if a.shape[0] not in (3,) and a.shape[-1] == 3:
        a = np.moveaxis(a, -1, 0)
    if a.shape[0] != 3:
        raise SystemExit(f"forme inattendue {a.shape} : attendu (3, H, W) ou (H, W, 3)")
    return np.ascontiguousarray(a)


def ciel_et_bruit(canal, bruit_impose=None):
    """Ciel = médiane des pixels sous le 60e centile ; bruit = écart robuste des différences de pixels voisins sur le ciel."""
    v = canal[::3, ::3].ravel()
    v = v[np.isfinite(v)]
    seuil = np.percentile(v, 60)
    ciel = float(np.median(v[v <= seuil]))
    if bruit_impose:
        return ciel, float(bruit_impose)
    d = (canal[:, 1:] - canal[:, :-1])
    d = d[::2, ::2]
    d = d[np.isfinite(d)]
    d = d[np.abs(d) < 5 * np.percentile(np.abs(d), 50) * 1.4826 + 1e-12]
    bruit = float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)) or 1.0
    return ciel, bruit


def apercu(entree, sans, etoiles, chemin):
    import cv2
    def etirer(a):
        v = np.arcsinh(np.clip(a, -3, None) / 3.0) / np.arcsinh(300.0 / 3.0)
        return np.clip(v * 255, 0, 255).astype(np.uint8)
    h = 1200
    vignettes = []
    for a in (entree, sans, etoiles):
        e = np.moveaxis(etirer(a), 0, -1)[..., ::-1]
        f = h / e.shape[0]
        vignettes.append(cv2.resize(e, (int(e.shape[1] * f), h), interpolation=cv2.INTER_AREA))
    cv2.imwrite(str(chemin), np.concatenate(vignettes, 1), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("image")
    ap.add_argument("--poids", default=os.environ.get("MODELE_RESEAU", "modeles/N7N5_075.pt"))
    ap.add_argument("--sortie", default="sortie")
    ap.add_argument("--bruit", type=float, default=None, help="bruit du ciel en unités de l'image (sinon estimé)")
    ap.add_argument("--carte", default="auto", help="cuda, cpu ou auto")
    ap.add_argument("--tuile", type=int, default=1024)
    ap.add_argument("--recouvrement", type=int, default=256)
    a = ap.parse_args(argv)

    import torch
    os.environ["MODELE_RESEAU"] = a.poids
    from astro_gpu.stages import star_removal as SR
    dev = torch.device("cuda" if (a.carte == "auto" and torch.cuda.is_available()) or a.carte == "cuda" else "cpu")
    print("carte :", dev, "| poids :", a.poids)
    img = lire(a.image)
    unites = np.empty_like(img)
    ciels, bruits = [], []
    for c in range(3):
        ciel, bruit = ciel_et_bruit(img[c], a.bruit)
        ciels.append(ciel)
        bruits.append(bruit)
        unites[c] = (img[c] - ciel) / bruit
    print("ciel par couleur :", [round(x, 5) for x in ciels], "| bruit par couleur :", [round(x, 5) for x in bruits])
    net = SR.network(dev)
    sans = SR.apply(net, unites, dev, None, tile=a.tuile, overlap=a.recouvrement)
    sortie = Path(a.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    sans_lin = np.empty_like(img)
    for c in range(3):
        sans_lin[c] = sans[c] * bruits[c] + ciels[c]
    etoiles = img - sans_lin
    np.save(sortie / "starless.npy", sans)
    np.save(sortie / "stars.npy", unites - sans)
    try:
        import tifffile
        tifffile.imwrite(sortie / "starless.tif", np.moveaxis(sans_lin, 0, -1).astype(np.float32))
        tifffile.imwrite(sortie / "stars.tif", np.moveaxis(etoiles, 0, -1).astype(np.float32))
    except Exception as e:  # noqa: BLE001
        print("TIFF non écrit :", e)
    apercu(unites, sans, unites - sans, sortie / "apercu.jpg")
    print("écrit dans", sortie)


if __name__ == "__main__":
    main()
