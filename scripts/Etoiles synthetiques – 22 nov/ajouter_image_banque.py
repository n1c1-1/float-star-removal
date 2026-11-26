# ======================================================================
# ajouter_image_banque.py
# Rôle     : Ajoute une image à la banque d'étoiles individuelles, à partir de ses SEULES mesures : l'image et son astrométrie (positions et G des étoiles de son champ, comme astrometrie.npz des méthodes) ; aucun catalogue profond, aucun téléc
# Étape    : 4. Étoiles synthétiques réalistes (22 nov) (voir scripts/LISEZMOI.md)
# Entrées  : fonds sans étoiles (cibles de paires), banques de formes d'étoiles
# Sorties  : paires (entrée, cible), planches, test à l'aveugle
# Exemple  : python ajouter_image_banque.py <image .npy|.fits> <astrometrie .npz | -> <nom>
# ======================================================================
"""Ajoute une image à la banque d'étoiles individuelles, à partir de ses SEULES mesures : l'image et son astrométrie
(positions et G des étoiles de son champ, comme astrometrie.npz des méthodes) ; aucun catalogue profond, aucun
téléchargement. Largeur, point zéro, loi du halo et angles des aigrettes sont mesurés sur l'image elle-même
(banque_individuelle.construire). Les unités de l'image n'importent pas : chaque forme est divisée par son flux.

    python ajouter_image_banque.py <image .npy|.fits> <astrometrie .npz | -> <nom>
      sans astrométrie (« - ») : les brillantes sont trouvées par détection (banque_detection.py).
      image : (C, H, W) ou (H, W), fond quelconque (retiré autour de chaque étoile) ; astrométrie : clé « cat » = dict
      avec x, y, g (ou les clés x, y, g à la racine).
    -> banque\\banque2_<nom>.npz ; la banque est prise par etoiles2 au prochain chargement.
"""
from __future__ import annotations

import os as _os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    _os.environ[_k] = "2"                                  # au plus 2 fils (consigne du 22 nov après la saturation)

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import banque_individuelle as BI                                     # noqa: E402


def lire_image(p: Path) -> np.ndarray:
    if p.suffix.lower() in (".fits", ".fit", ".fts"):
        from astropy.io import fits
        a = fits.getdata(p).astype(np.float32)
    else:
        a = np.array(np.load(p, mmap_mode="r"), np.float32)
    if a.ndim == 2:
        a = a[None]
    if a.shape[-1] in (1, 3) and a.shape[0] not in (1, 3):              # (H, W, C) -> (C, H, W)
        a = np.moveaxis(a, -1, 0)
    a[~np.isfinite(a)] = 0
    return np.ascontiguousarray(a)


def lire_astrometrie(p: Path):
    z = np.load(p, allow_pickle=True)
    cat = z["cat"].item() if "cat" in z.files else {k: z[k] for k in ("x", "y", "g")}
    return tuple(np.asarray(cat[k], float) for k in ("x", "y", "g"))


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    img, ast, nom = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    a = lire_image(img)
    if ast in ("-", "") or not Path(ast).exists():                    # tour 5 : pas d'astrométrie -> détection
        import banque_detection as BD
        b = BD.construire(nom, a)
    else:
        b = BI.construire(nom, a=a, cat=lire_astrometrie(Path(ast)))
    if b is None:
        print(nom, ": rien ajouté")
        return 1
    BI.sauver(b)
    print(nom, ": ajoutée,", len(b["G"]), "étoiles ->", BI.OUTB / f"banque2_{nom}.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
