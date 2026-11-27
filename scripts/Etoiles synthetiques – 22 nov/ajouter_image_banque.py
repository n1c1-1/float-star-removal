# ======================================================================
# ajouter_image_banque.py
# Role     : Adds an image to the bank of individual stars, from its ONLY measurements: the image and its astrometry (positions and G of the stars of its field, like the methods' astrometrie.npz); no deep catalogue, no download.
# Step     : 4. Realistic synthetic stars (22 Nov) (see scripts/LISEZMOI.md)
# Inputs   : starless backgrounds (pair targets), star shape banks
# Outputs  : pairs (input, target), sheets, blind test
# Example  : python ajouter_image_banque.py <image .npy|.fits> <astrometry .npz | -> <name>
# ======================================================================
"""Adds an image to the bank of individual stars, from its ONLY measurements: the image and its astrometry
(positions and G of the stars of its field, like the methods' astrometrie.npz); no deep catalogue, no
download. Width, zero point, halo law and spike angles are measured on the image itself
(banque_individuelle.construire). The image's units do not matter: each shape is divided by its flux.

    python ajouter_image_banque.py <image .npy|.fits> <astrometry .npz | -> <name>
      without astrometry (“-”): the bright stars are found by detection (banque_detection.py).
      image: (C, H, W) or (H, W), any background (removed around each star); astrometry: key “cat” = dict
      with x, y, g (or the keys x, y, g at the root).
    -> banque\\banque2_<name>.npz; the bank is picked up by etoiles2 at the next load.
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
