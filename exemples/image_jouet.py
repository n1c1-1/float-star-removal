# ======================================================================
# image_jouet.py
# Rôle     : fabrique une toute petite image linéaire de test (ciel + bruit + 4 étoiles de brillances différentes), pour
#            essayer le réseau en quelques secondes avant de l'appliquer à une vraie image.
# Entrées  : aucune (graine fixe).
# Sorties  : le fichier .npy demandé : tableau (3, 512, 512) float32, en unités brutes d'image (ciel ≈ 1000, bruit ≈ 5).
# Exemple  : python exemples/image_jouet.py image_jouet.npy
# ======================================================================
"""Image jouet : fond plat bruité et quatre étoiles gaussiennes (pics de 60, 200, 800 et 3000 fois le bruit)."""
import sys

import numpy as np


def main(sortie="image_jouet.npy"):
    rng = np.random.default_rng(1)
    h = w = 512
    img = 1000 + rng.normal(0, 5, (3, h, w)).astype(np.float32)
    yy, xx = np.mgrid[:h, :w]
    for x, y, amplitude in [(100, 120, 3000), (300, 200, 800), (400, 400, 200), (200, 350, 60)]:
        g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 2.5 ** 2))
        img += amplitude * g[None] * np.array([1, 0.9, 0.8], np.float32)[:, None, None]
    np.save(sortie, img)
    print("écrit", sortie, img.shape)


if __name__ == "__main__":
    main(*(sys.argv[1:2]))
