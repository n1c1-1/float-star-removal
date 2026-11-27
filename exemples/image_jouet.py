# ======================================================================
# image_jouet.py
# Role     : makes a very small linear test image (sky + noise + 4 stars of different brightnesses), to try the
#            network in a few seconds before applying it to a real image.
# Inputs   : none (fixed seed).
# Outputs  : the requested .npy file: array (3, 512, 512) float32, in raw image units (sky ~ 1000, noise ~ 5).
# Example  : python exemples/image_jouet.py image_jouet.npy
# ======================================================================
"""Toy image: a flat noisy background and four Gaussian stars (peaks of 60, 200, 800 and 3000 times the noise)."""
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
