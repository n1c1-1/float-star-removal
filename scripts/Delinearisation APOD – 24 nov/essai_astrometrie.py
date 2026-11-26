# ======================================================================
# essai_astrometrie.py
# Rôle     : Essai : résolution astrométrique AVEUGLE (tout le ciel, échelle libre) d'images web, par le solveur local de FLOAT (asterion, Tycho-2), hors ligne.
# Étape    : 3. Délinéarisation et astrométrie des images web APOD (24 nov) (voir scripts/LISEZMOI.md)
# Entrées  : images web APOD téléchargées
# Sorties  : images linéaires, astrométrie, tuiles
# Exemple  : python essai_astrometrie.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Essai : résolution astrométrique AVEUGLE (tout le ciel, échelle libre) d'images web, par le solveur local de FLOAT
(asterion, Tycho-2), hors ligne. Usage : essai_astrometrie.py <numéros>"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, r"<FLOAT>")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import os  # noqa: E402
os.environ["ASTRO_GPU_GAIA_EXTRACTS"] = "<CACHE_FLOAT>/cache/gaia_extracts"
import numpy as np  # noqa: E402
from astro_gpu.core import asterion  # noqa: E402

from outils_apod import CTRL, choix, lire  # noqa: E402

tmp = CTRL / "tmp_astro"
tmp.mkdir(exist_ok=True)
os.environ["TMP"] = os.environ["TEMP"] = str(tmp)
ch = {n: f for n, _, f, _ in choix()}
for n in map(int, sys.argv[1:]):
    y = lire(ch[n])
    t0 = time.time(); info = {}
    s = asterion.solve(y, tmp / str(n), timeout=120.0, info=info)
    print(n, ch[n], y.shape, "RESOLU %.3f arcsec/px, %d etoiles, centre %.4f %.4f" % (s.scale, s.matched, s.ra, s.dec) if s else "ECHEC %s" % str(info)[:200],
          "%.1f s" % (time.time() - t0), flush=True)
