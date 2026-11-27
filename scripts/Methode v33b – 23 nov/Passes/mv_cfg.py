# ======================================================================
# mv_cfg.py
# Role     : Parameters of a step B run (final passes), read from the JSON file given by the variable MV_CFG.
# Step     : 2. Classic Gaia-guided subtraction chain (v33b) (see scripts/LISEZMOI.md)
# Inputs   : stacked image (noise units), astrometry, offline deep Gaia extract
# Outputs  : star-free image, class maps
# Example  : python mv_cfg.py [arguments]  (see the docstring; paths through the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Parameters of a step B run (final passes), read from the JSON file given by the variable MV_CFG.
No path or crop name is hard-coded in passe_finale_p.py / passe_finale_t2_p.py / passe_finale_t3_p.py.

JSON keys:
  case      name of the case in mesure.CASES (e.g. "rosette_decoupe6", "mv_filaments")
  crop      [x0, y0, x1, y1] crop in the whole denoised Rosette image
  tile      [x0, y0, x1, y1] tile judged within the crop
  a_out     output folder of step A (sans_etoiles_tuile.npy, carte_remplissage_tuile.npy, mesures.json)
  base_csv  etoiles.csv of the track 1 base (window copy from step A)
  b_root    output folder of step B (t1/, t2/, t3/)
  p1_dir    frozen "Piste1" folder (astrometrie.npz, images.py, juger.py)
  six       list [[x, y], ...] of stars to show (views only; [] if none)
  angles    the instrument's spike angles, measured by step A (log / mesures.json)
  named_fixes  true only for decoupe6: the t3 corrections tied to named positions of this crop
"""
import json
import os
from pathlib import Path

CFG = json.loads(Path(os.environ["MV_CFG"]).read_text(encoding="utf-8"))
CASE = CFG["case"]
CROP = tuple(CFG["crop"])
TILE = tuple(CFG["tile"])
A_OUT = Path(CFG["a_out"])
BASE_CSV = Path(CFG["base_csv"])
B_ROOT = Path(CFG["b_root"])
P1_DIR = Path(CFG["p1_dir"])
SIX = [tuple(v) for v in CFG.get("six", [])]
ANGLES = [float(a) for a in CFG["angles"]]
NAMED_FIXES = bool(CFG.get("named_fixes", False))

import mesure as M  # noqa: E402  (version finale de mesure.py, copiee dans Passes/)

# SHO 21 nov (plomberie) : nom de l'image (defaut rosette, inchange) et catalogue profond (absent -> aucun, comme la piste 3)
IMAGE = CFG.get("image", "rosette")
DEEP = Path(CFG.get("deep", r"<SCRIPTS>\Catalogue Gaia profond – 20 nov\gaia_dr3_profond_rosette.npz"))
M.CASES[CASE] = (IMAGE, CROP, TILE)
M.P1 = P1_DIR          # astrometrie.npz et images.py / juger.py du dossier Piste1 gele
if IMAGE != "rosette":
    # chargeur d'images du dossier Piste1 de cette copie (qui declare l'image), au lieu de celui du dossier de travail
    import importlib.util as _ilu
    _sp = _ilu.spec_from_file_location("images_copie", P1_DIR / "images.py")
    M.images = _ilu.module_from_spec(_sp)
    _sp.loader.exec_module(M.images)
