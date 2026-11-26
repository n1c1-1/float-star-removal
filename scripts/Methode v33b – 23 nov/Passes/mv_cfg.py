# ======================================================================
# mv_cfg.py
# Rôle     : Parametres d'une execution de l'etape B (passes finales), lus dans le fichier JSON donne par la variable MV_CFG.
# Étape    : 2. Chaîne classique de soustraction guidée par Gaia (v33b) (voir scripts/LISEZMOI.md)
# Entrées  : image empilée (unités de bruit), astrométrie, extrait Gaia profond hors ligne
# Sorties  : image sans étoiles, cartes de classes
# Exemple  : python mv_cfg.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Parametres d'une execution de l'etape B (passes finales), lus dans le fichier JSON donne par la variable MV_CFG.
Aucun chemin ni nom de decoupe n'est ecrit en dur dans passe_finale_p.py / passe_finale_t2_p.py / passe_finale_t3_p.py.

Cle du JSON :
  case      nom du cas dans mesure.CASES (ex. "rosette_decoupe6", "mv_filaments")
  crop      [x0, y0, x1, y1] decoupe dans l'image Rosette debruitee entiere
  tile      [x0, y0, x1, y1] tuile jugee dans la decoupe
  a_out     dossier de sortie de l'etape A (sans_etoiles_tuile.npy, carte_remplissage_tuile.npy, mesures.json)
  base_csv  etoiles.csv de la base piste 1 (copie de fenetre de l'etape A)
  b_root    dossier de sortie de l'etape B (t1/, t2/, t3/)
  p1_dir    dossier "Piste1" gele (astrometrie.npz, images.py, juger.py)
  six       liste [[x, y], ...] d'etoiles a montrer (vues seulement ; [] si aucune)
  angles    angles des aigrettes de l'instrument, mesures par l'etape A (journal / mesures.json)
  named_fixes  true seulement pour decoupe6 : les corrections de t3 liees a des positions nommees de cette decoupe
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
