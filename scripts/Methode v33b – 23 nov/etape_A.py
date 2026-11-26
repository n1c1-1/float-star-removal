# ======================================================================
# etape_A.py
# Rôle     : Etape A de la methode validee : piste 3 "hybride_c9" (soustraction des halos et aigrettes, remplissage des coeurs, tour 4, remplisseur par patchs) sur la base piste 1 d'une decoupe.
# Étape    : 2. Chaîne classique de soustraction guidée par Gaia (v33b) (voir scripts/LISEZMOI.md)
# Entrées  : image empilée (unités de bruit), astrométrie, extrait Gaia profond hors ligne
# Sorties  : image sans étoiles, cartes de classes
# Exemple  : Usage : python etape_A.py <nom> <tag_piste1> x0,y0,x1,y1
# ======================================================================
"""Etape A de la methode validee : piste 3 "hybride_c9" (soustraction des halos et aigrettes, remplissage des coeurs, tour 4,
remplisseur par patchs) sur la base piste 1 d'une decoupe. Les scripts de Piste3/ (etat gele de c9) ne sont pas modifies ;
ce lanceur ne fait que choisir la decoupe, la base et le dossier de sortie.
Usage : python etape_A.py <nom> <tag_piste1> x0,y0,x1,y1
  nom         nom de la decoupe (sorties dans sorties/<nom>/A/)
  tag_piste1  dossier de la piste 1 : Piste1/sorties/rosette/<tag_piste1>/
  crop        decoupe 768 px dans l'image Rosette debruitee ; tuile jugee = 512 px centraux (128..640)
Pour decoupe6 le cas est celui de c9 ("rosette_decoupe6", avec ses etoiles de reference nommees) ; sinon un cas "mv_<nom>"
sans etoiles de reference. Commande equivalente de c9 : lancer.py hybride_c9 rosette_decoupe6 --tour4 --remplisseur=patch."""
import os, sys, json
for k, v in dict(PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                 NUMBA_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                 ASTRO_GPU_GAIA_EXTRACTS="<CACHE_FLOAT>/cache/gaia_extracts").items():
    os.environ[k] = v
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Piste3"))
import cv2
cv2.setNumThreads(4)
import piste3 as P3
import mesure as M

nom, tag_p1 = sys.argv[1], sys.argv[2]
crop = tuple(int(v) for v in sys.argv[3].split(","))
image = sys.argv[4] if len(sys.argv) > 4 else "rosette"      # SHO 21 nov : nom de l'image (defaut inchange)
case = "rosette_decoupe6" if nom == "decoupe6" else "mv_" + nom
if case not in M.CASES:
    M.CASES[case] = (image, crop, (128, 128, 640, 640))
    P3.REF_ROSETTE = []                         # references nommees de decoupe6 : sans objet ailleurs
else:
    assert tuple(M.CASES[case][1]) == crop, "decoupe6 : le crop est fixe"
if image != "rosette":
    # plomberie SHO : l'image est declaree dans Piste1/images.py de cette copie, son astrometrie dans Piste1/sorties/<image>/
    import importlib.util as _ilu
    _sp = _ilu.spec_from_file_location("images_copie", ROOT / "Piste1" / "images.py")
    M.images = _ilu.module_from_spec(_sp); _sp.loader.exec_module(M.images)
    M.P1 = ROOT / "Piste1"
    _deep = ROOT / "Catalogue Gaia profond – 20 nov" / f"gaia_dr3_profond_{image}.npz"
    if _deep.exists():
        P3.DEEP[image] = _deep                  # absent : pas d'interdits profonds (comportement existant)
p1_dir = ROOT / "Piste1" / "sorties" / image / tag_p1
M.p1_dir = lambda name, c: p1_dir
A = ROOT / "sorties" / nom / "A"
A.mkdir(parents=True, exist_ok=True)
P3.HERE = A                                      # sorties/hybride_c9/<cas>/ et copie de la fenetre de la base piste 1
logf = A / "sorties" / "hybride_c9" / "journal.txt"
logf.parent.mkdir(parents=True, exist_ok=True)


def log(s):
    print(s, flush=True)
    with open(logf, "a", encoding="utf-8") as f:
        f.write(s + "\n")


# v28 (22 nov) : try_sigma (mode "geantes" de la piste 3 : fenetre de 520 px, geante traitee comme une brillante) n'est plus
# fige : True seulement si une etoile du catalogue de G < GIANT_G (4,5) se trouve dans la decoupe + 520 px, sinon False
# (une image sans geante se comporte exactement comme v26). Le catalogue est celui de l'astrometrie de l'image (avec Alnitak
# ajoutee a la main pour horsehead). ETAPE_A_TRY_SIGMA=0|1 force la valeur (controle de fusion seulement).
import numpy as _np
_cat = _np.load(ROOT / "Piste1" / "sorties" / image / "astrometrie.npz", allow_pickle=True)["cat"].item()
_m = (_cat["g"] < P3.GIANT_G) & (_cat["x"] > crop[0] - P3.MARGIN_GIANT) & (_cat["x"] < crop[2] + P3.MARGIN_GIANT)      & (_cat["y"] > crop[1] - P3.MARGIN_GIANT) & (_cat["y"] < crop[3] + P3.MARGIN_GIANT)
try_sigma = bool(_m.any())
if os.environ.get("ETAPE_A_TRY_SIGMA") in ("0", "1"):
    try_sigma = os.environ["ETAPE_A_TRY_SIGMA"] == "1"
log(f"v28 : try_sigma = {try_sigma} ({int(_m.sum())} etoile(s) G < {P3.GIANT_G} dans la decoupe + {P3.MARGIN_GIANT} px)")
P3.run("hybride_c9", case, try_sigma=try_sigma, log=log, mode4=True, filler="patch")
(A / "cas.json").write_text(json.dumps({"case": case, "crop": list(crop), "tag_piste1": tag_p1, "image": image}), encoding="utf-8")
