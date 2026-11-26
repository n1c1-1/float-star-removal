# ======================================================================
# etape_piste1.py
# Rôle     : Etape 1 de la methode validee : piste 1 (Gaia profond + soustraction de PSF, sans remplissage), reglages de c9.
# Étape    : 2. Chaîne classique de soustraction guidée par Gaia (v33b) (voir scripts/LISEZMOI.md)
# Entrées  : image empilée (unités de bruit), astrométrie, extrait Gaia profond hors ligne
# Sorties  : image sans étoiles, cartes de classes
# Exemple  : Usage : python etape_piste1.py <tag> x0,y0,x1,y1
# ======================================================================
"""Etape 1 de la methode validee : piste 1 (Gaia profond + soustraction de PSF, sans remplissage), reglages de c9.
Usage : python etape_piste1.py <tag> x0,y0,x1,y1
  <tag>  nom de la sortie (Piste1/sorties/rosette/<tag>/)
  crop   decoupe dans l'image Rosette debruitee entiere (768 px pour decoupe6 : 1530,790,2298,1558)
Meme appel que piste1_decoupe.py au lancement de c9 : RM.run("rosette", tag, {"no_fill": True, "crop": crop, "region_grid": 1}).
Le code appele est Piste1/retrait_modele.py, copie gelee de l'etat de ce fichier au lancement de c9 (voir JOURNAL.md)."""
import os, sys, json
for k, v in dict(PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                 NUMBA_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4").items():
    os.environ[k] = v
sys.dont_write_bytecode = True
from pathlib import Path
P1 = Path(__file__).resolve().parent / "Piste1"
sys.path.insert(0, str(P1))
import retrait_modele as RM

tag = sys.argv[1]
crop = [int(v) for v in sys.argv[2].split(",")]
image = sys.argv[3] if len(sys.argv) > 3 else "rosette"      # SHO 21 nov : nom de l'image (defaut inchange)
ov = {"no_fill": True, "crop": crop, "region_grid": 1}
# v15 (SHO 21 nov) : le détecteur d'aigrettes par étoile de la piste 1 (spikes(), remplissage) ne voit pas les 6 aigrettes
# en pointillés de la G 5,23 de M8 (z 3,1-3,8 < 4 sur les 3 premières tranches) ; le modèle d'aigrettes de l'instrument
# (spike_model : angles sur les brillantes, jusqu'à 8, jumelle à 180 deg exigée ; gabarit soustrait) est mis en marche pour
# m8_sho. La Rosette garde la décision des essais 9-10 (coupé : croix sombres fines).
if image in ("m8_sho", "horsehead"):   # v23 : horsehead declare comme m8_sho (aigrettes d'instrument mesurees)
    ov["spikes"] = True
RM.run(image, tag, ov)
