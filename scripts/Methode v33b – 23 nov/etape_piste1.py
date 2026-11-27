# ======================================================================
# etape_piste1.py
# Role     : Step 1 of the validated method: track 1 (deep Gaia + PSF subtraction, without filling), c9 settings.
# Step     : 2. Classic Gaia-guided subtraction chain (v33b) (see scripts/LISEZMOI.md)
# Inputs   : stacked image (noise units), astrometry, offline deep Gaia extract
# Outputs  : star-free image, class maps
# Example  : Usage: python etape_piste1.py <tag> x0,y0,x1,y1
# ======================================================================
"""Step 1 of the validated method: track 1 (deep Gaia + PSF subtraction, without filling), c9 settings.
Usage: python etape_piste1.py <tag> x0,y0,x1,y1
  <tag>  name of the output (Piste1/sorties/rosette/<tag>/)
  crop   crop in the whole denoised Rosette image (768 px for decoupe6: 1530,790,2298,1558)
Same call as piste1_decoupe.py at the launch of c9: RM.run("rosette", tag, {"no_fill": True, "crop": crop, "region_grid": 1}).
The code called is Piste1/retrait_modele.py, frozen copy of the state of this file at the launch of c9 (see JOURNAL.md)."""
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
