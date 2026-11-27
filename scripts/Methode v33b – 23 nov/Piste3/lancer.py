# ======================================================================
# lancer.py
# Role     : Chain module (see its docstring below and scripts/LISEZMOI.md).
# Step     : 2. Classic Gaia-guided subtraction chain (v33b) (see scripts/LISEZMOI.md)
# Inputs   : stacked image (noise units), astrometry, offline deep Gaia extract
# Outputs  : star-free image, class maps
# Example  : python lancer.py [arguments]  (see the docstring; paths through the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
import os, sys
for k, v in dict(PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="5", MKL_NUM_THREADS="5",
                 NUMBA_NUM_THREADS="5", OPENBLAS_NUM_THREADS="5",
                 ASTRO_GPU_GAIA_EXTRACTS="<CACHE_FLOAT>/cache/gaia_extracts").items():
    os.environ[k] = v
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
cv2.setNumThreads(5)
import piste3
tour = sys.argv[1]
cases = sys.argv[2].split(",") if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else list(piste3.M.CASES)
logf = Path(__file__).resolve().parent / "sorties" / tour / "journal.txt"
logf.parent.mkdir(parents=True, exist_ok=True)
def log(s):
    print(s, flush=True)
    with open(logf, "a", encoding="utf-8") as f:
        f.write(s + "\n")
for c in cases:
    fil = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--remplisseur=")), "patch")
    piste3.run(tour, c, try_sigma="--sigma" in sys.argv, log=log, mode4="--tour4" in sys.argv, filler=fil)
