# ======================================================================
# lancer.py
# Rôle     : Module de la chaîne (voir sa docstring ci-dessous et scripts/LISEZMOI.md).
# Étape    : 2. Chaîne classique de soustraction guidée par Gaia (v33b) (voir scripts/LISEZMOI.md)
# Entrées  : image empilée (unités de bruit), astrométrie, extrait Gaia profond hors ligne
# Sorties  : image sans étoiles, cartes de classes
# Exemple  : python lancer.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
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
