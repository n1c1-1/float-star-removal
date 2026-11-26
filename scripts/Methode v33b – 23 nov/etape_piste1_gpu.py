# ======================================================================
# etape_piste1_gpu.py
# Rôle     : Etape 1 (piste 1) sur la carte graphique : meme appel que etape_piste1.py, meme algorithme (methode v26), code rapide.
# Étape    : 2. Chaîne classique de soustraction guidée par Gaia (v33b) (voir scripts/LISEZMOI.md)
# Entrées  : image empilée (unités de bruit), astrométrie, extrait Gaia profond hors ligne
# Sorties  : image sans étoiles, cartes de classes
# Exemple  : Usage : python etape_piste1_gpu.py <tag> x0,y0,x1,y1 [image] [cuda|cpu|ref] [--verif]
# ======================================================================
"""Etape 1 (piste 1) sur la carte graphique : meme appel que etape_piste1.py, meme algorithme (methode v26), code rapide.
Usage : python etape_piste1_gpu.py <tag> x0,y0,x1,y1 [image] [cuda|cpu|ref] [--verif]
  <tag>   nom de la sortie (Piste1/sorties/<image>/<tag>/) ; <image> : rosette (defaut), m8_sho, m57
  cuda    lots torch sur la RTX 4060 Ti (carte 0 seulement, jamais la 3070) ; defaut
  cpu     memes lots torch sur le processeur ; ref : retrait_rapide sans lots (code de la piste 1, coefficients groupes)
  Une seule execution GPU a la fois. Compatible methode_validee.py --piste1-gpu (memes arguments positionnels)."""
import os, sys
args = [a for a in sys.argv[1:] if not a.startswith("--")]
tag, crop_s = args[0], args[1]
image = "rosette"
dev = "cuda"
for a in args[2:]:
    if a in ("cuda", "cpu", "ref"):
        dev = a
    else:
        image = a
env = dict(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", NUMBA_NUM_THREADS="4",
           OPENBLAS_NUM_THREADS="4", CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0" if dev == "cuda" else "-1")
for k, v in env.items():
    os.environ[k] = v
sys.dont_write_bytecode = True
from pathlib import Path
P1 = Path(__file__).resolve().parent / "Piste1"
sys.path.insert(0, str(P1))
import torch
torch.set_num_threads(4)
if dev == "cuda":
    assert torch.cuda.device_count() == 1 and "4060" in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
import retrait_rapide as RM

crop = [int(v) for v in crop_s.split(",")]
ov = {"no_fill": True, "crop": crop, "region_grid": 1}
if image in ("m8_sho", "horsehead"):   # v15, v23 : meme reglage que etape_piste1.py
    ov["spikes"] = True
if dev in ("cuda", "cpu"):
    ov["device"] = dev
if "--verif" in sys.argv:
    ov["verif_lot"] = True
RM.run(image, tag, ov)
