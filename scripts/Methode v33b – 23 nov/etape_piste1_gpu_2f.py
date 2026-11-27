# ======================================================================
# etape_piste1_gpu_2f.py
# Role     : Step 1 (track 1) on the graphics card: same call as etape_piste1.py, same algorithm (method v26), fast code.
# Stage    : 2. Classical Gaia-guided subtraction chain (v33b) (see scripts/LISEZMOI.md)
# Inputs   : stacked image (noise units), astrometry, offline deep Gaia extract
# Outputs  : starless image, class maps
# Example  : Usage : python etape_piste1_gpu.py <tag> x0,y0,x1,y1 [image] [cuda|cpu|ref] [--verif]
# ======================================================================
"""Step 1 (track 1) on the graphics card: same call as etape_piste1.py, same algorithm (method v26), fast code.
Usage : python etape_piste1_gpu.py <tag> x0,y0,x1,y1 [image] [cuda|cpu|ref] [--verif]
  <tag>   name of the output (Piste1/sorties/<image>/<tag>/) ; <image> : rosette (default), m8_sho, m57
  cuda    torch batches on the RTX 4060 Ti (card 0 only, never the 3070) ; default
  cpu     same torch batches on the CPU ; ref : retrait_rapide without batches (track 1 code, grouped coefficients)
  Only one GPU run at a time. Compatible with methode_validee.py --piste1-gpu (same positional arguments)."""
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
env = dict(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMBA_NUM_THREADS="2",
           OPENBLAS_NUM_THREADS="2", CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0" if dev == "cuda" else "-1")
for k, v in env.items():
    os.environ[k] = v
sys.dont_write_bytecode = True
from pathlib import Path
P1 = Path(__file__).resolve().parent / "Piste1"
sys.path.insert(0, str(P1))
import torch
torch.set_num_threads(2)
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
