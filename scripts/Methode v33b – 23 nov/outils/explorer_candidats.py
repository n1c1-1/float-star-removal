# ======================================================================
# explorer_candidats.py
# Role     : Chain module (see its docstring below and scripts/LISEZMOI.md).
# Stage    : 2. Classical Gaia-guided subtraction chain (v33b) (see scripts/LISEZMOI.md)
# Inputs   : stacked image (noise units), astrometry, offline deep Gaia extract
# Outputs  : starless image, class maps
# Example  : python explorer_candidats.py [arguments]  (see the docstring; paths via variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
# Read-only: size of the whole image, deep catalogue, number of candidate stars for the star model according to
# isolation, per G slice (to choose the rules of model v7).
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
HERE = Path(__file__).resolve().parents[1] / "Piste1"
sys.path.insert(0, str(HERE))
import images
t = time.time()
img = images.load("rosette"); C, H, W = img.shape
print("image", img.shape, f"{time.time() - t:.0f} s", "max par couleur", img.reshape(C, -1).max(1))
z = np.load(HERE.parent.parent / "Catalogue Gaia profond – 20 nov" / "gaia_dr3_profond_rosette.npz")
ok = np.isfinite(z["Gmag"]) & np.isfinite(z["x_pix"]) & np.isfinite(z["y_pix"])
x, y, G = z["x_pix"][ok], z["y_pix"][ok], z["Gmag"][ok]
ins = (x > 40) & (x < W - 40) & (y > 40) & (y < H - 40)
print("catalogue", ok.sum(), "dans l'image", ins.sum())
tr = cKDTree(np.c_[x, y])
L = img.mean(0)
pk = np.array([L[int(round(b)), int(round(a))] if 0 <= round(a) < W and 0 <= round(b) < H else 0 for a, b in zip(x, y)])
for g0, g1 in ((8, 10), (10, 11), (11, 12), (12, 13.5), (13.5, 14.5), (14.5, 15.5), (15.5, 16.5), (16.5, 17.5)):
    s = np.where(ins & (G >= g0) & (G < g1))[0]
    res = []
    for rad, dg in ((15, 2.5), (20, 3.0), (30, 3.0), (30, 4.0)):
        n = 0
        for k in s:
            if not any(j != k and G[j] < G[k] + dg for j in tr.query_ball_point([x[k], y[k]], rad)):
                n += 1
        res.append(n)
    print(f"G {g0}-{g1}: {len(s)} étoiles, pic L médian {np.median(pk[s]):.1f} ; isolées (15px,2.5) (20,3) (30,3) (30,4) : {res}")
