# ======================================================================
# explorer_candidats.py
# Rôle     : Module de la chaîne (voir sa docstring ci-dessous et scripts/LISEZMOI.md).
# Étape    : 2. Chaîne classique de soustraction guidée par Gaia (v33b) (voir scripts/LISEZMOI.md)
# Entrées  : image empilée (unités de bruit), astrométrie, extrait Gaia profond hors ligne
# Sorties  : image sans étoiles, cartes de classes
# Exemple  : python explorer_candidats.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
# Lecture seule : taille de l'image entière, catalogue profond, nombre d'étoiles candidates au modèle d'étoile selon
# l'isolement, par tranche de G (pour choisir les règles du modèle v7).
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
