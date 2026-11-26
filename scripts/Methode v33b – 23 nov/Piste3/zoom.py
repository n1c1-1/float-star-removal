# ======================================================================
# zoom.py
# Rôle     : Module de la chaîne (voir sa docstring ci-dessous et scripts/LISEZMOI.md).
# Étape    : 2. Chaîne classique de soustraction guidée par Gaia (v33b) (voir scripts/LISEZMOI.md)
# Entrées  : image empilée (unités de bruit), astrométrie, extrait Gaia profond hors ligne
# Sorties  : image sans étoiles, cartes de classes
# Exemple  : python zoom.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
import sys
sys.dont_write_bytecode = True
import numpy as np, cv2
from pathlib import Path
import mesure as M
sys.path.insert(1, str(M.P1))
from juger import save_img
tour, case, cx, cy, hw = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
name, crop, tile = M.CASES[case]
full = M.images.load(name); img = full[:, crop[1]:crop[3], crop[0]:crop[2]]
p1 = np.load(M.p1_dir(name, case) / "starless.npy")
mine = np.load(Path("sorties") / tour / case / "sans_etoiles_tuile.npy")
g = full[1]; bk = float(np.median(g[::7, ::7])); t998 = float(np.percentile(g[::5, ::5], 99.8))
def post(a):
    x = np.clip((a - bk) / max(t998 - bk, 1e-9), 0, None); v = np.arcsinh(40 * x) / np.arcsinh(40)
    return (np.transpose(np.clip(v, 0, 1), (1, 2, 0)) * 255 + 0.5).astype(np.uint8)
X, Y = tile[0] + cx, tile[1] + cy
a = post(img[:, Y-hw:Y+hw, X-hw:X+hw]); b = post(p1[:, Y-hw:Y+hw, X-hw:X+hw]); c = post(mine[:, cy-hw:cy+hw, cx-hw:cx+hw])
k = max(1, 300 // (2 * hw))
up = lambda u: cv2.resize(u, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)
sp = np.full((2 * hw * k, 4, 3), 255, np.uint8)
out = Path("sorties") / tour / case / f"zoom_x{cx}_y{cy}.png"
save_img(out, np.concatenate([up(a), sp, up(b), sp, up(c)], 1)); print(out)
