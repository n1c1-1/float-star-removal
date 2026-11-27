# ======================================================================
# choisir_tuiles.py
# Role     : Chooses, for each validation image, the 512 px tile on which the classical chain and the networks are compared.
# Stage    : 15. Classical baseline versus networks on 512 px tiles (26-27 Nov) (see scripts/README.md)
# Inputs   : gaia_dr3_profond.npz and input.npy of each validation image (liste.py)
# Outputs  : tuiles.json
# Example  : python choisir_tuiles.py
# ======================================================================
r"""Choice of the baseline tiles (RULE WRITTEN BEFORE THE COMPUTATIONS, deterministic, not tuned on the results).
For each validation image that has gaia_dr3_profond.npz (LISTE order): one 512x512 tile whose origin (x0, y0) is on the 128 px grid, with
128 <= x0, x0+512 <= W-128 (same for y) (so the chain's tile +/-128 crop is inside the image), at least 95 % of non-zero pixels
(mean of the 3 channels != 0); the number of Gaia stars (gaia_dr3_profond.npz) with 8 <= G < 17 in the tile is maximised; tie: lowest (y, x).
Writes tuiles.json. usage: choisir_tuiles.py"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, r"C:\Users\Nicolas\sandbox\starRemoval\Validation – 22 nov")
from liste import LISTE
DATA = Path(r"F:\float_ia\Validation étoiles – 22 nov")
T, G = 512, 128
res = {}
for nom, v in LISTE.items():
    if not (DATA / nom / "gaia_dr3_profond.npz").exists():
        continue
    a = np.load(Path(v[0]) / "input.npy", mmap_mode="r")
    C, H, W = a.shape
    valid = np.zeros((H, W), bool)
    for y in range(0, H, 256):
        valid[y:y + 256] = np.any(np.asarray(a[:, y:y + 256]) != 0, axis=0)
    z = np.load(DATA / nom / "gaia_dr3_profond.npz")
    x, y, g = z["x_pix"], z["y_pix"], z["Gmag"]
    k = np.isfinite(g) & (g >= 8) & (g < 17) & np.isfinite(x) & np.isfinite(y)
    cnt = np.zeros((H, W), np.int32)
    xi, yi = np.round(x[k]).astype(int), np.round(y[k]).astype(int)
    ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    np.add.at(cnt, (yi[ok], xi[ok]), 1)
    ic = np.pad(cnt.cumsum(0).cumsum(1), ((1, 0), (1, 0))).astype(np.int64)
    iv = np.pad(valid.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0))).astype(np.int64)
    best = None
    for y0 in range(G, H - G - T + 1, G):
        for x0 in range(G, W - G - T + 1, G):
            fv = (iv[y0 + T, x0 + T] - iv[y0, x0 + T] - iv[y0 + T, x0] + iv[y0, x0]) / T ** 2
            if fv < 0.95:
                continue
            n = int(ic[y0 + T, x0 + T] - ic[y0, x0 + T] - ic[y0 + T, x0] + ic[y0, x0])
            if best is None or n > best[0]:          # parcours par (y, x) croissants : strict > garde le plus bas en cas d'egalite
                best = (n, x0, y0, float(fv))
    if best is None:
        res[nom] = {"tuile": None, "raison": "aucune tuile valide", "forme": [C, H, W]}
    else:
        res[nom] = {"x0": best[1], "y0": best[2], "n_gaia_8_17": best[0], "part_valide": round(best[3], 4), "forme": [C, H, W]}
    print(nom, res[nom], flush=True)
Path(__file__).with_name("tuiles.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
