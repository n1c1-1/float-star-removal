# ======================================================================
# arc.py
# Role     : The pale arc near (277,345) of the Rosette tile: is it in the low frequencies of the input (nebula) or does it only appear after track 1's subtraction?
# Step     : 2. Classic Gaia-guided subtraction chain (v33b) (see scripts/LISEZMOI.md)
# Inputs   : stacked image (noise units), astrometry, offline deep Gaia extract
# Outputs  : star-free image, class maps
# Example  : python arc.py [arguments]  (see the docstring; paths through the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""The pale arc near (277,345) of the Rosette tile: is it in the low frequencies of the input (nebula) or does it only appear after track 1's subtraction?
Thumbnails 100 % (and x3): input | input without stars by masked blur (8 px, catalogue and
track 1 stars excluded) | track 1 base | track 3 result; and profile along a cut across the arc."""
import sys, csv
sys.dont_write_bytecode = True
from pathlib import Path
import cv2
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mesure as M
import remplir as F
sys.path.insert(1, str(M.P1))
from juger import save_img
case = "rosette_etoile_sur_nebuleuse"
name, crop, tile = M.CASES[case]
full = M.images.load(name); img = full[:, crop[1]:crop[3], crop[0]:crop[2]]
cb = sorted((HERE / "sorties" / "base_piste1" / case).glob("essai16*"))[-1]
B = np.load(cb / "fenetre_sans_etoiles.npy")
wx0, wy0 = tile[0] - 200, tile[1] - 200
I = img[:, wy0:wy0 + B.shape[1], wx0:wx0 + B.shape[2]]
Rz = np.load(HERE / "sorties" / (sys.argv[1] if len(sys.argv) > 1 else "verif_hp") / case / "sans_etoiles_tuile.npy")
H, W = B.shape[1:]
stars = np.zeros((H, W), np.uint8)
z = np.load(M.P1 / "sorties" / name / "astrometrie.npz", allow_pickle=True)["cat"].item()
for x, y, g in zip(z["x"] - crop[0] - wx0, z["y"] - crop[1] - wy0, z["g"]):
    if -30 < x < W + 30 and -30 < y < H + 30:
        cv2.circle(stars, (int(round(x)), int(round(y))), int(max(4, (15 - g) * 3)), 1, -1)
for row in csv.DictReader(open(cb / "etoiles.csv", encoding="utf-8")):
    x, y = float(row["x"]) - wx0, float(row["y"]) - wy0
    if 0 <= x < W and 0 <= y < H:
        cv2.circle(stars, (int(round(x)), int(round(y))), 5, 1, -1)
Ilow = F.lowpass_masked(I, stars.astype(bool), 8.0)
Blow = F.lowpass_masked(B, stars.astype(bool), 8.0)
cx, cy, hw = 277 + 200, 345 + 200, 60
g_ = full[1]; bk = float(np.median(g_[::7, ::7])); t998 = float(np.percentile(g_[::5, ::5], 99.8))
def post(a):
    x = np.clip((a - bk) / (t998 - bk), 0, None); v = np.arcsinh(40 * x) / np.arcsinh(40)
    return (np.transpose(np.clip(v, 0, 1), (1, 2, 0)) * 255 + 0.5).astype(np.uint8)
def ext(a, lo, hi):
    v = np.clip((a - lo) / (hi - lo), 0, 1)
    return (np.transpose(v, (1, 2, 0)) * 255 + 0.5).astype(np.uint8)
c = lambda a: a[:, cy - hw:cy + hw, cx - hw:cx + hw]
Rw = np.zeros_like(B); Rw[:, 200:712, 200:712] = Rz
lo = float(np.percentile(c(Ilow).mean(0), 2)); hi = float(np.percentile(c(Ilow).mean(0), 98))
sp = np.full((2 * hw, 4, 3), 255, np.uint8)
row1 = np.concatenate([post(c(I)), sp, post(c(Ilow)), sp, post(c(B)), sp, post(c(Rw))], 1)
row2 = np.concatenate([ext(c(I), lo, hi), sp, ext(c(Ilow), lo, hi), sp, ext(c(B), lo, hi), sp, ext(c(Rw), lo, hi)], 1)
out = HERE / "sorties" / "arc_277_345" / (sys.argv[1] if len(sys.argv) > 1 else "verif_hp")
save_img(out / "arc_post_100pc_entree_entreeSansEtoiles_p1_p3.png", row1)
save_img(out / "arc_contraste_x3_entree_entreeSansEtoiles_p1_p3.png", cv2.resize(row2, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
# coupe horizontale à travers l'arc (y = 345), basse fréquence de l'entrée sans étoiles contre base et résultat
for nm, A in (("entree_sans_etoiles", Ilow), ("base_p1_flou", Blow), ("resultat_flou", F.lowpass_masked(Rw, stars.astype(bool), 8.0))):
    v = A.mean(0)[cy, cx - 40:cx + 41:5]
    print(nm, np.round(v, 2))
