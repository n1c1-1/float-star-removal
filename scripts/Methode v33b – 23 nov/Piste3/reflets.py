# ======================================================================
# reflets.py
# Role     : Search for the offset reflections (ghosts) of the bright stars: for each bright star of the crop, the smoothed (4 px) input minus its radial profile (median per ring) is rotated so that the direction of the optical centre (centre of the whole image) points towards +x, then normalised by the Gaia flux; the median of these thumbnails shows a common reflection (blob offset along this direction). Then, for G 5.60 of the Rosette tile, the thumbnail itself.
# Step     : 2. Classic Gaia-guided subtraction chain (v33b) (see scripts/LISEZMOI.md)
# Inputs   : stacked image (noise units), astrometry, offline deep Gaia extract
# Outputs  : star-free image, class maps
# Example  : python reflets.py
# ======================================================================
"""Search for the offset reflections (ghosts) of the bright stars: for each bright star of the crop, the smoothed (4 px)
input minus its radial profile (median per ring) is rotated so that the direction of the optical centre (centre of
the whole image) points towards +x, then normalised by the Gaia flux; the median of these thumbnails shows a common reflection
(blob offset along this direction). Then, for G 5.60 of the Rosette tile, the thumbnail itself.

    python reflets.py
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import cv2
import numpy as np
from scipy import ndimage as ndi
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mesure as M
sys.path.insert(1, str(M.P1))
from juger import save_img

name, crop, tile = M.CASES["rosette_etoile_sur_nebuleuse"]
full = M.images.load(name)
Hf, Wf = full.shape[1:]
cx0, cy0 = Wf / 2, Hf / 2
L = cv2.GaussianBlur(full.mean(0), (0, 0), 4.0)
z = np.load(M.P1 / "sorties" / name / "astrometrie.npz", allow_pickle=True)["cat"].item()
R = 180
sel = [(x, y, g) for x, y, g in zip(z["x"], z["y"], z["g"]) if 4.5 < g < 8.0 and R < x < Wf - R and R < y < Hf - R]
vigs = []
for x, y, g in sel:
    X, Y = int(round(x)), int(round(y))
    cut = L[Y - R:Y + R + 1, X - R:X + R + 1].astype(np.float32)
    p = M.radial_profile(L, x, y, int(R * 1.42))
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    rr = np.hypot(xx, yy)
    res = cut - np.interp(rr, np.arange(len(p)), np.nan_to_num(p))
    ang = np.degrees(np.arctan2(cy0 - y, cx0 - x))
    rot = cv2.warpAffine(res, cv2.getRotationMatrix2D((R, R), ang, 1.0), (2 * R + 1, 2 * R + 1))
    vigs.append(rot / 10 ** (-0.4 * (g - 6)))
    print(f"G {g:.2f} ({x:.0f},{y:.0f}) distance au centre {np.hypot(x - cx0, y - cy0):.0f} px")
S = np.median(vigs, 0)
sd = float(np.median(np.abs(S - np.median(S))) * 1.4826)
m = np.clip(0.5 + S / (8 * sd), 0, 1)
out = HERE / "sorties" / "reflets"
save_img(out / "mediane_vignettes_vers_centre_a_droite.png", (np.stack([m] * 3, -1) * 255).astype(np.uint8))
# pic positif hors du coeur (r > 25 px)
yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
Sm = np.where(np.hypot(xx, yy) > 25, ndi.uniform_filter(S, 9), -1e9)
iy, ix = np.unravel_index(np.argmax(Sm), Sm.shape)
print(f"médiane de {len(vigs)} étoiles : pic hors coeur à dx={ix - R}, dy={iy - R} (vers le centre = +x), {Sm[iy, ix] / sd:.1f} écarts")

# reflet à une distance proportionnelle à la distance au centre optique : vignettes centrées sur la position prédite
# x_r = x + k (centre - x), pour k de 0.05 à 0.6 ; le bon k donne une tache commune nette
Lr = full.mean(0)
Ls_ = cv2.GaussianBlur(Lr, (0, 0), 3.0) - cv2.GaussianBlur(Lr, (0, 0), 30.0)    # bande 3-30 px : reflet, pas la nébuleuse
sel2 = [(x, y, g) for x, y, g in zip(z["x"], z["y"], z["g"]) if 4.5 < g < 7.6 and np.hypot(x - cx0, y - cy0) > 250]
best = None
for k in np.arange(0.05, 0.61, 0.01):
    vs = []
    for x, y, g in sel2:
        gx_, gy_ = x + k * (cx0 - x), y + k * (cy0 - y)
        X, Y = int(round(gx_)), int(round(gy_))
        if 40 <= X < Wf - 40 and 40 <= Y < Hf - 40 and np.hypot(gx_ - x, gy_ - y) > 40:
            vs.append(Ls_[Y - 40:Y + 41, X - 40:X + 41] / 10 ** (-0.4 * (g - 6)))
    if len(vs) < 4:
        continue
    S2 = np.median(vs, 0)
    yy, xx = np.mgrid[-40:41, -40:41]
    c = S2[np.hypot(xx, yy) <= 10].mean() - S2[(np.hypot(xx, yy) > 20)].mean()
    sd2 = S2[np.hypot(xx, yy) > 20].std() + 1e-9
    if best is None or c / sd2 > best[0]:
        best = (c / sd2, k, len(vs), S2)
print(f"meilleur rapport k = {best[1]:.2f} : contraste {best[0]:.1f} écarts sur {best[2]} étoiles")
m2 = np.clip(0.5 + best[3] / (6 * best[3].std()), 0, 1)
save_img(out / f"mediane_reflet_k{best[1]:.2f}.png", cv2.resize((np.stack([m2] * 3, -1) * 255).astype(np.uint8), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
x5, y5 = 1941.0, 1127.0
k = best[1]
print(f"G 5.60 : reflet prédit en image ({x5 + k * (cx0 - x5):.0f},{y5 + k * (cy0 - y5):.0f}), tuile "
      f"({x5 + k * (cx0 - x5) - crop[0] - tile[0]:.0f},{y5 + k * (cy0 - y5) - crop[1] - tile[1]:.0f})")
