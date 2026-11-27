# ======================================================================
# planches_tete.py
# Role     : Horsehead Nebula contact sheets: same strong stretch for all (asinh((x-background)/unit/2)/asinh(1000), x = linear without gradient).
# Stage    : 13. Trial on the Horsehead Nebula (26 Nov) (see scripts/LISEZMOI.md)
# Inputs   : linear stack of the Horsehead Nebula (not published)
# Outputs  : starless images (not published)
# Example  : python planches_tete.py [arguments]  (see the docstring; paths via variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
r"""Horsehead Nebula contact sheets: same strong stretch for all (asinh((x-background)/unit/2)/asinh(1000), x = linear without gradient).
Writes entiere_<version>.jpg (full resolution, q92), planche_entiere.jpg, planche_tete.jpg, 16-bit TIFF of our network."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np, cv2
D = next((Path(__import__("os").environ.get("DONNEES", "donnees"))).glob("Tete de cheval*")); T = D / "travail"
z = np.load(T / "entree.npz"); lin = z["lin"]; bk = z["bk"]; unit = float(z["unit"])
V = {"entree": lin}
for cle, f in (("notre reseau", "N7N5_075"), ("modele FLOAT", "FLOAT"), ("outil de comparaison", "RefExt")):
    p = T / f"starless_{f}.npy"
    if p.exists(): V[cle] = np.load(p)
def vue(a):
    v = np.arcsinh(np.clip((a - bk) / unit, 0, None) / 2) / np.arcsinh(1000.0)
    return np.ascontiguousarray((np.clip(v, 0, 1).transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8))
def txt(im, s):
    im = im.copy(); cv2.putText(im, s, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6); cv2.putText(im, s, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2); return im
slug = {"entree": "entree", "notre reseau": "notre_reseau", "modele FLOAT": "modele_FLOAT", "outil de comparaison": "outil_comparaison"}
ims = {}
for k, a in V.items():
    ims[k] = vue(a); cv2.imencode(".jpg", ims[k], [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(str(D / f"entiere_{slug[k]}.jpg"))
H, W = lin.shape[1:]
def planche(nom, box, w):
    x0, y0, x1, y1 = box
    cols = []
    for k, im in ims.items():
        c = im[y0:y1, x0:x1]; c = cv2.resize(c, (w, int(w * (y1 - y0) / (x1 - x0))), interpolation=cv2.INTER_AREA); cols.append(txt(c, k))
    sep = np.full((cols[0].shape[0], 6, 3), 255, np.uint8)
    out = cols[0]
    for c in cols[1:]: out = np.hstack([out, sep, c])
    cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(str(D / nom))
planche("planche_entiere.jpg", (0, 0, W, H), 1000)
planche("planche_tete.jpg", (2590 - 500, 1320 - 500, 2590 + 500, 1320 + 500), 1000)
s = V["notre reseau"]
for nom, a in (("starless_notre_reseau_16bits.tif", s), ("etoiles_notre_reseau_16bits.tif", lin - s)):
    cv2.imencode(".tif", np.round(np.clip(a / 1.25, 0, 1) * 65535).astype(np.uint16).transpose(1, 2, 0)[:, :, ::-1])[1].tofile(str(D / nom))
print("fini", list(V))
