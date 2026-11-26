# ======================================================================
# planches.py
# Rôle     : Planches de controle : une par image, <DONNEES>/Paires étoiles APOD – 24 nov/controle/<image>.jpg ; une ligne par tuile : entree debruitee | cible | poids (blanc 1, gris 0,5, noir 0), arcsinh 60, 3 x 400 px.
# Étape    : 3. Production des paires web APOD (24 nov) (voir scripts/LISEZMOI.md)
# Entrées  : images APOD délinéarisées
# Sorties  : paires _debruitee .npz
# Exemple  : python planches.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Planches de controle : une par image, <DONNEES>/Paires étoiles APOD – 24 nov/controle/<image>.jpg ; une ligne par tuile :
entree debruitee | cible | poids (blanc 1, gris 0,5, noir 0), arcsinh 60, 3 x 400 px. usage : planches.py [image ...]"""
import sys
from pathlib import Path
import cv2, numpy as np
sys.stdout.reconfigure(encoding="utf-8")
S = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Paires étoiles APOD – 24 nov"); (S / "controle").mkdir(exist_ok=True)
noms = sorted({p.name.rsplit("_g", 1)[0] for p in S.glob("*_debruitee.npz")})
if len(sys.argv) > 1:
    noms = [n for n in noms if n in sys.argv[1:]]
for im in noms:
    lignes = []
    for f in sorted(S.glob(f"{im}_g*_debruitee.npz")):
        if f.name.rsplit("_g", 1)[0] != im:
            continue
        z = np.load(f); I, T, W = z["input"], z["target"], z["weight"]
        bk = [float(np.median(I[c])) for c in range(3)]; top = [max(float(np.percentile(I[c], 99.8)), bk[c] + 1e-3) for c in range(3)]
        st = lambda a: (np.clip(np.stack([np.arcsinh(60 * np.clip((a[c] - bk[c]) / (top[c] - bk[c]), 0, None)) / np.arcsinh(60) for c in range(3)], -1), 0, 1) * 255 + .5).astype(np.uint8)
        w = (np.clip(W, 0, 1) * 255).astype(np.uint8)
        pan = [cv2.resize(x, (400, 400), interpolation=cv2.INTER_AREA) for x in (st(I), st(T), np.stack([w] * 3, -1))]
        l = np.concatenate(pan, 1).copy()
        cv2.putText(l, f.name[len(im) + 1:-14], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(l, f"poids0 {100*(W==0).mean():.1f}%", (404, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        lignes.append(l)
    if lignes:
        cv2.imencode(".jpg", np.concatenate(lignes, 0)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(str(S / "controle" / f"{im}.jpg"))
        print(im, len(lignes), flush=True)
