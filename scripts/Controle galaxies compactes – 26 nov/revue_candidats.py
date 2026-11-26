# ======================================================================
# revue_candidats.py
# Rôle     : Planches de revue des candidats (vignettes 64x64 agrandies x3, numerotees) pour choisir a l'oeil.
# Étape    : 13. Jeu de contrôle des petites galaxies (26 nov) (voir scripts/LISEZMOI.md)
# Entrées  : images web et extraits Gaia (non publiés)
# Sorties  : controle.json, mesures par réseau
# Exemple  : python revue_candidats.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
r"""Planches de revue des candidats (vignettes 64x64 agrandies x3, numerotees) pour choisir a l'oeil. Une planche par image, 40 candidats au plus
(les plus larges, puis un tirage regulier). revue_candidats.py [image ...]"""
import os, sys, json, glob
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np, cv2

APOD = Path(glob.glob(r"<DONNEES>\Paires ?toiles APOD*")[0])
OUT = Path(glob.glob(r"<DONNEES>\Controle galaxies compactes*")[0])
(OUT / "revue").mkdir(exist_ok=True)


def vign(L, x, y, h=32, k=3):
    H, W = L.shape
    x0, y0 = int(round(x)) - h, int(round(y)) - h
    v = L[max(y0, 0):y0 + 2 * h, max(x0, 0):x0 + 2 * h]
    v = np.pad(v, ((0, 2 * h - v.shape[0]), (0, 2 * h - v.shape[1])))
    g = np.clip(np.arcsinh(np.clip(v, 0, None) / 8) / np.arcsinh(600 / 8), 0, 1)
    return cv2.resize((g * 255).astype(np.uint8), (2 * h * k, 2 * h * k), interpolation=cv2.INTER_CUBIC)


for nom in (sys.argv[1:] or [Path(f).stem for f in glob.glob(str(OUT / "candidats" / "*.json"))]):
    d = json.loads((OUT / "candidats" / f"{nom}.json").read_text(encoding="utf-8"))
    c = [r for r in d["candidats"] if r["ellipticite"] < 2.2]
    if not c:
        continue
    c.sort(key=lambda r: -r["crete_bruits"])
    sel = c if len(c) <= 40 else [c[int(i)] for i in np.linspace(0, len(c) - 1, 40)]
    L = np.load(APOD / "entrees" / nom / "input.npy", mmap_mode="r").mean(0)
    tiles = []
    for i, r in enumerate(sel):
        t = cv2.cvtColor(vign(L, r["x"], r["y"]), cv2.COLOR_GRAY2BGR)
        cv2.putText(t, f"{i} {r['rapport_fwhm']:.1f}x p{r['crete_bruits']:.0f}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        tiles.append(t)
    while len(tiles) % 8:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]
    cv2.imencode(".jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(str(OUT / "revue" / f"{nom}.jpg"))
    (OUT / "revue" / f"{nom}.json").write_text(json.dumps(sel), encoding="utf-8")
    print(nom, len(c), "->", len(sel))
