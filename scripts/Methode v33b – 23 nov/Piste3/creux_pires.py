# ======================================================================
# creux_pires.py
# Rôle     : Les pires creux d'un tour : étoiles Gaia G < 13 de la tuile, mesure de juger.py (médiane 0-4 px contre anneau 8-15 px, en % de l'anneau), pour la piste 1, la référence avant remplissage (après soustractions) et le résultat ; class
# Étape    : 2. Chaîne classique de soustraction guidée par Gaia (v33b) (voir scripts/LISEZMOI.md)
# Entrées  : image empilée (unités de bruit), astrométrie, extrait Gaia profond hors ligne
# Sorties  : image sans étoiles, cartes de classes
# Exemple  : python creux_pires.py <tour> <tuile> [n]
# ======================================================================
"""Les pires creux d'un tour : étoiles Gaia G < 13 de la tuile, mesure de juger.py (médiane 0-4 px contre anneau 8-15 px,
en % de l'anneau), pour la piste 1, la référence avant remplissage (après soustractions) et le résultat ; classe de la
carte au centre, étoile brillante dont le halo couvre le point, distance à l'axe d'aigrette le plus proche.

    python creux_pires.py <tour> <tuile> [n]
"""
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mesure as M

tour, case = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
name, crop, tile = M.CASES[case]
d = HERE / "sorties" / tour / case
meas = json.loads((d / "mesures.json").read_text(encoding="utf-8"))
base = meas["base_piste1"]
cb = HERE / "sorties" / "base_piste1" / case / base
win = np.load(cb / "fenetre_sans_etoiles.npy")
P1 = win[:, 200:712, 200:712]
R = np.load(d / "sans_etoiles_tuile.npy")
REF = np.load(d / "reference_entree_moins_modeles_tuile.npy")
cls = np.load(d / "carte_remplissage_tuile.npy")
z = np.load(M.P1 / "sorties" / name / "astrometrie.npz", allow_pickle=True)["cat"].item()
gx = z["x"] - crop[0] - tile[0]; gy = z["y"] - crop[1] - tile[1]; gG = z["g"]
gy_, gx_ = np.mgrid[-15:16, -15:16]


def hole(img, x, y):
    L = img.mean(0)
    xi, yi = int(round(x)), int(round(y))
    cut = L[yi - 15:yi + 16, xi - 15:xi + 16]
    r = np.hypot(gx_ + xi - x, gy_ + yi - y)
    ring = np.median(cut[(r >= 8) & (r <= 15)])
    if ring <= 0.5:
        return None
    return 100 * (np.median(cut[r <= 4]) - ring) / ring


rows = []
for x, y, g in zip(gx, gy, gG):
    if not (16 <= x < 496 and 16 <= y < 496 and g < 13):
        continue
    h3 = hole(R, x, y)
    if h3 is None:
        continue
    near = []
    for e in meas["etoiles"]:
        dd = np.hypot(x - e["x_tuile"], y - e["y_tuile"])
        if dd <= e["R"]:
            ax = []
            for sp in e["aigrettes"]:
                t = np.deg2rad(sp["angle"])
                along = (x - e["x_tuile"]) * np.cos(t) + (y - e["y_tuile"]) * np.sin(t)
                if 0 < along <= sp["length"] + 6:
                    ax.append(abs(-(x - e["x_tuile"]) * np.sin(t) + (y - e["y_tuile"]) * np.cos(t)))
            near.append(dict(G=e["G"], dist=round(float(dd)), R=e["R"], aigrette_px=round(min(ax), 1) if ax else None))
    rows.append(dict(x=round(float(x)), y=round(float(y)), G=round(float(g), 2), p3=round(float(h3), 1),
                     ref=round(float(hole(REF, x, y) or 0), 1), p1=round(float(hole(P1, x, y) or 0), 1),
                     classe=int(cls[int(round(y)), int(round(x))]), halos=near))
rows.sort(key=lambda r: r["p3"])
for r in rows[:n]:
    print(r)
(d / "creux_pires.json").write_text(json.dumps(rows[:n], indent=1, ensure_ascii=False), encoding="utf-8")
