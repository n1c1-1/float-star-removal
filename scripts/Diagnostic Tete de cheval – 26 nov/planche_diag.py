# ======================================================================
# planche_diag.py
# Rôle     : Planche et tableau du diagnostic : par etoile, entree | melange (V2 x2, comme la chaine) | meilleur test | modele FLOAT, meme etirement (asinh(max(x,0)/2)/asinh(1000), x en unites de bruit).
# Étape    : 13. Contrôle des étoiles saturées à faible signal (Tête de Cheval, 26 nov) (voir scripts/LISEZMOI.md)
# Entrées  : découpes de l'image réelle et des images de validation, réseaux .pt
# Sorties  : tableau de flux restant par étoile
# Exemple  : python planche_diag.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
r"""Planche et tableau du diagnostic : par etoile, entree | melange (V2 x2, comme la chaine) | meilleur test | modele FLOAT, meme etirement
(asinh(max(x,0)/2)/asinh(1000), x en unites de bruit). Ecrit planche_diagnostic.jpg et tableau.md dans <DONNEES>\Diagnostic..."""
import sys, json
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np, cv2
OUT = next((Path(__import__("os").environ.get("DONNEES", "donnees"))).glob("Diagnostic Tete de cheval*")); TR = OUT / "travail"
res = json.load(open(OUT / "resultats.json", encoding="utf-8")); tab = res["tableau"]
noms = list(tab)
def vue(a):
    v = np.arcsinh(np.clip(a, 0, None) / 2) / np.arcsinh(1000.0)
    return np.ascontiguousarray((np.clip(v, 0, 1).transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8))
def txt(im, s, y=22):
    im = im.copy(); cv2.putText(im, s, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4); cv2.putText(im, s, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1); return im
H = 160; rows = []
for i, n in enumerate(noms, 1):
    z = np.load(TR / f"crop_{i}.npz"); cx, cy = [int(v) for v in z["c"]]
    t = tab[n]; best = min((k for k in t if k != "x1 FLOAT"), key=lambda k: t[k][1])
    def c(a): return vue(a[:, max(0, cy - H):cy + H, max(0, cx - H):cx + H])
    cols = [txt(c(z["entree"]), n[:30]), txt(c(z["melange_V2_x2"]), f"melange V2 x2  {t['melange V2 x2'][1]:.2f}"),
            txt(c(z[best.replace(" ", "_")]), f"{best}  {t[best][1]:.2f}"), txt(c(z["x1_FLOAT"]), f"modele FLOAT  {t['x1 FLOAT'][1]:.2f}")]
    cols = [cv2.resize(x, (320, 320), interpolation=cv2.INTER_AREA) if x.shape[0] != 320 or x.shape[1] != 320 else x for x in cols]
    rows.append(np.hstack([np.hstack([x, np.full((320, 4, 3), 255, np.uint8)]) for x in cols]))
cv2.imencode(".jpg", np.vstack([np.vstack([r, np.full((4, r.shape[1], 3), 255, np.uint8)]) for r in rows]), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(OUT / "planche_diagnostic.jpg"))
cles = ["x1 E3", "x1 N5", "x1 N7", "x1 melange", "x1 melange50", "melange V2 x2", "melange V2 x4", "melange gain x0.5", "melange gain x4", "melange gain x8",
        "melange debruite x1", "melange debruite V2", "N5 V2 x2", "melange50 V2 x2", "x1 FLOAT"]
ent = ["E3 x1", "N5 x1", "N7 x1", "mel 0,75 x1", "mel 0,5 x1", "mel V2 x2", "mel V2 x4", "gain 0,5", "gain 4", "gain 8", "debr. x1", "debr. V2", "N5 V2", "mel0,5 V2", "FLOAT"]
L = ["Flux restant (disque de 2 FWHM, minimum 6 px ; entre parentheses : disque de 25 px). 0 = etoile retiree, 1 = intacte.", "",
     "| etoile | " + " | ".join(ent) + " |", "|" + "---|" * (len(ent) + 1)]
for n in noms:
    t = tab[n]; L.append(f"| {n} | " + " | ".join(f"{t[k][0]:.2f} ({t[k][1]:.2f})" if k in t else "-" for k in cles) + " |")
L += ["", "Fiches (pic en bruits ; pic lineaire 0-1 pour la Tete ; pixels >= 0,95 pic dans 3 FWHM ; FWHM de la mesure en px ; aigrettes = p99,5/mediane de l'anneau 4-10 FWHM ; bruit local) :", ""]
for f in res["fiches"]: L.append("- " + json.dumps(f, ensure_ascii=False))
(OUT / "tableau.md").write_text("\n".join(L), encoding="utf-8"); print("\n".join(L))
