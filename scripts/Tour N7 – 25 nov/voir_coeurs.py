# ======================================================================
# voir_coeurs.py
# Rôle     : Module de la chaîne (voir sa docstring ci-dessous et scripts/LISEZMOI.md).
# Étape    : 9. N7 : cœurs de galaxie synthétiques (25 nov) (voir scripts/LISEZMOI.md)
# Entrées  : paires v33g + cœurs synthétiques
# Sorties  : instantanés N7, tableau_n7.md
# Exemple  : python voir_coeurs.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
import sys, numpy as np, cv2, json
from pathlib import Path
sys.path.insert(0, r"<SCRIPTS>\Validation – 22 nov")
from liste import LISTE
L = json.load(open(Path(sys.argv[0]).parent / "coeurs_liste.json", encoding="utf-8-sig"))
tl = []
for nom, x, y in L:
    d0 = Path(LISTE[nom][0]); im = np.load(d0 / "input.npy", mmap_mode="r"); H, W = im.shape[1:]
    x0, y0 = max(0, min(W - 200, x - 100)), max(0, min(H - 200, y - 100))
    a = np.asarray(im[:, y0:y0+200, x0:x0+200], np.float32); top = np.arcsinh(np.percentile(a.mean(0), 99.9) / 60) + 1e-6
    v = np.clip(np.arcsinh(np.clip(a, 0, None) / 60) / top, 0, 1)
    t = cv2.resize((v.transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8), (300, 300)).copy()
    cv2.putText(t, f"{nom[:12]} {x},{y}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1); tl.append(t)
cv2.imencode(".jpg", np.hstack(tl))[1].tofile(r"<DONNEES>\Entrainement étoiles N7 – 25 nov\coeurs_candidats.jpg")
