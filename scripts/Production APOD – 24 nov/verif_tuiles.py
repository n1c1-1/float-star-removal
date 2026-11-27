# ======================================================================
# verif_tuiles.py
# Role     : Chain module (see its docstring below and scripts/LISEZMOI.md).
# Stage    : 3. Production of the APOD web pairs (24 Nov) (see scripts/LISEZMOI.md)
# Inputs   : de-linearised APOD images
# Outputs  : _debruitee .npz pairs
# Example  : python verif_tuiles.py [arguments]  (see the docstring; paths via variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
import csv, sys, numpy as np, cv2
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
E_ = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Paires étoiles APOD – 24 nov" / "entrees")
rows = list(csv.DictReader(open("tuiles.csv", encoding="utf-8"), delimiter=";"))
cells = []
for r in rows:
    a = np.load(E_ / r["image"] / "input.npy", mmap_mode="r"); x, y = int(r["x"]), int(r["y"])
    t = np.array(a[:, y:y + 512, x:x + 512]); bk = np.median(t, axis=(1, 2)); top = np.percentile(t.reshape(3, -1), 99.7, axis=1)
    o = np.stack([np.arcsinh(40 * np.clip((t[c] - bk[c]) / max(top[c] - bk[c], 1e-3), 0, None)) / np.arcsinh(40) for c in range(3)], -1)
    im = (np.clip(o, 0, 1) * 255).astype(np.uint8)
    im = cv2.resize(im, (320, 320), interpolation=cv2.INTER_AREA).copy()
    cv2.putText(im, r["image"][5:] + " " + r["choix"][:9], (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
    cells.append(im)
for k in range(0, len(cells), 20):
    ch = cells[k:k + 20]; ch += [np.zeros_like(cells[0])] * (20 - len(ch))
    g = np.concatenate([np.concatenate(ch[i:i + 5], 1) for i in range(0, 20, 5)], 0)
    cv2.imencode(".jpg", g[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(f"verif_{k//20+1}.jpg")
print(len(cells))
