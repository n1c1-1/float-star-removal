# ======================================================================
# mosaiques.py
# Role     : Mosaics by group of 4 images -> <DONNEES>/Choix des tuiles v2 - 23 nov/mosaique_NN.jpg.
# Step     : 2. Tile selection (23 Nov) (see scripts/LISEZMOI.md)
# Inputs   : images, mosaics
# Outputs  : list of tiles
# Example  : python mosaiques.py [arguments]  (see the docstring; paths via the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Mosaics by group of 4 images -> <DONNEES>/Choix des tuiles v2 - 23 nov/mosaique_NN.jpg. One image at a time, 2 threads, no GPU."""
import os, sys, csv
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np, cv2
cv2.setNumThreads(2)
ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parent / "Paires jeu 2 \u2013 23 nov"))
import planches_choix as pc
OUT = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Choix des tuiles v2 " / "u2013 23 nov")
COL = (0, 215, 255)   # jaune (BGR)
cands = {}
for r in csv.DictReader(open(OUT / "candidats.csv", encoding="utf-8-sig"), delimiter=";"):
    cands.setdefault(r["image"], []).append((int(r["num"]), int(r["x"]), int(r["y"])))
noms = sorted(cands)


def txt(im, s, x, y, sc, th):
    cv2.putText(im, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(im, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, COL, th, cv2.LINE_AA)


def ligne(nom):
    a = pc.charger(nom); a = a[None] if a.ndim == 2 else a
    C, H, W = a.shape
    st = max(1, int(np.ceil(max(H, W) / 2500)))
    ch = []
    for c in range(min(C, 3)):
        p = np.nan_to_num(np.asarray(a[c][::1, ::1], np.float32))
        ch.append(p)
    while len(ch) < 3:
        ch.append(ch[0])
    bk = [float(np.median(c[::4, ::4])) for c in ch]
    top = [float(np.percentile(c[::4, ::4], 99.8)) for c in ch]

    def etire(sub):
        o = [np.arcsinh(60 * np.clip((sub[c] - bk[c]) / max(top[c] - bk[c], 1e-9), 0, None)) / np.arcsinh(60) for c in range(3)]
        return np.ascontiguousarray((np.clip(np.stack(o, -1), 0, 1) * 255 + .5).astype(np.uint8)[:, :, ::-1])
    s = min(1.0, 450 / max(H, W))
    w, h = max(1, round(W * s)), max(1, round(H * s))
    ov = etire([cv2.resize(c, (w, h), interpolation=cv2.INTER_AREA) for c in ch])
    for n, x, y in cands[nom]:
        x0, y0, x1, y1 = round(x * s), round(y * s), round((x + 512) * s), round((y + 512) * s)
        cv2.rectangle(ov, (x0, y0), (x1, y1), COL, 2)
        txt(ov, str(n), x0 + 4, y0 + 24, 0.9, 2)
    th = []
    for n, x, y in sorted(cands[nom]):
        t = etire([c[y:y + 512, x:x + 512] for c in ch])
        t = cv2.resize(t, (160, 160), interpolation=cv2.INTER_AREA)
        txt(t, str(n), 4, 28, 1.0, 2)
        th.append(t)
    while len(th) < 8:
        th.append(np.zeros((160, 160, 3), np.uint8))
    g = 4
    gh = 2 * 160 + g
    R = max(h, gh + 24)
    row = np.full((R, 450 + 10 + 4 * 160 + 3 * g, 3), 30, np.uint8)
    row[:h, :w] = ov
    for i, t in enumerate(th):
        r_, c_ = divmod(i, 4)
        row[24 + r_ * (160 + g):24 + r_ * (160 + g) + 160, 460 + c_ * (160 + g):460 + c_ * (160 + g) + 160] = t
    cv2.putText(row, nom, (462, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return row


for k in range(0, len(noms), 4):
    rows = []
    for nom in noms[k:k + 4]:
        try:
            rows.append(ligne(nom))
        except Exception as e:
            print(nom, "ECHEC", repr(e)[:200], flush=True)
        print(nom, flush=True)
    if not rows:
        continue
    Wm = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 6), (0, Wm - r.shape[1]), (0, 0)), constant_values=0) for r in rows]
    m = np.vstack(rows)
    cv2.imencode(".jpg", m, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(OUT / f"mosaique_{k // 4 + 1:02d}.jpg"))
    print("mosaique", k // 4 + 1, m.shape, flush=True)
