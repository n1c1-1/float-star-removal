# ======================================================================
# planche_coeurs.py
# Rôle     : Planche de 8 paires synthétiques à CŒUR (entrée | cible), arcsinh 60, tuiles 256 sur les deux fonds réels du banc de vitesse.
# Étape    : 9. N7 : cœurs de galaxie synthétiques (25 nov) (voir scripts/LISEZMOI.md)
# Entrées  : paires v33g + cœurs synthétiques
# Sorties  : instantanés N7, tableau_n7.md
# Exemple  : python planche_coeurs.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
r"""Planche de 8 paires synthétiques à CŒUR (entrée | cible), arcsinh 60, tuiles 256 sur les deux fonds réels du banc de vitesse.
Sortie : <DONNEES>\Entrainement étoiles N7 – 25 nov\planche_coeurs.jpg (+ coeurs_infos.json). Processeur seul, 2 fils."""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[k] = "2"
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np, cv2
ICI = Path(__file__).resolve().parent
SR = (Path(__import__("os").environ.get("RACINE_SCRIPTS") or Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ICI)); sys.path.insert(0, str(SR / "Etoiles synthetiques – 22 nov"))
from paires_synthetiques_n7 import paire_synthetique_coeurs
OUT = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles N7 – 25 nov")
FONDS = [SR / "Methode v22 M57 – 22 nov" / "sorties" / "d_fond_calme" / "B" / "t3" / "sans_etoiles_tuile.npy",
         SR / "Methode v21 SHO – 22 nov" / "sorties" / "decoupe6" / "B" / "t3" / "sans_etoiles_tuile.npy"]
tiles = [np.load(p).astype(np.float32) for p in FONDS]


def etire(a, bk, wt):
    v = np.arcsinh(np.clip(a - bk, 0, None) / 60.0) / np.arcsinh(max(wt - bk, 1.0) / 60.0)
    return (np.clip(v, 0, 1).transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)       # RGB -> BGR pour cv2


rows, infos = [], []
for g in range(8):
    t = tiles[g % 2]
    oy, ox = (0, 0) if g < 2 else (int(t.shape[1] - 256) * (g % 4) // 3, int(t.shape[2] - 256) * ((g // 2) % 3) // 2)
    f = np.ascontiguousarray(t[:, oy:oy + 256, ox:ox + 256])
    e, c, co, inf = paire_synthetique_coeurs(f, 1000 + g, p_coeurs=1.0)
    assert np.array_equal(c, f + co) and len(inf) in (1, 2)
    assert not np.any((e - c)[:, :2, :2] > 1e9)
    bk = float(np.median(c[1])); wt = float(np.percentile(e[1], 99.8))
    p1, p2 = etire(e, bk, wt), etire(c, bk, wt)
    p1 = cv2.resize(p1, (384, 384), interpolation=cv2.INTER_NEAREST); p2 = cv2.resize(p2, (384, 384), interpolation=cv2.INTER_NEAREST)
    txt = " + ".join(f"n{i['sersic_n']:.1f} Re{i['re']:.0f} e{i['ell']:.2f} pic{i['pic']:.0f}{' sat' if i['sature'] else ''} {i['type'][:3]}" for i in inf)
    cv2.putText(p1, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    rows.append(np.concatenate([p1, np.full((384, 4, 3), 255, np.uint8), p2], 1))
    infos.append([{k: (round(v, 3) if isinstance(v, float) else v) for k, v in i.items()} for i in inf])
    print(g, txt, flush=True)
sep = np.full((384, 14, 3), 255, np.uint8)
grid = [np.concatenate([rows[i], sep, rows[i + 1]], 1) for i in range(0, 8, 2)]
sheet = np.concatenate(sum([[r, np.full((10, grid[0].shape[1], 3), 255, np.uint8)] for r in grid], [])[:-1], 0)
ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 90]); assert ok
buf.tofile(str(OUT / "planche_coeurs.jpg"))
(OUT / "coeurs_infos.json").write_text(json.dumps(infos, indent=1, ensure_ascii=False), encoding="utf-8")
print("fini", sheet.shape)
