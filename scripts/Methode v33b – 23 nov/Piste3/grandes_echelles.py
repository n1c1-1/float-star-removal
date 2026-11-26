# ======================================================================
# grandes_echelles.py
# Rôle     : Mesure des grandes échelles : flou gaussien (16 et 48 px) de l'image sans étoiles contre celui de la référence (entrée moins modèles d'étoiles, avant remplissage, d'un tour 4), hors coeurs (zones dures dilatées) ; écart en bruits 
# Étape    : 2. Chaîne classique de soustraction guidée par Gaia (v33b) (voir scripts/LISEZMOI.md)
# Entrées  : image empilée (unités de bruit), astrométrie, extrait Gaia profond hors ligne
# Sorties  : image sans étoiles, cartes de classes
# Exemple  : python grandes_echelles.py <dossier_reference> <dossier_resultat> [<dossier_resultat> ...]
# ======================================================================
"""Mesure des grandes échelles : flou gaussien (16 et 48 px) de l'image sans étoiles contre celui de la référence
(entrée moins modèles d'étoiles, avant remplissage, d'un tour 4), hors coeurs (zones dures dilatées) ; écart en bruits
(médiane, max, 99e centile) et carte. Compare plusieurs résultats à la même référence.

    python grandes_echelles.py <dossier_reference> <dossier_resultat> [<dossier_resultat> ...]
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import cv2
import numpy as np
from scipy import ndimage as ndi
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mesure as M
sys.path.insert(1, str(M.P1))
from juger import save_img

ref_dir = Path(sys.argv[1])
ref = np.load(ref_dir / "reference_entree_moins_modeles_tuile.npy")
hard = np.unpackbits(np.load(ref_dir / "zones_dures_tuile.npy"))[:512 * 512].reshape(512, 512).astype(bool)
excl = ndi.binary_dilation(hard, iterations=4)
L0 = ref.mean(0)
sig = float(np.median(np.abs((L0 - cv2.GaussianBlur(L0, (0, 0), 6)) - np.median(L0 - cv2.GaussianBlur(L0, (0, 0), 6)))) * 1.4826)


def lp(a, s):
    w = cv2.GaussianBlur((~excl).astype(np.float32), (0, 0), s)
    return cv2.GaussianBlur(np.where(excl, 0, a).astype(np.float32), (0, 0), s) / np.maximum(w, 1e-6)


res = {}
for d in sys.argv[2:]:
    d = Path(d)
    R = np.load(d / "sans_etoiles_tuile.npy").mean(0)
    row = {}
    maps = []
    for s in (16, 48):
        diff = (lp(R, s) - lp(L0, s)) / sig
        v = np.abs(diff[~excl])
        row[f"flou{s}"] = dict(mediane=round(float(np.median(v)), 3), p99=round(float(np.percentile(v, 99)), 3), max=round(float(v.max()), 3))
        m = np.clip(0.5 + diff / 4.0, 0, 1)          # +-2 bruits = blanc / noir
        maps.append((np.stack([m, m, m], -1) * 255).astype(np.uint8))
    sep = np.full((512, 6, 3), 255, np.uint8)
    save_img(d / "8_grandes_echelles_ecart_flou16_flou48.png", np.concatenate([maps[0], sep, maps[1]], 1))
    res[str(d)] = row
    print(d.name, d.parent.name, row)
(HERE / "sorties" / "grandes_echelles.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
