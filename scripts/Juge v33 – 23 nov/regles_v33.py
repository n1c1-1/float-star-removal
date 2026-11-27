# ======================================================================
# regles_v33.py
# Role     : Judge v33 (23 Nov): two LOCAL rules added to set 1's judge (construire_paire.py), nothing else changes.
# Step     : 2. Zone judge: weight 0 where the chain fails (v33) (see scripts/LISEZMOI.md)
# Inputs   : input, chain target, zones
# Outputs  : weight map
# Example  : python regles_v33.py [arguments]  (see the docstring; paths through the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Judge v33 (23 Nov): two LOCAL rules added to set 1's judge (construire_paire.py), nothing else changes.
 (a) target brighter than the input: blob >= 25 px where (target - input), mean luminance of the 3 colours smoothed 1.5 px,
     exceeds SEUIL_A noises -> weight 0 on the blob dilated by 3 px;
 (b) star left in the target: at a catalogue position (the image's deep Gaia), compact peak of the target above the local
     background by more than SEUIL_B noises -> weight 0 on a disc of radius max(3 x FWHM, measured remnant radius).
Noise = robust spread of (L - L blurred 4 px) of the input, as in analyse_v33.py. The two rules only set to 0 (idempotent)."""
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

SEUIL_A, TAILLE_A, DIL_A = 5.0, 25, 3
SEUIL_B, R_MIN_FW, G_MAX = 20.0, 2.5, 21.5
SEUIL_REST, CAP_FW = 5.0, 4.0
DEEP = (Path(__import__("os").environ.get("RACINE_SCRIPTS") or Path(__file__).resolve().parents[1]) / "Methode v33 production – 23 nov" / "Catalogue Gaia profond – 20 nov")
SORT = (Path(__import__("os").environ.get("RACINE_SCRIPTS") or Path(__file__).resolve().parents[1]) / "Methode v33 production – 23 nov" / "Piste1" / "sorties")


def mad(v):
    v = np.asarray(v, np.float64)
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) + 1e-12


def catalogue(image):
    f = DEEP / f"gaia_dr3_profond_{image}.npz"
    if f.exists():
        z = np.load(f)
        ok = np.isfinite(z["Gmag"]) & np.isfinite(z["x_pix"]) & np.isfinite(z["y_pix"])
        return z["x_pix"][ok].astype(float), z["y_pix"][ok].astype(float), z["Gmag"][ok].astype(float)
    cat = np.load(SORT / image / "astrometrie.npz", allow_pickle=True)["cat"].item()
    return np.asarray(cat["x"], float), np.asarray(cat["y"], float), np.asarray(cat["g"], float)


def fwhm_image(image, defaut=3.0):
    try:
        return float(json.loads((SORT / image / "modele_psf_image.json").read_text(encoding="utf-8"))["fwhm"])
    except Exception:
        return defaut


def regle_a(I, T, W, bruit):
    L, TL = I.mean(0), T.mean(0)
    d = cv2.GaussianBlur(np.nan_to_num(TL - L, posinf=1e30, neginf=-1e30).astype(np.float32), (0, 0), 1.5)
    lab, n = ndi.label(d > SEUIL_A * bruit)
    if not n:
        return 0
    sz = ndi.sum(np.ones_like(d), lab, np.arange(1, n + 1))
    garde = 1 + np.nonzero(sz >= TAILLE_A)[0]
    if not len(garde):
        return 0
    m = ndi.binary_dilation(np.isin(lab, garde), iterations=DIL_A)
    n_av = int((W > 0).sum())
    W[m] = 0.0
    return int(len(garde))


def regle_b(I, T, W, cl, bruit, image, x0, y0, fw):
    """Candidats : (1) positions du catalogue dans la tuile ; (2) pics compacts de la cible hors catalogue (geante, etoile hors
    Gaia) seulement la ou la chaine a touche (classe != 0 sur plus de la moitie d'un disque de 2 FWHM) : un noeud de nebuleuse
    intact (classe 0) n'est pas pris."""
    H, Wd = T.shape[1:]
    X, Y, G = catalogue(image)
    Lt = np.nan_to_num(T.mean(0)).astype(np.float32)
    sm = cv2.GaussianBlur(Lt, (0, 0), 0.7)
    k = max(15, int(round(17 * fw)) | 1)
    bg = ndi.median_filter(Lt[::2, ::2], size=max(5, k // 2) | 1)
    bg = cv2.resize(bg, (Wd, H), interpolation=cv2.INTER_LINEAR)
    ex = sm - bg
    lab, _ = ndi.label(ex > SEUIL_REST * bruit)
    yy, xx = np.mgrid[0:H, 0:Wd]
    touche = (cl != 0).astype(np.float32)
    s = (X >= x0 - 3) & (X < x0 + Wd + 3) & (Y >= y0 - 3) & (Y < y0 + H + 3) & (G <= G_MAX)
    cand = []
    for x, y in zip(X[s] - x0, Y[s] - y0):
        xi, yi = int(round(x)), int(round(y))
        a, b, c, d = max(yi - 2, 0), min(yi + 3, H), max(xi - 2, 0), min(xi + 3, Wd)
        if a >= b or c >= d:
            continue
        w = ex[a:b, c:d]
        j = np.unravel_index(np.argmax(w), w.shape)
        if w[j] > SEUIL_B * bruit:
            cand.append((float(w[j]), a + j[0], c + j[1], x, y))
    mx = (ex == ndi.maximum_filter(ex, size=max(5, int(round(2 * fw)) | 1))) & (ex > SEUIL_B * bruit)
    for py, px in zip(*np.nonzero(mx)):
        r2 = int(round(2 * fw))
        tz = touche[max(py - r2, 0):py + r2 + 1, max(px - r2, 0):px + r2 + 1]
        if tz.mean() > 0.5:
            cand.append((float(ex[py, px]), int(py), int(px), float(px), float(py)))
    cand.sort(reverse=True)
    n, rayons = 0, []
    for e, py, px, x, y in cand:
        l = lab[py, px]
        reste = float(np.sqrt((lab == l).sum() / np.pi)) if l else 0.0
        r = max(R_MIN_FW * fw, min(reste, CAP_FW * fw))
        disque = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
        if not (W[disque] > 0).any():
            continue                                  # deja a 0 sur tout le disque
        W[disque] = 0.0
        n += 1
        rayons.append((round(float(x), 1), round(float(y), 1), round(e / bruit, 1), round(r, 1)))
    return n, rayons


def appliquer(I, T, W, cl, image, x0, y0, fw=None):
    """Met a 0 le poids W (en place). Rend le nombre de taches (a) et d'etoiles restees (b)."""
    L = I.mean(0)
    bruit = mad(L - cv2.GaussianBlur(L, (0, 0), 4.0))
    fwh = fw if fw else fwhm_image(image)
    na = regle_a(I, T, W, bruit)
    nb, ray = regle_b(I, T, W, cl, bruit, image, x0, y0, fwh)
    return dict(taches_plus_claires=na, etoiles_restees=nb, restes=ray[:30])
