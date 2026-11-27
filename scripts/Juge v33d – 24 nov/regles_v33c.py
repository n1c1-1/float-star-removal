# ======================================================================
# regles_v33c.py
# Role     : Judge v33 (23 Nov): two LOCAL rules added to set 1's judge (construire_paire.py), nothing else changes.
# Step     : 2. Judge v33c/v33d: halos and faint stars left (24 Nov) (see scripts/LISEZMOI.md)
# Inputs   : v33 pairs
# Outputs  : weights set to 0 (_v33d files)
# Example  : python regles_v33c.py [arguments]  (see the docstring; paths through the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
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


# ---------------- v33c (24 nov) : trois corrections des poids, mises a 0 seulement ----------------
ILOT_MAX = 400          # regle 1 : ilot de poids > 0 plus petit que ca et enferme dans du poids 0
SEUIL_PLAT = 3.0        # regle 2 : ecart disque / anneau en bruits
R_PLAT_FW = (1.0, 1.5, 2.0, 2.5)
ANNEAU_FW = 1.5         # largeur de l'anneau de reference (en FWHM)
REGLE_3 = True
TEX = 0.5               # poids de la texture locale de l anneau dans le bruit de reference (0 = bruit global seul)


def regle_1(W):
    """Ilots de poids > 0 (8-connexes) plus petits que ILOT_MAX qui ne touchent pas le bord de la tuile : donc entoures de poids 0."""
    lab, n = ndi.label(W > 0, structure=np.ones((3, 3)))
    if not n:
        return 0, 0
    sz = ndi.sum(np.ones_like(W), lab, np.arange(1, n + 1))
    bord = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    petit = np.zeros(n + 1, bool)
    petit[1:] = sz < ILOT_MAX
    petit[bord] = False
    m = petit[lab]
    W[m] = 0.0
    return int(petit.sum()), int(m.sum())


def regle_2(I, T, W, cl, bruit, image, x0, y0, fw, seuil=SEUIL_PLAT):
    """Plateau (disque plus clair) ou creux a une position du catalogue ou la chaine a touche (classe != 0 dans le disque de 1 FWHM) :
    moyenne cible du disque contre mediane de l'anneau autour ; si |ecart| > seuil bruits, disque a 0. Rayon 1 a 2,5 FWHM : celui d'ecart maximal."""
    H, Wd = T.shape[1:]
    X, Y, G = catalogue(image)
    Lt = np.nan_to_num(T).astype(np.float32)                       # par canal : un plateau peut ne changer que la couleur
    bc = np.array([mad(I[c] - cv2.GaussianBlur(I[c], (0, 0), 4.0)) for c in range(3)])[:, None, None]
    s = (X >= x0 - 3) & (X < x0 + Wd + 3) & (Y >= y0 - 3) & (Y < y0 + H + 3) & (G <= G_MAX)
    touche = cv2.dilate((cl != 0).astype(np.uint8), np.ones((3, 3), np.uint8))
    rmax = R_PLAT_FW[-1] * fw
    pris = []
    for x, y in zip(X[s] - x0, Y[s] - y0):
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < Wd and 0 <= yi < H) or not touche[yi, xi]:
            continue
        R = int(np.ceil(rmax + ANNEAU_FW * fw)) + 1
        a, b, c, d = max(yi - R, 0), min(yi + R + 1, H), max(xi - R, 0), min(xi + R + 1, Wd)
        yy, xx = np.mgrid[a:b, c:d]
        r2 = (xx - x) ** 2 + (yy - y) ** 2
        best = (0.0, 0.0)
        for kf in R_PLAT_FW:
            r = kf * fw
            din = r2 <= r * r
            ann = (r2 > r * r) & (r2 <= (r + ANNEAU_FW * fw) ** 2)
            if din.sum() < 3 or ann.sum() < 8:
                continue
            ra = Lt[:, a:b, c:d][:, ann]
            md = np.median(ra, 1)
            loc = np.maximum(bc[:, 0, 0], TEX * 1.4826 * np.median(np.abs(ra - md[:, None]), 1))   # texture locale de l'anneau
            ec = (Lt[:, a:b, c:d][:, din].mean(1) - md) / loc
            e = float(ec[np.argmax(np.abs(ec))])
            if abs(e) > abs(best[0]):
                best = (e, r)
        if abs(best[0]) > seuil:
            disque = (np.mgrid[0:H, 0:Wd][1] - x) ** 2 + (np.mgrid[0:H, 0:Wd][0] - y) ** 2 <= best[1] ** 2
            if (W[disque] > 0).any():
                W[disque] = 0.0
                pris.append((round(float(x), 1), round(float(y), 1), round(best[0], 1), round(best[1], 1)))
    return len(pris), pris


def regle_3(I, T, W, bruit_c):
    """Zones de poids 0,5 (reparations) d'au moins 100 px dont la couleur ou le niveau (par canal) s'ecarte de l'anneau 3-12 px autour de plus
    de SEUIL_PLAT bruits de reference (meme mesure que la regle 2, a l'echelle de la zone) : zone a 0."""
    lab, n = ndi.label(np.abs(W - 0.5) < 1e-6)
    n_pris = 0
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        comp = lab[sl] == k
        if comp.sum() < 100:
            continue
        a = max(sl[0].start - 14, 0); b = min(sl[0].stop + 14, W.shape[0]); c = max(sl[1].start - 14, 0); d = min(sl[1].stop + 14, W.shape[1])
        cm = lab[a:b, c:d] == k
        ring = ndi.binary_dilation(cm, iterations=12) & ~ndi.binary_dilation(cm, iterations=3) & (W[a:b, c:d] > 0) & ~((lab[a:b, c:d] > 0) & ~cm)
        if ring.sum() < 50:
            continue
        Tz = np.nan_to_num(T[:, a:b, c:d])
        md = np.median(Tz[:, ring], 1)
        loc = np.maximum(bruit_c, TEX * 1.4826 * np.median(np.abs(Tz[:, ring] - md[:, None]), 1))
        ec = (Tz[:, cm].mean(1) - md) / loc
        if np.abs(ec).max() > SEUIL_PLAT:
            W[a:b, c:d][cm] = 0.0
            n_pris += 1
    return n_pris


def appliquer_v33c(I, T, W, cl, image, x0, y0, fw=None, seuil=SEUIL_PLAT):
    """Poids W (deja passe par v33) mis a 0 en place. Rend un dict de comptes."""
    L = I.mean(0)
    bruit = mad(L - cv2.GaussianBlur(L, (0, 0), 4.0))
    fwh = fw if fw else fwhm_image(image)
    n2, pris = regle_2(I, T, W, cl, bruit, image, x0, y0, fwh, seuil)   # d'abord : peut creer des ilots
    bc = np.array([mad(I[c] - cv2.GaussianBlur(I[c], (0, 0), 4.0)) for c in range(3)])
    n3 = regle_3(I, T, W, bc) if REGLE_3 else 0
    ni, px = regle_1(W)
    return dict(plateaux=n2, ilots=ni, ilots_px=px, zones05=n3, pris=pris[:40])
