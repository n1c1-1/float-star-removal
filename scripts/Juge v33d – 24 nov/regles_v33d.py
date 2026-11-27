# ======================================================================
# regles_v33d.py
# Role     : Judge v33d (24 Nov): v33c UNCHANGED + two more rules, which only set weights to 0 (idempotent).
# Step     : 2. Judge v33c/v33d: halos and faint stars left (24 Nov) (see scripts/LISEZMOI.md)
# Inputs   : v33 pairs
# Outputs  : weights set to 0 (_v33d files)
# Example  : python regles_v33d.py [arguments]  (see the docstring; paths through the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Judge v33d (24 Nov): v33c UNCHANGED + two more rules, which only set weights to 0 (idempotent).
 rule H (halo): bright catalogue star: halo extent measured in the INPUT; if the target keeps light on this disc,
                weight 0 up to that extent (without the 4 FWHM cap of rule b).
 rule F (faint): catalogue position where the target keeps a peak of more than SEUIL_F noises: small disc at 0.
Use on another folder of pairs: see rejuger_v33d.py (--dossier)."""
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import regles_v33c as C  # noqa: E402
from regles_v33c import mad, catalogue, fwhm_image, G_MAX  # noqa: E402,F401

# ---- reglages (valeurs fixees par mesure, voir JOURNAL.md) ----
G_HALO = 12.0           # etoiles assez brillantes pour un halo (magnitude Gaia G)
SEUIL_FOND = 2.0        # le halo finit ou le profil retombe au fond local + 2 bruits
R_MIN_HALO_FW = 3.0     # on ne cherche pas la fin du halo en dessous
R_MAX_HALO = 40.0       # plafond en pixels (mesure : au dela le « halo » est surtout de la nebuleuse, ex. NGC 6543 : 850 bruits)
SEUIL_H = 2.0           # cible : moyenne dans le rayon au-dessus du fond juste au-dela, en bruits
SEUIL_F, R_FAIBLE_FW, TEX_F = 5.0, 1.5, 0.5   # TEX_F : poids de la texture de la couronne dans le bruit local (comme TEX de v33c)


def profil(A, x, y, rmax):
    """Mediane par anneaux de 1 px autour de (x, y), rayons 0..rmax-1 ; nan si moins de 12 pixels dans la tuile."""
    H, W = A.shape
    a, b, c, d = max(int(y - rmax), 0), min(int(y + rmax) + 2, H), max(int(x - rmax), 0), min(int(x + rmax) + 2, W)
    yy, xx = np.mgrid[a:b, c:d]
    rb = np.hypot(xx - x, yy - y).astype(int).ravel()
    v = A[a:b, c:d].ravel()
    o = np.argsort(rb, kind="stable"); rb, v = rb[o], v[o]
    lim = np.searchsorted(rb, np.arange(rmax + 1))
    return np.array([np.median(v[lim[k]:lim[k + 1]]) if lim[k + 1] - lim[k] >= 12 else np.nan for k in range(rmax)])


def etendue_halo(L, x, y, fw, bruit, rmax=None):
    """Rayon (px) ou le profil median de l'entree L retombe au fond local + SEUIL_FOND bruits ; fond local = mediane de L sur
    l'anneau [r, 1,5 r] (a l'echelle du rayon : suit une nebuleuse). None si le profil n'est pas mesurable."""
    rmax = int(rmax or R_MAX_HALO)
    R = int(rmax * 1.5) + 2
    H, W = L.shape
    a, b, c, d = max(int(y - R), 0), min(int(y + R) + 2, H), max(int(x - R), 0), min(int(x + R) + 2, W)
    yy, xx = np.mgrid[a:b, c:d]
    r = np.hypot(xx - x, yy - y).ravel(); v = L[a:b, c:d].ravel()
    p = profil(L, x, y, rmax)
    k0 = int(round(R_MIN_HALO_FW * fw))
    for k in range(k0, rmax):
        m = (r >= k) & (r < 1.5 * k)
        if m.sum() < 60 or not np.isfinite(p[k]):
            return None
        if p[k] - np.median(v[m]) <= SEUIL_FOND * bruit:
            return float(k)
    return float(rmax)


def _exmap(A, fw):
    """Excès local d'une image : lissage 0,7 px moins fond median (comme la regle b)."""
    Lt = np.nan_to_num(A).astype(np.float32)
    sm = cv2.GaussianBlur(Lt, (0, 0), 0.7)
    k = max(15, int(round(17 * fw)) | 1)
    bg = ndi.median_filter(Lt[::2, ::2], size=max(5, k // 2) | 1)
    bg = cv2.resize(bg, (Lt.shape[1], Lt.shape[0]), interpolation=cv2.INTER_LINEAR)
    return sm - bg


def regle_h(I, T, W, bruit, image, x0, y0, fw):
    """Halo : pour chaque etoile du catalogue G <= G_HALO (dans la tuile +- 20 px) : etendue du halo dans l'entree (etendue_halo).
    Si la moyenne de la cible entre 2 FWHM et cette etendue depasse la mediane de la couronne [rh ; 1,3 rh] de plus de SEUIL_H bruits,
    poids 0 sur le disque de cette etendue. Rend (nombre, liste (x, y, G, rh, exces))."""
    H, Wd = T.shape[1:]
    X, Y, G = catalogue(image)
    L = I.mean(0); TL = np.nan_to_num(T.mean(0))
    s = (X >= x0 - 20) & (X < x0 + Wd + 20) & (Y >= y0 - 20) & (Y < y0 + H + 20) & (G <= G_HALO)
    yy, xx = np.mgrid[0:H, 0:Wd]
    pris = []
    for k in np.argsort(G[s]):
        x, y, g = X[s][k] - x0, Y[s][k] - y0, G[s][k]
        rh = etendue_halo(L, x, y, fw, bruit)
        if rh is None:
            continue
        r = np.hypot(xx - x, yy - y)
        dans = (r >= 2 * fw) & (r < rh)
        cour = (r >= rh) & (r < 1.3 * rh + 1)
        if dans.sum() < 30 or cour.sum() < 30:
            continue
        ex = float((TL[dans].mean() - np.median(TL[cour])) / bruit)
        if ex > SEUIL_H and (W[r < rh] > 0).any():
            W[r < rh] = 0.0
            pris.append((round(float(x), 1), round(float(y), 1), round(float(g), 1), rh, round(ex, 1)))
    return len(pris), pris


def regle_f(I, T, W, bruit, image, x0, y0, fw):
    """Etoiles faibles restees : a chaque position du catalogue (toutes magnitudes), pic compact (noyau de rayon 1,5 px contre la mediane
    de la couronne [2 ; 3,5 FWHM]) de plus de SEUIL_F bruits LOCAUX, dans la cible (etoile restee) ET dans l'entree (etoile visible) :
    disque de poids 0, rayon R_FAIBLE_FW x FWHM. Bruit local = max(bruit global, dispersion robuste de la couronne, dans l'image testee) :
    une nebuleuse brillante ou texturee ne declenche pas la regle."""
    H, Wd = T.shape[1:]
    X, Y, G = catalogue(image)
    L = I.mean(0); TL = np.nan_to_num(T.mean(0)).astype(np.float32)
    sI, sT = cv2.GaussianBlur(L.astype(np.float32), (0, 0), 0.7), cv2.GaussianBlur(TL, (0, 0), 0.7)
    s = (X >= x0) & (X < x0 + Wd) & (Y >= y0) & (Y < y0 + H)
    R = int(np.ceil(3.5 * fw)) + 1
    rr = int(np.ceil(R_FAIBLE_FW * fw)) + 1
    n, pris = 0, []
    for x, y in zip(X[s] - x0, Y[s] - y0):
        xi, yi = int(round(x)), int(round(y))
        a, b, c, d = max(yi - R, 0), min(yi + R + 1, H), max(xi - R, 0), min(xi + R + 1, Wd)
        yy, xx = np.mgrid[a:b, c:d]
        r = np.hypot(xx - x, yy - y)
        ring, core = (r >= 2 * fw) & (r < 3.5 * fw), r <= 1.5
        if ring.sum() < 12 or core.sum() < 3:
            continue
        ok = True
        for A in (sT[a:b, c:d], sI[a:b, c:d]):
            rv = A[ring]; md = np.median(rv)
            loc = max(bruit, TEX_F * 1.4826 * np.median(np.abs(rv - md)))
            if A[core].max() - md <= SEUIL_F * loc:
                ok = False
                break
        if not ok:
            continue
        a2, b2, c2, d2 = max(yi - rr, 0), min(yi + rr + 1, H), max(xi - rr, 0), min(xi + rr + 1, Wd)
        yy, xx = np.mgrid[a2:b2, c2:d2]
        disque = (xx - x) ** 2 + (yy - y) ** 2 <= (R_FAIBLE_FW * fw) ** 2
        w = W[a2:b2, c2:d2]
        if (w[disque] > 0).any():
            w[disque] = 0.0
            n += 1
            if len(pris) < 40:
                pris.append((round(float(x), 1), round(float(y), 1)))
    return n, pris


def appliquer_v33d(I, T, W, image, x0, y0, fw=None):
    """Poids W (deja v33c) mis a 0 en place par les regles H puis F. Rend un dict de comptes."""
    L = I.mean(0)
    bruit = mad(L - cv2.GaussianBlur(L, (0, 0), 4.0))
    fwh = fw if fw else fwhm_image(image)
    nh, ph = regle_h(I, T, W, bruit, image, x0, y0, fwh)
    nf, pf = regle_f(I, T, W, bruit, image, x0, y0, fwh)
    return dict(halos=nh, faibles=nf, halos_pris=ph, faibles_pris=pf)
