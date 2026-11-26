# ======================================================================
# etoiles_outils.py
# Rôle     : Détection d'étoiles isolées, photométrie d'ouverture et rayon de demi-flux (banc de délinéarisation).
# Étape    : 3. Délinéarisation des images web (23 nov) (voir scripts/LISEZMOI.md)
# Entrées  : image étirée
# Sorties  : image rendue linéaire
# Exemple  : python etoiles_outils.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Détection d'étoiles isolées, photométrie d'ouverture et rayon de demi-flux (banc de délinéarisation)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def fond_carte(x: np.ndarray, bloc: int = 64) -> np.ndarray:
    """Carte de fond lisse : médianes par blocs, agrandies."""
    H, W = x.shape
    h, w = H // bloc, W // bloc
    b = x[: h * bloc, : w * bloc].reshape(h, bloc, w, bloc).swapaxes(1, 2).reshape(h, w, -1)
    med = np.median(b, axis=2)
    med = ndi.median_filter(med, size=3, mode="nearest")
    return ndi.zoom(med, (H / h, W / w), order=1, mode="nearest")[:H, :W]


def detecter(x: np.ndarray, sigma: float, k: float = 12.0, isol: int = 12, bord: int = 24, max_n: int = 3000,
             plafond: float | None = None):
    """Étoiles isolées de l'image 2D x (pic au-dessus du fond local > k sigma). -> (ys, xs) centroïdes."""
    fond = fond_carte(x)
    s = ndi.gaussian_filter(x - fond, 1.2)
    ss = sigma / (2 * np.sqrt(np.pi) * 1.2)            # bruit blanc lissé
    pic = (s == ndi.maximum_filter(s, size=7)) & (s > k * max(ss, 1e-12))
    ys, xs = np.nonzero(pic)
    ok = (ys > bord) & (ys < x.shape[0] - bord) & (xs > bord) & (xs < x.shape[1] - bord)
    ys, xs = ys[ok], xs[ok]
    if plafond is not None:
        keep = x[ys, xs] < plafond
        ys, xs = ys[keep], xs[keep]
    # isolées : aucun autre pic détecté (même plus faible, au-dessus de 4 sigma) à moins de isol px
    pic4 = (s == ndi.maximum_filter(s, size=5)) & (s > 4 * max(ss, 1e-12))
    yy, xx = np.nonzero(pic4)
    from scipy.spatial import cKDTree
    t = cKDTree(np.c_[yy, xx])
    cnt = np.array([len(t.query_ball_point((a, b), isol)) for a, b in zip(ys, xs)])
    ys, xs = ys[cnt == 1], xs[cnt == 1]
    order = np.argsort(-s[ys, xs])[:max_n]
    ys, xs = ys[order].astype(float), xs[order].astype(float)
    # centroïde 5x5 au-dessus du fond
    cy, cx = [], []
    for a, b in zip(ys.astype(int), xs.astype(int)):
        c = (x[a - 2:a + 3, b - 2:b + 3] - fond[a, b]).clip(0)
        t_ = c.sum()
        if t_ <= 0:
            cy.append(a); cx.append(b); continue
        g = np.mgrid[-2:3, -2:3]
        cy.append(a + (c * g[0]).sum() / t_); cx.append(b + (c * g[1]).sum() / t_)
    return np.array(cy), np.array(cx)


def photometrie(x: np.ndarray, ys, xs, ra: float, rin: float, rout: float):
    """Flux d'ouverture (fond = médiane de l'anneau) et rayon de demi-flux, pour chaque étoile. -> (flux, r50, fond)."""
    R = int(np.ceil(rout)) + 1
    g = np.mgrid[-R:R + 1, -R:R + 1].astype(float)
    flux, r50, bg = [], [], []
    for cy, cx in zip(ys, xs):
        a, b = int(round(cy)), int(round(cx))
        c = x[a - R:a + R + 1, b - R:b + R + 1]
        if c.shape != g[0].shape:
            flux.append(np.nan); r50.append(np.nan); bg.append(np.nan); continue
        r = np.hypot(g[0] - (cy - a), g[1] - (cx - b))
        ann = c[(r >= rin) & (r < rout)]
        f0 = float(np.median(ann))
        ins = r < ra
        v = (c[ins] - f0)
        F = float(v.sum())
        rr = r[ins]
        o = np.argsort(rr)
        cum = np.cumsum(v[o])
        if F > 0:
            j = np.searchsorted(cum, 0.5 * F)
            r50.append(float(rr[o][min(j, len(o) - 1)]))
        else:
            r50.append(np.nan)
        flux.append(F); bg.append(f0)
    return np.array(flux), np.array(r50), np.array(bg)


def fwhm_median(x: np.ndarray, ys, xs, n: int = 200) -> float:
    """FWHM médiane (moments du second ordre au-dessus du fond local, étoiles moyennes)."""
    vals = []
    for cy, cx in list(zip(ys, xs))[len(ys) // 4: len(ys) // 4 + n]:
        a, b = int(round(cy)), int(round(cx))
        c = x[a - 6:a + 7, b - 6:b + 7]
        if c.shape != (13, 13):
            continue
        f0 = np.median(np.r_[c[0], c[-1], c[:, 0], c[:, -1]])
        c = (c - f0).clip(0)
        g = np.mgrid[-6:7, -6:7]
        t = c.sum()
        if t <= 0:
            continue
        my, mx = (c * g[0]).sum() / t, (c * g[1]).sum() / t
        v = (c * ((g[0] - my) ** 2 + (g[1] - mx) ** 2)).sum() / t / 2
        if v > 0:
            vals.append(2.3548 * np.sqrt(v))
    return float(np.median(vals)) if vals else 3.0
