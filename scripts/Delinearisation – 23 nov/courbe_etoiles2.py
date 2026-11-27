# ======================================================================
# courbe_etoiles2.py
# Role     : Transfer curve estimated from the stars of the stretched image, version 2 (improvement 1).
# Step     : 3. De-linearisation of web images (23 Nov) (see scripts/LISEZMOI.md)
# Inputs   : stretched image
# Outputs  : image made linear
# Example  : python courbe_etoiles2.py [arguments]  (see the docstring; paths via the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Transfer curve estimated from the stars of the stretched image, version 2 (improvement 1).

Differences from version 1 (courbe_etoiles.py, trial of 23 Nov 3 pm): NON-parametric common profile P (one value
per ring, shared by all stars) instead of a Moffat, whose departure from the true wings bent the
curve at the top; log g' parameterised directly at the knots (wide bounds); start = the best of several MTF curves.

Idea (radiometric calibration from several “exposures” of the same scene: Debevec and Malik, SIGGRAPH 1997, “Recovering
High Dynamic Range Radiance Maps from Photographs”; Mitsunaga and Nayar, CVPR 1999, “Radiometric Self Calibration”):
in a linear image, all stars share the same profile (the point spread function, here a Moffat, Moffat
1969, A&A 3, 455), up to their flux, sitting on their local background. Stars of different brightness play the role of the exposures.
The inverse g of the stretch is thus the monotonic curve that yields, for each star i and each radius r,
    g(y_i(r)) = B_i + F_i * P(r ; alpha, beta)
with the same P for all of them. The background B_i (different from one star to the next, nebula) removes the known power
ambiguity of this problem (Grossberg and Nayar, PAMI 2003, “What can be known about the radiometric response from images”),
and the faint stars, almost in the linear part of the curve around their background, fix the shape of P.

Profile of each star: median of the pixels per ring. A median commutes with any monotonic curve, so the median
of a ring of the stretched image is exactly the image under the stretch of the linear median of the ring (up to noise),
and neither spikes nor a neighbouring star move it.

Curve: g(t) = integral of exp(s(t)), s = log g' piecewise linear on knots placed at the quantiles of the
data (constant beyond); g is monotonic by construction. B_i, F_i and P: alternating least squares at each
evaluation (variable projection, Golub and Pereyra 1973); s: scipy least_squares (soft-L1 loss), smoothing by
second differences."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

import float_unstretch_copie as fu
from commun import bruit_laplacien, mad_sigma
from etoiles_outils import detecter

GRILLE = np.linspace(0.0, 1.0, 4097)
BORDS = np.array([0, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 4.0, 5.0, 6.0, 7.5, 9.0, 11.0])
ANNEAU = (12.0, 16.0)


def profils(Y, ys, xs):
    """(N, nb+1) médianes par anneau (le dernier = anneau de fond), rayons médians, effectifs ; pixels écrêtés exclus."""
    R = int(ANNEAU[1]) + 1
    g = np.mgrid[-R:R + 1, -R:R + 1].astype(float)
    nb = len(BORDS) - 1
    Pm, Rm, Nm = (np.full((len(ys), nb + 1), np.nan) for _ in range(3))
    for i, (cy, cx) in enumerate(zip(ys, xs)):
        a, b = int(round(cy)), int(round(cx))
        c = Y[a - R:a + R + 1, b - R:b + R + 1]
        if c.shape != g[0].shape:
            continue
        r = np.hypot(g[0] - (cy - a), g[1] - (cx - b))
        for k in range(nb + 1):
            lo, hi = (BORDS[k], BORDS[k + 1]) if k < nb else ANNEAU
            m = (r >= lo) & (r < hi)
            v, rv = c[m], r[m]
            bon = (v < 0.995) & (v > 0.0005)
            if bon.sum() < max(1, 0.8 * m.sum()):
                continue
            Pm[i, k] = np.median(v[bon]); Rm[i, k] = np.median(rv[bon]); Nm[i, k] = bon.sum()
    return Pm, Rm, Nm


class Ajustement:
    def __init__(self, Yp, Np, sigma, noeuds, s_dep, lam=1.0):
        ok = np.isfinite(Yp)
        self.Y, self.ok = np.where(ok, Yp, 0.5), ok
        self.sig = np.where(ok, np.sqrt((1.25 * sigma) ** 2 / np.maximum(Np, 1) + 0.002 ** 2), 1.0)
        self.noeuds, self.lam, self.s_dep = noeuds, lam, s_dep
        self.P = None

    def courbe(self, s_n):
        s = np.interp(GRILLE, self.noeuds, s_n)
        e = np.exp(s - s.max())
        g = np.concatenate([[0.0], np.cumsum(0.5 * (e[1:] + e[:-1]) * np.diff(GRILLE))])
        return g / g[-1], e / g[-1]

    def interne(self, G, w):
        """B_i, F_i, P_k : G_ik = B_i + F_i P_k, P_0 = 1 ; moindres carrés pondérés alternés."""
        P = self.P if self.P is not None else np.exp(-0.5 * (np.arange(G.shape[1]) / 2.0) ** 2)
        for _ in range(6):
            S0, S1, S2 = w.sum(1), (w * P).sum(1), (w * P * P).sum(1)
            T0, T1 = (w * G).sum(1), (w * G * P).sum(1)
            det = S0 * S2 - S1 ** 2
            F = np.where(det > 0, (S0 * T1 - S1 * T0) / np.where(det > 0, det, 1), 0.0)
            F = np.maximum(F, 0.0)
            B = (T0 - F * S1) / np.maximum(S0, 1e-30)
            num = (w * F[:, None] * (G - B[:, None])).sum(0)
            den = (w * F[:, None] ** 2).sum(0)
            P = np.where(den > 0, num / np.where(den > 0, den, 1), P)
            P = P / P[0] if P[0] > 0 else P
        return B, F, P

    def residus(self, s_n, rendre=False):
        g, gp = self.courbe(s_n)
        G = np.interp(self.Y, GRILLE, g)
        Gp = np.interp(self.Y, GRILLE, gp)
        w = np.where(self.ok, 1.0 / (Gp * self.sig) ** 2, 0.0)
        B, F, P = self.interne(G, w)
        self.P = P
        res = np.where(self.ok, (G - B[:, None] - F[:, None] * P[None]) / (Gp * self.sig), 0.0)
        if rendre:
            return g, B, F, P
        lis = self.lam * np.diff(s_n, 2)
        ancre = 0.01 * np.array([np.mean(s_n - self.s_dep)])       # niveau de s : sans effet (échelle), fixé
        return np.concatenate([res.ravel(), lis, ancre])


def s_mtf(m, noeuds):
    g0 = fu.mtf_inverse(GRILLE, m, 0.0, 1.0)
    gp = np.maximum(np.gradient(g0, GRILLE), 1e-300)
    return np.interp(noeuds, GRILLE, np.log(gp))


def estimer(y: np.ndarray, max_etoiles: int = 600, canal: int | None = None):
    """y (3, H, W) étiré 0-1 -> (g sur GRILLE, info). Étoiles trouvées sur la luminance ; profils et courbe sur la
    luminance (canal None) ou sur un canal (amélioration 2 : étirement délié). Repli : une MTF si les étoiles manquent."""
    Yd = y.mean(0)
    sd = max(mad_sigma(bruit_laplacien(Yd)[::2, ::2]), 1.0 / 255 / np.sqrt(12))
    ys, xs = detecter(Yd, sd, k=10.0, isol=14, bord=20, max_n=5000)
    Y = Yd if canal is None else y[canal]
    sig = max(mad_sigma(bruit_laplacien(Y)[::2, ::2]), 1.0 / 255 / np.sqrt(12))
    info = {"etoiles_vues": int(len(ys))}
    try:
        _l, m_fl, _u = fu.estimate_channel(Y)
    except ValueError:
        m_fl = None
    if len(ys) > max_etoiles:
        idx = np.linspace(0, len(ys) - 1, max_etoiles).astype(int)
        ys, xs = ys[idx], xs[idx]
    Yp, _Rp, Np = profils(Y, ys, xs)
    bon = np.isfinite(Yp[:, -1]) & (np.isfinite(Yp[:, :-1]).sum(1) >= 6) & np.isfinite(Yp[:, 0])
    Yp, Np = Yp[bon], Np[bon]
    info["etoiles_utilisees"] = int(bon.sum())
    if bon.sum() < 30:
        m = m_fl if m_fl is not None else 0.05
        info["repli"] = "trop peu d'étoiles : MTF"
        g = fu.mtf_inverse(GRILLE, m, 0.0, 1.0)
        return g / g[-1], info
    v = Yp[np.isfinite(Yp)]
    noeuds = np.unique(np.quantile(v, np.linspace(0.01, 0.99, 12)))
    # départ : la meilleure de plusieurs MTF (et celle de FLOAT quand elle existe)
    cands = [m_pour_sky(float(np.median(Y)), x0) for x0 in (1e-4, 5e-4, 2e-3, 1e-2, 5e-2, 0.2)]
    if m_fl is not None:
        cands.append(m_fl)
    best = None
    for m in cands:
        if not 0 < m < 1:
            continue
        s0 = s_mtf(m, noeuds)
        aj = Ajustement(Yp, Np, sig, noeuds, s0)
        c = float(np.mean(aj.residus(s0)[: aj.ok.size] ** 2))
        if best is None or c < best[0]:
            best = (c, m, s0)
    c0, m0, s0 = best
    aj = Ajustement(Yp, Np, sig, noeuds, s0)
    sol = least_squares(aj.residus, s0, bounds=(s0 - 12, s0 + 12), loss="soft_l1", f_scale=2.0, max_nfev=400)
    g, B, F, P = aj.residus(sol.x, rendre=True)
    info.update(m_depart=round(float(m0), 6), m_float=None if m_fl is None else round(float(m_fl), 6),
                cout_depart=round(c0, 3), cout_fin=round(float(np.mean(sol.fun[: aj.ok.size] ** 2)), 3),
                nfev=int(sol.nfev), noeuds=np.round(noeuds, 4).tolist(), P=np.round(P, 4).tolist())
    return g, info


def m_pour_sky(s, x0):
    """m de la MTF qui envoie le niveau linéaire x0 (au-dessus du noir) sur le fond étiré s."""
    return x0 * (s - 1.0) / (2.0 * s * x0 - s - x0)


def appliquer(y, g):
    return np.interp(y, GRILLE, g)
