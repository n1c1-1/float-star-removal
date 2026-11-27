# ======================================================================
# courbe_etoiles.py
# Role     : Transfer curve estimated from the stars of the stretched image (improvement 1).
# Step     : 3. De-linearisation of web images (23 Nov) (see scripts/LISEZMOI.md)
# Inputs   : stretched image
# Outputs  : image made linear
# Example  : python courbe_etoiles.py [arguments]  (see the docstring; paths via the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Transfer curve estimated from the stars of the stretched image (improvement 1).

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

Curve: g(t) = integral of exp(s(t)), s = log g' = that of the FLOAT function (starting point) + a piecewise-linear
correction on knots placed at the quantiles of the data; so g is monotonic by construction. For B_i,
F_i: exact weighted least squares at each evaluation (variable projection, Golub and Pereyra 1973); for the
correction, alpha and beta: scipy least_squares (soft-L1 loss), smoothing by second differences."""
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
    def __init__(self, Yp, Rp, Np, sigma, s0, noeuds, lam=3.0):
        ok = np.isfinite(Yp)
        self.Y, self.R, self.ok = np.where(ok, Yp, 0.5), np.where(ok, Rp, 1.0), ok
        self.sig = np.where(ok, np.sqrt((1.25 * sigma) ** 2 / np.maximum(Np, 1) + 0.003 ** 2), 1.0)
        self.s0, self.noeuds, self.lam = s0, noeuds, lam

    def courbe(self, d):
        s = self.s0 + np.interp(GRILLE, self.noeuds, d)
        e = np.exp(s - s.max())
        g = np.concatenate([[0.0], np.cumsum(0.5 * (e[1:] + e[:-1]) * np.diff(GRILLE))])
        return g / g[-1], e / g[-1]

    def residus(self, p, rendre=False):
        K = len(self.noeuds)
        d, la, lb = p[:K], p[K], p[K + 1]
        al, be = np.exp(la), np.exp(lb)
        g, gp = self.courbe(d)
        G = np.interp(self.Y, GRILLE, g)
        Gp = np.interp(self.Y, GRILLE, gp)
        P = (1.0 + (self.R / al) ** 2) ** (-be)
        w = np.where(self.ok, 1.0 / (Gp * self.sig) ** 2, 0.0)
        # moindres carrés pondérés exacts de G = B + F P, étoile par étoile
        S0, S1, S2 = w.sum(1), (w * P).sum(1), (w * P * P).sum(1)
        T0, T1 = (w * G).sum(1), (w * G * P).sum(1)
        det = S0 * S2 - S1 ** 2
        F = np.where(det > 0, (S0 * T1 - S1 * T0) / np.where(det > 0, det, 1), 0.0)
        F = np.maximum(F, 0.0)
        B = (T0 - F * S1) / np.maximum(S0, 1e-30)
        res = np.where(self.ok, (G - B[:, None] - F[:, None] * P) / (Gp * self.sig), 0.0)
        lis = self.lam * np.diff(d, 2)
        prior = 0.05 * d
        if rendre:
            return g, B, F, al, be
        return np.concatenate([res.ravel(), lis, prior])


def estimer(y: np.ndarray, max_etoiles: int = 600):
    """y (3, H, W) étiré 0-1 -> (g sur GRILLE, info). Repli : la courbe FLOAT si les étoiles manquent."""
    Y = y.mean(0)
    try:
        _lin, m0, _u = fu.estimate_channel(Y)
    except ValueError:
        from methodes import m_force
        m0 = m_force(Y)
    t = np.clip(GRILLE, 0, 1)
    g0 = fu.mtf_inverse(t, m0, 0.0, 1.0)
    s0 = np.log(np.maximum(np.gradient(g0, GRILLE), 1e-12))
    sig = mad_sigma(bruit_laplacien(Y)[::2, ::2])
    sig = max(sig, 1.0 / 255 / np.sqrt(12))
    ys, xs = detecter(Y, sig, k=10.0, isol=14, bord=20, max_n=5000)
    info = {"m_float": round(float(m0), 5), "etoiles_vues": int(len(ys))}
    if len(ys) < 30:
        info["repli"] = "trop peu d'étoiles"
        return g0 / g0[-1], info
    # garder des étoiles de tous les éclats : tirage régulier dans l'ordre des pics
    if len(ys) > max_etoiles:
        idx = np.linspace(0, len(ys) - 1, max_etoiles).astype(int)
        ys, xs = ys[idx], xs[idx]
    Yp, Rp, Np = profils(Y, ys, xs)
    bon = np.isfinite(Yp[:, -1]) & (np.isfinite(Yp[:, :-1]).sum(1) >= 6)
    Yp, Rp, Np = Yp[bon], Rp[bon], Np[bon]
    info["etoiles_utilisees"] = int(bon.sum())
    if bon.sum() < 30:
        info["repli"] = "trop peu de profils"
        return g0 / g0[-1], info
    v = Yp[np.isfinite(Yp)]
    # nœuds dans la plage des données seulement : au-delà, la correction reste celle du dernier nœud (pas d'extrapolation
    # libre, qui faisait exploser la courbe au-dessus de 0,97 au premier essai)
    noeuds = np.unique(np.quantile(v, np.linspace(0.02, 0.98, 10)))
    aj = Ajustement(Yp, Rp, Np, sig, s0, noeuds)
    p0 = np.r_[np.zeros(len(noeuds)), np.log(2.0), np.log(2.5)]
    lo = np.r_[np.full(len(noeuds), -6.0), np.log(0.3), np.log(0.8)]
    hi = np.r_[np.full(len(noeuds), 6.0), np.log(30.0), np.log(15.0)]
    sol = least_squares(aj.residus, p0, bounds=(lo, hi), loss="soft_l1", f_scale=2.0, x_scale=1.0, max_nfev=300)
    g, B, F, al, be = aj.residus(sol.x, rendre=True)
    r0 = aj.residus(p0)[: aj.ok.size]
    r1 = sol.fun[: aj.ok.size]
    info.update(alpha=round(float(al), 3), beta=round(float(be), 3), cout_depart=round(float(np.mean(r0 ** 2)), 3),
                cout_fin=round(float(np.mean(r1 ** 2)), 3), nfev=int(sol.nfev), noeuds=np.round(noeuds, 4).tolist(),
                correction=np.round(sol.x[: len(noeuds)], 3).tolist())
    return g, info


def appliquer(y, g):
    return np.interp(y, GRILLE, g)
