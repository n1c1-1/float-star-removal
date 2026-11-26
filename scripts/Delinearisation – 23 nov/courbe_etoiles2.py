# ======================================================================
# courbe_etoiles2.py
# Rôle     : Courbe de transfert estimée depuis les étoiles de l'image étirée, version 2 (amélioration 1).
# Étape    : 3. Délinéarisation des images web (23 nov) (voir scripts/LISEZMOI.md)
# Entrées  : image étirée
# Sorties  : image rendue linéaire
# Exemple  : python courbe_etoiles2.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Courbe de transfert estimée depuis les étoiles de l'image étirée, version 2 (amélioration 1).

Différences avec la version 1 (courbe_etoiles.py, essai du 23 nov 15 h) : profil commun P NON paramétrique (une valeur
par anneau, partagée par toutes les étoiles) au lieu d'une Moffat, dont l'écart aux vraies ailes faisait tordre la
courbe en haut ; log g' paramétré directement aux nœuds (bornes larges) ; départ = la meilleure de plusieurs courbes MTF.

Idée (étalonnage radiométrique par plusieurs « poses » d'une même scène : Debevec et Malik, SIGGRAPH 1997, « Recovering
High Dynamic Range Radiance Maps from Photographs » ; Mitsunaga et Nayar, CVPR 1999, « Radiometric Self Calibration ») :
dans une image linéaire, toutes les étoiles ont le même profil (la fonction d'étalement du point, ici une Moffat, Moffat
1969, A&A 3, 455), à leur flux près, posé sur leur fond local. Les étoiles d'éclats différents jouent le rôle des poses.
L'inverse g de l'étirement est donc la courbe monotone qui rend, pour chaque étoile i et chaque rayon r,
    g(y_i(r)) = B_i + F_i * P(r ; alpha, beta)
avec un même P pour toutes. Le fond B_i (différent d'une étoile à l'autre, nébuleuse) lève l'ambiguïté de puissance
connue de ce problème (Grossberg et Nayar, PAMI 2003, « What can be known about the radiometric response from images »),
et les étoiles faibles, presque dans la partie linéaire de la courbe autour de leur fond, fixent la forme de P.

Profil de chaque étoile : médiane des pixels par anneau. Une médiane commute avec toute courbe monotone, donc la médiane
d'un anneau de l'image étirée est exactement l'image par l'étirement de la médiane linéaire de l'anneau (au bruit près),
et les aigrettes ou une voisine ne la déplacent pas.

Courbe : g(t) = intégrale de exp(s(t)), s = log g' linéaire par morceaux sur des nœuds placés aux quantiles des
données (constante au-delà) ; g est monotone par construction. B_i, F_i et P : moindres carrés alternés à chaque
évaluation (projection des variables, Golub et Pereyra 1973) ; s : scipy least_squares (perte soft-L1), lissage par
différences secondes."""
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
