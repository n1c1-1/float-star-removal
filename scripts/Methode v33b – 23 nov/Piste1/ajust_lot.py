# ======================================================================
# ajust_lot.py
# Role     : Batch fitting (torch, CUDA or CPU) of the medium stars of track 1 (v5 GPU, 21 Nov).
# Step     : 2. Classic Gaia-guided subtraction chain (v33b) (see scripts/LISEZMOI.md)
# Inputs   : stacked image (noise units), astrometry, offline deep Gaia extract
# Outputs  : star-free image, class maps
# Example  : python ajust_lot.py [arguments]  (see the docstring; paths through the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Batch fitting (torch, CUDA or CPU) of the medium stars of track 1 (v5 GPU, 21 Nov).

Same computation as ``fit_star`` + the end of ``refit`` of retrait_modele.py (v5), but for a batch of independent stars
(a « wave »: none reads what another of the batch writes), all solved together:
- PSF: cubic B-spline (coefficients of scipy.ndimage.spline_filter, mirror edge, off-grid = 0), identical to
  map_coordinates(order=3, mode="constant");
- weighted least squares (masked rows) by normal equations; exact medians (mean of the two middle ones);
- loops with a clean stop for each star (Gauss-Newton, width, ellipse, position per colour) in common steps with
  a mask of the stars still active: each star follows exactly the sequence of steps of the original code.
All in float64 (like numpy) except the rendered model and the « no hole » guard, in float32 like the original.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from scipy import ndimage as ndi


class CF:
    """v30 : coefficients de PSF par étoile quand le modèle dépend de l'éclat G (modèle de l'image, v7) :
    PSF(étoile) = S + (1 - t) rho[j] + t rho[j + 1] ; la spline est linéaire, donc trois lectures de tables partagées
    (rien n'est recopié par étoile). Se comporte comme un tenseur (B, C, .) pour Lot.sub et le découpage par couleur."""

    def __init__(self, base, rho, j, t, cs=slice(None)):
        self.base, self.rho, self.j, self.t, self.cs = base, rho, j, t, cs
        self.shape = (j.shape[0],)

    def __getitem__(self, key):
        if isinstance(key, tuple):                  # [:, c:c+1] : une couleur
            return CF(self.base, self.rho, self.j, self.t, key[1])
        return CF(self.base, self.rho, self.j[key], self.t[key], self.cs)

    def gather(self, idx):
        B = self.j.shape[0]
        Wp2 = self.base.shape[-1]
        bs = self.base[self.cs][None].expand(B, -1, -1)
        rh = self.rho[self.cs].reshape(self.rho[self.cs].shape[0], -1)[None].expand(B, -1, -1)
        j = (self.j * Wp2)[:, None, None]
        t = self.t[:, None, None]
        return (torch.gather(bs, 2, idx) + (1 - t) * torch.gather(rh, 2, idx + j)
                + t * torch.gather(rh, 2, idx + j + Wp2))


def _bs(t):
    t2 = t * t
    t3 = t2 * t
    return torch.stack([(1 - t) ** 3 / 6, (3 * t3 - 6 * t2 + 4) / 6, (-3 * t3 + 3 * t2 + 3 * t + 1) / 6, t3 / 6], -1)


def mmedian(v, m):
    """Médiane de v sur le masque m (dernier axe), comme np.median (moyenne des deux du milieu) ; nan si vide."""
    P = v.shape[-1]
    s, _ = torch.sort(torch.where(m, v, torch.full_like(v, float("inf"))), -1)
    n = m.sum(-1)
    lo = ((n - 1).clamp(min=0) // 2).clamp(max=P - 1)
    hi = (n // 2).clamp(max=P - 1)
    a = s.gather(-1, lo[..., None])[..., 0]
    b = s.gather(-1, hi[..., None])[..., 0]
    return torch.where(n > 0, (a + b) / 2, torch.full_like(a, float("nan")))


EXACT = {"pinv": False}          # True : systèmes singuliers résolus par pseudo-inverse (norme minimale, comme lstsq)


def wls(A, y, w):
    """Moindres carrés sur les lignes w (équations normales) ; rend (solution, singulier ?). Sans synchronisation : un
    système singulier donne 0 (l'appelant note s'il servait, et refait alors le lot en mode exact) ; en mode exact,
    solution de norme minimale (pseudo-inverse), comme np.linalg.lstsq."""
    Aw = A * w[..., None].to(A.dtype)
    # v30 : produits de petites matrices (k = 3 ou 4) par somme directe : le noyau de produit matriciel de torch en
    # float64 (tuiles 32x32) prenait 42 % du temps GPU pour ces matrices minuscules
    M = (Aw.unsqueeze(-1) * A.unsqueeze(-2)).sum(-3)
    r = (Aw * y[..., None]).sum(-2)[..., None]
    sol, info = torch.linalg.solve_ex(M, r)
    bad = (info != 0) | ~torch.isfinite(sol).all(-1).all(-1)
    if EXACT["pinv"]:
        if bool(bad.any()):
            sol = sol.clone()
            sol[bad] = __import__('operator').matmul(torch.linalg.pinv(M[bad]), r[bad])
        return sol[..., 0], torch.zeros_like(bad)
    return torch.where(bad[..., None, None], torch.zeros_like(sol), sol)[..., 0], bad


class Lot:
    def __init__(self, device="cuda"):
        self.dev = torch.device(device)
        self.dt = torch.float64
        self.coef = None
        self.fast32 = os.environ.get("PISTE1_RENDU_SIMPLE") == "1"     # 1 : rendu du modele en simple precision (plus rapide, ecart jusqu'a 0,13 bruit)

    # ------------------------------------------------------------------ PSF
    def set_stamps(self, stamps, os_, R, rho=None, gc=None):
        """stamps[c][g] (n, n) -> table des coefficients B-spline (C, G, (n+4)^2), bord en miroir.
        v30 : rho (C, nb, n, n) et gc (nb,) : correction par tranche d'éclat du modèle de l'image (G = 1 région)."""
        C, G = len(stamps), len(stamps[0])
        n = stamps[0][0].shape[0]
        tab = np.zeros((C * G, 1, n, n))
        for c in range(C):
            for g in range(G):
                tab[c * G + g, 0] = ndi.spline_filter(np.asarray(stamps[c][g], np.float64), 3, output=np.float64, mode="constant")
        t = torch.as_tensor(tab, device=self.dev, dtype=self.dt)
        t = torch.nn.functional.pad(t, (2, 2, 2, 2), mode="reflect")
        self.coef = t.reshape(C, G, -1)
        self.rho = self.gc = None
        if rho is not None:
            nb = rho.shape[1]
            tr = np.zeros((C * nb, 1, n, n))
            for c in range(C):
                for b in range(nb):
                    tr[c * nb + b, 0] = ndi.spline_filter(np.asarray(rho[c][b], np.float64), 3, output=np.float64, mode="constant")
            tr = torch.nn.functional.pad(torch.as_tensor(tr, device=self.dev, dtype=self.dt), (2, 2, 2, 2), mode="reflect")
            self.rho = tr.reshape(C, nb, -1)
            self.gc = np.asarray(gc, float)
        self.coef32 = self.coef.float()
        self.rho32 = self.rho.float() if self.rho is not None else None
        self.n, self.Wp, self.os, self.R, self.G, self.C = n, n + 4, os_, R, G, C
        a4 = torch.arange(4, device=self.dev)
        self.offs = (a4[:, None] * self.Wp + a4[None]).reshape(16)

    def coefs_G(self, gG):
        """v30 : interp_rho de modele_psf_image (mêmes j et t) pour l'éclat G de chaque étoile du lot."""
        gc = self.gc
        j = np.clip(np.searchsorted(gc, np.asarray(gG, float)) - 1, 0, len(gc) - 2)
        t = np.clip((np.asarray(gG, float) - gc[j]) / np.maximum(gc[j + 1] - gc[j], 1e-6), 0, 1)
        return CF(self.coef[:, 0], self.rho, torch.as_tensor(j, device=self.dev),
                  torch.as_tensor(t, device=self.dev, dtype=self.dt))

    def coefs_32(self):
        """Mêmes coefficients en simple précision (rendu du modèle, déjà rendu en float32 par v5)."""
        if self.cf32 is None:
            if isinstance(self.cf, CF):
                self.cf32 = CF(self.coef32[:, 0], self.rho32, self.cf.j, self.cf.t.float())
            else:
                self.cf32 = self.coef32[:, 0][None].expand(self.B, -1, -1) if self.G == 1 else self.cf.float()
        return self.cf32

    def coefs_for(self, wts):
        """wts (B, G) poids bilinéaires des régions -> coefficients (B, C, Wp^2) (la spline est linéaire)."""
        if self.G == 1:
            return self.coef[:, 0][None].expand(wts.shape[0], -1, -1)
        return torch.einsum("bg,cgk->bck", wts, self.coef)

    def eval(self, cf, dx, dy, sc, a, b):
        """EPSF.eval (forme elliptique ; a = b = 0 donne exactement la forme à échelle seule) : cf (B, C, Wp^2),
        dx, dy (B, C, P), sc, a, b (B,) -> (B, C, P)."""
        B, C, P = dx.shape
        sc_, a_, b_ = sc[:, None, None], a[:, None, None], b[:, None, None]
        u = ((1 - a_) * dx - b_ * dy) / sc_
        w = (-b_ * dx + (1 + a_) * dy) / sc_
        cn = (self.n - 1) / 2
        row = w * self.os + cn
        col = u * self.os + cn
        n = self.n
        inside = (row >= 0) & (row <= n - 1) & (col >= 0) & (col <= n - 1)
        r0 = torch.floor(row)
        c0 = torch.floor(col)
        wr = _bs(row - r0)
        wc = _bs(col - c0)
        r0 = r0.clamp(-1, n).long() + 1
        c0 = c0.clamp(-1, n).long() + 1
        base = r0 * self.Wp + c0
        idx = (base[..., None] + self.offs).clamp(0, self.Wp * self.Wp - 1).reshape(B, C, P * 16)
        vals = (cf.gather(idx) if isinstance(cf, CF) else torch.gather(cf, 2, idx)).reshape(B, C, P, 4, 4)
        v = ((wr[..., :, None] * wc[..., None, :]) * vals).sum((-1, -2))
        v = torch.where(inside, v, torch.zeros_like(v))
        return v * (1 - a_ * a_ - b_ * b_) / (sc_ * sc_)

    # ------------------------------------------------------------------ lot
    def load(self, d):
        """d : dictionnaire numpy du lot (voir retrait_rapide.passe_lot)."""
        T = lambda v, dt=None: torch.as_tensor(np.ascontiguousarray(v), device=self.dev, dtype=dt)
        S = d["S"]
        self.S = S
        jj = torch.arange(S, device=self.dev, dtype=self.dt)
        gx0 = T(d["gx0"], self.dt)
        gy0 = T(d["gy0"], self.dt)
        B = gx0.shape[0]
        self.B = B
        self.Xg = (gx0[:, None, None] + jj[None, None, :]).expand(B, S, S).reshape(B, 1, S * S)
        self.Yg = (gy0[:, None, None] + jj[None, :, None]).expand(B, S, S).reshape(B, 1, S * S)
        self.win32 = T(d["win32"]).reshape(B, self.C, S * S)
        self.valid = T(d["valid"]).reshape(B, 1, S * S)
        self.sat = T(d["sat"]).reshape(B, self.C, S * S)
        self.box = T(d["modbox"]).reshape(B, 1, S * S)
        self.rf = T(d["rf"], self.dt)
        self.aw = float(d["aw"])
        self.sk32 = T(d["sk"], torch.float32)                      # bruit local (float32 comme sig)
        self.sk = self.sk32.to(self.dt)
        self.s2 = (self.sk32.clamp(min=1e-6) ** 2).to(self.dt)     # max(sig, 1e-6) ** 2 en float32 comme numpy
        self.cf = self.coefs_G(d["gG"]) if self.rho is not None else self.coefs_for(T(d["wts"], self.dt))
        self.cf32 = None                                                   # tables simple précision (rendu du modèle)
        # fenêtre d'ajustement seule (les pixels hors de la boîte d'ajustement ne servent pas à fit_star) : sous-carré
        # centré de demi-côté hf (la fenêtre du modèle, plus grande, ne sert qu'au rendu)
        h = (S - 1) // 2
        hf = min(h, int(d["hf"]))
        o, Sf = h - hf, 2 * hf + 1
        cut = lambda a_: a_[..., o:o + Sf, o:o + Sf]
        fb = T(cut(d["fitbox"])).reshape(B, 1, Sf * Sf)
        self.V = dict(Xg=(gx0[:, None, None] + o + jj[None, None, :Sf]).expand(B, Sf, Sf).reshape(B, 1, Sf * Sf),
                      Yg=(gy0[:, None, None] + o + jj[None, :Sf, None]).expand(B, Sf, Sf).reshape(B, 1, Sf * Sf),
                      Rv=T(cut(d["win32"])).reshape(B, self.C, Sf * Sf).to(self.dt),
                      vf=T(cut(d["valid"])).reshape(B, 1, Sf * Sf) & fb,
                      sat=T(cut(d["sat"])).reshape(B, self.C, Sf * Sf),
                      rf=self.rf, s2=self.s2, cf=self.cf, sk=self.sk)

    def sub(self, idx):
        return {k: (v[idx] if v.shape[0] == self.B else v) for k, v in self.V.items()}

    def solve(self, V, x, y, sc, a, b, co):
        dx = V["Xg"] - x[:, None, None]
        dy = V["Yg"] - y[:, None, None]
        r = torch.hypot(dx, dy)
        rf = V["rf"][:, None, None]
        inn = (r <= rf) & V["vf"]
        ann = (r > rf) & (r <= rf + self.aw) & V["vf"]
        ok = inn & ~V["sat"]
        okA = ann & ~V["sat"]
        M = self.eval(V["cf"], dx - co[:, :, 0:1], dy - co[:, :, 1:2], sc, a, b)
        nok = ok.sum(-1)
        nA = okA.sum(-1)
        use = nok >= 6
        rob = use & (nA >= 12)
        Rv = V["Rv"]
        one = torch.ones_like(dx)
        Ab = torch.stack([one, dx / rf, dy / rf], -1)               # (B, 1, P, 3)
        okf = ok.to(self.dt)
        f = torch.zeros(M.shape[:2], device=self.dev, dtype=self.dt)
        keep = okA.clone()
        for _ in range(3):
            yb = Rv - f[..., None] * M
            bsol, bad = wls(Ab.expand(-1, self.C, -1, -1), yb, keep)
            self._bad = self._bad | (bad & rob).any()
            bg = (Ab * bsol[..., None, :]).sum(-1)
            rb = yb - bg
            sd = 1.4826 * mmedian(rb.abs(), keep) + 1e-9
            keep = okA & (rb.abs() < 3 * sd[..., None])
            keep = torch.where((keep.sum(-1) < 8)[..., None], okA, keep)
            f = (M * (Rv - bg) * okf).sum(-1) / ((M * M) * okf).sum(-1).clamp(min=1e-12)
        res_r = Rv - bg - f[..., None] * M
        A4 = torch.stack([M, one.expand_as(M), (dx / rf).expand_as(M), (dy / rf).expand_as(M)], -1)
        sol, bad = wls(A4, Rv, ok)
        self._bad = self._bad | (bad & use & ~rob).any()
        res_p = Rv - (A4 * sol[..., None, :]).sum(-1)
        zero = torch.zeros_like(f)
        F = torch.where(rob, f, torch.where(use, sol[..., 0], zero))
        res = torch.where(rob[..., None], res_r, res_p)
        chi_c = ((res * res) * okf).sum(-1) / V["s2"] / nok.clamp(min=1).to(self.dt)
        chi = torch.where(use, chi_c, zero).sum(-1)
        return F, chi, M, res, ok, use

    def _deriv(self, V, x, y, sc, a, b, co, F, c=None):
        """Dérivées de position du modèle (différences centrées à 0,05 px) ; toutes couleurs ou la couleur c seule."""
        cf = V["cf"] if c is None else V["cf"][:, c:c + 1]
        co_ = co if c is None else co[:, c:c + 1]
        F_ = F if c is None else F[:, c:c + 1]
        dx = V["Xg"] - x[:, None, None] - co_[:, :, 0:1]
        dy = V["Yg"] - y[:, None, None] - co_[:, :, 1:2]
        mx = (self.eval(cf, dx - 0.05, dy, sc, a, b) - self.eval(cf, dx + 0.05, dy, sc, a, b)) / 0.1 * F_[..., None]
        my = (self.eval(cf, dx, dy - 0.05, sc, a, b) - self.eval(cf, dx, dy + 0.05, sc, a, b)) / 0.1 * F_[..., None]
        return mx, my

    def fit(self, x, y, s0, xcat, ycat, co0, fixp, freew, ellip, colpos, P):
        """Rend (F, x, y, sc, a, b, ell, chi, co) : fit_star de v5 pour chaque étoile du lot. Les pas itératifs
        (Gauss-Newton, ellipse, position par couleur) ne calculent que les étoiles encore actives."""
        dev, dt = self.dev, self.dt
        B, C = self.B, self.C
        I2 = torch.eye(2, device=dev, dtype=dt) * 1e-6
        x, y = x.clone(), y.clone()
        sc = s0.clone()
        a = torch.zeros(B, device=dev, dtype=dt)
        b = torch.zeros(B, device=dev, dtype=dt)
        co = co0.clone()
        cobase = co0.clone()
        pm = float(P["pos_margin"])
        self._bad = torch.zeros((), dtype=torch.bool, device=dev)
        F, chi, M, res, ok, use = self.solve(self.V, x, y, sc, a, b, co)
        F, chi, M, res, ok, use = F.clone(), chi.clone(), M.clone(), res.clone(), ok.clone(), use.clone()

        def solve_i(idx, sc_=None, a_=None, b_=None):
            sc_ = sc[idx] if sc_ is None else sc_
            a_ = a[idx] if a_ is None else a_
            b_ = b[idx] if b_ is None else b_
            return self.solve(self.sub(idx), x[idx], y[idx], sc_, a_, b_, co[idx])

        def put(idx, F2, chi2, M2=None, res2=None, ok2=None, use2=None):
            F[idx] = F2; chi[idx] = chi2
            if M2 is not None:
                M[idx] = M2; res[idx] = res2; ok[idx] = ok2; use[idx] = use2

        def J2(mx, my, sel, rr, sg, dims):
            jx, jy = mx / sg * sel, my / sg * sel
            rr = rr / sg
            JTJ = torch.stack([torch.stack([(jx * jx).sum(dims), (jx * jy).sum(dims)], -1),
                               torch.stack([(jy * jx).sum(dims), (jy * jy).sum(dims)], -1)], -2)
            JTr = torch.stack([(jx * rr).sum(dims), (jy * rr).sum(dims)], -1)
            d, info = torch.linalg.solve_ex(JTJ + I2, JTr[..., None])
            # v33b : un décalage non fini (fenêtre sans donnée utilisable, bord de l'image) n'est pas une solution : position gardée
            return d[..., 0], (info == 0) & torch.isfinite(d[..., 0]).all(-1)

        # Gauss-Newton sur la position, commun aux couleurs
        idx = (~fixp).nonzero()[:, 0]
        for _ in range(3):
            if idx.numel() == 0:
                break
            V = self.sub(idx)
            mx, my = self._deriv(V, x[idx], y[idx], sc[idx], a[idx], b[idx], co[idx], F[idx])
            sel = ((use[idx] & (F[idx] > 0))[..., None] & ok[idx]).to(dt)
            d, good = J2(mx, my, sel, res[idx], V["sk"][..., None], (1, 2))
            d = d.clamp(-0.5, 0.5)
            nx = torch.minimum(torch.maximum(x[idx] + d[:, 0], xcat[idx] - pm), xcat[idx] + pm)
            ny = torch.minimum(torch.maximum(y[idx] + d[:, 1], ycat[idx] - pm), ycat[idx] + pm)
            conv = ((nx - x[idx]).abs() < 0.01) & ((ny - y[idx]).abs() < 0.01)
            still = good & ~conv
            idx, nx, ny = idx[still], nx[still], ny[still]
            if idx.numel() == 0:
                break
            x[idx] = nx; y[idx] = ny
            put(idx, *solve_i(idx))
        # largeur
        if bool(freew.any()):
            idx = freew.nonzero()[:, 0]
            bchi, bs, bF = chi[idx].clone(), sc[idx].clone(), F[idx].clone()
            wm = P["width_margin"]
            n_ = idx.numel()
            sv = np.linspace(1 - wm, 1 + wm, 5)
            # les 5 largeurs essayées en un seul calcul (5 copies du lot), comparées ensuite dans l'ordre de v5
            i5 = idx.repeat(5)
            s5 = torch.as_tensor(np.repeat(sv, n_), device=dev, dtype=dt)
            z5 = torch.zeros(5 * n_, device=dev, dtype=dt)
            F5, chi5, *_ = self.solve(self.sub(i5), x[i5], y[i5], s5, z5, z5, co[i5])
            for j_, s1 in enumerate(sv):
                s1t = torch.full((n_,), float(s1), device=dev, dtype=dt)
                F1, chi1 = F5[j_ * n_:(j_ + 1) * n_], chi5[j_ * n_:(j_ + 1) * n_]
                better = chi1 < bchi
                bchi = torch.where(better, chi1, bchi)
                bs = torch.where(better, s1t, bs)
                bF = torch.where(better[:, None], F1, bF)
            chi[idx] = bchi; sc[idx] = bs; F[idx] = bF
        # ellipse : recherche par coordonnées à pas décroissant
        ell = freew & ellip
        if bool(ell.any()):
            lim = float(P.get("ellipse_max", 0.15))
            for step in (0.06, 0.03, 0.015):
                impr = ell.clone()
                while bool(impr.any()):
                    rnd = torch.zeros_like(impr)
                    for da, db in ((step, 0), (-step, 0), (0, step), (0, -step)):
                        na = (a + da).clamp(-lim, lim)
                        nb = (b + db).clamp(-lim, lim)
                        cand = impr & ~((na == a) & (nb == b))
                        idx = cand.nonzero()[:, 0]
                        if idx.numel() == 0:
                            continue
                        F1, chi1, *_ = solve_i(idx, None, na[idx], nb[idx])
                        acc = chi1 < chi[idx] * 0.999
                        ia = idx[acc]
                        F[ia] = F1[acc]; chi[ia] = chi1[acc]; a[ia] = na[ia]; b[ia] = nb[ia]
                        rnd[ia] = True
                    impr = impr & rnd
        # position propre à chaque couleur
        cp = colpos & ~fixp
        if C == 3 and bool(cp.any()):
            idx = cp.nonzero()[:, 0]
            _, _, M2, res2, ok2, use2 = solve_i(idx)
            M[idx] = M2; res[idx] = res2; ok[idx] = ok2; use[idx] = use2
            snr_s = (float(P["colour_pos_snr"]) * self.sk32).to(dt)      # 20 x sig[c] en float32 comme numpy
            for c in range(C):
                mmax = torch.where(ok[:, c], M[:, c], torch.full_like(M[:, c], -float("inf"))).max(-1).values
                act = cp & use[:, c] & (F[:, c] > 0) & ~(F[:, c] * mmax <= snr_s[:, c])
                idx = act.nonzero()[:, 0]
                for _ in range(4):
                    if idx.numel() == 0:
                        break
                    V = self.sub(idx)
                    mx, my = self._deriv(V, x[idx], y[idx], sc[idx], a[idx], b[idx], co[idx], F[idx], c=c)
                    d, good = J2(mx[:, 0], my[:, 0], ok[idx, c].to(dt), res[idx, c], V["sk"][:, c, None], -1)
                    idx, d = idx[good], d[good]
                    d = d.clamp(-0.3, 0.3)
                    new = torch.minimum(torch.maximum(co[idx, c] + d, cobase[idx, c] - 0.3), cobase[idx, c] + 0.3)
                    small = (new - co[idx, c]).abs().max(-1).values < 0.005
                    co[idx, c] = new
                    idx = idx[~small]
                    if idx.numel() == 0:
                        break
                    put(idx, *solve_i(idx))
                    idx = idx[use[idx, c]]
            idx = cp.nonzero()[:, 0]
            F1, chi1, *_ = solve_i(idx)
            F[idx] = F1; chi[idx] = chi1
        return F, x, y, sc, a, b, ell, chi, co

    def model_guard(self, F, nx, ny, sc, a, b, co, Fexp, R1, flux_guard_hole=False, guard=True):
        """star_model (halo nul) sur la fenêtre du modèle, puis la garde « pas de trou » de refit (v5 : seule la
        remontée du flux d'un reste positif est active). Rend (mod float32 (B, C, P), F)."""
        dt = self.dt
        dx = self.Xg - nx[:, None, None]
        dy = self.Yg - ny[:, None, None]
        r = torch.hypot(dx, dy)
        inner = (r <= R1) & self.box
        if not guard and self.fast32:
            # v30 : rendu du modèle en simple précision (le résultat est de toute façon écrit en float32) ; la position
            # relative est calculée en double puis arrondie
            f32 = torch.float32
            m = self.eval(self.coefs_32(), (dx - co[:, :, 0:1]).to(f32), (dy - co[:, :, 1:2]).to(f32), sc.to(f32), a.to(f32), b.to(f32))
            m = torch.where(inner, m, torch.zeros_like(m))
            mod = F.to(f32)[..., None] * m
        else:
            m = self.eval(self.cf, dx - co[:, :, 0:1], dy - co[:, :, 1:2], sc, a, b)
            m = torch.where(inner, m, torch.zeros_like(m))
            mod = (F[..., None] * m).to(torch.float32)
        mod = torch.where((F != 0)[..., None], mod, torch.zeros_like(mod))
        if not guard:
            return mod, F
        rf = self.rf[:, None, None]
        annz = (r > rf) & (r <= rf + self.aw) & self.valid & self.box            # (B, 1, P)
        core = (mod > self.sk32[..., None]) & self.valid & ~self.sat & self.box   # (B, C, P)
        after = self.win32 - mod
        annzC = annz.expand_as(core)
        bgv = mmedian(after, annzC).to(dt)
        ncore = core.sum(-1)
        coref = core.to(torch.float32)
        mean_after = ((after * coref).sum(-1) / ncore.clamp(min=1)).to(dt)
        mc = ((mod * coref).sum(-1) / ncore.clamp(min=1)).to(dt)
        deficit = bgv - mean_after
        cond = (ncore >= 3) & (annz.sum(-1) >= 8) & (F > 0)
        half = (0.5 * self.sk32).to(dt)
        up = cond & (deficit < -half) & (F < 3.0 * Fexp)
        if flux_guard_hole:
            raise NotImplementedError("flux_guard_hole")
        a_ = torch.minimum((-deficit - half) / mc.clamp(min=1e-9), 3.0 * Fexp / F.clamp(min=1e-9) - 1)
        fac = 1 + a_
        mod = torch.where(up[..., None], (mod.to(dt) * fac[..., None]).to(torch.float32), mod)
        F = torch.where(up, F * fac, F)
        return mod, F
