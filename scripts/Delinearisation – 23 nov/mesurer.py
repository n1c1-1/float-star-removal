# ======================================================================
# mesurer.py
# Rôle     : Comparaison d'une image délinéarisée D à la pile linéaire d'origine L (même cadrage).
# Étape    : 3. Délinéarisation des images web (23 nov) (voir scripts/LISEZMOI.md)
# Entrées  : image étirée
# Sorties  : image rendue linéaire
# Exemple  : python mesurer.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Comparaison d'une image délinéarisée D à la pile linéaire d'origine L (même cadrage).

Mise à la même échelle : D_c = a L_c + b_c (une échelle commune, un décalage par canal), ajustée sur les médianes de
blocs 8x8 (le bruit et les étoiles n'y pèsent pas), blocs sous le 99e centile. Les mesures d'étoiles, de bruit et de
couleur ci-dessous sont des rapports : elles ne dépendent ni de a ni de b, sauf les rapports de flux par tranche (qui
dépendent de a, la mesure sans unité est « compression » = rapport brillantes / rapport faibles, 1 = linéaire)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from commun import bruit_laplacien, mad_sigma
from etoiles_outils import detecter, fwhm_median, photometrie


def blocs(x, b):
    H, W = x.shape[-2:]
    h, w = H // b, W // b
    y = x[..., : h * b, : w * b].reshape(*x.shape[:-2], h, b, w, b)
    return np.median(np.swapaxes(y, -3, -2).reshape(*x.shape[:-2], h, w, b * b), axis=-1)


def echelle(L, D):
    bL, bD = blocs(L, 8), blocs(D, 8)
    lum = bL.mean(0)
    sel = lum < np.percentile(lum, 99)
    for _ in range(3):
        rows, rhs = [], []
        for c in range(3):
            v = bL[c][sel]
            A = np.zeros((v.size, 4)); A[:, 0] = v; A[:, 1 + c] = 1
            rows.append(A); rhs.append(bD[c][sel])
        A, y = np.vstack(rows), np.concatenate(rhs)
        p, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = y - __import__('operator').matmul(A, p)
        s = mad_sigma(r)
        keep = (np.abs(r) < 4 * s + 1e-15).reshape(3, -1).all(0)
        idx = np.flatnonzero(sel)
        sel = sel.copy(); sel.flat[idx[~keep]] = False
    return float(p[0]), p[1:]


def plan(z, m):
    yy, xx = np.nonzero(m)
    A = np.c_[np.ones(yy.size), yy, xx]
    p, *_ = np.linalg.lstsq(A, z[m], rcond=None)
    return p


class Reference:
    """Ce qui ne dépend que de L : étoiles, blocs de fond et de signal fort."""

    def __init__(self, L):
        self.L = L
        lumL = L.mean(0)
        self.sigL = mad_sigma(bruit_laplacien(lumL)[::2, ::2])
        self.ys, self.xs = detecter(lumL, self.sigL, k=15.0, plafond=0.85)
        self.fw = fwhm_median(lumL, self.ys, self.xs)
        fw = self.fw
        self.ap = (max(3.0, 2.5 * fw), max(4.0, 3.5 * fw), max(6.0, 5.0 * fw))
        self.FL = np.array([photometrie(c, self.ys, self.xs, *self.ap)[0] for c in L])
        self.FLl, self.r50L, _ = photometrie(lumL, self.ys, self.xs, *self.ap)
        # masque des étoiles (toutes, même faibles) pour les blocs de bruit
        from etoiles_outils import fond_carte
        s = ndi.gaussian_filter(lumL - fond_carte(lumL), 1.2)
        ss = self.sigL / (2 * np.sqrt(np.pi) * 1.2)
        m = ndi.binary_dilation(s > 5 * ss, iterations=int(1.5 * fw) + 1)
        B = 32
        self.B = B
        self.masque = m
        self.lvl = self.bloc_masque(lumL)
        frac = m[: self.lvl.shape[0] * B, : self.lvl.shape[1] * B].reshape(self.lvl.shape[0], B, self.lvl.shape[1], B)
        fr = frac.mean(axis=(1, 3))
        self.libre = fr < 0.5                               # au moins la moitié du bloc hors étoiles
        if self.libre.sum() < 40:                           # champ très dense (amas)
            self.libre = fr <= np.sort(fr.ravel())[min(fr.size - 1, 99)]
        self.sigblocL = self._sig_blocs(lumL)
        lv = self.lvl[self.libre]
        self.ciel = self.libre & (self.lvl <= np.percentile(lv, 30))
        self.fort = self.libre & (self.lvl >= np.percentile(lv, 95))
        self.fond_bas = np.isfinite(self.lvl) & (self.lvl <= np.nanpercentile(self.lvl, 40))
        self.signal_fort = float(np.median(self.lvl[self.fort]) - np.median(self.lvl[self.ciel])) / self.sigL

    def _empile(self, x):
        B = self.B
        H, W = x.shape
        h, w = H // B, W // B
        x = np.where(self.masque, np.nan, x)
        return x[: h * B, : w * B].reshape(h, B, w, B).swapaxes(1, 2).reshape(h, w, -1)

    def bloc_masque(self, x):
        """Médiane par bloc des pixels hors étoiles."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.nanmedian(self._empile(x), axis=2)

    def _sig_blocs(self, x):
        import warnings
        rb = self._empile(bruit_laplacien(x))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med = np.nanmedian(rb, axis=2)
            return 1.4826 * np.nanmedian(np.abs(rb - med[..., None]), axis=2)


def mesurer(ref: Reference, D: np.ndarray) -> dict:
    L = ref.L
    a, b = echelle(L, D)
    Dp = (D - b[:, None, None]) / a
    lumL, lumD = L.mean(0), Dp.mean(0)
    out = {}
    # fond : niveau (unités de bruit de L) et pente
    zl, zd = blocs(lumL, 32), blocs(lumD, 32)
    m = ref.fond_bas
    pl, pd = plan(zl, m), plan(zd, m)
    gl, gd = np.hypot(*pl[1:]), np.hypot(*pd[1:])
    out["fond_ecart_sigma"] = float(np.median(zd[ref.ciel] - zl[ref.ciel]) / ref.sigL)
    out["_gl"], out["_gd"] = gl, gd
    # étoiles
    FD, r50D, _ = photometrie(lumD, ref.ys, ref.xs, *ref.ap)
    ok = (ref.FLl > 0) & np.isfinite(FD)
    lf = np.log10(ref.FLl[ok])
    rat = FD[ok] / ref.FLl[ok]
    rr = r50D[ok] / ref.r50L[ok]
    q33, q66, q90 = np.percentile(lf, [33, 66, 90])
    tr = {"faibles": lf <= q33, "moyennes": (lf > q33) & (lf <= q66), "brillantes": lf >= q90}
    for k, s in tr.items():
        out[f"flux_{k}"] = float(np.median(rat[s]))
        out[f"r50_{k}"] = float(np.nanmedian(rr[s]))
    out["compression"] = out["flux_brillantes"] / out["flux_faibles"]
    # pente log-log flux délinéarisé / flux d'origine (1 = linéaire), indépendante de l'échelle
    pos = FD[ok] > 0
    if pos.sum() > 20:
        xx, yy = lf[pos], np.log10(FD[ok][pos])
        keep = np.ones(xx.size, bool)
        for _ in range(3):
            k_, c_ = np.polyfit(xx[keep], yy[keep], 1)
            r_ = yy - (k_ * xx + c_)
            keep = np.abs(r_) < 3 * mad_sigma(r_[keep]) + 1e-9
        out["pente_flux"] = float(k_)
    else:
        out["pente_flux"] = np.nan
    out["n_etoiles"] = int(ok.sum())
    # bruit
    sD = ref._sig_blocs(lumD)
    sky_L, sky_D = np.median(ref.sigblocL[ref.ciel]), np.median(sD[ref.ciel])
    f_L, f_D = np.median(ref.sigblocL[ref.fort]), np.median(sD[ref.fort])
    out["bruit_ciel_rapport"] = float(sky_D / sky_L)
    out["bruit_fort_rapport"] = float(f_D / f_L)
    out["bruit_dependance"] = float((f_D / sky_D) / (f_L / sky_L))
    # pente du fond rapportée au bruit du ciel de chaque image : indépendante de l'échelle a
    gl, gd = out.pop("_gl"), out.pop("_gd")
    out["pente_rapport"] = float((gd / sky_D) / (gl / sky_L)) if gl > 0 and sky_D > 0 else np.nan
    out["signal_fort_sigma"] = ref.signal_fort
    # couleurs : étoiles moyennes et brillantes, et zones de signal fort
    FDc = np.array([photometrie(c, ref.ys, ref.xs, *ref.ap)[0] for c in Dp])
    s = ok.copy(); s[ok] = lf > q33
    s &= np.all(ref.FL > 0, 0) & np.all(FDc > 0, 0)
    dlog = [np.median(np.abs(np.log(FDc[c, s] / FDc[1, s]) - np.log(ref.FL[c, s] / ref.FL[1, s]))) for c in (0, 2)]
    out["couleur_etoiles_pct"] = float(100 * (np.exp(np.mean(dlog)) - 1))
    bl = np.stack([ref.bloc_masque(c) for c in L])
    bd = np.stack([ref.bloc_masque(c) for c in Dp])
    cl = np.median(bl[:, ref.ciel], axis=1)
    cd = np.median(bd[:, ref.ciel], axis=1)
    nl = bl[:, ref.fort] - cl[:, None]
    nd = bd[:, ref.fort] - cd[:, None]
    good = np.all(nl > 0, 0) & np.all(nd > 0, 0)
    if good.sum() > 5 and ref.signal_fort > 10:
        dl = [np.median(np.abs(np.log(nd[c, good] / nd[1, good]) - np.log(nl[c, good] / nl[1, good]))) for c in (0, 2)]
        out["couleur_neb_pct"] = float(100 * (np.exp(np.mean(dl)) - 1))
    else:
        out["couleur_neb_pct"] = np.nan
    out["echelle_a"] = a
    return out
