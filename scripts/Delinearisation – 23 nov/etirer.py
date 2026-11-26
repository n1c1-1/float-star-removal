# ======================================================================
# etirer.py
# Rôle     : Étape 2 du banc : six traitements d'amateur courants et variés appliqués à chaque pile linéaire.
# Étape    : 3. Délinéarisation des images web (23 nov) (voir scripts/LISEZMOI.md)
# Entrées  : image étirée
# Sorties  : image rendue linéaire
# Exemple  : python etirer.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Étape 2 du banc : six traitements d'amateur courants et variés appliqués à chaque pile linéaire.
Écrit banc/<pile>__<recette>.tif (16 bits) ou .jpg (8 bits) et banc/<pile>__<recette>.json (la recette)."""
from __future__ import annotations

import sys
import time

import numpy as np
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, __import__("os").path.dirname(__file__))
from commun import BANC, PILES, bruit_laplacien, ecrire_json, m_pour, mad_sigma, mtf  # noqa: E402
from float_unstretch_copie import ght  # noqa: E402

REC = ("R1_auto_liee", "R2_stf_deliee_sat", "R3_asinh", "R4_asinh_debruit_jpeg", "R5_courbes_jpeg", "R6_ghs")
QUOI = {
    "R1_auto_liee": "étirement automatique lié (MTF, noir à -2,8 MAD, fond 0,25), TIFF 16 bits",
    "R2_stf_deliee_sat": "STF automatique par canal (délié) + saturation x1,5, JPEG qualité 92",
    "R3_asinh": "arcsinh 300 à couleurs gardées, noir à -3 MAD, puis MTF fond 0,15, TIFF 16 bits",
    "R4_asinh_debruit_jpeg": "arcsinh 1000, noir à -1,5 MAD, MTF fond 0,18, débruitage (luminance variation totale, chrominance floue), saturation x1,3, JPEG 90",
    "R5_courbes_jpeg": "niveaux (noir -2 MAD, blanc p99,9) + gamma 1/2,4 + courbe en S + saturation x1,2, JPEG 85",
    "R6_ghs": "étirement hyperbolique généralisé (D=e^7-1, b=6, pivot au fond) puis MTF fond 0,20, TIFF 16 bits",
}


def stats(x):
    """Médiane et MAD normalisé (1,4826 MAD) de chaque canal entier, comme les étirements automatiques des logiciels
    courants les lisent ([logiciel tiers] : find_linked_midtones_balance ; [logiciel tiers] : STF automatique)."""
    med = np.array([np.median(c) for c in x])
    sig = np.array([mad_sigma(c[::2, ::2]) for c in x])
    return med, sig


def sat(y, s):
    Y = y.mean(0, keepdims=True)
    return np.clip(Y + s * (y - Y), 0, 1)


def asinh_couleur(x, beta, bp):
    """arcsinh de [logiciel tiers] (filters/asinh.c), luminance de Rec.709, couleurs gardées."""
    p = np.clip((x - bp) / (1 - bp), 0, None)
    w = np.array([0.2126, 0.7152, 0.0722])[:, None, None]
    L = (w * p).sum(0)
    with np.errstate(all="ignore"):
        k = np.where(L > 0, np.arcsinh(beta * L) / (L * np.arcsinh(beta)), 1.0)
    return np.clip(p * k, 0, 1)


def vers_fond(y, cible):
    """MTF liée qui amène la médiane de luminance sur cible."""
    s = float(np.median(y.mean(0)))
    return mtf(y, m_pour(s, cible))


def recette(nom, x):
    med, sig = stats(x)
    mL, sL = float(np.median(x.mean(0))), float(np.mean(sig))
    if nom == "R1_auto_liee":
        c0 = float(np.mean(med) - 2.8 * np.mean(sig))
        m = m_pour((float(np.mean(med)) - c0) / (1 - c0), 0.25)
        return mtf(x, m, c0, 1.0), dict(c0=c0, m=m)
    if nom == "R2_stf_deliee_sat":
        out, pp = [], []
        for c, md, sg in zip(x, med, sig):
            c0 = md - 2.8 * sg
            m = m_pour((md - c0) / (1 - c0), 0.25)
            out.append(mtf(c, m, c0, 1.0)); pp.append((c0, m))
        return sat(np.array(out), 1.5), dict(par_canal=pp, saturation=1.5)
    if nom == "R3_asinh":
        bp = mL - 3 * sL
        y = asinh_couleur(x, 300.0, bp)
        return vers_fond(y, 0.15), dict(beta=300, bp=bp)
    if nom == "R4_asinh_debruit_jpeg":
        from scipy.ndimage import gaussian_filter
        from skimage.restoration import denoise_tv_chambolle
        bp = mL - 1.5 * sL
        y = vers_fond(asinh_couleur(x, 1000.0, bp), 0.18)
        Y = y.mean(0)
        sY = mad_sigma(bruit_laplacien(Y)[::2, ::2])
        Yd = denoise_tv_chambolle(Y, weight=1.5 * sY)            # luminance : variation totale (Chambolle 2004)
        ch = np.stack([gaussian_filter(c - Y, 1.5) for c in y])  # chrominance : flou 1,5 px
        y = Yd[None] + ch
        return sat(np.clip(y, 0, 1), 1.3), dict(beta=1000, bp=bp, saturation=1.3,
                                                debruitage="luminance variation totale, chrominance floue")
    if nom == "R5_courbes_jpeg":
        lo = mL - 2 * sL
        hi = float(np.percentile(x.mean(0), 99.9))
        y = np.clip((x - lo) / (hi - lo), 0, 1) ** (1 / 2.4)
        s_curve = PchipInterpolator([0, 0.25, 0.5, 0.75, 1], [0, 0.17, 0.52, 0.84, 1])
        y = s_curve(y)
        s_curve2 = PchipInterpolator([0, 0.3, 0.7, 1], [0, 0.26, 0.74, 1])     # second passage, contraste léger
        y = s_curve2(y)
        return sat(np.clip(y, 0, 1), 1.2), dict(lo=lo, hi=hi, gamma=1 / 2.4, saturation=1.2)
    if nom == "R6_ghs":
        D, b = float(np.expm1(7.0)), 6.0
        SP = max(mL, 1e-6)
        y = np.stack([ght(np.clip(c, 0, 1), b, D, 0.0, SP, 1.0, "normal") for c in x])
        y = np.clip(y, 0, 1)
        return vers_fond(y, 0.20), dict(D=D, b=b, SP=SP)
    raise ValueError(nom)


def ecrire(chemin_sans_ext, y, jpeg_q=None):
    y = np.clip(y, 0, 1)
    if jpeg_q is None:
        import tifffile
        p = chemin_sans_ext.with_suffix(".tif")
        tifffile.imwrite(str(p), np.round(np.moveaxis(y, 0, -1) * 65535).astype(np.uint16), photometric="rgb")
    else:
        from PIL import Image
        p = chemin_sans_ext.with_suffix(".jpg")
        Image.fromarray(np.round(np.moveaxis(y, 0, -1) * 255).astype(np.uint8)).save(p, quality=jpeg_q, subsampling=0)
    return p


JPEG = {"R2_stf_deliee_sat": 92, "R4_asinh_debruit_jpeg": 90, "R5_courbes_jpeg": 85}


def main(noms):
    for nom in noms:
        x = np.load(BANC / f"{nom}_lin.npy").astype(np.float64)
        for r in REC:
            t0 = time.time()
            y, par = recette(r, x)
            p = ecrire(BANC / f"{nom}__{r}", y, JPEG.get(r))
            par = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in par.items()}
            ecrire_json(BANC / f"{nom}__{r}.json", dict(pile=nom, recette=r, quoi=QUOI[r], fichier=p.name,
                                                        parametres=str(par),
                                                        fond_etire=float(np.median(y.mean(0))),
                                                        noirs_pct=float(100 * np.mean(np.any(y <= 0, 0))),
                                                        blancs_pct=float(100 * np.mean(np.any(y >= 1, 0)))))
            print(nom, r, f"{time.time() - t0:.1f}s fond {np.median(y.mean(0)):.3f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(PILES))
