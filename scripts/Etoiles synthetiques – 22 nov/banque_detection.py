# ======================================================================
# banque_detection.py
# Role     : Individual-star bank WITHOUT catalogue (round 5): the bright stars are found by DETECTION on the image.
# Step     : 4. Realistic synthetic stars (22 Nov) (see scripts/LISEZMOI.md)
# Inputs   : star-free backgrounds (pair targets), star shape banks
# Outputs  : pairs (input, target), sheets, blind test
# Example  : python banque_detection.py [arguments]  (see the docstring; paths through the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Individual-star bank WITHOUT catalogue (round 5): the bright stars are found by DETECTION on the image.

- background: median over 32 px blocks (luminance), smoothed; noise: robust MAD of the remainder;
- stars: local maxima (9 px) of the smoothed luminance (sigma 1.5) above 30 noises;
- width: 2nd-order moment of the isolated unclipped stars (peak 100 noises at 20 % of the max), median;
- flux: 3 FWHM box (unclipped); clipped ones: the wing (8 FWHM box minus 3 FWHM box) divided by the median wing
  share of the unclipped ones;
- bright ones: the strongest in flux, isolated (no detection of more than a tenth of its flux within 150 px, nor a quarter
  within 40 px), far from the edge; all neighbouring detections masked (3 FWHM + 10 px per magnitude above the
  median of the width stars); then the same cleaning as banque_individuelle (smoothed decreasing round profile +
  structures linked to the core) and the same checks.
A reference "G" is given: -2.5 log10(flux) + constant, the median of the width stars at 12.5 (display only).

    construire(nom, a, verbose) -> dict (same shape as banque_individuelle.construire)
"""
from __future__ import annotations

import os as _os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    _os.environ[_k] = "2"                                  # au plus 2 fils (consigne du 22 nov après la saturation)

import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import maximum_filter, median_filter
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_etoiles as A                                            # noqa: E402
import banque_individuelle as BI                                     # noqa: E402

H = BI.H


def boite(I, x, y, r):
    """Somme dans la boîte carrée de demi-côté r centrée en (x, y) (image intégrale I)."""
    x0 = np.clip(np.rint(x - r).astype(int), 0, I.shape[1] - 1); x1 = np.clip(np.rint(x + r + 1).astype(int), 0, I.shape[1] - 1)
    y0 = np.clip(np.rint(y - r).astype(int), 0, I.shape[0] - 1); y1 = np.clip(np.rint(y + r + 1).astype(int), 0, I.shape[0] - 1)
    return I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0]


def detecter(a, verbose=True, nom=""):
    C, Hh, W = a.shape
    L = a.sum(0).astype(np.float32)
    small = cv2.resize(L, (max(W // 32, 4), max(Hh // 32, 4)), interpolation=cv2.INTER_AREA)
    small = median_filter(small, size=5)
    bg = cv2.resize(small, (W, Hh), interpolation=cv2.INTER_LINEAR)
    R = L - bg
    del L
    samp = R[::7, ::7].ravel()
    samp = samp[samp < np.percentile(samp, 99)]
    sig = 1.4826 * float(np.median(np.abs(samp - np.median(samp)))) + 1e-12
    Rs = cv2.GaussianBlur(R, (0, 0), 1.5)
    pk = (Rs == maximum_filter(Rs, size=9)) & (Rs > 30 * sig)
    yy, xx = np.nonzero(pk)
    del pk
    pv = Rs[yy, xx]
    Lmax = float(np.max(R))
    if verbose:
        print(nom, f"détections {len(xx)}, bruit {sig:.3g}, max {Lmax:.3g}", flush=True)
    return R, Rs, sig, xx.astype(float), yy.astype(float), pv, Lmax


def construire(nom, a, verbose=True, exclure=()):
    if a.shape[0] == 1:
        a = np.repeat(a, 3, 0)
    R, Rs, sigL, x, y, pv, Lmax = detecter(a, verbose, nom)
    Hh, W = R.shape
    if len(x) < 30:
        if verbose:
            print(nom, "trop peu de détections", flush=True)
        return None
    tree = cKDTree(np.c_[x, y])
    # écrêtée = plateau propre (>= 9 px à >= 98 % de son pic) : un seuil sur le max de l'image classait presque toutes
    # les étoiles écrêtées sur certaines piles (tour 5, E022 : 90 % des pics au-dessus de 20 % du max)
    def plateau(i, h=12):
        st_ = A.stamp(a, x[i], y[i], h)
        if np.isnan(st_).any():
            return 99
        Lc = st_.sum(0)
        return int(np.sum(Lc >= 0.98 * Lc[h - 2:h + 3, h - 2:h + 3].max()))
    imax = a.reshape(3, -1).max(1)
    order_pv = np.argsort(-pv)
    cand = [i for i in order_pv[len(pv) // 100:] if pv[i] > 100 * sigL][:3000]
    fw = 4.0
    fws, unsat_ids = [], []
    for _ in range(2):
        fws, unsat_ids = [], []
        h = int(math.ceil(3 * fw)) + 3
        for i in cand:
            if not (h < x[i] < W - h and h < y[i] < Hh - h):
                continue
            if any(j != i and pv[j] > 0.2 * pv[i] for j in tree.query_ball_point([x[i], y[i]], 4 * fw)):
                continue                                              # champ dense toléré
            if plateau(i) >= 5:
                continue
            s = A.stamp(a, x[i], y[i], h)
            pr = A.ring_median(s)
            bgc = np.nanmedian(pr[:, int(2.5 * fw):int(3 * fw) + 2], axis=1)
            f, *_ = A.second_moment_fwhm(s, fw, bgc)
            if np.isfinite(f[1]) and 0.8 < f[1] < 40:
                fws.append(f[1]); unsat_ids.append(i)
            if len(fws) >= 300:
                break
        if len(fws) < 20:
            if verbose:
                print(nom, "trop peu d'étoiles isolées pour la largeur :", len(fws), flush=True)
            return None
        fw = float(np.median(fws))
    I = cv2.integral(R.astype(np.float64))
    r3 = int(math.ceil(3 * fw)); r8 = int(math.ceil(8 * fw))
    b3 = BD_boite = boite(I, x, y, r3); b8 = boite(I, x, y, r8)
    ui = np.array(unsat_ids)
    okw = b3[ui] > 0
    wing = float(np.median((b8[ui][okw] - b3[ui][okw]) / b3[ui][okw])) if okw.sum() > 10 else 0.1
    a3, a8 = (2 * r3 + 1) ** 2, (2 * r8 + 1) ** 2
    Floc = np.clip(b3 - (b8 - b3) * a3 / max(a8 - a3, 1), 1e-12, None)   # flux local (sans le halo d'une voisine)
    F = Floc.copy()
    satur = np.zeros(len(x), bool)
    for i in np.argsort(-b8)[:400]:                                   # les plus fortes : écrêtées ? flux par l'aile
        if plateau(i, int(max(12, 2 * fw))) >= 9:
            satur[i] = True
            F[i] = max((b8[i] - b3[i]) / max(wing, 1e-3), Floc[i])
    Fref = float(np.median(Floc[ui]))
    G = -2.5 * np.log10(F / Fref) + 12.5
    ex = np.array(exclure, float).reshape(-1, 2)
    yyg, xxg = np.mgrid[-H:H + 1, -H:H + 1].astype(np.float32)
    rr = np.hypot(xxg, yyg)
    formes, Gs, niv, xy = [], [], [], []
    rej = dict(bord=0, exclue=0, voisine=0, vignette=0, aile=0, coeur_masque=0, coeur=0, halo=0)
    rang = 0
    for i in np.argsort(-F):
        if len(formes) >= BI.MAXI or rang >= 150:
            break
        rang += 1
        if not (H + 10 < x[i] < W - H - 10 and H + 10 < y[i] < Hh - H - 10):
            rej["bord"] += 1; continue
        if len(ex) and np.min(np.hypot(ex[:, 0] - x[i], ex[:, 1] - y[i])) < 5:
            rej["exclue"] += 1; continue
        nb = tree.query_ball_point([x[i], y[i]], 150)
        # isolement relâché (tour 5 : champs denses à 135-275 mm) : les voisines sont masquées de toute façon
        if any(j != i and (Floc[j] > F[i] or (Floc[j] > 0.3 * F[i] and math.hypot(x[j] - x[i], y[j] - y[i]) < 60))
               for j in nb):
            rej["voisine"] += 1; continue
        # centre : barycentre du cœur (plateau compris)
        c0 = A.stamp(a, x[i], y[i], int(2 * fw) + 2)
        Lc = np.nan_to_num(c0.sum(0)); Lc = np.clip(Lc - np.median(Lc), 0, None)
        hh = Lc.shape[0] // 2
        gy, gx = np.mgrid[-hh:hh + 1, -hh:hh + 1]
        wsum = Lc.sum()
        cx = x[i] + (float((Lc * gx).sum() / wsum) if wsum > 0 else 0.0)
        cy = y[i] + (float((Lc * gy).sum() / wsum) if wsum > 0 else 0.0)
        s = A.stamp(a, cx, cy, H)
        if np.isnan(s).mean() > 0.01 or np.mean(s == 0) > 0.01:
            rej["vignette"] += 1; continue
        s = np.nan_to_num(s)
        ring = (rr >= 185) & (rr <= 200)
        s = s - np.median(s[:, ring], axis=1)[:, None, None]
        fx, fy = cx - round(cx), cy - round(cy)
        M = np.float32([[1, 0, -fx], [0, 1, -fy]])
        s = np.stack([cv2.warpAffine(c, M, (2 * H + 1, 2 * H + 1), flags=cv2.INTER_LINEAR) for c in s])
        pr = A.ring_median(s[:, H - 60:H + 61, H - 60:H + 61])
        sh = np.nanmean(pr[:, int(3 * fw):int(8 * fw)], axis=1)
        if not np.all(np.isfinite(sh)) or np.any(sh <= 0):
            rej["aile"] += 1; continue
        parts = sh / sh.sum()
        shape = s / (F[i] * parts[:, None, None])
        rcoeur = 2 * fw + 1.5 * math.sqrt(plateau(i, int(max(12, 2 * fw))) / math.pi)
        for j in tree.query_ball_point([x[i], y[i]], H * 1.42):
            if j == i or math.hypot(x[j] - x[i], y[j] - y[i]) < rcoeur:
                continue                                              # pic dans l'étoile même (plateau, pied d'aigrette)
            gj = -2.5 * math.log10(max(Floc[j], 1e-12) / Fref) + 12.5
            rad = 3 * fw + 10 * max(0.0, 12.5 - gj)
            mk = (np.hypot(xxg - (x[j] - round(cx)), yyg - (y[j] - round(cy))) < rad) & (rr > rcoeur)
            shape[:, mk] = np.nan                                     # jamais sur le cœur de l'étoile
        if np.isnan(shape[:, rr <= 2 * fw]).any():
            rej["coeur_masque"] += 1; continue
        cl = BI.nettoyer(shape, fw)
        if any(float(cl[c][rr <= 1.5].max()) < 0.9 * float(np.abs(cl[c]).max()) for c in range(3)):
            rej["coeur"] += 1; continue
        pr1 = [float(cl[1][(rr >= a_) & (rr < a_ + 3)].mean()) for a_ in (15, 30, 60)]
        if not (pr1[1] < 0.5 * pr1[0] and pr1[2] < 0.5 * pr1[1]) or min(
                float(cl[c][rr < 60].min()) for c in range(3)) < -0.1 * pr1[0]:
            rej["halo"] += 1; continue
        pkv = np.nanmax(s[:, H - 3:H + 4, H - 3:H + 4], axis=(1, 2))
        formes.append(cl.astype(np.float16)); Gs.append(float(G[i])); niv.append((pkv / (F[i] * parts)).tolist())
        xy.append([float(cx), float(cy), float(F[i])])
    if verbose:
        print(nom, "rejets :", rej, flush=True)
    if len(formes) < 3:
        if verbose:
            print(nom, "trop peu de brillantes pour la banque :", len(formes), flush=True)
        return None
    F32 = np.stack(formes).astype(np.float32)
    med = np.percentile(F32[:, 1], 30, axis=0)
    ri = np.rint(rr).astype(int)
    prof = np.bincount(ri.ravel(), med.ravel()) / np.maximum(np.bincount(ri.ravel()), 1)
    p30, p50, p80 = float(np.mean(prof[27:34])), float(np.mean(prof[45:56])), float(np.mean(prof[72:89]))
    gam = float(np.clip(math.log(max(p30, 1e-30) / max(p80, 1e-30)) / math.log(80 / 30), 1.6, 3.0)) \
        if p30 > 0 and p80 > 0 else 2.2
    sig_c = A.sky_noise(a)
    res = dict(image=nom, formes=np.stack(formes), G=Gs, niveau=niv, fwhm=fw, zp=None, etoiles_largeur=len(fws),
               a50=float(np.clip(p50, 0.0, 4e-6)), a50_mesure=max(p50, 0.0), gamma=gam, angles=BI.angles_aigrettes(F32, fw), bruit=sig_c.tolist(),
               max_image=imax.tolist(), xy=xy, detection=True)
    if verbose:
        print(nom, f"fwhm {fw:.2f} : {len(formes)} étoiles ; halo a50 {res['a50']:.2e} pente {gam:.2f} ; "
                   f"aigrettes {[round(v) for v in res['angles']]}", flush=True)
    return res
