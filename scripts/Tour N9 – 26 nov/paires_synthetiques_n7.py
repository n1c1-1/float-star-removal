# ======================================================================
# paires_synthetiques_n7.py
# Role     : COPY of 24 Nov (trial N2): only addition, the extras parameter of paire_synthetique (imposed very bright stars).
# Stage    : 10. Discarded trial N9 (compact cores, 26 Nov) (see scripts/LISEZMOI.md)
# Inputs   : v33g pairs
# Outputs  : snapshots, tableau_n9.md
# Example  : python paires_synthetiques.py cache [n_styles=32]   -> cache_styles.npz (once, ~5 min on 2 threads)
# ======================================================================
"""COPY of 24 Nov (trial N2): only addition, the extras parameter of paire_synthetique (imposed very bright stars).
Original unchanged: Etoiles synthetiques – 22 nov/paires_synthetiques.py.

COPY of 25 Nov (trial N7): paire_synthetique unchanged; added coeur_synthetique and paire_synthetique_coeurs (extended compact CORES,
present identically in the input and the target: a network that never sees them takes a galaxy core for a star).

Training hook-up (round 6): FAST synthetic pair on a real starless background.

    from paires_synthetiques import paire_synthetique
    entree, cible = paire_synthetique(fond, graine)
      fond   : (3, H, W) linear float32, the starless target of a real pair (256 or 512 crop), in sky noise units
               (convention of the method images: background 0, noise 1) ; "bruit=" otherwise
      cible  : the EXACT background (copy)
      entree : background + drawn stars + their photon noise, each clipped at its own level around it only
               (entree - cible is never non-zero outside the stars)
    Reproducible by seed.

Speed: the full generator (etoiles2.ajouter_etoiles, 7-40 s per 512 tile) is too slow for training.
Here, "image styles" are precomputed once and cached (cache_styles.npz, ~50 MB) from
etoiles2.tirer_image (free draws: real instrument, Fourier optics, halo, spikes, beads, plateau,
colours, bank):
  - core PSF (65 px, per colour); "halo + spikes" template at unit flux (257 px, per colour), with the same
    law as etoiles2.vignette; 3 shapes from the bank (median of 5-8 real stars, rotated, at the width of the style);
Per tile: power-law counts (density 50-6000 / Mpx above a noise, as stars.draw_field), BP-RP colours
of our fields, all cores in a single bilinear deposit + convolution, halo and spikes of stars above
20 noises by template cut at their reach, shapes from the bank for the clipped ones, own plateau per star (soft
edge), photon noise.

    python paires_synthetiques.py cache [n_styles=32]   -> cache_styles.npz (once, ~5 min on 2 threads)
    python paires_synthetiques.py vitesse               -> time per 256 and 512 tile (2 threads), Synth_20 contact sheet
"""
from __future__ import annotations

import os as _os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    _os.environ.setdefault(_k, "2")                        # au plus 2 fils (l'appelant peut fixer moins)

import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = (Path(__import__("os").environ.get("RACINE_SCRIPTS") or Path(__file__).resolve().parents[1]) / "Etoiles synthetiques – 22 nov") / "cache_styles.npz"   # copie nuit 24 nov : cache de l'original
#: tour 8 : les changements du tour 7 (aigrettes amincies et inégales, formes de banque relissées avec queue continue)
#: font reculer les très brillantes au juge ; le cache est fait sans eux (rendu du tour 6), emprunt 0,7-1,4
T7 = False
K = 65                                                      # PSF du cœur
RT = 128                                                    # demi-côté des gabarits
SEUIL = 20.0                                                # pic (bruits) au-dessus duquel une étoile a halo et aigrettes
WAVE_RGB = (640.0, 530.0, 460.0)
WAVE_NB = (672.0, 656.0, 501.0)
LAM0 = 530.0
BPRP = np.array([[0.0, 0.32, 0.65, 0.97, 1.17, 1.49, 2.45, 3.12, 4.0],
                 [0.1, 0.37, 0.85, 1.16, 1.5, 2.25, 3.48, 5.85, 6.5],
                 [0.2, 0.47, 0.7, 0.85, 0.96, 1.19, 1.5, 2.09, 3.0],
                 [-0.2, 0.15, 0.82, 1.23, 1.52, 1.92, 2.74, 3.44, 4.5]])
CENTILES = np.array([0, 1, 5, 25, 50, 75, 95, 99, 100], float)
_S = None


# ------------------------------------------------------------------------------------------------ cache des styles
def _bord(g):
    """Niveau moyen par couleur à 0,75-0,8 RT (pour prolonger la queue au-delà du gabarit)."""
    yy, xx = np.mgrid[-RT:RT + 1, -RT:RT + 1]
    rr = np.hypot(xx, yy)
    m = (rr >= 0.75 * RT) & (rr <= 0.8 * RT)
    return [max(float(g[c][m].mean()), 0.0) for c in range(3)]


def _gabarit(P, rng=None):
    """Auréole + aigrettes à flux unité par couleur (3, 2RT+1, 2RT+1), sans le cœur ; part de l'auréole."""
    fw = P["fwhm"]
    yy, xx = np.mgrid[-RT:RT + 1, -RT:RT + 1].astype(np.float32)
    rr = np.hypot(xx, yy); th = np.arctan2(yy, xx)
    a50, gam, rc = P["a50"], P["gamma"], P["rc"]
    aur = a50 * ((rr ** 2 + rc ** 2) / (50 ** 2 + rc ** 2)) ** (-gam / 2)
    er = P["allongement"] * np.clip(rr / 100.0, 0, 2.5) ** 0.8
    aur = aur * (1 + np.clip(er, 0, 0.6) * np.cos(2 * (th - math.radians(P["axe"]))))
    share = float(min(aur.sum(), 0.4))
    aur *= share / max(float(aur.sum()), 1e-12)
    out = np.repeat(aur[None], 3, 0).astype(np.float32)
    ag = P["aigrettes"]
    waves = WAVE_NB if P["mode"] == "nb" else WAVE_RGB
    w = P["larg_sigma"]
    rng = rng or np.random.default_rng(0)
    for a in ag["angles"]:
        fbr = float(np.exp(rng.normal(0, 0.25))) if T7 else 1.0        # tour 7 : branches inégales
        for split in ((0.0,) if not ag["dedoublement"] else (-ag["dedoublement"] / 2, ag["dedoublement"] / 2)):
            t = math.radians(a + split)
            along = xx * math.cos(t) + yy * math.sin(t)
            across = -xx * math.sin(t) + yy * math.cos(t)
            wr = w * (0.5 + 0.5 * np.exp(-np.maximum(along, 0) / (10 * fw))) if T7 else w   # amincie (tour 7)
            band = np.exp(-0.5 * (across / wr) ** 2) * (along > 0)
            fac = (0.5 if ag["dedoublement"] else 1.0) * fbr
            for c in range(3):
                prof = P["s30"] * ((1 + (along / fw) ** 2) / (1 + (30 / fw) ** 2)) ** (-P["pente"] / 2)
                if P["L"]:
                    prof = prof * np.exp(-(along / P["L"]) ** 2)
                prof = prof + P["s11"] * ((along ** 2 + fw ** 2) / (121.0 + fw ** 2)) ** -2.0
                if P["pointilles"]:
                    prof = prof * (1 + P["pointilles"] * np.cos(2 * np.pi * along / (P["pas"] * waves[c] / LAM0)))
                out[c] += (fac * prof * band).astype(np.float32)
    tw = np.clip((RT - rr) / (0.15 * RT), 0, 1)
    return (out * tw).astype(np.float32), share


def _formes_banque(P, rng, n=3):
    """Jusqu'à n formes de banque (médiane de 5-8 étoiles) à la largeur et à l'orientation du style, flux unité."""
    b = P.get("banque")
    if b is None or "formes" not in b:
        return np.zeros((0, 3, 2 * RT + 1, 2 * RT + 1), np.float16), [], [], 2.2
    F = b["formes"]
    h0 = F.shape[-1] // 2
    s = float(np.clip(P["fwhm"] / b["fwhm"], 0.7, 1.4))
    out, pk = [], []
    for _ in range(n):
        m = int(min(len(F), rng.integers(5, 9)))
        pick = rng.choice(len(F), m, replace=False)
        f = np.median(np.stack([F[int(k)].astype(np.float32) for k in pick]), axis=0)
        th = math.radians(P["banque_rotation"] + float(rng.uniform(-2, 2)))
        A2 = s * np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        M = np.zeros((2, 3), np.float32); M[:, :2] = A2; M[:, 2] = np.array([RT, RT]) - __import__('operator').matmul(A2, np.array([h0, h0]))
        g = np.stack([cv2.warpAffine(f[c], M, (2 * RT + 1, 2 * RT + 1), flags=cv2.INTER_LINEAR) / (s * s)
                      for c in range(3)])
        # tour 7 : comme etoiles2.vignette_banque2 : rond lissé, reste non rond, queue continue depuis un raccord
        # adapté à la forme ; éteint au tour 8 (T7 = False) : forme telle quelle, fondue au bord
        import etoiles2 as E
        yy, xx = np.mgrid[-RT:RT + 1, -RT:RT + 1]
        rr = np.hypot(xx, yy)
        if not T7:
            g *= np.clip((RT - rr) / (0.15 * RT), 0, 1)
            out.append(g.astype(np.float16)); pk.append(g[:, RT - 1:RT + 2, RT - 1:RT + 2].max(axis=(1, 2)).tolist())
            continue
        npl = int(np.sum(g[1] >= 0.98 * g[1].max()))
        r0 = max(2 * P["fwhm"], 1.5 * math.sqrt(npl / math.pi))
        rg = math.sqrt(npl / math.pi) + P["fwhm"]
        rond, res, plum = E.rond_et_reste(g, RT, min(r0, rg), r_garde=rg, fondu=2 * P["fwhm"])
        i0 = int(math.ceil(r0)); zr = np.flatnonzero(plum[i0:] <= 1e-3 * max(plum[i0], 1e-30))
        r_zero = float(i0 + zr[0]) if zr.size else float(RT)
        rj = float(np.clip(0.6 * r_zero, r0 + 5, 0.6 * RT))
        yy, xx = np.mgrid[-RT:RT + 1, -RT:RT + 1]
        rr = np.hypot(xx, yy)
        sel = (rr >= 0.9 * rj) & (rr <= rj)
        gam = float(b.get("gamma", 2.2))
        wj = np.clip((rr - 0.6 * rj) / (0.4 * rj), 0, 1)
        for c in range(3):
            lev = max(float(rond[c][sel].mean()), 0.0)
            g[c] = (1 - wj) * rond[c] + wj * lev * (np.maximum(rr, 1.0) / rj) ** (-gam) + res[c]
        g *= np.clip((RT - rr) / (0.15 * RT), 0, 1)
        out.append(g.astype(np.float16)); pk.append(g[:, RT - 1:RT + 2, RT - 1:RT + 2].max(axis=(1, 2)).tolist())
    return np.stack(out), pk, [_bord(g_.astype(np.float32)) for g_ in out], float(b.get("gamma", 2.2))


def construire_cache(n_styles=32, graine=2026):
    sys.path.insert(0, str(HERE))
    import etoiles2 as E
    rng = np.random.default_rng(graine)
    cores, gabs, banq, meta = [], [], [], []
    for k in range(n_styles):
        t0 = time.time()
        P = E.tirer_image(np.random.default_rng([graine, k]), 512)
        # banque de même configuration que le style (même nombre d'aigrettes), tournée vers ses angles ; sinon aucune
        # (1re construction : un style à 6 aigrettes avec la banque d'un Newton à 2 : deux sortes d'aigrettes par tuile)
        ang = list(P["aigrettes"]["angles"])
        comp = [b_ for b_ in E.banques_dispo() if len(b_.get("angles", [])) == len(ang) and len(b_["G"]) >= 5
                and 0.7 <= P["fwhm"] / b_["fwhm"] <= 1.4]                # tour 8 : emprunt d'échelle 0,7-1,4
        if P.get("banque") is not None and comp:
            b_ = comp[int(rng.integers(len(comp)))]
            if ang:
                per = 360.0 / len(ang)
                d = (min(a_ % per for a_ in ang) - min(a_ % per for a_ in b_["angles"])) % per
                rot = d - per if d > per / 2 else d
            else:
                rot = float(rng.uniform(0, 360))
            P["banque"], P["banque_rotation"] = b_, float(rot)
        elif P.get("banque") is not None:
            P["banque"] = None
        c = P["grille"][(1, 1)]
        h = c.shape[-1] // 2
        core = c[:, h - K // 2:h + K // 2 + 1, h - K // 2:h + K // 2 + 1].astype(np.float32)
        core /= core.sum(axis=(1, 2), keepdims=True)
        gab, share = _gabarit(P, rng)
        fb, pk, bord_b, gam_b = _formes_banque(P, rng)
        cores.append(core); gabs.append(gab.astype(np.float16))
        banq.append(fb)
        wb_gains = np.asarray(P["gains"], float)
        meta.append(dict(fwhm=P["fwhm"], share=share, sat=[float(v) for v in P["sat"]], plateau=list(P["plateau"]),
                         plateau_couleur_p=P["plateau_couleur_p"], bord=P["bord"], photon_k=P["photon_k"],
                         mode=P["mode"], gains=wb_gains.tolist(), champ=["rosette", "m8_sho", "m57", "horsehead"].index(
                             P["champ_bprp"]), banque=(P["banque"]["image"] if P.get("banque") else None),
                         banque_pic=pk, densite=P["densite"], aigrettes=len(P["aigrettes"]["angles"]),
                         psf_pic=float(core[1].max()),
                         # tour 7 (méta seulement, tableaux inchangés) : queue au-delà du gabarit
                         gamma=float(P["gamma"]), bord_gabarit=_bord(gab), bord_banque=bord_b, gamma_banque=gam_b))
        print(f"style {k}: fwhm {P['fwhm']:.2f} aigrettes {meta[-1]['aigrettes']} banque {meta[-1]['banque']} "
              f"{time.time() - t0:.1f} s", flush=True)
    nb = max(len(b) for b in banq)
    B = np.zeros((n_styles, max(nb, 1), 3, 2 * RT + 1, 2 * RT + 1), np.float16)
    for k, b in enumerate(banq):
        if len(b):
            B[k, :len(b)] = b
    np.savez_compressed(CACHE, cores=np.stack(cores), gabarits=np.stack(gabs), banque=B,
                        meta=json.dumps(meta))


def _charger():
    global _S
    if _S is None:
        if not CACHE.exists():
            construire_cache()
        z = np.load(CACHE, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        gab = z["gabarits"].astype(np.float32)
        # portée de chaque gabarit : rayon où son maximum par anneau passe sous une valeur (pour couper au tirage)
        yy, xx = np.mgrid[-RT:RT + 1, -RT:RT + 1]
        ri = np.rint(np.hypot(xx, yy)).astype(int).ravel()
        prof = []
        for g in gab:
            m = g.max(0).ravel()
            pmax = np.zeros(RT * 2 + 2); np.maximum.at(pmax, ri, m)
            prof.append(np.maximum.accumulate(pmax[::-1])[::-1][:RT + 1])
        _S = dict(cores=z["cores"].astype(np.float32), gabarits=gab, banque=z["banque"].astype(np.float32),
                  meta=meta, prof=np.stack(prof))
    return _S


# ------------------------------------------------------------------------------------------------ une paire
def _planck_parts(bp, waves, gains):
    c = np.clip(bp, -0.3, 4.5)
    T = 5040.0 / (0.4929 + 0.5092 * c - 0.0353 * c * c)
    f = np.stack([lam ** -5 / np.expm1(np.clip(1.4388e7 / (lam * T), 1e-6, 700)) * g for lam, g in zip(waves, gains)])
    return (f / f.sum(0, keepdims=True)).astype(np.float32)


def paire_synthetique(fond, graine, bruit=1.0, extras=None):
    """(entree, cible) : voir le module. fond (3, H, W) float32, en bruits du ciel (sinon donner « bruit »)."""
    S = _charger()
    rng = np.random.default_rng(graine)
    C, Hh, W = fond.shape
    k = int(rng.integers(len(S["meta"])))
    m_ = S["meta"][k]
    core = S["cores"][k]
    gab = S["gabarits"][k]
    fw = m_["fwhm"]
    # le champ : comptes en loi de puissance sur la tuile et une marge (les halos de dehors entrent)
    mg = RT
    Cw, Ch = W + 2 * mg, Hh + 2 * mg
    dens = float(np.exp(rng.uniform(math.log(50), math.log(6000))))
    alpha = float(rng.uniform(0.55, 0.75))
    pmin = 0.2
    nst = int(min(dens * Cw * Ch / 1e6 * pmin ** -alpha, 200_000))
    peak = pmin * rng.random(nst) ** (-1 / alpha)
    bright = float(rng.uniform(0.7, 1.0))
    peak = np.where(peak > 100, 100 * (peak / 100) ** (alpha / bright), peak)
    peak = np.minimum(peak, 3e5)
    x = rng.uniform(0, Cw, nst); y = rng.uniform(0, Ch, nst)
    if extras is not None and len(extras):           # copie nuit 24 nov (N2) : étoiles très brillantes imposées (x, y dans la tuile, pic en bruits)
        ex = np.asarray(extras, np.float64).reshape(-1, 3)
        peak = np.concatenate([peak, ex[:, 2]]); x = np.concatenate([x, ex[:, 0] + mg]); y = np.concatenate([y, ex[:, 1] + mg])
        nst = len(peak)
    bp = np.interp(rng.random(nst) * 100, CENTILES, BPRP[m_["champ"]])
    parts = _planck_parts(bp, WAVE_NB if m_["mode"] == "nb" else WAVE_RGB, m_["gains"])
    F = peak * 3.0 / m_["psf_pic"] * bruit                                # flux total (somme des couleurs)
    sat = np.asarray(m_["sat"], np.float32) * bruit
    lo, hi = m_["plateau"]
    w_core = 1.0 - m_["share"]
    # étoiles à halo et aigrettes : niveau d'écrêtage propre, écrêtée ou non ; une écrêtée prise dans la banque n'a
    # pas son cœur dans la convolution (la forme de la banque porte le sien)
    big = np.flatnonzero(peak > SEUIL)
    nbq = len(m_["banque_pic"])
    lv = rng.uniform(lo, hi, len(big))[:, None] * sat[None, :]
    drop = rng.random(len(big)) < m_["plateau_couleur_p"]
    if drop.any():
        cc = rng.integers(0, 3, len(big))
        lv[np.flatnonzero(drop), cc[drop]] *= rng.uniform(0.55, 1.0, int(drop.sum()))
    satur = F[big] * parts[1, big] * m_["psf_pic"] * w_core > lv[:, 1]
    de_banque = satur & (nbq > 0)
    wc = np.full(nst, w_core, np.float32)
    wc[big[de_banque]] = 0.0
    # 1) les cœurs : un dépôt bilinéaire par couleur, une convolution
    x0 = np.floor(x).astype(np.int64); y0 = np.floor(y).astype(np.int64)
    fx, fy = x - x0, y - y0
    idx, wts = [], []
    for dy, dx, w in ((0, 0, (1 - fx) * (1 - fy)), (0, 1, fx * (1 - fy)), (1, 0, (1 - fx) * fy), (1, 1, fx * fy)):
        xx_, yy_ = x0 + dx, y0 + dy
        ok = (xx_ >= 0) & (xx_ < Cw) & (yy_ >= 0) & (yy_ < Ch)
        idx.append(yy_[ok] * Cw + xx_[ok]); wts.append((F * wc * w)[ok]); 
        if dy == 0 and dx == 0:
            ok0 = ok
    idx = np.concatenate(idx)
    sel = [np.flatnonzero(((x0 + dx >= 0) & (x0 + dx < Cw) & (y0 + dy >= 0) & (y0 + dy < Ch)))
           for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1))]
    stars = np.empty((3, Ch, Cw), np.float32)
    wall = np.concatenate(wts)
    pall = np.concatenate([parts[:, s_] for s_ in sel], axis=1)
    for c in range(3):
        dep = np.bincount(idx, wall * pall[c], minlength=Cw * Ch).astype(np.float32).reshape(Ch, Cw)
        stars[c] = cv2.filter2D(dep, -1, core[c], borderType=cv2.BORDER_CONSTANT)
    # 2) auréole et aigrettes par gabarit coupé à la portée ; formes de la banque pour les écrêtées ; plateau propre
    Lmap = None
    bq = S["banque"][k]
    prof = S["prof"][k]
    for j, i in enumerate(big):
        Fi = F[i]
        R = int(np.searchsorted(-prof, -0.01 * bruit / Fi))                # portée : gabarit > 0,01 bruit
        R = int(min(max(R, 4), RT))
        xi, yi = int(round(x[i])), int(round(y[i]))
        if xi + R < mg or xi - R >= mg + W or yi + R < mg or yi - R >= mg + Hh:
            continue
        ib = 0
        if de_banque[j]:
            R = RT if R < RT // 2 else R
            ib = int(rng.integers(nbq))
            g = bq[ib][:, RT - R:RT + R + 1, RT - R:RT + R + 1]
        else:
            g = gab[:, RT - R:RT + R + 1, RT - R:RT + R + 1]
        y0_, x0_ = yi - R, xi - R
        sy0, sx0 = max(0, y0_), max(0, x0_)
        sy1, sx1 = min(Ch, yi + R + 1), min(Cw, xi + R + 1)
        stars[:, sy0:sy1, sx0:sx1] += g[:, sy0 - y0_:sy1 - y0_, sx0 - x0_:sx1 - x0_] * (Fi * parts[:, i])[:, None, None]
        # tour 7 : au-delà du gabarit, la queue continue (sinon un bord net autour des plus brillantes)
        if R >= RT and "bord_gabarit" in m_:
            bd = np.asarray(m_["bord_banque"][ib] if de_banque[j] else m_["bord_gabarit"], np.float32)
            gq = m_["gamma_banque"] if de_banque[j] else m_["gamma"]
            r_a = 0.775 * RT
            R2 = int(min(r_a * (max(Fi * float(bd.max()) / (0.01 * bruit), 1.0)) ** (1 / gq), 3 * RT))
            if R2 > RT:
                qy0, qx0 = max(0, yi - R2), max(0, xi - R2)
                qy1, qx1 = min(Ch, yi + R2 + 1), min(Cw, xi + R2 + 1)
                if qy0 < qy1 and qx0 < qx1:
                    gy, gx = np.mgrid[qy0 - yi:qy1 - yi, qx0 - xi:qx1 - xi]
                    rq = np.hypot(gx, gy).astype(np.float32)
                    ramp = np.clip((rq - 0.85 * RT) / (0.15 * RT), 0, 1) * np.clip((R2 - rq) / (0.15 * R2), 0, 1)
                    q = (np.maximum(rq, 1.0) / r_a) ** (-gq) * ramp
                    stars[:, qy0:qy1, qx0:qx1] += q[None] * (Fi * parts[:, i] * bd)[:, None, None]
        if satur[j]:
            if Lmap is None:
                Lmap = np.full((3, Ch, Cw), np.inf, np.float32)
            r2 = int(3 * fw + 20)
            ty0, tx0, ty1, tx1 = max(0, yi - r2), max(0, xi - r2), min(Ch, yi + r2 + 1), min(Cw, xi + r2 + 1)
            sub = stars[:, ty0:ty1, tx0:tx1]
            foot = sub > 0.5 * lv[j][:, None, None]
            ls = Lmap[:, ty0:ty1, tx0:tx1]
            ls[foot] = np.minimum(ls, np.broadcast_to(lv[j][:, None, None], ls.shape))[foot]
    st = stars[:, mg:mg + Hh, mg:mg + W]
    # 3) bruit de photons des étoiles (le fond garde le sien)
    if m_["photon_k"] > 0:
        st = st + np.sqrt(np.clip(st, 0, None) * m_["photon_k"] * bruit) * rng.standard_normal(st.shape, dtype=np.float32)
    entree = fond + st
    if Lmap is not None:
        L = Lmap[:, mg:mg + Hh, mg:mg + W]
        m = np.isfinite(L)
        if m.any():
            s = m_["bord"] * L[m]
            entree[m] = -s * np.logaddexp(-entree[m] / s, -L[m] / s)             # écrêtage à bord doux
    return entree.astype(np.float32), fond.copy()


# ------------------------------------------------------------------------------------------------ cœurs étendus (N7)
FACTEUR_LARGEUR = 2.5     # largeur à mi-hauteur du cœur >= 2,5 x celle des étoiles de la tuile


def _fwhm_equivalente(img):
    """Largeur à mi-hauteur équivalente : diamètre du disque de même surface que la zone au-dessus de la moitié du maximum."""
    return 2.0 * math.sqrt(float((img >= 0.5 * img.max()).sum()) / math.pi)


def coeur_synthetique(shape, rng, fw_etoile, bruit=1.0, photon_k=0.0, compact=False):
    """Un cœur étendu compact (3, H, W) : profil de Sérsic n 1-4, rayon effectif 4-60 px, ellipticité 0-0,6, éclat central 50-5000 bruits
    (log-uniforme), couleur de galaxie (jaune-blanc) ou de planétaire (bleu-vert), parfois écrêté (cœur saturé, plateau).
    Lissé par un noyau gaussien au besoin pour que sa largeur à mi-hauteur dépasse FACTEUR_LARGEUR x celle des étoiles (fw_etoile).
    Retourne (coeur, info)."""
    Hh, W = shape
    if compact:                                                 # N9 : cœur COMPACT (comme M32), Sérsic pointu, largeur 1,3-1,5 x celle des étoiles
        n = float(rng.uniform(3.0, 6.0))
        re = float(np.exp(rng.uniform(math.log(1.5), math.log(6.0))))
    else:
        n = float(rng.uniform(1.0, 4.0))
        re = float(np.exp(rng.uniform(math.log(4.0), math.log(60.0))))
    ell = float(rng.uniform(0.0, 0.6))
    th = float(rng.uniform(0, math.pi))
    pic = float(np.exp(rng.uniform(math.log(50.0), math.log(5000.0))))
    cx, cy = float(rng.uniform(0.15, 0.85) * W), float(rng.uniform(0.15, 0.85) * Hh)
    bn = 2.0 * n - 1.0 / 3.0 + 0.009876 / n
    yy, xx = np.mgrid[0:Hh, 0:W].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    u = dx * math.cos(th) + dy * math.sin(th)
    v = -dx * math.sin(th) + dy * math.cos(th)
    r = np.sqrt(u * u + (v / (1.0 - ell)) ** 2 + 0.25)
    prof = np.exp(-bn * (r / re) ** (1.0 / n)).astype(np.float32)
    if compact:
        cible_fw = float(rng.uniform(1.3, 1.5)) * fw_etoile
    else:
        cible_fw = FACTEUR_LARGEUR * fw_etoile * float(rng.uniform(1.0, 1.4))
    sig = 0.0
    p = prof
    for _ in range(40):                                        # lissage croissant jusqu'à la largeur voulue
        if _fwhm_equivalente(p) >= cible_fw:
            break
        sig = max(sig * 1.25, 0.4 * fw_etoile) if not compact else max(sig * 1.1, 0.1 * fw_etoile)   # compact : pas de lissage plus fin
        p = cv2.GaussianBlur(prof, (0, 0), sig)
    p = p / float(p.max())
    if rng.random() < 0.5:                                      # galaxie jaune-blanc ou planétaire bleu-vert (rouge, vert, bleu)
        col = np.array([rng.uniform(0.95, 1.1), 1.0, rng.uniform(0.65, 0.9)], np.float32)
        kind = "galaxie"
    else:
        col = np.array([rng.uniform(0.15, 0.5), 1.0, rng.uniform(0.8, 1.15)], np.float32)
        kind = "planetaire"
    coeur = pic * bruit * col[:, None, None] * p[None]
    if photon_k > 0:
        coeur = coeur + np.sqrt(np.clip(coeur, 0, None) * photon_k * bruit) * rng.standard_normal(coeur.shape, dtype=np.float32)
    sature = bool(rng.random() < 0.3)
    if sature:                                                  # écrêté au maximum : plateau, bord doux
        niv = (pic * bruit * col * float(rng.uniform(0.25, 0.7)))[:, None, None]
        s = 0.04 * niv
        coeur = -s * np.logaddexp(-coeur / s, -niv / s)
    info = dict(compact=compact, sersic_n=n, re=re, ell=ell, pic=pic, sature=sature, type=kind, fwhm=_fwhm_equivalente(p), sigma_lissage=sig,
                x=cx, y=cy)
    return coeur.astype(np.float32), info


PART_COMPACTS = 0.40      # N9 : part des cœurs synthétiques compacts


def paire_synthetique_coeurs(fond, graine, bruit=1.0, extras=None, p_coeurs=0.25, part_compacts=PART_COMPACTS):
    """Comme paire_synthetique, mais sur une part p_coeurs des tuiles, 1 ou 2 cœurs étendus sont ajoutés au fond : ils sont dans
    l'entrée ET dans la cible (on les garde) ; les étoiles synthétiques restent posées par-dessus et retirées dans la cible comme avant.
    Retourne (entree, cible, coeur, infos) ; coeur = (3, H, W) additionné à fond dans les deux (zéros si aucune tuile à cœur)."""
    rng = np.random.default_rng([int(graine) & 0x7FFFFFFF, 424242])        # tirage à part : le flux des étoiles ne change pas
    if rng.random() >= p_coeurs:
        e, c = paire_synthetique(fond, graine, bruit=bruit, extras=extras)
        return e, c, np.zeros_like(fond), []
    S = _charger()
    k = int(np.random.default_rng(graine).integers(len(S["meta"])))        # même style que paire_synthetique (premier tirage)
    m_ = S["meta"][k]
    Hh, W = fond.shape[-2:]
    coeur = np.zeros(fond.shape, np.float32)
    infos = []
    rng_c = np.random.default_rng([int(graine) & 0x7FFFFFFF, 777])         # N9 : choix compact / étendu à part, le tirage des 60 % étendus ne change pas
    for _ in range(int(rng.integers(1, 3))):
        compact = bool(rng_c.random() < part_compacts)
        c_, i_ = coeur_synthetique((Hh, W), rng if not compact else rng_c, m_["fwhm"], bruit, m_["photon_k"], compact=compact)
        coeur += c_
        infos.append(i_)
    e, c = paire_synthetique((fond + coeur).astype(np.float32), graine, bruit=bruit, extras=extras)
    return e, c, coeur, infos


def vitesse():
    import audit_etoiles as A
    _charger()
    sys.path.insert(0, str(HERE))
    tiles = []
    for d, t in (("Methode v22 M57 – 22 nov", "d_fond_calme"), ("Methode v21 SHO – 22 nov", "decoupe6")):
        p = A.SR / d / "sorties" / t / "B" / "t3" / "sans_etoiles_tuile.npy"
        tiles.append(np.load(p).astype(np.float32))
    res = {}
    for n in (256, 512):
        ts = []
        for g in range(60):
            f = np.ascontiguousarray(tiles[g % 2][:, :n, :n])
            t0 = time.perf_counter()
            e, c = paire_synthetique(f, [6, n, g])
            ts.append(time.perf_counter() - t0)
            assert np.array_equal(c, f)
        ts = np.array(ts[5:]) * 1000
        res[n] = (float(np.median(ts)), float(np.percentile(ts, 90)), float(ts.max()))
        print(f"tuile {n} : médiane {res[n][0]:.1f} ms, 90e centile {res[n][1]:.1f} ms, max {res[n][2]:.1f} ms",
              flush=True)
    # reproductibilité
    f = np.ascontiguousarray(tiles[0][:, :256, :256])
    assert np.array_equal(paire_synthetique(f, 7)[0], paire_synthetique(f, 7)[0])
    # planche : 12 paires 256 (entrée | cible), arcsinh 60
    rows = []
    for g in range(12):
        f = np.ascontiguousarray(tiles[g % 2][:, :256, :256])
        e, c = paire_synthetique(f, [9, g])
        bk = float(np.median(c[1])); wt = float(np.percentile(e[1], 99.8))
        p1 = (A.stretch60(e, bk, wt) * 255).astype(np.uint8); p2 = (A.stretch60(c, bk, wt) * 255).astype(np.uint8)
        rows.append(np.concatenate([p1, np.full((256, 4, 3), 255, np.uint8), p2], 1))
    grid = [np.concatenate([rows[i], np.full((256, 12, 3), 255, np.uint8), rows[i + 1],
                            np.full((256, 12, 3), 255, np.uint8), rows[i + 2]], 1) for i in range(0, 12, 3)]
    sheet = np.concatenate(sum([[g_, np.full((12, grid[0].shape[1], 3), 255, np.uint8)] for g_ in grid], [])[:-1], 0)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)); assert ok
    buf.tofile(str(A.OUT / "Synth_20_paires_rapides.png"))
    return res


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    if len(sys.argv) > 1 and sys.argv[1] == "cache":
        construire_cache(int(sys.argv[2]) if len(sys.argv) > 2 else 32)
    else:
        vitesse()
