# ======================================================================
# modele_psf_image.py
# Role     : Method v7 (21 Nov): the track 1 star model built on the WHOLE IMAGE, once per image, kept on disk.
# Step     : 2. Classic Gaia-guided subtraction chain (v33b) (see scripts/LISEZMOI.md)
# Inputs   : stacked image (noise units), astrometry, offline deep Gaia extract
# Outputs  : star-free image, class maps
# Example  : python modele_psf_image.py [arguments]  (see the docstring; paths through the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Method v7 (21 Nov): the track 1 star model built on the WHOLE IMAGE, once per image, kept on disk.

Why (Recul – 21 nov soir\\RECUL.md): in v5 the model was built on ~9 stars of the crop, edge forced to zero at
r 13-15, halo 0; the remainder stacked around the isolated stars is negative at r 2-4 and positive at r 6-12 (and still at r 16-22),
in proportion to brightness, of opposite sign for G > 16.5.

Method (sources):
  - iterated effective PSF (Anderson & King 2000; photutils' EPSFBuilder): model -> star fit -> mean
    of the flux-normalised remainders -> added to the model -> start again;
  - model = function + table of deviations (DAOPHOT, Stetson 1987): here the table is the sum of the stacked remainders;
  - extended PSF (Infante-Sainz, Trujillo & Román 2020): the wing beyond ~12 px is taken from brighter
    stars and joined to the core of each slice; radius 28 px; the edge is not brought back to zero (only a
    fade of the last 2 px to avoid a step).
Form:  model(star) = S (starting 2D shape, stacked on the G < 14.5 stars of the whole image)
                          + rho(r ; G) (radial correction per half-magnitude of G, interpolated in G for each star; denoised
                            image: the shape changes with brightness, even inside a 1-1.5 mag slice),
  rho corrects the WING only (r >= 5 px, full at 7 px); beyond 12 px drawn from the stars of the slice AND of the
  brighter slices (extended wing). The CORE (r < 5 px) does not converge in this iteration (tests of 21 Nov: overshoot
  then blocking at -10 / +6 noises, see JOURNAL): it is left to the core correction of track 1 on each crop.
Stars: deep Gaia catalogue over the whole image, unsaturated, with no neighbour brighter than G+1 within 24 px nor neighbour
less than 3 mag apart in the core, visible neighbours masked (disc according to brightness), flat local background (ring 29-35 px).
Positions: the track 1 fit (fit_star, position per colour v4); flux of each colour refitted against the sky
far from the star (plane on the 29-35 px ring, like EPSFBuilder); a background taken in the wing made the model drift.
Stop: stacked profiles flat to +-0.1 sky noise in each slice G 12-13.5 / 13.5-14.5 / 14.5-15.5 / 15.5-16.5 and
each colour (r 5-24 px), or 4 updates.
Cache: Piste1/sorties/<image>/modele_psf_image.npz (+ .json: settings, star counts, profiles at each round).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
import images  # noqa: E402
import retrait_modele as RM  # noqa: E402

VERSION = "v7-3"
R_MOD = 28                     # rayon du modèle (px)
BG_RING = (29.0, 35.0)         # anneau du fond local (au-delà du modèle)
RB = 36                        # demi-fenêtre des restes
EDGES = np.r_[-np.inf, 11.0, np.arange(11.5, 17.01, 0.5), np.inf]   # tranches du modèle : demi-magnitudes G
CHECK = ((12.0, 13.5), (13.5, 14.5), (14.5, 15.5), (15.5, 16.5))     # tranches contrôlées (celles de RECUL)
R_CORE = 12.0                  # (terme régional 2D : abandonné, voir AILE)
AILE = (5.0, 7.0)              # le modèle de l'image corrige r >= 5 px (plein à 7 px) ; le coeur est laissé à la correction du
                               # coeur de la piste 1 sur la découpe (r < 2,5 largeurs ~ 7,5 px)
JOIN = (10.0, 14.0)            # raccord : propre à la tranche avant 10 px, aile des plus brillantes après 14 px
MAX_UPDATES = int(__import__("os").environ.get("MPI_U", 4))
FLAT = 0.1                     # bruits de ciel
CHECK_R = 24                   # rayon contrôlé
PER_BIN_MAX = int(__import__("os").environ.get("MPI_N", 250))   # étoiles par tranche au plus (les plus calmes) ; MPI_N : essai réduit
G_MAX = 17.5


def bin_of(G):
    return np.clip(np.searchsorted(EDGES, np.asarray(G, float), side="right") - 1, 0, len(EDGES) - 2)


def check_of(G):
    for i, (a, b) in enumerate(CHECK):
        if a <= G < b:
            return i
    return -1


def interp_rho(rho_c, gc, G):
    """Profil d'une couleur pour l'éclat G : interpolation linéaire entre les centres des tranches (bornée aux extrêmes)."""
    if G is None:
        return rho_c[-1]
    j = int(np.clip(np.searchsorted(gc, G) - 1, 0, len(gc) - 2))
    t_ = float(np.clip((G - gc[j]) / max(gc[j + 1] - gc[j], 1e-6), 0, 1))
    return (1 - t_) * rho_c[j] + t_ * rho_c[j + 1]


class EPSFImage(RM.EPSF):
    """Modèle v7 : S (C, ng, n, n) par région + rho (C, nb, n, n) par tranche d'éclat ; blend(c, x, y, G)."""

    def __init__(self, S, rho, gc, centres, os_, R, halo, R_ancien):
        C, ng = S.shape[:2]
        super().__init__([[S[c, g] for g in range(ng)] for c in range(C)], centres, os_, R, halo)
        self.rho = rho
        self.gc = gc
        self.R_ancien = R_ancien

    def blend(self, c, x, y, G=None):
        return RM.EPSF.blend(self, c, x, y) + interp_rho(self.rho[c], self.gc, G)


def _taper_edge(n, os_, R):
    c0 = (n - 1) // 2
    rr = np.hypot(*np.mgrid[-c0:c0 + 1, -c0:c0 + 1]) / os_
    t = np.clip((R - rr) / 2.0, 0, 1)
    return rr, 0.5 - 0.5 * np.cos(np.pi * t)          # 1 jusqu'à R-2, 0 à R (fondu en cosinus)


def _radialize(P_, rr, a, b):
    """Au-delà de a px : moyenne azimutale (fondue de a à b), comme la v5."""
    prof_r = np.arange(0, rr.max() + 0.51, 0.25)
    k = np.digitize(rr, prof_r)
    prof = ndi.mean(P_, labels=k, index=np.arange(1, len(prof_r) + 1))
    prof = np.nan_to_num(np.asarray(prof, float))
    rad = np.interp(rr, prof_r[:len(prof)] + 0.125, prof)
    t = np.clip((rr - a) / max(b - a, 1e-6), 0, 1)
    return (1 - t) * P_ + t * rad


def construire(name, log=print):
    t0 = time.time()
    img = images.load(name)
    C, H, W = img.shape
    valid = ndi.binary_erosion(np.any(img != 0, axis=0), iterations=3)
    xg, yg, G = RM.catalogue_image(name, log)       # SHO 21 nov : profond, sinon extrait hors ligne
    log(f"[modèle image] image {img.shape}, catalogue profond {len(G)} étoiles")
    # saturation (même règle que la piste 1, sur l'image entière)
    inb = (xg > 3) & (xg < W - 3) & (yg > 3) & (yg < H - 3)
    brightest = np.argsort(np.where(inb, G, 99))[:25]
    sat_lvl = np.zeros(C)
    for c in range(C):
        pk_ = np.array([img[c, int(yg[k]) - 2:int(yg[k]) + 3, int(xg[k]) - 2:int(xg[k]) + 3].max() for k in brightest])
        top_ = np.sort(pk_)[::-1][:12]
        plateau = len(top_) >= 5 and np.std(top_[1:]) / max(np.median(top_), 1e-9) < 0.12
        sat_lvl[c] = float(np.median(top_[1:])) * RM.P["sat_frac"] if plateau else np.inf
    sat = np.stack([ndi.binary_dilation(img[c] >= sat_lvl[c], iterations=2) for c in range(C)])
    # bruit (même mesure que la piste 1)
    star_mask = np.zeros((H, W), bool)
    for x, y, g in zip(xg, yg, G):
        if g > 19:
            continue
        rr = int(np.clip(3 + 2.5 * (16 - g), 3, 60))
        cv2.circle(star_mask.view(np.uint8), (int(round(x)), int(round(y))), rr, 1, -1)
    sig, ncorr = RM.noise_stats(img, star_mask | ~valid)
    SIG = np.array([float(np.median(s)) for s in sig])
    log(f"[modèle image] saturation {np.round(sat_lvl, 1)}, bruit du ciel médian {np.round(SIG, 3)}, {time.time() - t0:.0f} s")
    # largeur : même règle que la piste 1 (étoiles de pic > psf_snr bruits et < psf_lin x niveau de référence, 400 premières
    # du catalogue), pour que les rayons d'ajustement (2 largeurs, anneau de fond) soient ceux de la piste 1
    L = img.mean(0)
    inside = (xg > RB + 2) & (xg < W - RB - 3) & (yg > RB + 2) & (yg < H - RB - 3)
    pk = np.zeros(len(G))
    pk[inside] = L[np.round(yg[inside]).astype(int), np.round(xg[inside]).astype(int)]
    ref_lvl = np.array([sat_lvl[c] if np.isfinite(sat_lvl[c]) else float(img[c].max()) for c in range(C)])
    fw = []
    for k in np.where(inside & (pk > RM.P["psf_snr"] * float(np.median(sig))) & (pk < RM.P["psf_lin"] * ref_lvl.min()))[0][:400]:
        xi, yi = int(round(xg[k])), int(round(yg[k]))
        cut = L[yi - 8:yi + 9, xi - 8:xi + 9]
        b = np.median(np.r_[cut[0], cut[-1], cut[:, 0], cut[:, -1]])
        v = cut - b
        if v.max() > 0:
            fw.append(2 * np.sqrt((v >= v.max() / 2).sum() / np.pi))
    fwhm = float(np.median(fw))
    os_ = 4 if fwhm < 4 else 2
    R = R_MOD
    n = 2 * R * os_ + 1
    c0 = R * os_
    ann_w = max(4.0, 1.5 * fwhm)
    P_fit = dict(RM.P, ann_w=ann_w, choff=np.zeros((C, 2)), colour_pos_snr=RM.COLOUR_POS_SNR, ellipse=False)
    log(f"[modèle image] largeur à mi-hauteur {fwhm:.2f} px ({len(fw)} étoiles), rayon du modèle {R} px, suréchantillonnage {os_}")

    # ---- étoiles du modèle
    tree = cKDTree(np.c_[xg, yg])
    cand = np.where(inside & (G >= 8) & (G < G_MAX))[0]
    # v33b : si aucune étoile ne passe les critères d'isolement (champ très dense, ou largeur mesurée grande : image sans étoile
    # isolée à 2,5 largeurs), les distances d'isolement sont réduites par paliers (1 : critères de v33, inchangés) jusqu'à en trouver.
    for rel in (1.0, 0.7, 0.5, 0.35):
        sel, neigh = [], {}
        rough = []
        yy, xx = np.mgrid[-RB:RB + 1, -RB:RB + 1]
        ang = ((np.arctan2(yy, xx) + np.pi) / (2 * np.pi) * 8).astype(int) % 8
        rr0 = np.hypot(xx, yy)
        ringm = (rr0 >= BG_RING[0]) & (rr0 <= BG_RING[1])
        n_rej = dict(saturee=0, voisine_brillante=0, coeur=0, bord=0, masque=0)
        for k in cand:
            xi, yi = int(round(xg[k])), int(round(yg[k]))
            if sat[:, yi - 3:yi + 4, xi - 3:xi + 4].any():
                n_rej["saturee"] += 1
                continue
            if not valid[yi - RB:yi + RB + 1, xi - RB:xi + RB + 1].all():
                n_rej["bord"] += 1
                continue
            nb_ = [j for j in tree.query_ball_point([xg[k], yg[k]], RB * 1.42) if j != k]
            bad = False
            lst = []
            for j in nb_:
                d = np.hypot(xg[j] - xg[k], yg[j] - yg[k])
                if (d < 24 * rel and G[j] < G[k] + 1.0) or (d < 2.5 * fwhm * rel and G[j] < G[k] + 3.0):
                    bad = True
                    n_rej["voisine_brillante" if d >= 2.5 * fwhm * rel else "coeur"] += 1
                    break
                if G[j] < 18.5:
                    rm_ = float(np.clip(3 + 2.5 * (16 - G[j]), 3, 25))
                    if d - rm_ < 2.0 * fwhm * rel:
                        bad = True
                        n_rej["masque"] += 1
                        break
                    lst.append((xg[j] - xg[k], yg[j] - yg[k], rm_))
            if bad:
                continue
            cut = L[yi - RB:yi + RB + 1, xi - RB:xi + RB + 1]
            m = np.ones(cut.shape, bool)
            for dx_, dy_, rm_ in lst:
                m &= np.hypot(xx + xi - xg[k] - dx_, yy + yi - yg[k] - dy_) > rm_
            secs = [np.median(cut[ringm & m & (ang == a)]) for a in range(8) if (ringm & m & (ang == a)).sum() > 8]
            if len(secs) < 6:
                n_rej["masque"] += 1
                continue
            sel.append(k)
            neigh[k] = lst
            rough.append(float(np.std(secs)) / (SIG.mean() / np.sqrt(3)))
        if len(sel) > 0:
            break
        log(f"[modèle image] aucune étoile isolée avec les critères au palier {rel} ; écartées {n_rej}")
    sel = np.array(sel, int)
    rough = np.array(rough)
    log(f"[modèle image] DIAG candidates {len(cand)} retenues {len(sel)} écartées {n_rej}")
    if len(sel) == 0:
        raise RuntimeError("aucune étoile pour le modèle")
    bsel = bin_of(G[sel])
    keep = []
    for b in range(len(EDGES) - 1):
        ib = np.where(bsel == b)[0]
        ib = ib[np.argsort(rough[ib])][:PER_BIN_MAX]
        keep.extend(ib.tolist())
    keep = np.sort(np.array(keep))
    sel, rough, bsel = sel[keep], rough[keep], bsel[keep]
    counts = {f"{EDGES[b]}-{EDGES[b + 1]}": int((bsel == b).sum()) for b in range(len(EDGES) - 1)}
    # centre de chaque tranche : G médian de ses étoiles (milieu de la tranche si vide)
    gc = np.array([float(np.median(G[sel][bsel == b])) if (bsel == b).any()
                   else float(np.clip(0.5 * (EDGES[b] + EDGES[b + 1]), 10.5, 17.5)) for b in range(len(EDGES) - 1)])
    gc = np.maximum.accumulate(gc + np.arange(len(gc)) * 1e-6)
    chk = np.array([check_of(G[k]) for k in sel])
    log(f"[modèle image] étoiles retenues {len(sel)} par tranche {counts} ; écartées {n_rej} ; "
        f"rugosité du fond (bruits L) médiane {np.median(rough):.2f}, {time.time() - t0:.0f} s")

    # régions : la plus fine grille g x g (g <= 4) où chaque région a au moins 60 étoiles G < 15.5
    for g in (1,):                 # une région : le coeur (seul à varier dans le champ) est corrigé sur chaque découpe
        ci = np.minimum((xg[sel] / (W / g)).astype(int), g - 1) + g * np.minimum((yg[sel] / (H / g)).astype(int), g - 1)
        nreg = np.bincount(ci[G[sel] < 15.5], minlength=g * g)
        if nreg.min() >= 60 or g == 1:
            break
    reg = ci
    ng = g * g
    centres = np.array([((j + 0.5) * W / g, (i + 0.5) * H / g) for i in range(g) for j in range(g)])
    log(f"[modèle image] régions du coeur : {g}x{g}, étoiles G < 15.5 par région {nreg.tolist()}")

    rr, taper = _taper_edge(n, os_, R)
    E1 = RM.EPSF([[None]] * C, np.zeros((1, 2)), os_, R, [(0.0, 3.0, R)] * C)

    def nmask(k, xi, yi, xs, ys):
        m = valid[yi - RB:yi + RB + 1, xi - RB:xi + RB + 1].copy()
        for dx_, dy_, rm_ in neigh[k]:
            m &= np.hypot(xx + xi - xg[k] - dx_, yy + yi - yg[k] - dy_) > rm_
        return m

    # ---- modèle de départ : empilement des étoiles G < 14.5 (positions Gaia affinées par barycentre), normé au flux
    def stack_initial(ks, pos):
        acc = np.zeros((C, n * n)); cnt = np.zeros((C, n * n))
        for k in ks:
            x, y = pos[k]
            xi, yi = int(round(x)), int(round(y))
            m = nmask(k, xi, yi, x, y)
            cut = img[:, yi - RB:yi + RB + 1, xi - RB:xi + RB + 1].astype(np.float64)
            dx = xx + xi - x; dy = yy + yi - y
            r = np.hypot(dx, dy)
            ring = ringm & m
            for c in range(C):
                v = cut[c] - np.median(cut[c][ring])
                f = v[(r <= R) & m].sum() / max(((r <= R) & m).mean() / (r <= R).mean(), 0.3)
                if f <= 0:
                    continue
                ix = np.round(dx * os_).astype(int) + c0; iy = np.round(dy * os_).astype(int) + c0
                ok = (r <= R + 0.5) & m & (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n) & ~sat[c, yi - RB:yi + RB + 1, xi - RB:xi + RB + 1]
                np.add.at(acc[c], iy[ok] * n + ix[ok], v[ok] / f); np.add.at(cnt[c], iy[ok] * n + ix[ok], 1)
        S0 = np.zeros((C, n, n))
        for c in range(C):
            P_ = (acc[c] / np.maximum(cnt[c], 1)).reshape(n, n)
            hole = cnt[c].reshape(n, n) == 0
            if hole.any():
                P_[hole] = ndi.uniform_filter(P_, 3)[hole]
            P_ = ndi.gaussian_filter(P_, 0.6 * os_ / 2)
            fw_os = RM._fwhm_of(P_, os_)
            P_ = _radialize(P_, rr, 2.0 * fw_os, 3.5 * fw_os) * taper * (rr <= R)
            S0[c] = P_ / (P_.sum() / os_ ** 2)
        return S0

    pos = {}
    for k in sel:
        xi, yi = int(round(xg[k])), int(round(yg[k]))
        w_ = np.clip(L[yi - 2:yi + 3, xi - 2:xi + 3] - np.median(L[yi - 6:yi + 7, xi - 6:xi + 7]), 0, None)
        s_ = w_.sum()
        gy_, gx_ = np.mgrid[-2:3, -2:3]
        pos[k] = (xi + (w_ * gx_).sum() / s_, yi + (w_ * gy_).sum() / s_) if s_ > 0 else (xg[k], yg[k])
    init = [k for k in sel if G[k] < 14.5]
    S0 = stack_initial(init, pos)

    # ---- ajustement d'une étoile (fit_star de la piste 1) et ses restes
    zp = [None]
    med_sig = SIG

    def fit_and_residual(k, stamps):
        x, y = pos[k]
        xi, yi = int(round(x)), int(round(y))
        Fp = 10 ** (-0.4 * (G[k] - zp[0])) if zp[0] is not None else np.inf
        rf = max(3.0, 2.0 * fwhm) * (1.6 if Fp > 200 * med_sig.min() * fwhm ** 2 else 1.0)
        rf = min(rf, max(3.0, RB - 1 - ann_w))     # v33b : la fenêtre (rf + anneau) tient dans la vignette du masque (demi-côté RB)
        rw = int(np.ceil(rf + ann_w)) + 1
        m = nmask(k, xi, yi, x, y)
        sl = (slice(None), slice(yi - rw, yi + rw + 1), slice(xi - rw, xi + rw + 1))
        mw = m[RB - rw:RB + rw + 1, RB - rw:RB + rw + 1]
        co_out = [None]
        F, nx, ny, s, chi = RM.fit_star(img[sl], mw, sat[sl], E1, stamps, x - (xi - rw), y - (yi - rw), None, 1.0, rf,
                                        False, False, sig[:, yi, xi], xg[k] - (xi - rw), yg[k] - (yi - rw),
                                        dict(P_fit, _co_out=co_out))
        nx += xi - rw; ny += yi - rw
        co = co_out[0] if co_out[0] is not None else np.zeros((C, 2))
        if not (np.isfinite(nx) and np.isfinite(ny)):      # 22 nov (58_hoo) : ajustement sans solution -> étoile écartée
            return None
        xi, yi = int(round(nx)), int(round(ny))
        if xi - RB < 0 or yi - RB < 0 or xi + RB >= W or yi + RB >= H:
            return None
        m = nmask(k, xi, yi, nx, ny)
        cut = img[:, yi - RB:yi + RB + 1, xi - RB:xi + RB + 1].astype(np.float64)
        dx = xx + xi - nx; dy = yy + yi - ny
        r = np.hypot(dx, dy)
        inner = r <= R + 1
        res = np.zeros_like(cut); okc = np.zeros(cut.shape, bool)
        sk = sig[:, yi, xi]
        F = np.array(F, float)
        for c in range(C):
            mod = np.zeros(r.shape)
            mod[inner] = E1.eval(stamps[c], dx[inner] - co[c, 0], dy[inner] - co[c, 1], 1.0)
            ok = m & ~sat[c, yi - RB:yi + RB + 1, xi - RB:xi + RB + 1]
            # fond = plan sur l'anneau 29-35 px, au-delà du modèle (le ciel loin de l'étoile, comme EPSFBuilder / DAOPHOT) ;
            # le flux de la couleur est réajusté contre ce fond (même coeur r <= rf que la piste 1, positions de la piste 1) :
            # un fond pris dans l'aile (anneau 10-14 px de la piste 1) ferait dériver le modèle (aile négative, flux qui monte)
            ring = ok & ringm
            Ab = np.c_[np.ones(ring.sum()), dx[ring] / RB, dy[ring] / RB]
            kp = np.ones(ring.sum(), bool)
            for _ in range(3):
                bs = np.linalg.lstsq(Ab[kp], cut[c][ring][kp], rcond=None)[0]
                rb = cut[c][ring] - __import__('operator').matmul(Ab, bs)
                sd = 1.4826 * np.median(np.abs(rb[kp])) + 1e-9
                kp = np.abs(rb) < 3 * sd
            d_ = cut[c] - (bs[0] + bs[1] * dx / RB + bs[2] * dy / RB)
            cr = ok & (np.hypot(dx - co[c, 0], dy - co[c, 1]) <= rf)
            if cr.sum() >= 6 and (mod[cr] ** 2).sum() > 0:
                F[c] = float((mod[cr] * d_[cr]).sum() / (mod[cr] ** 2).sum())
            v = d_ - F[c] * mod
            ok &= ~((r > 2.5 * fwhm) & (np.abs(v) > 5 * sk[c]))
            res[c] = v; okc[c] = ok
        return dict(k=k, x=nx, y=ny, F=F, co=co, dx=dx, dy=dy, r=r, res=res, resf=res, ok=okc, sk=sk)

    # ---- itérations
    rbins = np.arange(0, R + RB, 0.25)               # profil radial (0.25 px)
    nrb = len(rbins)
    S = np.repeat(S0[:, None], ng, axis=1)          # (C, ng, n, n)
    rho = np.zeros((C, len(EDGES) - 1, nrb))         # corrections radiales par tranche (sur rbins)
    hist = []
    nb_ = len(EDGES) - 1

    def model_stamps(g_, Gk):
        return [S[c, g_] + np.interp(rr, rbins, interp_rho(rho[c], gc, Gk)) * taper * (rr <= R) for c in range(C)]

    it_done = 0
    for it in range(MAX_UPDATES + 1):
        ti = time.time()
        # étoile -> modèle de sa région et de sa tranche (au plus proche centre de région, pour l'empilement)
        prof = {b: [] for b in range(len(CHECK))}
        rad = []                         # par étoile et couleur : (couleur, tranche, région, somme e, nombre, poids)
        samples = []
        Ft_list = []
        for i_, k in enumerate(sel):
            out = fit_and_residual(k, model_stamps(reg[i_], G[k]))
            if out is None:
                continue
            pos[k] = (out["x"], out["y"])
            F = out["F"]
            Ft_list.append((G[k], float(F.sum()), float(F[1]), bsel[i_]))
            r = out["r"]
            ri = np.clip((r / 0.25).astype(int), 0, nrb - 1)
            r1 = np.clip(r.astype(int), 0, RB)
            p_ = np.full((C, RB + 1), np.nan)
            for c in range(C):
                ok = out["ok"][c]
                v = out["res"][c]
                cnt_ = np.bincount(r1[ok], minlength=RB + 1)
                sm_ = np.bincount(r1[ok], weights=v[ok], minlength=RB + 1)
                p_[c] = np.where(cnt_ > 0, sm_ / np.maximum(cnt_, 1), np.nan) / SIG[c]
                if F[c] <= 0:
                    continue
                w = F[c] ** 2 / out["sk"][c] ** 2
                vf = out["resf"][c]
                # empilement autour du centre de CETTE couleur (position par couleur v4), là où son modèle a été évalué
                dxc = out["dx"] - out["co"][c, 0]; dyc = out["dy"] - out["co"][c, 1]
                rc = np.hypot(dxc, dyc)
                ric = np.clip((rc / 0.25).astype(int), 0, nrb - 1)
                rad.append((c, bsel[i_], reg[i_], np.bincount(ric[ok], weights=vf[ok] / F[c], minlength=nrb), np.bincount(ric[ok], minlength=nrb), w))
                core = ok & (rc <= R_CORE + 0.5)
                ix = np.round(dxc[core] * os_).astype(int) + c0; iy = np.round(dyc[core] * os_).astype(int) + c0
                samples.append((c, reg[i_], bsel[i_], iy * n + ix, rc[core], vf[core] / F[c], w))
            if chk[i_] >= 0:
                prof[chk[i_]].append(p_)
        Ft_list = np.array(Ft_list)
        # poids F^2 / bruit^2 plafonnés à 4 x la médiane de la tranche (sinon 2 ou 3 étoiles très brillantes font la moyenne)
        wcap = {}
        for c in range(C):
            for b in range(nb_):
                ws = [t_[5] for t_ in rad if t_[0] == c and t_[1] == b]
                wcap[c, b] = 4.0 * float(np.median(ws)) if ws else 0.0
        Sr = np.zeros((C, nb_, nrb)); Wr = np.zeros((C, nb_, nrb))
        Wcell = np.zeros((C, ng, nb_))
        for c, b, g_, sk_, nk_, w in rad:
            we = min(w, wcap[c, b])
            Sr[c, b] += we * sk_; Wr[c, b] += we * nk_
        for c, g_, b_, idx_, r_, e_, w in samples:
            Wcell[c, g_, b_] += min(w, wcap[c, b_]) * len(e_)
        if zp[0] is None and len(Ft_list):
            okz = Ft_list[:, 1] > 0
            zp[0] = float(np.median(Ft_list[okz, 0] + 2.5 * np.log10(Ft_list[okz, 1])))
        # contrôle : profils empilés par tranche (bruits de ciel), r 0..CHECK_R
        stat = {}
        worst = 0.0
        for b in range(len(CHECK)):
            if not prof[b]:
                continue
            a = np.array(prof[b])
            mn = np.nanmean(a, 0); se = np.nanstd(a, 0) / np.sqrt(np.isfinite(a).sum(0).clip(1))
            stat[b] = dict(n=len(a), moy=np.round(mn[:, :CHECK_R + 1], 3).tolist(), se=np.round(se[:, :CHECK_R + 1], 3).tolist())
            worst = max(worst, float(np.nanmax(np.abs(mn[:, int(AILE[0]):CHECK_R + 1]))))
        hist.append(dict(tour=it, pire=round(worst, 3), zp=zp[0], profils={f"{CHECK[b][0]}-{CHECK[b][1]}": s for b, s in stat.items()},
                         secondes=round(time.time() - ti, 1)))
        lines = []
        for b in range(len(CHECK)):
            if b in stat:
                mn = np.array(stat[b]["moy"])
                lines.append(f"G {CHECK[b][0]}-{CHECK[b][1]} ({stat[b]['n']}) r0-4 " + f"{mn[:, 0].mean():+.2f}/"
                             + "/".join(f"{mn[:, r_].mean():+.2f}" for r_ in (1, 2, 3, 4))
                             + " r6-12 " + "/".join(f"{mn[:, r_].mean():+.2f}" for r_ in (6, 8, 10, 12))
                             + " r16,20 " + "/".join(f"{mn[:, r_].mean():+.2f}" for r_ in (16, 20)))
        st_dbg = []
        for gk in (12.75, 14.0, 15.0, 16.0):
            ms = model_stamps(ng // 2, gk)[1]
            st_dbg.append(f"G{gk}: somme {ms.sum() / os_ ** 2:.4f} pic {ms[c0, c0]:.5f}")
        log("[modèle image]   modèle (V, région centrale) : " + " ; ".join(st_dbg))
        log(f"[modèle image] tour {it} : pire |reste| tranches 12-16.5, r {int(AILE[0])}-{CHECK_R} : {worst:.3f} bruit, "
            f"{time.time() - ti:.0f} s ; moyenne des 3 couleurs : " + " | ".join(lines))
        it_done = it
        if worst <= FLAT or it == MAX_UPDATES:
            break
        # ---- mise à jour : radiale par tranche (aile des plus brillantes au-delà de 14 px), puis coeur 2D par région
        drho = np.zeros_like(rho)
        for c in range(C):
            for b in range(nb_):
                own = np.where(Wr[c, b] > 0, Sr[c, b] / np.maximum(Wr[c, b], 1e-30), 0.0)
                ext_S = Sr[c, :b + 1].sum(0); ext_W = Wr[c, :b + 1].sum(0)
                ext = np.where(ext_W > 0, ext_S / np.maximum(ext_W, 1e-30), 0.0)
                own_s = np.where(rbins < 4, ndi.gaussian_filter1d(own, 0.5), ndi.gaussian_filter1d(own, 2.0))
                ext_s = ndi.gaussian_filter1d(ext, 4.0)
                t = np.clip((rbins - JOIN[0]) / (JOIN[1] - JOIN[0]), 0, 1)
                d = (1 - t) * own_s + t * ext_s
                d[rbins > R] = 0.0
                d *= np.clip((rbins - AILE[0]) / (AILE[1] - AILE[0]), 0, 1)     # coeur non touché
                drho[c, b] = d
        accS = np.zeros((C, ng, n * n)); accW = np.zeros((C, ng, n * n)); accN = np.zeros((C, ng, n * n))
        for c, g_, b_, idx, r_, e_, w in samples:
            e2 = e_ - np.interp(r_, rbins, drho[c, b_])
            w = min(w, wcap[c, b_]) / max(wcap[c, b_], 1e-30)      # terme régional : chaque étoile compte à peu près autant
            np.add.at(accS[c, g_], idx, e2 * w); np.add.at(accW[c, g_], idx, w); np.add.at(accN[c, g_], idx, 1)
        coref = np.clip((R_CORE - rr) / 2.0, 0, 1)
        for c in range(C):
            for g_ in range(ng):
                D = (accS[c, g_] / np.maximum(accW[c, g_], 1e-30)).reshape(n, n)
                low = accN[c, g_].reshape(n, n) < 3
                D[low] = 0
                # terme régional = forme NON radiale seulement (la partie radiale est celle des tranches)
                lab_ = np.digitize(rr, np.arange(0, R + 1.01, 0.25))
                mr_ = np.asarray(ndi.mean(D, labels=np.where(low, 0, lab_), index=np.arange(1, lab_.max() + 1)), float)
                D = np.where(low, 0.0, D - np.nan_to_num(mr_)[lab_ - 1])
                D = ndi.gaussian_filter(D, 0.6 * os_ / 2) * coref
                pass
        rho += drho
        it_done = it + 1
    secs = time.time() - t0
    rho_img = np.array([[np.interp(rr, rbins, rho[c, b]) * taper * (rr <= R) for b in range(nb_)] for c in range(C)], np.float32)
    info_gc = np.round(gc, 3).tolist()
    info = dict(version=VERSION, image=name, R=R, os=os_, fwhm=fwhm, regions=g, etoiles=int(len(sel)), par_tranche=counts,
                ecartees=n_rej, mises_a_jour=int(min(it_done, MAX_UPDATES)), pire_final=hist[-1]["pire"], plat=hist[-1]["pire"] <= FLAT,
                secondes=round(secs), tranches=EDGES.tolist(), centres_G=info_gc, controle=[list(c_) for c_ in CHECK], bruit=SIG.tolist(), zp=zp[0], tours=hist)
    return dict(S=S.astype(np.float32), rho=rho_img, gc=gc, centres=centres, os=os_, R=R, info=info)


def obtenir(name, log=print):
    """Modèle de l'image (construit une fois, gardé sur disque)."""
    d = HERE / "sorties" / name
    d.mkdir(parents=True, exist_ok=True)
    f = d / "modele_psf_image.npz"
    if f.exists():
        z = np.load(f, allow_pickle=False)
        info = json.loads((d / "modele_psf_image.json").read_text(encoding="utf-8"))
        if info.get("version") == VERSION:
            log(f"[modèle image] relu du cache : {info['etoiles']} étoiles, {info['mises_a_jour']} mises à jour, "
                f"pire reste {info['pire_final']} bruit, bâti en {info['secondes']} s")
            return dict(S=z["S"], rho=z["rho"], gc=z["gc"], centres=z["centres"], os=int(z["os"]), R=int(z["R"]), info=info)
    m = construire(name, log)
    np.savez(f, S=m["S"], rho=m["rho"], gc=m["gc"], centres=m["centres"], os=m["os"], R=m["R"])
    (d / "modele_psf_image.json").write_text(json.dumps(m["info"], ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[modèle image] bâti en {m['info']['secondes']} s, gardé : {f}")
    return m


def epsf_pour_decoupe(name, offset, R_ancien, log=print):
    m = obtenir(name, log)
    C = m["S"].shape[0]
    cen = m["centres"] - np.asarray(offset, float)[None]
    return EPSFImage(m["S"].astype(np.float64), m["rho"].astype(np.float64), m["gc"], cen, m["os"], m["R"],
                     [(0.0, 3.0, float(m["R"]))] * C, R_ancien), m["info"]
