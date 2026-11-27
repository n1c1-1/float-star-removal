# ======================================================================
# retrait_modele.py
# Role     : Star removal by model (no network), 2026-11-20.
# Step     : 2. Classic Gaia-guided subtraction chain (v33b) (see scripts/LISEZMOI.md)
# Inputs   : stacked image (noise units), astrometry, offline deep Gaia extract
# Outputs  : star-free image, class maps
# Example  : python retrait_modele.py <image> [--tag v1]
# ======================================================================
"""Star removal by model (no network), 2026-11-20.

    python retrait_modele.py <image> [--tag v1]

Method (sources):
1. Star positions: astrometry of the image then Gaia DR3 stars (offline extracts) at their position, with their expected
   brightness (astrometrie.py). A nebula knot is not in Gaia: it is not taken for a star.
   Stars fainter than the catalogue's completeness are then searched in the residual, kept only if
   they have the point-like shape of the PSF (sharpness and roundness, Stetson 1987 DAOFIND): an extended knot does not pass.
2. Subtract, do not erase: effective empirical PSF (Anderson & King 2000) measured on the image, per colour and per
   region of the field (interpolated between regions), extended by a radial power-law halo (Racine 1996), fitted
   with a local background plane (Stetson 1987, DAOPHOT); neighbours are fitted together by successive passes on the
   common residual (Gauss-Seidel: at convergence, it is the simultaneous fit of the group).
3. Closed loop per star: residual in rings (core -> halo) against the background just beyond; correction by the shape
   of the residual (flux, position, width, halo) until a flat residual at noise level; final quality score per star.
   Safeguards: shape shared per region (per star: flux, position, width within +-8 %), flux checked against the predicted
   Gaia brightness (a star that wants to be much brighter than predicted eats the nebula: blocked and flagged).
4. What the model cannot do (saturated cores, spikes, non-converged loop): filling by the smooth local
   background plus noise at the image's grain (same power spectrum as the sky).
Outputs: sorties/<image>/<tag>/starless.npy (C, H, W float32, same units as the input), etoiles.csv, journal.txt.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from scipy.ndimage import map_coordinates  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import images  # noqa: E402

DEEP_DIR = HERE.parent / "Catalogue Gaia profond – 20 nov"     # lecture seule


def catalogue_image(name, log=print):
    """SHO 21 nov (plomberie, pas de changement de méthode) : (x, y, G) de l'image entière pour les modèles bâtis sur
    l'image (modele_psf_image, psf_etendue). Catalogue profond s'il existe (comme avant) ; sinon l'extrait hors ligne
    de l'astrométrie en cache (sorties/<image>/astrometrie.npz), comme la piste 1 quand le fichier profond manque."""
    deep = DEEP_DIR / f"gaia_dr3_profond_{name}.npz"
    if deep.exists():
        z = np.load(deep)
        ok_ = np.isfinite(z["Gmag"]) & np.isfinite(z["x_pix"]) & np.isfinite(z["y_pix"])
        return z["x_pix"][ok_].astype(float), z["y_pix"][ok_].astype(float), z["Gmag"][ok_].astype(float)
    cat = np.load(HERE / "sorties" / name / "astrometrie.npz", allow_pickle=True)["cat"].item()
    log(f"pas de catalogue profond pour {name} : extrait hors ligne de l'astrométrie, {len(cat['x'])} étoiles, G <= {float(np.max(cat['g'])):.2f}")
    return np.asarray(cat["x"], float), np.asarray(cat["y"], float), np.asarray(cat["g"], float)

# ----------------------------------------------------------------------------- réglages
P = dict(
    region_grid=3,          # régions de PSF (n x n), réduites si trop peu d'étoiles
    region_px=512,          # côté maximal d'une région de PSF (px)
    os_fine=4,              # suréchantillonnage de la PSF effective (étoiles fines)
    min_psf_stars=12,       # étoiles par région pour une PSF (la correction de coeur la rend ensuite locale)
    psf_snr=40.0,           # pic minimal (bruits) d'une étoile de PSF
    sat_frac=0.85,          # au-dessus de cette part du niveau d'écrêtage d'une couleur : pixel saturé (pas ajusté, rempli)
    passes=3,               # passes Gauss-Seidel
    loop_iters=4,           # tours de boucle fermée par étoile non plate
    width_margin=0.08,      # largeur libre par étoile (+-)
    pos_margin=1.5,         # px autour de la position Gaia (mouvement propre depuis 2016, astrométrie à 0.1-0.2 px)
    flux_guard=2.5,         # x l'éclat prédit (après étalonnage) : au-delà, bloqué et signalé
    halo_level=0.1,         # le halo modélisé jusqu'où il vaut cette part du bruit
    halo_rmax=400,
    q_flat=3.0,             # résidu plat : |moyenne d'anneau| < q_flat x son erreur ...
    q_floor=0.08,           # ... ou < q_floor x bruit du ciel (le fond n'est connu qu'à ça près)
    faint_sigma=6.0,        # détection d'étoiles hors catalogue dans le résidu
    faint_passes=(6.0, 5.0, 4.0, 3.0),   # seuils des passes successives (trouver, ajuster, soustraire, recommencer)
    fill_grow=1.5,          # px ajoutés autour d'une zone remplie
    psf_lin=0.15,           # pic maximal (part du niveau de saturation) d'une étoile de PSF
    profile_sat=0.12,
    giant_G=4.5,            # plus brillante : géante laissée entière, zone exclue
    giant_sat_area=60.0,    # zone saturée sans étoile Gaia plus grande que ça (x largeur^2) : géante hors catalogue       # pic prédit (part du niveau de saturation) au-delà duquel une étoile prend son propre profil
    fail_fill_fwhm=3.0,     # boucle non convergée : remplissage limité à ce rayon (largeurs) au-delà du coeur saturé
    # v14 (SHO 21 nov) : saturation par couleur aussi lue contre le MAXIMUM de chaque couleur sur l'image entière :
    # pixel saturé si >= (1 - sat_eps) x max de sa couleur ET dans un plateau connexe (8 voisins) d'au moins sat_min_px.
    # sat_eps 8 % : après empilement le niveau d'écrêtage varie d'une étoile à l'autre (M8 SHO : sommets des coeurs
    # écrêtés à 93,5-100 % du maximum de la couleur ; à 0,5-1 % seule l'étoile du maximum est vue).
    sat_eps=0.08,
    sat_min_px=9,
)


def plateaux_satures(full, eps, min_px):
    """v14 : par couleur, niveau (1 - eps) x maximum de la couleur sur l'image entière (zone valide) et masque des
    plateaux connexes d'au moins min_px pixels au-dessus de ce niveau ; niveau inf si aucun plateau."""
    Cf = full.shape[0]
    vf = ndi.binary_erosion(np.any(full != 0, axis=0), iterations=3)
    lvl = np.full(Cf, np.inf)
    masks = np.zeros(full.shape, bool)
    info = []
    for c in range(Cf):
        mx = float(full[c][vf].max())
        m = (full[c] >= (1 - eps) * mx) & vf
        lab, n = ndi.label(m, structure=np.ones((3, 3), int))
        sz = np.bincount(lab.ravel(), minlength=n + 1)
        keep = sz >= min_px
        keep[0] = False
        if keep[1:].any():
            lvl[c] = (1 - eps) * mx
            masks[c] = keep[lab]
        info.append((round(mx, 1), int(keep[1:].sum()), int(masks[c].sum())))
    return lvl, masks, info


def log_to(path):
    f = open(path, "a", encoding="utf-8")

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        f.write(time.strftime("%H:%M:%S ") + s + "\n")
        f.flush()
    return log


def blur(a, s):
    return cv2.GaussianBlur(a.astype(np.float32), (0, 0), float(s))


# ----------------------------------------------------------------------------- bruit et fond
def noise_stats(img, star_mask):
    """Bruit du ciel par couleur (carte lissée, blocs de 64 px) et aire de corrélation du grain (px)."""
    C, H, W = img.shape
    sig = np.zeros_like(img)
    ncorr = []
    for c in range(C):
        hp = img[c] - blur(img[c], 6)
        hp = np.where(star_mask, np.nan, hp)
        B = 64
        hb, wb = H // B, W // B
        blk = hp[:hb * B, :wb * B].reshape(hb, B, wb, B).transpose(0, 2, 1, 3).reshape(hb, wb, -1)
        with np.errstate(all="ignore"):
            med = np.nanmedian(blk, axis=2)
            mad = np.nanmedian(np.abs(blk - med[..., None]), axis=2) * 1.4826
        good = np.isfinite(mad)
        if not good.any():
            mad[:] = 1.0
        else:
            mad[~good] = np.nanmedian(mad[good])
        # le passe-haut perd une part du bruit corrélé : corrigé plus bas par la mesure du grain
        mad = ndi.median_filter(mad, 3)
        sig[c] = cv2.resize(mad.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        # grain : autocorrélation d'un passe-haut sur une zone calme
        ys, xs = np.unravel_index(np.argsort(mad.ravel())[: max(1, mad.size // 10)], mad.shape)
        acfs = []
        for yy, xx in list(zip(ys, xs))[:12]:
            p = hp[yy * B:(yy + 1) * B, xx * B:(xx + 1) * B]
            if np.isnan(p).mean() > 0.2:
                continue
            p = np.nan_to_num(p - np.nanmean(p))
            F = np.fft.fft2(p)
            acf = np.real(np.fft.ifft2(np.abs(F) ** 2))
            acf = np.fft.fftshift(acf) / acf[0, 0]
            acfs.append(acf)
        if acfs:
            acf = np.mean(acfs, 0)
            k = acf.shape[0] // 2
            core = acf[k - 6:k + 7, k - 6:k + 7]
            ncorr.append(float(np.clip(core[core > 0].sum(), 1.0, 50.0)))
        else:
            ncorr.append(1.0)
    return sig, ncorr


def grain_kernel(img, star_mask, size=64):
    """Noyau qui donne au bruit blanc le spectre de puissance du ciel calme de l'image (par couleur)."""
    C, H, W = img.shape
    out = []
    for c in range(C):
        hp = img[c] - blur(img[c], 8)
        best = None
        rng = np.random.default_rng(0)
        for _ in range(200):
            y = int(rng.integers(0, H - size)); x = int(rng.integers(0, W - size))
            if star_mask[y:y + size, x:x + size].any():
                continue
            p = hp[y:y + size, x:x + size]
            v = float(np.std(p))
            if best is None or v < best[0]:
                best = (v, p.copy())
        if best is None:
            out.append(None)
            continue
        ps = []
        for _ in range(60):
            y = int(rng.integers(0, H - size)); x = int(rng.integers(0, W - size))
            if star_mask[y:y + size, x:x + size].mean() > 0.02:
                continue
            p = hp[y:y + size, x:x + size]
            ps.append(np.abs(np.fft.fft2((p - p.mean()) * np.hanning(size)[:, None] * np.hanning(size)[None])) ** 2)
        psd = np.median(ps, 0) if ps else np.abs(np.fft.fft2(best[1] - best[1].mean())) ** 2
        amp = np.sqrt(psd)
        amp /= np.sqrt((amp ** 2).mean())
        out.append(amp.astype(np.float32))
    return out


def synth_grain(amp, shape, rng):
    """Bruit de variance 1 au spectre ``amp`` (tuiles de 64 px recousues par fondu)."""
    H, W = shape
    if amp is None:
        return rng.standard_normal(shape).astype(np.float32)
    n = rng.standard_normal((H + 64, W + 64)).astype(np.float32)
    k = np.real(np.fft.ifft2(amp)).astype(np.float32)
    k = np.fft.fftshift(k)
    g = cv2.filter2D(n, -1, k)[32:32 + H, 32:32 + W]
    return (g / (g.std() + 1e-9)).astype(np.float32)


# ----------------------------------------------------------------------------- PSF effective
class EPSF:
    """PSF effective (Anderson & King 2000) suréchantillonnée, une par couleur et par région, interpolée entre régions ;
    prolongée au-delà de son rayon par un halo radial r^-n."""

    def __init__(self, stamps, centres, os_, R, halo):
        self.stamps = stamps          # [c][region] -> (n, n) array
        self.centres = centres        # (nreg, 2) x, y du centre de chaque région
        self.os = os_
        self.R = R                    # rayon de la PSF effective (px)
        self.halo = halo              # [c] -> (A, n) : h(r) = A r^-n au-delà de R (part du flux par px)
        g = int(round(np.sqrt(len(centres))))
        self.g = g

    def blend(self, c, x, y):
        """PSF de la couleur c à la position (x, y) : interpolation bilinéaire entre centres de régions."""
        st = self.stamps[c]
        if len(st) == 1:
            return st[0]
        g = self.g
        xs = np.unique(self.centres[:, 0]); ys = np.unique(self.centres[:, 1])
        fx = np.clip(np.interp(x, xs, np.arange(g)), 0, g - 1)
        fy = np.clip(np.interp(y, ys, np.arange(g)), 0, g - 1)
        i0, j0 = int(min(fy, g - 2)), int(min(fx, g - 2))
        ty, tx = fy - i0, fx - j0
        k = lambda i, j: st[i * g + j]
        return ((1 - ty) * (1 - tx) * k(i0, j0) + (1 - ty) * tx * k(i0, j0 + 1)
                + ty * (1 - tx) * k(i0 + 1, j0) + ty * tx * k(i0 + 1, j0 + 1))

    def eval(self, stamp, dx, dy, s=1.0):
        """``s`` : échelle, ou (échelle, a, b) : déformation elliptique bornée de la PSF locale, les coordonnées passées
        par (1/s) [[1 - a, -b], [-b, 1 + a]] (allongement le long de deux axes, angle libre), flux conservé."""
        c = (stamp.shape[0] - 1) / 2
        if np.ndim(s) == 0:
            v = map_coordinates(stamp, [dy / s * self.os + c, dx / s * self.os + c], order=3, mode="constant", cval=0.0)
            return v / (s * s)
        sc, a, b = s
        u = ((1 - a) * dx - b * dy) / sc
        w = (-b * dx + (1 + a) * dy) / sc
        v = map_coordinates(stamp, [w * self.os + c, u * self.os + c], order=3, mode="constant", cval=0.0)
        return v * (1 - a * a - b * b) / (sc * sc)

    def halo_eval(self, c, r):
        A, n, R0 = self.halo[c]
        return np.where(r > R0, A * np.maximum(r, 1.0) ** (-n), 0.0)


def build_epsf(img_c, sat_c, stars, R, os_, iters=3, log=print):
    """PSF effective d'une couleur à partir d'étoiles isolées : (stamp, positions affinées)."""
    n = 2 * R * os_ + 1
    c0 = R * os_
    xs, ys = stars[:, 0].copy(), stars[:, 1].copy()
    H, W = img_c.shape
    Rw = R + 2
    yy, xx = np.mgrid[-Rw:Rw + 1, -Rw:Rw + 1]
    stamp = None
    flux = np.ones(len(xs))
    for it in range(iters):
        acc = np.zeros(n * n); cnt = np.zeros(n * n); acc2 = np.zeros(n * n)
        cuts = []
        for k in range(len(xs)):
            xi, yi = int(round(xs[k])), int(round(ys[k]))
            if xi - Rw < 0 or yi - Rw < 0 or xi + Rw >= W or yi + Rw >= H:
                cuts.append(None)
                continue
            cut = img_c[yi - Rw:yi + Rw + 1, xi - Rw:xi + Rw + 1].astype(np.float64)
            if (cut >= sat_c).any():
                cuts.append(None)
                continue
            dx = xx + xi - xs[k]; dy = yy + yi - ys[k]
            r = np.hypot(dx, dy)
            ring = (r > R - 1) & (r <= R + 2)
            b = np.median(cut[ring])
            v = cut - b
            if stamp is None:
                f = v[r <= R].sum()
            else:
                m = EPSF.eval(_E1(os_), stamp, dx, dy)
                inn = r <= max(3, R / 2)
                f = (v[inn] * m[inn]).sum() / max((m[inn] ** 2).sum(), 1e-12)
            if f <= 0:
                cuts.append(None)
                continue
            flux[k] = f
            cuts.append((dx, dy, v / f))
            ix = np.round(dx * os_).astype(int) + c0; iy = np.round(dy * os_).astype(int) + c0
            ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n) & (r <= R + 0.5)
            idx = iy[ok] * n + ix[ok]
            np.add.at(acc, idx, (v / f)[ok]); np.add.at(acc2, idx, ((v / f) ** 2)[ok]); np.add.at(cnt, idx, 1)
        mean = acc / np.maximum(cnt, 1)
        # second passage : écarts > 3 sigma écartés (une voisine, un rayon cosmique, un noeud)
        sd = np.sqrt(np.maximum(acc2 / np.maximum(cnt, 1) - mean ** 2, 0))
        acc[:] = 0; cnt[:] = 0
        for cu in cuts:
            if cu is None:
                continue
            dx, dy, v = cu
            ix = np.round(dx * os_).astype(int) + c0; iy = np.round(dy * os_).astype(int) + c0
            r = np.hypot(dx, dy)
            ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n) & (r <= R + 0.5)
            idx = iy[ok] * n + ix[ok]
            good = np.abs(v[ok] - mean[idx]) <= 3 * sd[idx] + 1e-9
            np.add.at(acc, idx[good], v[ok][good]); np.add.at(cnt, idx[good], 1)
        P_ = (acc / np.maximum(cnt, 1)).reshape(n, n)
        hole = (cnt == 0).reshape(n, n)
        if hole.any():
            P_[hole] = ndi.generic_filter(P_, np.nanmean, size=3)[hole] if hole.mean() < 0.3 else 0
        P_ = ndi.gaussian_filter(P_, 0.6 * os_ / 2)
        # ailes : moyenne azimutale (bruit réduit) au-delà de 2.5 largeurs, fondue
        rr = np.hypot(*np.mgrid[-c0:c0 + 1, -c0:c0 + 1]) / os_
        prof_r = np.arange(0, R + 1.01, 0.5)
        k = np.digitize(rr, prof_r)
        prof = ndi.mean(P_, labels=k, index=np.arange(1, len(prof_r) + 1))
        rad = np.interp(rr, prof_r[:len(prof)] + 0.25, prof)
        fwhm_os = _fwhm_of(P_, os_)
        a, b = 2.0 * fwhm_os, 3.5 * fwhm_os
        t = np.clip((rr - a) / max(b - a, 1e-6), 0, 1)
        P_ = (1 - t) * P_ + t * rad
        # bord de la PSF ramené à zéro (moyenne du bord retirée, ancien FLOAT optics.py) : un plancher positif laissé
        # par un fond sous-estimé (nébuleuse dans l'anneau) devient, à l'échelle du flux, un disque sombre de rayon R
        edge = (rr > R - 2) & (rr <= R)
        P_ = P_ - float(P_[edge].mean())
        P_ *= (rr <= R)
        P_ /= P_.sum() / os_ ** 2
        stamp = P_
        # positions affinées par ajustement de la nouvelle PSF
        for k in range(len(xs)):
            if cuts[k] is None:
                continue
            xi, yi = int(round(xs[k])), int(round(ys[k]))
            cut = img_c[yi - Rw:yi + Rw + 1, xi - Rw:xi + Rw + 1].astype(np.float64)
            for _ in range(2):
                dx = xx + xi - xs[k]; dy = yy + yi - ys[k]
                inn = np.hypot(dx, dy) <= max(2.5, 1.5 * fwhm_os)
                m = EPSF.eval(_E1(os_), stamp, dx, dy)
                mx = (EPSF.eval(_E1(os_), stamp, dx - 0.05, dy) - EPSF.eval(_E1(os_), stamp, dx + 0.05, dy)) / 0.1
                my = (EPSF.eval(_E1(os_), stamp, dx, dy - 0.05) - EPSF.eval(_E1(os_), stamp, dx, dy + 0.05)) / 0.1
                A = np.c_[m[inn], np.ones(inn.sum()), mx[inn], my[inn]]
                sol = np.linalg.lstsq(A, cut[inn], rcond=None)[0]
                if sol[0] <= 0:
                    break
                ddx, ddy = np.clip(sol[2] / sol[0], -0.5, 0.5), np.clip(sol[3] / sol[0], -0.5, 0.5)
                xs[k] += ddx; ys[k] += ddy
    nused = sum(cu is not None for cu in cuts)
    return stamp, np.c_[xs, ys], nused


class _E1:
    def __init__(self, os_):
        self.os = os_


def _fwhm_of(P_, os_):
    """Largeur à mi-hauteur (en px de la grille suréchantillonnée ramenée en px image)."""
    c = (P_.shape[0] - 1) // 2
    pk = P_[c, c]
    if pk <= 0:
        return 3.0
    area = (P_ >= pk / 2).sum() / os_ ** 2
    return float(2 * np.sqrt(area / np.pi))


# ----------------------------------------------------------------------------- ajustement d'une étoile
COLOUR_POS_SNR = 20.0   # v4 : seuil du pic (en bruits, par couleur) pour ajuster la position de cette couleur

def star_model(E, stamps, x, y, F, s, x0, x1, y0, y1, halo_amp=1.0, with_halo=True, choff=None):
    """Modèle (C, h, w) de l'étoile de flux F (C,) au centre (x, y) dans la fenêtre [y0:y1, x0:x1] ; ``choff`` :
    décalage (C, 2) du centre de chaque couleur."""
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx = xx - x; dy = yy - y
    r = np.hypot(dx, dy)
    out = np.zeros((len(F), y1 - y0, x1 - x0), np.float32)
    inner = r <= E.R + 1
    co = np.zeros((len(F), 2)) if choff is None else np.asarray(choff, float)
    for c in range(len(F)):
        if F[c] == 0:
            continue
        m = np.zeros(dx.shape)
        if inner.any():
            m[inner] = E.eval(stamps[c], dx[inner] - co[c, 0], dy[inner] - co[c, 1], s)
        if with_halo:
            s0_ = s[0] if np.ndim(s) else s
            m += halo_amp * E.halo_eval(c, r / s0_) / (s0_ * s0_)
        out[c] = F[c] * m
    return out


def fit_star(R, valid, sat, E, stamps, x, y, F0, s0, rfit, fix_pos, free_width, sig, x_cat, y_cat, P):
    """Ajustement linéaire flux + plan de fond par couleur, position commune (Gauss-Newton), largeur en option.
    ``R`` : la fenêtre du résidu avec l'étoile remise. Rend (F, x, y, s, chi2)."""
    C, h, w = R.shape
    yy, xx = np.mgrid[0:h, 0:w]

    ann_w = P.get("ann_w", 4.0)
    co = np.array(P.get("choff", np.zeros((C, 2))), float)       # copie : v4 la modifie par couleur
    co_base = co.copy()

    def solve(x, y, s):
        """Fond : plan robuste ajusté sur l'anneau juste au-delà du rayon d'ajustement (DAOPHOT : le ciel se mesure
        autour de l'étoile, pas avec elle), sur le résidu moins l'aile de l'étoile ; puis le flux seul, linéaire.
        Un plan libre ajusté avec le flux échangeait une partie plate de la PSF contre le flux (fond trop bas, flux trop
        fort : le puits sombre au centre des étoiles moyennes)."""
        dx = xx - x; dy = yy - y
        r = np.hypot(dx, dy)
        inn = (r <= rfit) & valid
        ann = (r > rfit) & (r <= rfit + ann_w) & valid
        F = np.zeros(C); B = np.zeros((C, 3)); chi = 0.0
        Ms = []
        for c in range(C):
            ok = inn & ~sat[c]
            okA = ann & ~sat[c]
            if ok.sum() < 6:
                Ms.append(None)
                continue
            # c5 : décalage sous-pixel propre à chaque couleur (dématriçage, chromatisme latéral), champ mesuré
            dxc, dyc = dx - co[c, 0], dy - co[c, 1]
            m = E.eval(stamps[c], dxc[ok], dyc[ok], s)
            if okA.sum() >= 12:
                mA = E.eval(stamps[c], dxc[okA], dyc[okA], s)
                Ab = np.c_[np.ones(okA.sum()), dx[okA] / rfit, dy[okA] / rfit]
                f = 0.0
                keep = np.ones(okA.sum(), bool)
                for _ in range(3):
                    yb = R[c][okA] - f * mA
                    bsol, *_ = np.linalg.lstsq(Ab[keep], yb[keep], rcond=None)
                    rb = yb - __import__('operator').matmul(Ab, bsol)
                    sd = 1.4826 * np.median(np.abs(rb[keep])) + 1e-9
                    keep = np.abs(rb) < 3 * sd
                    if keep.sum() < 8:
                        keep[:] = True
                    bg = __import__('operator').matmul(np.c_[np.ones(ok.sum()), dx[ok] / rfit, dy[ok] / rfit], bsol)
                    f = float((m * (R[c][ok] - bg)).sum() / max((m * m).sum(), 1e-12))
                F[c] = f; B[c] = bsol
                res = R[c][ok] - bg - f * m
                sol_A = np.c_[m, np.ones(ok.sum()), dx[ok] / rfit, dy[ok] / rfit]
            else:
                A = np.c_[m, np.ones(ok.sum()), dx[ok] / rfit, dy[ok] / rfit]
                sol, *_ = np.linalg.lstsq(A, R[c][ok], rcond=None)
                F[c] = sol[0]; B[c] = sol[1:]
                res = R[c][ok] - __import__('operator').matmul(A, sol)
            chi += float((res ** 2).sum() / max(sig[c], 1e-6) ** 2) / ok.sum()
            Ms.append((ok, m, res))
        return F, B, chi, Ms

    s = s0
    F, B, chi, Ms = solve(x, y, s)
    if not fix_pos:
        for _ in range(3):
            # Gauss-Newton sur la position, commun aux couleurs (pondéré par le flux)
            JTJ = np.zeros((2, 2)); JTr = np.zeros(2)
            for c in range(C):
                if Ms[c] is None or F[c] <= 0:
                    continue
                ok, m, res = Ms[c]
                dx = xx[ok] - x - co[c, 0]; dy = yy[ok] - y - co[c, 1]
                mx = (E.eval(stamps[c], dx - 0.05, dy, s) - E.eval(stamps[c], dx + 0.05, dy, s)) / 0.1 * F[c]
                my = (E.eval(stamps[c], dx, dy - 0.05, s) - E.eval(stamps[c], dx, dy + 0.05, s)) / 0.1 * F[c]
                J = np.c_[mx, my] / sig[c]
                JTJ += __import__('operator').matmul(J.T, J); JTr += __import__('operator').matmul(J.T, (res / sig[c]))
            try:
                d = np.linalg.solve(JTJ + 1e-6 * np.eye(2), JTr)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(d)):      # v33b : fenêtre sans donnée utilisable (bord de l'image) : on garde la position
                break
            d = np.clip(d, -0.5, 0.5)
            nx = np.clip(x + d[0], x_cat - P["pos_margin"], x_cat + P["pos_margin"])
            ny = np.clip(y + d[1], y_cat - P["pos_margin"], y_cat + P["pos_margin"])
            if abs(nx - x) < 0.01 and abs(ny - y) < 0.01:
                break
            x, y = nx, ny
            F, B, chi, Ms = solve(x, y, s)
    if free_width:
        best = (chi, s, F)
        for s1 in np.linspace(1 - P["width_margin"], 1 + P["width_margin"], 5):
            F1, B1, chi1, _ = solve(x, y, s1)
            if chi1 < best[0]:
                best = (chi1, s1, F1)
        chi, s, F = best
        if P.get("ellipse", True):
            # liberté elliptique bornée autour de la PSF locale (a, b dans +-ellipse_max : allongement jusqu'à 15 %,
            # angle libre), recherche par coordonnées à pas décroissant sur le chi2 du coeur, flux réajusté à chaque essai
            a, b = 0.0, 0.0
            lim = P.get("ellipse_max", 0.15)
            for step in (0.06, 0.03, 0.015):
                improved = True
                while improved:
                    improved = False
                    for da, db in ((step, 0), (-step, 0), (0, step), (0, -step)):
                        na, nb = np.clip(a + da, -lim, lim), np.clip(b + db, -lim, lim)
                        if (na, nb) == (a, b):
                            continue
                        F1, B1, chi1, _ = solve(x, y, (s, na, nb))
                        if chi1 < chi * 0.999:
                            chi, F, a, b = chi1, F1, na, nb
                            improved = True
            s = (s, a, b)
    # v4 : position propre à chaque couleur (photométrie PSF par bande), pour les étoiles dont le pic dépasse
    # P["colour_pos_snr"] bruits dans cette couleur ; Gauss-Newton sur cette couleur seule, +-0.3 px autour de la
    # position commune ; le décalage est rendu par P["_co_out"] pour que le modèle soit évalué au même endroit
    if P.get("colour_pos_snr") and not fix_pos and C == 3:
        _, _, _, Ms = solve(x, y, s)
        for c in range(C):
            if Ms[c] is None or F[c] <= 0 or F[c] * float(Ms[c][1].max()) <= P["colour_pos_snr"] * sig[c]:
                continue
            for _ in range(4):
                ok, m, res = Ms[c]
                dx = xx[ok] - x - co[c, 0]; dy = yy[ok] - y - co[c, 1]
                sp = s
                mx = (E.eval(stamps[c], dx - 0.05, dy, sp) - E.eval(stamps[c], dx + 0.05, dy, sp)) / 0.1 * F[c]
                my = (E.eval(stamps[c], dx, dy - 0.05, sp) - E.eval(stamps[c], dx, dy + 0.05, sp)) / 0.1 * F[c]
                J = np.c_[mx, my] / sig[c]
                try:
                    d = np.linalg.solve(__import__('operator').matmul(J.T, J) + 1e-6 * np.eye(2), __import__('operator').matmul(J.T, (res / sig[c])))
                except np.linalg.LinAlgError:
                    break
                if not np.all(np.isfinite(d)):      # v33b : idem, décalage de couleur inchangé
                    break
                d = np.clip(d, -0.3, 0.3)
                new = np.clip(co[c] + d, co_base[c] - 0.3, co_base[c] + 0.3)
                if np.abs(new - co[c]).max() < 0.005:
                    co[c] = new
                    break
                co[c] = new
                F, B, chi, Ms = solve(x, y, s)
                if Ms[c] is None:
                    break
        F, B, chi, Ms = solve(x, y, s)
    if "_co_out" in P:
        P["_co_out"][0] = co
    return F, x, y, s, chi


# ----------------------------------------------------------------------------- annuli (boucle fermée)
def annuli_residual(Rimg, valid, x, y, fwhm, rmax, sig, ncorr, P):
    """Résidu moyen par anneau (coeur -> halo) moins le fond juste au-delà, par couleur, et son erreur.
    Rend (edges, res (C, k), err (C, k), ref (C,), score)."""
    C, H, W = Rimg.shape
    rout = rmax + max(4.0, 1.5 * fwhm)
    x0, x1 = int(max(0, x - rout - 1)), int(min(W, x + rout + 2))
    y0, y1 = int(max(0, y - rout - 1)), int(min(H, y + rout + 2))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    r = np.hypot(xx - x, yy - y)
    v = valid[y0:y1, x0:x1]
    e = [0, 0.6 * fwhm, 1.2 * fwhm, 2.0 * fwhm, 3.0 * fwhm]
    while e[-1] < rmax:
        e.append(min(rmax, e[-1] * 1.6))
    edges = np.array(e)
    ring = v & (r > rmax) & (r <= rout)
    res = np.zeros((C, len(edges) - 1)); err = np.zeros_like(res); ref = np.zeros(C)
    if ring.sum() < 5:
        return None
    lab = np.searchsorted(edges, r, side="left")          # anneau k+1 : edges[k] < r <= edges[k+1]
    lab[~v | (r > rmax)] = 0
    lab[r == 0] = 1
    nb = len(edges)
    cnt = np.bincount(lab.ravel(), minlength=nb + 1)[1:nb]
    ang = ((np.arctan2(yy - y, xx - x) + np.pi) / (2 * np.pi) * 8).astype(int) % 8
    for c in range(C):
        Rc = Rimg[c, y0:y1, x0:x1]
        ref[c] = np.median(Rc[ring])
        s = float(sig[c, int(np.clip(y, 0, H - 1)), int(np.clip(x, 0, W - 1))])
        # rugosité du fond : dispersion des 8 secteurs de l'anneau extérieur (nébuleuse) ; le résidu n'est pas jugé
        # plus finement que le fond n'est connu (sinon fausse alarme au bord d'une nébuleuse)
        secs = [np.median(Rc[ring & (ang == a)]) for a in range(8) if (ring & (ang == a)).sum() > 3]
        tau = float(np.std(secs)) if len(secs) >= 4 else 0.0
        sm = np.bincount(lab.ravel(), weights=Rc.ravel(), minlength=nb + 1)[1:nb]
        okk = cnt > 0
        res[c, okk] = sm[okk] / cnt[okk] - ref[c]
        err[c, okk] = np.sqrt((s * np.sqrt(ncorr[c] / cnt[okk]) + P["q_floor"] * s) ** 2 + tau ** 2)
    z = np.abs(res) / np.maximum(err, 1e-9)
    return edges, res, err, ref, float(z.max())


# ----------------------------------------------------------------------------- pipeline
def run(name, tag="v1", overrides=None, log=None):
    pr = dict(P)
    pr.update(overrides or {})
    out = HERE / "sorties" / name / tag
    out.mkdir(parents=True, exist_ok=True)
    if log is None:
        log = log_to(out / "journal.txt")
    t0 = time.time()
    log(f"=== {name} {tag} ===  réglages: {json.dumps(overrides or {})}")
    img = images.load(name)
    # v14 : plateaux saturés lus sur l'image ENTIÈRE (maximum de chaque couleur), avant la découpe
    sat14_lvl, sat14_m, sat14_info = plateaux_satures(img, pr["sat_eps"], pr["sat_min_px"])
    # v17 (22 nov) : réglages de l'image entière mesurés une fois et figés (etalonnage_image.py : largeur, forme avec la
    # correction du coeur, point zéro, part par couleur, saturation) ; la découpe n'ajuste plus que l'éclat et la
    # position de chaque étoile (photométrie forcée, Lang, Hogg & Schlegel 2016)
    CAL = None
    DEC17 = {}                       # v17 : valeurs que la découpe aurait prises (journal et resume.json seulement)
    if pr.get("etalonnage_image", True):
        import etalonnage_image as EI
        CAL = EI.obtenir(name, log)
    crop = pr.get("crop")
    if crop:
        cx0, cy0, cx1, cy1 = crop
        img = np.ascontiguousarray(img[:, cy0:cy1, cx0:cx1])
        sat14_m = np.ascontiguousarray(sat14_m[:, cy0:cy1, cx0:cx1])
    C, H, W = img.shape
    valid = np.any(img != 0, axis=0)
    valid = ndi.binary_erosion(valid, iterations=3)
    log(f"image {img.shape}, {time.time() - t0:.0f} s")

    # 1. astrométrie + Gaia (en cache : même solution d'une version à l'autre)
    import astrometrie
    cache = HERE / "sorties" / name / "astrometrie.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        cat = z["cat"].item(); scale = float(z["scale"]); wres = float(z["wres"])
        log(f"astrométrie relue du cache: {scale:.3f}\"/px, résidu {wres:.3f} px, {len(cat['x'])} étoiles Gaia")
    else:
        wcs, scale, wres = astrometrie.solve(img, images.IMAGES[name][1], HERE / "sorties" / name / "tmp", log)
        cat = astrometrie.gaia_stars(wcs, (H, W), scale, log=log)
        np.savez(cache, cat=np.array(cat, dtype=object), scale=scale, wres=wres, wcs=wcs.to_header_string(relax=True))
        try:
            (HERE / "sorties" / name / "tmp").rmdir()
        except OSError:
            pass
    # tour 19 : catalogue Gaia DR3 profond (G < 21, téléchargé une fois, lu ici en lecture seule) à la place de
    # l'extrait hors ligne quand il existe pour cette image ; indice de couleur BP - RP
    deep = DEEP_DIR / f"gaia_dr3_profond_{name}.npz"
    if pr.get("deep_gaia", True) and deep.exists():
        zd = np.load(deep)
        ok_ = np.isfinite(zd["Gmag"]) & np.isfinite(zd["x_pix"]) & np.isfinite(zd["y_pix"])
        n_ = int(ok_.sum())
        bprp = (zd["BPmag"] - zd["RPmag"])[ok_]
        cat = dict(x=zd["x_pix"][ok_].astype(float), y=zd["y_pix"][ok_].astype(float), g=zd["Gmag"][ok_].astype(float),
                   teff=np.zeros(n_), src=np.zeros(n_, int), sp=-np.ones(n_, int), rgb_xp=np.zeros((n_, 3)), ci=bprp.astype(float))
        log(f"catalogue Gaia profond : {n_} étoiles ({deep.name}), au lieu de l'extrait hors ligne")
        # catalogue complet à G 21 : hors catalogue, seulement les sources nettes au-dessus de 8 bruits (le reste était
        # du bruit corrélé ou de la nébuleuse : 3 865 « étoiles » sur NGC 6543 au tour 19), sans maxima locaux
        pr.setdefault("faint_passes", (8.0,))
        pr["faint_passes"] = tuple(t for t in pr["faint_passes"] if t >= 8.0) or (8.0,)
        pr["peak_finder"] = False
    if crop:
        sel = (cat["x"] > cx0 - 60) & (cat["x"] < cx1 + 60) & (cat["y"] > cy0 - 60) & (cat["y"] < cy1 + 60)
        cat = {k: v[sel] for k, v in cat.items()}
        cat["x"] = cat["x"] - cx0; cat["y"] = cat["y"] - cy0
    xg, yg, G = cat["x"].astype(float), cat["y"].astype(float), cat["g"].astype(float)
    # suivi pas à pas d'étoiles signalées (coordonnées de l'image entière)
    DEBUG_FULL = {"rosette": [(2045.1, 967.4), (2036.9, 1035.0)]}
    if name in DEBUG_FULL and "debug_xy" not in pr:
        ox_, oy_ = (crop[0], crop[1]) if crop else (0, 0)
        pr["debug_xy"] = [(x - ox_, y - oy_) for x, y in DEBUG_FULL[name]]

    # saturation par couleur
    # saturation par couleur, lue sur les données : les coeurs écrêtés des étoiles les plus brillantes s'empilent au même
    # niveau, différent par couleur (balance des blancs d'un capteur couleur) ; le maximum de l'image n'est pas ce niveau
    # (Rosette : coeurs à 0.52-0.56 du maximum de chaque couleur)
    inb = (xg > 3) & (xg < W - 3) & (yg > 3) & (yg < H - 3)
    brightest = np.argsort(np.where(inb, G, 99))[:25]
    sat_lvl = np.zeros(C)
    ref_lvl = np.zeros(C)
    sat_old = np.zeros(C)
    for c in range(C):
        pk_ = np.array([img[c, int(yg[k]) - 2:int(yg[k]) + 3, int(xg[k]) - 2:int(xg[k]) + 3].max() for k in brightest if inb[k]])
        top_ = np.sort(pk_)[::-1][:12]
        plateau = len(top_) >= 5 and np.std(top_[1:]) / max(np.median(top_), 1e-9) < 0.12
        if plateau:
            sat_lvl[c] = float(np.median(top_[1:])) * pr["sat_frac"]
        else:
            # pas de plateau : aucune étoile écrêtée ; seul le sommet de la plus brillante est tenu pour saturé (jamais le
            # maximum de l'image, qui peut être une nébuleuse : le coeur de NGC 6543)
            sat_lvl[c] = np.inf
        # niveau de référence (seuils relatifs : étoiles de PSF, profil propre) : la saturation, sinon le pic de la
        # plus brillante des étoiles hors la plus forte (qui peut être un coeur de nébuleuse)
        # v14 : plateau écrêté vu contre le maximum de la couleur (image entière) : son niveau compte aussi (le plus bas
        # des deux) ; ses pixels sont saturés même sous l'ancien seuil
        sat_old[c] = sat_lvl[c]
        sat_lvl[c] = min(sat_lvl[c], sat14_lvl[c])
        ref_lvl[c] = sat_lvl[c] if np.isfinite(sat_lvl[c]) else (float(top_[1]) if len(top_) > 1 else float(img[c].max()))
        if CAL is not None:
            # v17 : saturation de l'image entière (détection v14), figée ; le plateau des 12 plus brillantes de la
            # découpe n'est plus qu'écrit au journal
            sat_lvl[c] = float(CAL["info"]["sat_lvl"][c]); ref_lvl[c] = float(CAL["info"]["ref_lvl"][c])
        log(f"v14 saturation couleur {c} : plateau des coeurs {sat_old[c]:.1f}{' (découpe, non utilisé : v17)' if CAL is not None else ''} ; "
            f"max image entière {sat14_info[c][0]}, "
            f"seuil (1 - {pr['sat_eps']}) x max, plateaux >= {pr['sat_min_px']} px : {sat14_info[c][1]} "
            f"({sat14_info[c][2]} px, dont {int(sat14_m[c].sum())} dans la découpe) ; niveau retenu {sat_lvl[c]:.1f}")
    if CAL is not None:
        sat_old[:] = np.inf
    # pixels saturés : ancienne règle (>= plateau des coeurs) OU plateau v14 (>= (1 - eps) x max, assez grand) ;
    # v17 : plateau v14 seul (image entière)
    sat = np.stack([(img[c] >= sat_old[c]) | sat14_m[c] for c in range(C)])
    sat = np.stack([ndi.binary_dilation(s, iterations=2) for s in sat])
    log(f"saturation par couleur (seuil, part du maximum) : {np.round(sat_lvl, 1)}, "
        f"{np.round(sat_lvl / np.array([img[c].max() for c in range(C)]), 3)} ; pixels exclus {sat.any(0).mean() * 100:.3f} %")

    # masque grossier des étoiles (pour bruit et grain) : disque selon l'éclat Gaia
    L = img.mean(0)
    star_mask = np.zeros((H, W), bool)
    for x, y, g in zip(xg, yg, G):
        rr = int(np.clip(3 + 2.5 * (16 - g), 3, 60))
        xi, yi = int(round(x)), int(round(y))
        if -rr <= xi < W + rr and -rr <= yi < H + rr:
            cv2.circle(star_mask.view(np.uint8), (xi, yi), rr, 1, -1)
    star_mask = star_mask | ~valid
    sig, ncorr = noise_stats(img, star_mask)
    log(f"bruit du ciel médian par couleur {[round(float(np.median(s)), 3) for s in sig]}, aire de corrélation {[round(v, 1) for v in ncorr]} px")

    # 2. PSF : étoiles de PSF = Gaia isolées, assez brillantes, non saturées
    pk = np.array([L[int(np.clip(round(y), 0, H - 1)), int(np.clip(round(x), 0, W - 1))] for x, y in zip(xg, yg)])
    inside = (xg > 40) & (xg < W - 40) & (yg > 40) & (yg < H - 40)
    from scipy.spatial import cKDTree
    tree = cKDTree(np.c_[xg, yg])
    # largeur d'abord sur 200 étoiles moyennes
    # étoiles de PSF dans le régime linéaire seulement : un coeur aplati par la saturation (même partielle, lissée par le
    # drizzle) donne une PSF aux ailes trop fortes -> bulles sombres autour des étoiles moyennes
    cand = np.where(inside & (pk > pr["psf_snr"] * np.median(sig)) & (pk < pr["psf_lin"] * ref_lvl.min()))[0]
    fw = []
    for k in cand[:400]:
        xi, yi = int(round(xg[k])), int(round(yg[k]))
        cut = L[yi - 8:yi + 9, xi - 8:xi + 9]
        b = np.median(np.r_[cut[0], cut[-1], cut[:, 0], cut[:, -1]])
        v = cut - b
        if v.max() <= 0:
            continue
        fw.append(2 * np.sqrt((v >= v.max() / 2).sum() / np.pi))
    fwhm = float(np.median(fw)) if fw else 3.0
    if CAL is not None:
        # v17 : largeur de l'image entière (modèle v7), figée ; celle de la découpe n'est plus qu'écrite au journal
        log(f"v17 largeur mesurée sur la découpe {fwhm:.2f} px ({len(fw)} étoiles), non utilisée ; largeur de l'image "
            f"{float(CAL['info']['fwhm']):.2f} px")
        DEC17["fwhm"] = fwhm
        fwhm = float(CAL["info"]["fwhm"])
    R = int(np.clip(np.ceil(5 * fwhm), 10, 40))
    os_ = pr["os_fine"] if fwhm < 4 else 2
    pr["ann_w"] = max(4.0, 1.5 * fwhm)
    log(f"largeur à mi-hauteur ~{fwhm:.2f} px ({len(fw)} étoiles), rayon de PSF {R} px, suréchantillonnage {os_}")
    if pr.get("modele_image", True):
        # v7 : modèle d'étoile bâti sur l'IMAGE ENTIÈRE (modele_psf_image.py : centaines d'étoiles, restes empilés ajoutés
        # au modèle jusqu'à un profil plat par tranche de G, rayon 28 px, aile des plus brillantes), une fois par image,
        # relu du disque ; le choix du modèle d'une étoile se fait par sa région et sa tranche d'éclat (E.blend(c, x, y, G))
        import modele_psf_image as MPI
        E, minfo = MPI.epsf_pour_decoupe(name, (crop[0], crop[1]) if crop else (0, 0), R, log)
        R_ancien = R
        R, os_, g = E.R, E.os, E.g
        stamps = E.stamps
        good = np.zeros(0, int)
        log(f"PSF effective v7 (image entière) : {minfo['etoiles']} étoiles, {g}x{g} régions du coeur, tranches G "
            f"{minfo['tranches'][1:-1]}, rayon {R} px (ancien rayon {R_ancien} px gardé pour les profils des brillantes), "
            f"{minfo['mises_a_jour']} mises à jour, pire reste {minfo['pire_final']} bruit, {time.time() - t0:.0f} s")
        if CAL is not None:
            # v17 : correction du coeur de l'image entière (même calcul que core_correction, sur les étoiles
            # d'étalonnage de l'image), ajoutée une fois ; plus de correction du coeur sur la découpe
            corr_ = np.asarray(CAL["corr"], float)
            for c in range(C):
                E.stamps[c][0] = E.stamps[c][0] + corr_[c]
            log(f"v17 correction du coeur de l'image ajoutée au modèle : part de flux par couleur "
                f"{np.round(corr_.sum(axis=(1, 2)) / E.os ** 2, 4).tolist()} ; correction par découpe coupée")
    else:
      # (ancien modèle v5, sur la découpe)
      iso = np.ones(len(xg), bool)
      for k in cand:
          # c2 : avec le catalogue profond, la règle « aucune étoile de moins de 3 mag d'écart à 2 R » ne laissait que 5
          # étoiles de PSF sur la découpe Rosette ; voisines comptées jusqu'à 2.5 mag d'écart, à 1.2 R
          iso_r, iso_dg = (1.2 * R, 2.5) if pr.get("iso_v2", False) else (2 * R, 3.0)   # c2-c4 : v2 ; c1 : ancien
          nb = tree.query_ball_point([xg[k], yg[k]], iso_r)
          for j in nb:
              if j != k and G[j] < G[k] + iso_dg:
                  iso[k] = False
                  break
      good = cand[iso[cand]]
      # nébuleuse : le fond autour ne doit pas trop varier
      bgs = blur(L, 3 * R)
      flat = np.array([abs(float(bgs[int(yg[k]), int(xg[k])])) < 5 * float(np.median(sig)) + 0.05 * pk[k] for k in good])
      good = good[flat] if flat.any() else good
      # tour 14 : la forme des étoiles (allongement, orientation) varie vite sur le champ ; une région au plus de
      # region_px de côté (sur la Rosette, résidus en dipôle le long du grand axe avec 2 x 2 régions de 1024 px)
      g = max(pr["region_grid"], int(np.ceil(max(H, W) / pr["region_px"])))
      while g > 1 and len(good) < pr["min_psf_stars"] * g * g:
          g -= 1
      centres = []
      stamps = [[] for _ in range(C)]
      for i in range(g):
          for j in range(g):
              cx, cy = (j + 0.5) * W / g, (i + 0.5) * H / g
              centres.append((cx, cy))
              # étoiles de la région (élargie de 30 %) les plus proches du centre
              d = np.maximum(np.abs(xg[good] - cx) / (W / g), np.abs(yg[good] - cy) / (H / g))
              sel = good[np.argsort(d)][:max(pr["min_psf_stars"], 80)]
              for c in range(C):
                  st, _, nu = build_epsf(img[c], sat_lvl[c], np.c_[xg[sel], yg[sel]], R, os_)
                  stamps[c].append(st)
      centres = np.array(centres)
      log(f"PSF effective : {g}x{g} régions, {len(good)} étoiles candidates, {time.time() - t0:.0f} s")
      E = EPSF(stamps, centres, os_, R, [(0.0, 3.0, R)] * C)

    def _blend(c, x, y, g_=None):
        """v7 : modèle de la région et de la tranche d'éclat de l'étoile (v5 : de la région seule)."""
        return E.blend(c, x, y, g_) if hasattr(E, "rho") else E.blend(c, x, y)
    R_anc = getattr(E, "R_ancien", E.R)        # v7 : rayon de l'ancien modèle, gardé pour les profils des brillantes

    # 3. étalonnage flux <-> G : ajustement rapide (position Gaia, sans halo) sur les étoiles non saturées
    def quick_flux(k):
        x, y = xg[k], yg[k]
        xi, yi = int(round(x)), int(round(y))
        rf = max(3.0, 1.5 * fwhm)
        rw = int(np.ceil(rf + pr["ann_w"])) + 1
        if xi - rw < 0 or yi - rw < 0 or xi + rw >= W or yi + rw >= H:
            return None
        win = img[:, yi - rw:yi + rw + 1, xi - rw:xi + rw + 1]
        st = [_blend(c, x, y, G[k]) for c in range(C)]
        F, *_ = fit_star(win, valid[yi - rw:yi + rw + 1, xi - rw:xi + rw + 1], sat[:, yi - rw:yi + rw + 1, xi - rw:xi + rw + 1],
                         E, st, x - (xi - rw), y - (yi - rw), None, 1.0, rf, True, False, sig[:, yi, xi],
                         x - (xi - rw), y - (yi - rw), pr)
        return F
    calib = np.where(inside & (pk > 10 * np.median(sig)) & (pk < 0.4 * ref_lvl.min()))[0]
    # étoiles d'étalonnage isolées : aucune voisine de moins de 2 mag d'écart à moins de 2 largeurs (avec le catalogue
    # profond, les voisines faibles d'un amas faussaient l'étalonnage : dispersion 1.1 mag sur la découpe Rosette)
    trc = cKDTree(np.c_[xg, yg])
    iso_c = np.array([all(G[j] > G[k] + 2 for j in trc.query_ball_point([xg[k], yg[k]], 2 * fwhm) if j != k) for k in calib], bool)
    calib = calib[iso_c][:1500] if (iso_c.sum() >= 50 and pr.get("iso_v2", False)) else calib[:1500]
    Fq = []
    for k in calib:
        F = quick_flux(k)
        Fq.append(F if F is not None else np.full(C, np.nan))
    Fq = np.array(Fq)
    Ftot = np.nansum(Fq, 1)
    ok = np.isfinite(Ftot) & (Ftot > 0)
    zp = G[calib][ok] + 2.5 * np.log10(Ftot[ok])
    zp0 = float(np.median(zp)); zsd = float(1.4826 * np.median(np.abs(zp - zp0)))
    colour = np.nanmedian(Fq[ok] / Ftot[ok, None], 0)
    log(f"étalonnage : flux total = 10^(-0.4 (G - {zp0:.3f})), dispersion {zsd:.3f} mag sur {ok.sum()} étoiles ; couleur moyenne {np.round(colour, 3)}"
        + (" (découpe, non utilisé : v17)" if CAL is not None else ""))
    if CAL is not None:
        # v17 : point zéro, dispersion et part par couleur de l'image entière, figés
        DEC17.update(zp=zp0, zp_sd=zsd, couleur=np.round(colour, 4).tolist())
        zp0, zsd = float(CAL["info"]["zp"]), float(CAL["info"]["zp_sd"])
        colour = np.asarray(CAL["info"]["couleur"], float)
        log(f"v17 étalonnage de l'image entière : flux total = 10^(-0.4 (G - {zp0:.3f})), dispersion {zsd:.3f} mag sur "
            f"{CAL['info']['etoiles']} étoiles ; part par couleur {np.round(colour, 3)}")

    def predicted(k):
        return 10 ** (-0.4 * (G[k] - zp0))

    # flux attendu par couleur selon la couleur de l'étoile (ancien FLOAT, stars.py:predict_flux :
    # log10 F_c + 0.4 G = cubique(couleur), par couleur, écrêtage 3 sigma). Indice de couleur : -2.5 log10(bleu / rouge)
    # du spectre XP hors ligne ; sans spectre, tiré de la température effective de l'extrait astrométrique (relation
    # ajustée sur les étoiles qui ont les deux) ; sinon la médiane.
    rgbx = np.asarray(cat.get("rgb_xp", np.zeros((len(G), 3))), float)
    with np.errstate(all="ignore"):
        ci = np.where((rgbx[:, 0] > 0) & (rgbx[:, 2] > 0), -2.5 * np.log10(rgbx[:, 2] / rgbx[:, 0]), np.nan)
    if "ci" in cat:                              # catalogue profond : BP - RP directement
        ci = np.asarray(cat["ci"], float).copy()
    teff = np.asarray(cat.get("teff", np.zeros(len(G))), float)
    both = np.isfinite(ci) & (teff > 2500)
    if both.sum() > 30:
        cf_t = np.polyfit(np.log10(teff[both]), ci[both], 2)
        fromT = ~np.isfinite(ci) & (teff > 2500)
        ci[fromT] = np.polyval(cf_t, np.log10(teff[fromT]))
    ci_med = float(np.nanmedian(ci)) if np.isfinite(ci).any() else 0.0
    ci = np.where(np.isfinite(ci), ci, ci_med)
    lo_c, hi_c = np.percentile(ci, [1, 99])
    ci = np.clip(ci, lo_c, hi_c)
    Fexp = np.zeros((len(G), C))
    cal_ci = ci[calib]
    for c in range(C):
        with np.errstate(all="ignore"):
            yv = np.log10(Fq[:, c]) + 0.4 * G[calib]
        m_ = np.isfinite(yv) & (Fq[:, c] > 0)
        if m_.sum() < 20:
            Fexp[:, c] = 10 ** (-0.4 * (G - zp0)) * colour[c]
            continue
        keep = m_.copy()
        for _ in range(4):
            cf = np.polyfit(cal_ci[keep], yv[keep], 3 if keep.sum() > 60 else 1)
            rr_ = yv - np.polyval(cf, cal_ci)
            sd = 1.4826 * np.median(np.abs(rr_[keep]))
            keep = m_ & (np.abs(rr_) < 3 * sd)
        Fexp[:, c] = 10 ** (np.polyval(cf, ci) - 0.4 * G)
        log(f"flux attendu couleur {c} : dispersion {sd / 0.4:.3f} mag sur {keep.sum()} étoiles"
            + (" (découpe, non utilisé : v17)" if CAL is not None else ""))
    if CAL is not None:
        # v17 : relation flux par couleur / BP - RP de l'image entière (sans elle : éclat x part par couleur)
        for c in range(C):
            cf_ = np.asarray(CAL["info"]["cf_couleur"][c], float)
            Fexp[:, c] = 10 ** (np.polyval(cf_, ci) - 0.4 * G) if np.all(np.isfinite(cf_)) else 10 ** (-0.4 * (G - zp0)) * colour[c]
    # images en filtres assemblés (bandes étroites, couleurs recombinées : NGC 6543, 58 HOO) : la couleur Gaia BP - RP
    # ne dit pas la part de chaque canal ; l'attendu par canal est l'éclat Gaia x la couleur médiane des étoiles de
    # l'image, et la borne par couleur est élargie (0.2 - 5 x)
    narrow = images.IMAGES[name][2] == "filtres"
    if narrow:
        Fexp = (10 ** (-0.4 * (G - zp0)))[:, None] * colour[None, :]
        pr["colour_bound"] = (0.2, 5.0)
        log(f"image en filtres assemblés : flux attendu par canal = éclat Gaia x couleur médiane {np.round(colour, 3)}, borne 0.2-5 x")
    # tour 18 : étoiles brillantes absentes du catalogue (NGC 6543, x 1195 y 1141 : pic 137 000 en bleu) ajoutées à la
    # liste comme des étoiles Gaia, éclat tiré de leur flux d'ouverture : elles reçoivent le traitement des brillantes
    # (profil propre au coeur, profil partagé au-delà) au lieu de la PSF des étoiles moyennes
    n_add = 0
    if pr.get("bright_offcat", True):
        thr_b = pr["profile_sat"] * float(ref_lvl.min())
        mx = ndi.maximum_filter(L, size=7)
        ys_, xs_ = np.nonzero((L >= mx) & (L > thr_b) & valid)
        if len(ys_):
            from scipy.spatial import cKDTree
            dgx, _ = cKDTree(np.c_[xg, yg]).query(np.c_[xs_, ys_])
            ra_ = int(np.ceil(3 * fwhm))
            gy_, gx_ = np.mgrid[-ra_ - 4:ra_ + 5, -ra_ - 4:ra_ + 5]
            rr_a = np.hypot(gx_, gy_)
            add = []
            for xa, ya, dd in zip(xs_, ys_, dgx):
                if dd < 3 * fwhm or xa < ra_ + 5 or ya < ra_ + 5 or xa >= W - ra_ - 5 or ya >= H - ra_ - 5:
                    continue
                s_win = sat.any(0)[ya - ra_:ya + ra_ + 1, xa - ra_:xa + ra_ + 1]
                if s_win.sum() > 0.5 * s_win.size:                       # grande zone saturée : géante (traitée à part)
                    continue
                cut = img[:, ya - ra_ - 4:ya + ra_ + 5, xa - ra_ - 4:xa + ra_ + 5]
                bgc = np.median(cut[:, rr_a > ra_ + 1], axis=1)
                Lc = cut.mean(0) - bgc.mean()
                ring_ = Lc[(rr_a >= fwhm) & (rr_a < 2 * fwhm)].mean()
                if ring_ > 0.3 * Lc[ra_ + 4, ra_ + 4]:                   # pas ponctuelle : coeur de nébuleuse
                    continue
                Fa = np.array([float((cut[c][rr_a <= ra_] - bgc[c]).sum()) for c in range(C)])
                if Fa.sum() <= 0:
                    continue
                add.append((float(xa), float(ya), Fa))
            if add:
                xg = np.r_[xg, [a[0] for a in add]]; yg = np.r_[yg, [a[1] for a in add]]
                Fa_ = np.array([a[2] for a in add])
                G = np.r_[G, zp0 - 2.5 * np.log10(Fa_.sum(1))]
                Fexp = np.r_[Fexp, np.maximum(Fa_, 1e-6)]
                n_add = len(add)
    log(f"étoiles brillantes hors catalogue ajoutées à la liste : {n_add}")

    # halo : nul au départ ; mesuré après deux passes sur le résidu des étoiles brillantes (voir measure_halo)
    halo = [(0.0, 3.0, float(R))] * C
    E.halo = halo

    # 4. ajustement de toutes les étoiles Gaia : passes Gauss-Seidel sur le résidu commun
    Rimg = img.copy()
    N = len(xg)
    Fs = np.zeros((N, C)); Xs = xg.copy(); Ys = yg.copy(); Ss = np.ones(N); HA = np.ones(N)
    flag = np.zeros(N, int)          # 1 bloqué par Gaia, 2 non convergé (rempli), 4 hors image
    Fpred = 10 ** (-0.4 * (G - zp0))
    med_sig = np.array([float(np.median(sig[c])) for c in range(C)])

    # profil radial moyen de la PSF (toutes couleurs, région centrale) : la fenêtre d'une étoile faible s'arrête où son
    # modèle tombe sous halo_level x bruit
    # v7 : profil de référence = modèle de la tranche G 14.5-15.5 au centre de la découpe (v5 : région du milieu)
    st_c = _blend(1 if C > 1 else 0, W / 2, H / 2, 15.0) if hasattr(E, "rho") else stamps[1 if C > 1 else 0][len(stamps[0]) // 2]
    cc_ = (st_c.shape[0] - 1) // 2
    rr_ = np.hypot(*np.mgrid[-cc_:cc_ + 1, -cc_:cc_ + 1]) / os_
    prof_r = np.arange(0, R + 1)
    prof_v = np.array([st_c[(rr_ >= a) & (rr_ < a + 1)].max() if ((rr_ >= a) & (rr_ < a + 1)).any() else 0 for a in prof_r])

    def footprint(k, F):
        """Rayon où le modèle de l'étoile tombe sous halo_level x bruit."""
        Ft = float(np.sum(np.abs(F))) if F is not None else Fpred[k]
        lvl = pr["halo_level"] * med_sig.min()
        A, n_, _ = halo[int(np.argmax([h[0] for h in halo]))]
        if A > 0 and Ft * A * E.R ** (-n_) > lvl:
            r = (Ft * A / lvl) ** (1 / n_)
        else:
            above = np.where(Ft * prof_v > lvl)[0]
            r = (above[-1] + 2) if len(above) else 2 * fwhm
        return int(np.clip(r, max(3, 2 * fwhm), pr["halo_rmax"]))

    satany = sat.any(0)

    def sat_radius(x, y):
        """Rayon (px) de la zone saturée (toutes couleurs) centrée sur l'étoile, 0 si aucune."""
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < W and 0 <= yi < H) or not satany[max(0, yi - 1):yi + 2, max(0, xi - 1):xi + 2].any():
            return 0.0
        rmax = 0
        for dx_, dy_ in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
            r_ = 0
            while r_ < 300:
                xx_, yy_ = xi + dx_ * (r_ + 1), yi + dy_ * (r_ + 1)
                if not (0 <= xx_ < W and 0 <= yy_ < H) or not satany[yy_, xx_]:
                    break
                r_ += 1
            rmax = max(rmax, r_ * (1.414 if dx_ and dy_ else 1.0))
        return float(rmax + 1)

    models = {}
    CO_STAR = {}                        # v4 : (fix_pos, nsat, décalage par couleur) du dernier ajustement de chaque étoile
    stc: dict = {}
    SH: dict = {}                       # déformation elliptique (a, b) par étoile
    CHROMA = {"coef": None}             # champ des décalages R-G et B-G (polynôme d'ordre 2 en x, y)

    def _poly_terms(x, y):
        u = (np.asarray(x, float) - W / 2) / max(W, H); v = (np.asarray(y, float) - H / 2) / max(W, H)
        return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], -1)

    def chroma_at(x, y):
        """Décalage (C, 2) du centre de chaque couleur par rapport au vert, au point (x, y) ; zéro sans champ mesuré."""
        if CHROMA["coef"] is None:
            return np.zeros((C, 2))
        t_ = _poly_terms(x, y)
        cf = CHROMA["coef"]                     # (4, 6) : Rx, Ry, Bx, By
        o = __import__('operator').matmul(cf, t_)
        return np.array([[o[0], o[1]], [0.0, 0.0], [o[2], o[3]]])

    def measure_chroma():
        """Décalages sous-pixel R-G et B-G mesurés sur les étoiles moyennes isolées (chaque couleur ajustée seule, position
        libre), puis champ lisse : polynôme d'ordre 2 en x, y, écrêtage 3 sigma (le chromatisme latéral croît avec la
        distance au centre optique ; le dématriçage ajoute un décalage constant)."""
        from scipy.spatial import cKDTree
        tr_ = cKDTree(np.c_[Xs[stars_in], Ys[stars_in]])
        pts = []
        for k in stars_in:
            if bright_mode[k] or giant[k] or k in PAIR or k not in models:
                continue
            if Fexp[k].sum() * peak_frac / C < 30 * med_sig.min():
                continue
            if any(G[stars_in[j]] < G[k] + 2 and stars_in[j] != k for j in tr_.query_ball_point([Xs[k], Ys[k]], 2.5 * fwhm)):
                continue
            x, y = Xs[k], Ys[k]
            rf = max(3.0, 2.0 * fwhm); rw = int(np.ceil(rf + pr["ann_w"])) + 1
            xi, yi = int(round(x)), int(round(y))
            if xi - rw < 0 or yi - rw < 0 or xi + rw >= W or yi + rw >= H:
                continue
            (my0, my1, mx0, mx1), mod = models[k]
            win = Rimg[:, yi - rw:yi + rw + 1, xi - rw:xi + rw + 1].copy()
            # l'étoile remise dans la fenêtre
            oy0, ox0 = max(yi - rw, my0), max(xi - rw, mx0); oy1, ox1 = min(yi + rw + 1, my1), min(xi + rw + 1, mx1)
            win[:, oy0 - (yi - rw):oy1 - (yi - rw), ox0 - (xi - rw):ox1 - (xi - rw)] += mod[:, oy0 - my0:oy1 - my0, ox0 - mx0:ox1 - mx0]
            st = stc.get(k) or [_blend(c, x, y, G[k]) for c in range(C)]
            pos = []
            for c in range(C):
                Fc, cx_, cy_, _, _ = fit_star(win[c:c + 1], valid[yi - rw:yi + rw + 1, xi - rw:xi + rw + 1],
                                              sat[c:c + 1, yi - rw:yi + rw + 1, xi - rw:xi + rw + 1], E, [st[c]],
                                              x - (xi - rw), y - (yi - rw), None, 1.0, rf, False, False, sig[c:c + 1, yi, xi],
                                              x - (xi - rw), y - (yi - rw), dict(pr, pos_margin=1.0, choff=np.zeros((1, 2))))
                pos.append((cx_, cy_, Fc[0]))
            if min(p_[2] for p_ in pos) <= 0:
                continue
            pts.append((x, y, pos[0][0] - pos[1][0], pos[0][1] - pos[1][1], pos[2][0] - pos[1][0], pos[2][1] - pos[1][1]))
        if len(pts) < 15:
            return f"décalages par couleur : trop peu d'étoiles ({len(pts)})"
        pts = np.array(pts)
        T = _poly_terms(pts[:, 0], pts[:, 1])
        if len(pts) < 40:
            T = T[:, :3]                         # peu d'étoiles : plan seulement
        coef = np.zeros((4, 6))
        for i in range(4):
            v = pts[:, 2 + i]
            keep = np.ones(len(v), bool)
            for _ in range(3):
                cf, *_ = np.linalg.lstsq(T[keep], v[keep], rcond=None)
                rr = v - __import__('operator').matmul(T, cf)
                sd = 1.4826 * np.median(np.abs(rr[keep])) + 1e-6
                keep = np.abs(rr) < 3 * sd
            coef[i, :T.shape[1]] = cf
        CHROMA["coef"] = coef
        stc.clear()
        fld = np.array([chroma_at(x, y)[[0, 2]].ravel() for x, y in pts[:, :2]])
        return (f"décalages par couleur (px) sur {len(pts)} étoiles : R-G médiane ({np.median(pts[:, 2]):+.3f}, {np.median(pts[:, 3]):+.3f}), "
                f"B-G médiane ({np.median(pts[:, 4]):+.3f}, {np.median(pts[:, 5]):+.3f}) ; champ ajusté : R-G de "
                f"{np.hypot(fld[:, 0], fld[:, 1]).min():.3f} à {np.hypot(fld[:, 0], fld[:, 1]).max():.3f} px, B-G de "
                f"{np.hypot(fld[:, 2], fld[:, 3]).min():.3f} à {np.hypot(fld[:, 2], fld[:, 3]).max():.3f} px")
    OWNP: dict = {}                     # profil propre mesuré des étoiles brillantes (bords, profil, rayon saturé, portée)
    SHARED = {"P": None, "rg": None}    # profil étendu partagé par couleur, par unité de flux attendu
    BRIGHT_FIT: dict = {}               # v12 : ajustement d'amplitude de chaque brillante (anneau, pixels, amplitude)

    def build_shared():
        """Profil étendu partagé : médiane, sur les étoiles brillantes (hors géantes), de leur profil propre divisé par
        leur flux attendu par couleur, sur une grille radiale commune ; chaque rayon n'utilise que les étoiles mesurées
        jusque-là (au moins 5) ; décroissant vers l'extérieur."""
        rg = np.unique(np.r_[np.arange(0, 20, 1.0), np.geomspace(20, pr["halo_rmax"], 40)])
        ks = [k for k in OWNP if not giant[k] and bright_mode[k]]
        if len(ks) < 5:
            return f"profil partagé : trop peu d'étoiles brillantes ({len(ks)})"
        P = np.full((C, len(rg)), np.nan)
        for c in range(C):
            stack = []
            for k in ks:
                ed, pf, rs_, rm_ = OWNP[k]
                v = np.interp(rg, ed + 0.5, pf[c], right=np.nan) / max(Fexp[k][c], 1e-12)
                v[rg >= rm_] = np.nan
                v[rg < rs_ + 2 * fwhm] = np.nan             # coeur saturé : pas une mesure de la forme
                stack.append(v)
            stack = np.array(stack)
            n_ok = np.isfinite(stack).sum(0)
            with np.errstate(all="ignore"):
                med = np.nanmedian(stack, 0)
            med[n_ok < 5] = np.nan
            # coeur (r < 2 largeurs) : la PSF effective des moyennes, mise à la même échelle au raccord
            ok = np.where(np.isfinite(med))[0]
            if not len(ok):
                continue
            j0 = ok[0]
            core = np.interp(rg, prof_r, prof_v)
            if core[j0] > 0:
                med[:j0] = core[:j0] * med[j0] / core[j0]
            last = ok[-1]
            med[last + 1:] = 0.0
            med = np.nan_to_num(med)
            P[c] = np.minimum.accumulate(np.maximum(med, 0.0))
        SHARED["P"], SHARED["rg"] = np.nan_to_num(P), rg
        return (f"profil étendu partagé sur {len(ks)} étoiles brillantes, portée "
                f"{[int(rg[np.where(P[c] > 0)[0].max()]) if np.any(P[c] > 0) else 0 for c in range(C)]} px")
    order = np.argsort(G)
    inimg = (xg > -R_anc) & (xg < W + R_anc) & (yg > -R_anc) & (yg < H + R_anc)
    flag[~inimg] |= 4

    def pair_refit(k, j):
        """Deux étoiles serrées ajustées ensemble : flux des deux composantes (positions Gaia, forme locale) et plan de
        fond commun sur l'anneau autour du couple, par couleur ; flux bornés par couleur (0.35-3 x l'attendu)."""
        for q in (k, j):
            old = models.pop(q, None)
            if old is not None:
                (y0, y1, x0, x1), mod = old
                Rimg[:, y0:y1, x0:x1] += mod
        xm, ym = 0.5 * (Xs[k] + Xs[j]), 0.5 * (Ys[k] + Ys[j])
        sep_ = float(np.hypot(Xs[k] - Xs[j], Ys[k] - Ys[j]))
        rf = max(3.0, 2.0 * fwhm) + sep_ / 2
        aw = pr["ann_w"]
        fp = int(max(footprint(k, Fs[k] if Fs[k].any() else None), footprint(j, Fs[j] if Fs[j].any() else None)) + sep_)
        x0, x1 = int(max(0, np.floor(xm - fp))), int(min(W, np.ceil(xm + fp + 1)))
        y0, y1 = int(max(0, np.floor(ym - fp))), int(min(H, np.ceil(ym + fp + 1)))
        if x1 - x0 < 5 or y1 - y0 < 5:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        r = np.hypot(xx - xm, yy - ym)
        inn = (r <= rf) & valid[y0:y1, x0:x1]
        ann = (r > rf) & (r <= rf + aw) & valid[y0:y1, x0:x1]
        sts = []
        for q in (k, j):
            st = stc.get(q)
            if st is None:
                st = stc[q] = [_blend(c, min(max(Xs[q], 0), W - 1), min(max(Ys[q], 0), H - 1), G[q]) for c in range(C)]
            sts.append(st)
        mod = np.zeros((C, y1 - y0, x1 - x0), np.float32)
        mods = [np.zeros_like(mod), np.zeros_like(mod)]
        for c in range(C):
            ok = inn & ~sat[c, y0:y1, x0:x1]; okA = ann & ~sat[c, y0:y1, x0:x1]
            if ok.sum() < 8 or okA.sum() < 12:
                continue
            ms = [E.eval(sts[i][c], xx - Xs[q], yy - Ys[q]) for i, q in enumerate((k, j))]
            Rc = Rimg[c, y0:y1, x0:x1]
            f = np.zeros(2)
            Ab = np.c_[np.ones(okA.sum()), (xx[okA] - xm) / rf, (yy[okA] - ym) / rf]
            for _ in range(3):
                yb = Rc[okA] - f[0] * ms[0][okA] - f[1] * ms[1][okA]
                bsol = np.linalg.lstsq(Ab, yb, rcond=None)[0]
                bg = bsol[0] + bsol[1] * (xx - xm) / rf + bsol[2] * (yy - ym) / rf
                M = np.c_[ms[0][ok], ms[1][ok]]
                f = np.linalg.lstsq(M, (Rc - bg)[ok], rcond=None)[0]
            for i, q in enumerate((k, j)):
                f[i] = float(np.clip(f[i], pr.get("colour_bound", (0.35, 3.0))[0] * Fexp[q][c], pr.get("colour_bound", (0.35, 3.0))[1] * Fexp[q][c]))
                mods[i][c] = f[i] * ms[i]
            Fs[k][c], Fs[j][c] = f[0], f[1]
        for i, q in enumerate((k, j)):
            Rimg[:, y0:y1, x0:x1] -= mods[i]
            models[q] = ((y0, y1, x0, x1), mods[i])

    def refit(k, free_width, loop_round=0):
        if bright_mode[k]:
            profile_refit(k)
            return
        if k in PAIR and not bright_mode[PAIR[k][0]]:
            if PAIR[k][1]:
                pair_refit(k, PAIR[k][0])
            return
        x, y = Xs[k], Ys[k]
        old = models.get(k)
        if old is not None:
            (y0, y1, x0, x1), mod = old
            Rimg[:, y0:y1, x0:x1] += mod
        fp = footprint(k, Fs[k] if Fs[k].any() else None)
        x0, x1 = int(max(0, np.floor(x - fp))), int(min(W, np.ceil(x + fp + 1)))
        y0, y1 = int(max(0, np.floor(y - fp))), int(min(H, np.ceil(y + fp + 1)))
        if x1 - x0 < 3 or y1 - y0 < 3:
            models.pop(k, None)
            return
        # ajustement sur le coeur + un anneau de fond
        rf = max(3.0, 2.0 * fwhm) * (1.6 if Fpred[k] > 200 * med_sig.min() * fwhm ** 2 else 1.0)
        rs = sat_radius(x, y)
        if rs > 0:                      # coeur écrêté : le flux se mesure sur les ailes non saturées autour
            rf = max(rf, rs + 3.0 * fwhm)
        rw = int(np.ceil(rf + pr["ann_w"])) + 1
        fx0, fx1 = int(max(0, np.floor(x - rw))), int(min(W, np.ceil(x + rw + 1)))
        fy0, fy1 = int(max(0, np.floor(y - rw))), int(min(H, np.ceil(y + rw + 1)))
        if fx1 - fx0 < 3 or fy1 - fy0 < 3:
            models.pop(k, None)
            return
        st = stc.get(k)
        if st is None:
            st = stc[k] = [_blend(c, min(max(x, 0), W - 1), min(max(y, 0), H - 1), G[k]) for c in range(C)]
        win = Rimg[:, fy0:fy1, fx0:fx1]
        vwin = valid[fy0:fy1, fx0:fx1]
        swin = sat[:, fy0:fy1, fx0:fx1]
        # le halo des étoiles brillantes est soustrait de la fenêtre d'ajustement avec le flux précédent
        sk = sig[:, int(np.clip(y, 0, H - 1)), int(np.clip(x, 0, W - 1))]
        nsat = int(swin.any(0)[np.hypot(*np.meshgrid(np.arange(fx0, fx1) - x, np.arange(fy0, fy1) - y)) <= rf].sum())
        _fixp = (Fexp[k].sum() * peak_frac / C < 4 * med_sig.min()) or nsat > 0
        _co_out = [None]
        F, nx, ny, s, chi = fit_star(win, vwin, swin, E, st, x - fx0, y - fy0, None, Ss[k], rf,
                                     fix_pos=_fixp,
                                     free_width=free_width and nsat == 0, sig=sk,
                                     x_cat=xg[k] - fx0, y_cat=yg[k] - fy0,
                                     P=dict(pr, choff=chroma_at(x, y), colour_pos_snr=COLOUR_POS_SNR, _co_out=_co_out, ellipse=pr.get("ellipse", True) and
                                            Fexp[k].sum() * peak_frac / C > 20 * med_sig.min()))
        nx += fx0; ny += fy0
        # saturée : le flux du coeur n'est pas mesurable ; flux pris des ailes si possible, borné par Gaia
        Fg = Fexp[k]
        if nsat > 0 or not np.all(np.isfinite(F)):
            F = np.where(np.isfinite(F) & (F > 0), F, Fg)
            F = np.clip(F, 0.4 * Fg, pr["flux_guard"] * Fg)
        # borne par couleur autour du flux attendu (0.35 - 3 x, ancien FLOAT : sans elle, points verts et trous noirs),
        # là où l'attendu est mesurable (au-dessus de 5 bruits sur l'aire de l'étoile)
        meas = Fg > 5 * sk * np.sqrt(ncorr) * fwhm
        F = np.where(meas, np.clip(F, pr.get("colour_bound", (0.35, 3.0))[0] * Fg, pr.get("colour_bound", (0.35, 3.0))[1] * Fg), F)
        # garde-fou Gaia
        Ft = float(np.sum(F))
        lim = pr["flux_guard"] * 10 ** (0.4 * 2 * zsd) * Fpred[k]
        if Ft > lim and Fpred[k] > 3 * med_sig.min() * fwhm ** 2:
            F = F * (lim / Ft)
            flag[k] |= 1
        F = np.maximum(F, 0)
        Fs[k] = F; Xs[k] = nx; Ys[k] = ny
        if np.ndim(s):
            Ss[k] = s[0]; SH[k] = (s[1], s[2])
        else:
            Ss[k] = s
            if k in SH:                     # forme elliptique trouvée à une passe précédente : gardée
                s = (s, *SH[k])
        _co = _co_out[0] if _co_out[0] is not None else chroma_at(nx, ny)
        CO_STAR[k] = (_fixp, nsat, np.array(_co))
        mod = star_model(E, st, nx, ny, F, s, x0, x1, y0, y1, HA[k], choff=_co)
        # garde « pas de trou » : dans le coeur de l'étoile (modèle > 1 bruit), le sans-étoiles ne doit pas descendre
        # sous le fond local (anneau juste au-delà) de plus d'un demi-bruit en moyenne ; sinon le flux de cette couleur
        # est réduit d'autant
        yy_, xx_ = np.mgrid[y0:y1, x0:x1]
        rr_ = np.hypot(xx_ - nx, yy_ - ny)
        annz = (rr_ > rf) & (rr_ <= rf + pr["ann_w"]) & valid[y0:y1, x0:x1]
        for c in range(C):
            core = (mod[c] > sk[c]) & valid[y0:y1, x0:x1] & ~sat[c, y0:y1, x0:x1]
            if core.sum() < 3 or annz.sum() < 8 or F[c] <= 0:
                continue
            after = Rimg[c, y0:y1, x0:x1] - mod[c]
            bgv = float(np.median(after[annz]))
            deficit = bgv - float(after[core].mean())
            mc = float(mod[c][core].mean())
            if deficit > 0.5 * sk[c] and pr.get("flux_guard_hole", False):
                a_ = min(1.0, (deficit - 0.5 * sk[c]) / max(mc, 1e-9))
                mod[c] *= (1 - a_)
                F[c] *= (1 - a_)
                hole_guard[0] += 1
            elif deficit < -0.5 * sk[c] and F[c] < 3.0 * Fexp[k][c]:
                # reste positif : le flux remonte d'autant, sans passer 3 x l'attendu
                a_ = min((-deficit - 0.5 * sk[c]) / max(mc, 1e-9), 3.0 * Fexp[k][c] / max(F[c], 1e-9) - 1)
                mod[c] *= (1 + a_)
                F[c] *= (1 + a_)
            # au pixel : ce qui reste sous le plan de fond local moins 1 bruit dans le coeur est remplacé par ce plan
            # plus le grain de l'image (le modèle effectif de l'étoile inclut ce remplacement, pour les passes suivantes)
            if pr.get("pixel_guard", True) and not pr.get("no_fill", False) \
                    and Fexp[k].sum() * peak_frac / C >= pr.get("faint_noguard", 0.0) * med_sig.min():
                after = Rimg[c, y0:y1, x0:x1] - mod[c]
                Ab = np.c_[np.ones(annz.sum()), (xx_[annz] - nx) / 10.0, (yy_[annz] - ny) / 10.0]
                bsol = np.linalg.lstsq(Ab, after[annz], rcond=None)[0]
                plane = bsol[0] + bsol[1] * (xx_ - nx) / 10.0 + bsol[2] * (yy_ - ny) / 10.0
                # seuil sur l'écart lissé (1 px) à -3 bruits : le bruit seul n'y passe presque jamais (remplacer chaque
                # pixel négatif du bruit relèverait le fond) ; la zone trouvée est élargie d'un pixel
                dev = cv2.GaussianBlur((after - plane).astype(np.float32), (0, 0), 1.0)
                # tour 13 : seulement là où le modèle peut avoir causé le creux (modèle > 2 x le creux) ; sinon c'est la
                # texture de la nébuleuse (le remplacement de ses creux seuls relevait le fond : +8 % au coeur)
                low = core & ndi.binary_dilation((dev < -4 * sk[c]) & (mod[c] > 2 * (plane - after)))
                if low.any():
                    newv = plane + grain[c, y0:y1, x0:x1] * sk[c]
                    mod[c][low] = (Rimg[c, y0:y1, x0:x1] - newv)[low]
                    pix_guard[0] += int(low.sum())
        Fs[k] = F
        Rimg[:, y0:y1, x0:x1] -= mod
        models[k] = ((y0, y1, x0, x1), mod)

    hole_guard = [0]
    pix_guard = [0]
    spike_models: dict = {}
    # grain de l'image (spectre de puissance du ciel calme), variance 1, pour les pixels remplacés
    amp_g = grain_kernel(img, star_mask, 64)
    rng_g = np.random.default_rng(7)
    grain = np.stack([synth_grain(amp_g[c], (H, W), rng_g) for c in range(C)])

    def spike_model():
        sel = [k for k in stars_in if bright_mode[k] and not giant[k] and k in models
               and 50 < Xs[k] < W - 50 and 50 < Ys[k] < H - 50]
        if len(sel) < 2:
            return "aigrettes : trop peu d'étoiles brillantes"
        nA = 180
        w = max(1.5, 0.8 * fwhm)
        Z = np.zeros(nA)
        geo = {}
        top_sel = set(sorted(sel, key=lambda k: G[k])[:20])      # les angles se lisent sur les 20 plus brillantes
        for k in sel:
            x, y = Xs[k], Ys[k]
            (my0, my1, mx0, mx1), _ = models[k]
            r0 = sat_radius(x, y) + 2 * fwhm
            r1 = max(r0 + 15, 0.95 * min(x - mx0, mx1 - x, y - my0, my1 - y, 6 * R_anc))
            r1a = min(r1, r0 + 30)          # angles cherchés près de l'étoile, où l'aigrette domine
            y0, y1 = int(max(0, y - r1)), int(min(H, y + r1 + 1)); x0, x1 = int(max(0, x - r1)), int(min(W, x + r1 + 1))
            yy, xx = np.mgrid[y0:y1, x0:x1]
            r = np.hypot(xx - x, yy - y)
            ia = ((np.arctan2(yy - y, xx - x) + np.pi) / (2 * np.pi) * nA).astype(int) % nA
            ok = (r > r0) & (r < r1a) & valid[y0:y1, x0:x1]
            Lr = Rimg[:, y0:y1, x0:x1].mean(0)
            # médiane par secteur de 2 deg (une voisine dans un secteur ne fait pas une aigrette)
            o_ = np.argsort(ia[ok], kind="stable")
            ia_s, v_s = ia[ok][o_], Lr[ok][o_]
            cuts = np.searchsorted(ia_s, np.arange(nA + 1))
            mean = np.array([np.median(v_s[cuts[j]:cuts[j + 1]]) if cuts[j + 1] > cuts[j] else 0.0 for j in range(nA)])
            z = (mean - np.median(mean)) / (1.4826 * np.median(np.abs(mean - np.median(mean))) + 1e-9)
            if k in top_sel:
                Z += np.clip(z, -5, 20)
            geo[k] = (r0, r1)
        Zs = np.convolve(np.r_[Z[-2:], Z, Z[:2]], np.ones(5) / 5, mode="valid")
        # une aigrette de diffraction a sa jumelle à 180 deg : un angle n'est retenu que si son opposé dépasse aussi le
        # seuil (un pan de nébuleuse d'un seul côté n'est pas une aigrette)
        thr = max(np.median(Zs) + 1.4826 * np.median(np.abs(Zs - np.median(Zs))), 0.2 * Zs.max(), 3.0 * np.sqrt(len(top_sel)))
        half = nA // 2
        peaks = [j for j in range(nA) if Zs[j] > thr and Zs[j] >= Zs[(j - 1) % nA] and Zs[j] >= Zs[(j + 1) % nA]
                 and max(Zs[(j + half + d) % nA] for d in range(-3, 4)) > thr]
        # au plus 8 angles, séparés d'au moins 6 deg
        peaks = sorted(peaks, key=lambda j: -Zs[j])
        angs = []
        for j in peaks:
            if all(min(abs(j - a), nA - abs(j - a)) >= 6 for a in angs):
                angs.append(j)
            if len(angs) >= 8:
                break
        if not angs:
            return "aigrettes : aucun angle d'instrument trouvé"
        thetas = [-np.pi + (j + 0.5) * 2 * np.pi / nA for j in angs]
        # profils le long de chaque aigrette, par couleur, normalisés par le flux attendu
        ub = np.geomspace(1, 8 * R_anc, 25)
        prof_all = []
        meas = {}
        for k in sel:
            x, y = Xs[k], Ys[k]
            r0, r1 = geo[k]
            pk_ = []
            for th in thetas:
                ct, st_ = np.cos(th), np.sin(th)
                L_ = int(r1)
                y0, y1 = int(max(0, y - L_)), int(min(H, y + L_ + 1)); x0, x1 = int(max(0, x - L_)), int(min(W, x + L_ + 1))
                yy, xx = np.mgrid[y0:y1, x0:x1]
                u = (xx - x) * ct + (yy - y) * st_
                v = -(xx - x) * st_ + (yy - y) * ct
                core = (np.abs(v) <= w) & (u > r0) & (u < r1) & valid[y0:y1, x0:x1]
                flank = (np.abs(v) > 2 * w) & (np.abs(v) <= 4 * w) & (u > r0) & (u < r1) & valid[y0:y1, x0:x1]
                bi = np.searchsorted(ub, u) - 1
                p = np.full((C, len(ub) - 1), np.nan)
                for c in range(C):
                    Rc = Rimg[c, y0:y1, x0:x1]
                    okc = ~sat[c, y0:y1, x0:x1]
                    for b_ in range(len(ub) - 1):
                        m1 = core & okc & (bi == b_); m2 = flank & okc & (bi == b_)
                        if m1.sum() >= 3 and m2.sum() >= 6:
                            p[c, b_] = Rc[m1].mean() - np.median(Rc[m2])
                pk_.append(p)
            meas[k] = pk_
            for p in pk_:
                prof_all.append(p / np.maximum(Fexp[k][:, None], 1e-9))
        T = np.nanmedian(np.array(prof_all), 0)              # gabarit (C, bins) par unité de flux attendu
        T = np.where(np.isfinite(T), np.maximum(T, 0), 0)
        n_sub = 0
        for k in sel:
            x, y = Xs[k], Ys[k]
            r0, r1 = geo[k]
            for th, p in zip(thetas, meas[k]):
                tm = T * Fexp[k][:, None]
                okb = np.isfinite(p) & (tm > 0)
                if okb.sum() < 3:
                    continue
                a_ = float(np.clip((p[okb] * tm[okb]).sum() / max((tm[okb] ** 2).sum(), 1e-12), 0.3, 3.0))
                # profil propre lissé si mesuré, sinon gabarit ; jamais négatif ; borné à 3 x le gabarit
                prof_k = np.where(np.isfinite(p), np.clip(p, 0, 3 * tm), a_ * tm)
                prof_k = np.clip(prof_k, 0, None)
                # aigrette réellement vue chez cette étoile : 3 tranches de suite au-dessus de 2 bruits
                pl = np.nan_to_num(p).mean(0)
                seen = pl > 2 * med_sig.min()
                if not np.any(seen[:-2] & seen[1:-1] & seen[2:]):
                    continue
                ct, st_ = np.cos(th), np.sin(th)
                L_ = int(r1)
                y0, y1 = int(max(0, y - L_)), int(min(H, y + L_ + 1)); x0, x1 = int(max(0, x - L_)), int(min(W, x + L_ + 1))
                yy, xx = np.mgrid[y0:y1, x0:x1]
                u = (xx - x) * ct + (yy - y) * st_
                v = -(xx - x) * st_ + (yy - y) * ct
                um = 0.5 * (ub[:-1] + ub[1:])
                sv = fwhm / 2.355           # une aigrette a la largeur de l'étoile (diffraction du même faisceau)
                gm = float(np.mean(np.exp(-0.5 * (np.linspace(-w, w, 21) / sv) ** 2)))
                g_ = np.exp(-0.5 * (v / sv) ** 2) / gm * ((u > r0) & (u < r1) & (np.abs(v) <= 4 * w))
                mod = np.stack([np.interp(u, um, prof_k[c], left=0, right=0) * g_ for c in range(C)]).astype(np.float32)
                Rimg[:, y0:y1, x0:x1] -= mod
                spike_models.setdefault(k, []).append(((y0, y1, x0, x1), mod))
                n_sub += 1
        return (f"aigrettes : angles d'instrument {[round(float(np.degrees(t)), 1) for t in thetas]} deg, "
                f"gabarit sur {len(sel)} étoiles, {n_sub} aigrettes soustraites")

    stars_in = order[inimg[order]]
    log(f"{len(stars_in)} étoiles Gaia à ajuster")

    # étoiles brillantes : leur profil (coeur écrêté, ailes larges, halo) n'est pas celui des étoiles moyennes qui font la
    # PSF effective ; elles sont modélisées par leur propre profil radial mesuré sur l'image (médiane par anneau de la
    # médiane de 8 secteurs : un noeud de nébuleuse ou une voisine dans un secteur ne compte pas), moins le fond de
    # l'anneau au-delà, rendu décroissant vers l'extérieur et >= 0 (soustraction de profil radial des étoiles brillantes,
    # pratique de l'imagerie profonde ; l'ancien FLOAT prenait de même les ailes du profil médian des étoiles G 6-9).
    peak_frac = float(prof_v[0])
    # étoile Gaia invisible sur un fond très brillant (pic attendu < 5 % du fond, fond > 50 bruits : dans le coeur d'une
    # planétaire, des sources Gaia faibles, parfois fausses) : laissée telle quelle, l'ajuster ne ferait que manger la
    # nébuleuse
    keep_ = []
    n_inv = 0
    n_faint = 0
    for k in stars_in:
        # tour 19 (catalogue profond) : pic attendu sous min_peak bruits : rien à retirer de mesurable, pas ajustée
        if Fexp[k].sum() / C * peak_frac < pr.get("min_peak", 1.0) * med_sig.min():
            n_faint += 1
            continue
        xi, yi = int(round(xg[k])), int(round(yg[k]))
        if 0 <= xi < W and 0 <= yi < H:
            r_ = int(4 * fwhm)
            cut = L[max(0, yi - r_):yi + r_ + 1, max(0, xi - r_):xi + r_ + 1]
            bgl = float(np.median(cut))
            if bgl > 50 * med_sig.min() and Fexp[k].sum() / C * peak_frac < 0.05 * bgl:
                n_inv += 1
                continue
        keep_.append(k)
    stars_in = np.array(keep_, int)
    log(f"étoiles Gaia invisibles sur fond très brillant, laissées : {n_inv} ; trop faibles pour être vues "
        f"(pic attendu < {pr.get('min_peak', 1.0)} bruit) : {n_faint} ; à ajuster : {len(stars_in)}")
    # couples serrés (< 0.6 largeur, ~2 px) : une seule source à deux composantes non séparables à cette résolution,
    # au barycentre des flux attendus, avec la somme des flux attendus (sinon l'une prend tout, l'autre finit à 0 et le
    # coeur reste)
    from scipy.spatial import cKDTree
    trp = cKDTree(np.c_[xg[stars_in], yg[stars_in]])
    gone = set()
    n_pairs = 0
    for i_, j_ in sorted(trp.query_pairs(0.6 * fwhm), key=lambda p: min(G[stars_in[p[0]]], G[stars_in[p[1]]])):
        a_, b_ = stars_in[i_], stars_in[j_]
        if a_ in gone or b_ in gone:
            continue
        if G[b_] < G[a_]:
            a_, b_ = b_, a_
        fa, fb = Fexp[a_].sum(), Fexp[b_].sum()
        wa = fa / max(fa + fb, 1e-12)
        xg[a_] = wa * xg[a_] + (1 - wa) * xg[b_]; yg[a_] = wa * yg[a_] + (1 - wa) * yg[b_]
        Xs[a_], Ys[a_] = xg[a_], yg[a_]
        Fexp[a_] = Fexp[a_] + Fexp[b_]
        Fpred[a_] = Fpred[a_] + Fpred[b_]
        G[a_] = -2.5 * np.log10(10 ** (-0.4 * G[a_]) + 10 ** (-0.4 * G[b_]))
        gone.add(b_)
        n_pairs += 1
    stars_in = np.array([k for k in stars_in if k not in gone], int)
    log(f"couples serrés fondus en une source : {n_pairs}")
    # tour 18 : couples à moins de 3 px (au-delà de la fusion) ajustés ensemble, deux composantes aux positions Gaia,
    # fond commun (ALLSTAR : les voisines d'un groupe s'ajustent simultanément)
    trp2 = cKDTree(np.c_[xg[stars_in], yg[stars_in]])
    PAIR: dict = {}
    for i_, j_ in trp2.query_pairs(pr.get("pair_px", 3.0)):
        a_, b_ = stars_in[i_], stars_in[j_]
        if a_ in PAIR or b_ in PAIR:
            continue
        if G[b_] < G[a_]:
            a_, b_ = b_, a_
        PAIR[a_] = (b_, True); PAIR[b_] = (a_, False)       # (partenaire, principal ?)
    log(f"couples à moins de {pr.get('pair_px', 3.0)} px ajustés ensemble : {len(PAIR) // 2}")
    bright_mode = np.zeros(N, bool)
    for k in stars_in:
        bright_mode[k] = sat_radius(xg[k], yg[k]) > 0 or Fpred[k] * peak_frac / C > pr["profile_sat"] * ref_lvl.min()
    log(f"étoiles brillantes modélisées par leur profil : {int(bright_mode.sum())}")
    # géantes (classe sigma Ori, Alnitak) : laissées entières, leur zone exclue (décision du propriétaire : les effacer
    # est impossible même à la main et dégrade le reste). Gaia G < giant_G, plus les grandes zones saturées sans étoile
    # Gaia (Alnitak manque à l'extrait hors ligne). Leur lumière est modélisée pendant les passes (pour les voisines),
    # puis rendue à la fin.
    giant = np.zeros(N, bool)
    for k in stars_in:
        giant[k] = G[k] < pr["giant_G"]
    lab_s, ns = ndi.label(satany)
    giant_blobs = []
    if ns:
        areas = np.bincount(lab_s.ravel())[1:]
        coms = ndi.center_of_mass(satany, lab_s, np.arange(1, ns + 1))
        big = np.where(areas > pr["giant_sat_area"] * fwhm ** 2)[0]
        for i in big:
            cy_, cx_ = coms[i]
            rb = np.sqrt(areas[i] / np.pi)
            if not np.any(np.hypot(xg[stars_in] - cx_, yg[stars_in] - cy_) < rb + 2 * fwhm):
                giant_blobs.append((cx_, cy_, rb))
    log(f"géantes laissées entières : {int(giant.sum())} Gaia (G < {pr['giant_G']}), {len(giant_blobs)} zones saturées hors catalogue")

    def profile_refit(k):
        x, y = Xs[k], Ys[k]
        old = models.get(k)
        if old is not None:
            (y0, y1, x0, x1), mod = old
            Rimg[:, y0:y1, x0:x1] += mod
        rs = sat_radius(x, y)
        rmax = int(min(pr["halo_rmax"], max(4 * R_anc, 3 * rs + 4 * R_anc)))
        for attempt in range(2):
            ro = int(rmax * 1.25) + 4
            x0, x1 = int(max(0, x - ro)), int(min(W, x + ro + 1)); y0, y1 = int(max(0, y - ro)), int(min(H, y + ro + 1))
            x1, y1 = max(x1, x0), max(y1, y0)   # SHO 21 nov : brillante de la marge hors decoupe -> fenetre vide (garde)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            r = np.hypot(xx - x, yy - y)
            ang = ((np.arctan2(yy - y, xx - x) + np.pi) / (2 * np.pi) * 8).astype(int) % 8
            edges = np.r_[np.arange(0, min(20, rmax), 1.0), np.geomspace(20, max(rmax, 21), 40)] if rmax > 20 else np.arange(0, rmax + 1, 1.0)
            edges = np.unique(edges)
            b = np.searchsorted(edges, r, side="right") - 1
            vwin = valid[y0:y1, x0:x1]
            ring = vwin & (r > rmax) & (r <= ro)
            prof = np.zeros((C, len(edges)))
            # allongement : harmonique d'ordre 2 de l'anneau (16 secteurs, les 3 plus hauts écartés : aigrettes,
            # voisines), en part du profil ; une étoile réelle n'est pas ronde (ellipticité ~0.13 mesurée) et un profil
            # rond laisse alors un trèfle positif/négatif
            ang16 = ((np.arctan2(yy - y, xx - x) + np.pi) / (2 * np.pi) * 16).astype(int) % 16
            th16 = -np.pi + (np.arange(16) + 0.5) * 2 * np.pi / 16
            h2 = np.zeros((C, len(edges), 2))
            for c in range(C):
                Rc = Rimg[c, y0:y1, x0:x1]
                okc = vwin & ~sat[c, y0:y1, x0:x1]
                if ring.sum() < 20:
                    break
                ref = np.median(Rc[ring & okc]) if (ring & okc).sum() > 10 else 0.0
                p = np.full(len(edges), np.nan)
                for i in range(len(edges) - 1):
                    if edges[i] >= rmax:
                        break
                    inb = okc & (b == i)
                    if inb.sum() < 3:
                        continue
                    # tour 16 : 16 secteurs (avec 8, les 4 aigrettes en touchaient la moitié : profil trop haut entre
                    # elles, d'où les creux sombres en 4 lobes autour des étoiles à aigrettes)
                    if edges[i] >= 6:
                        secs = [np.median(Rc[inb & (ang16 == a)]) for a in range(16) if (inb & (ang16 == a)).sum() >= 2]
                    else:
                        secs = [np.median(Rc[inb & (ang == a)]) for a in range(8) if (inb & (ang == a)).sum() >= 2]
                    if len(secs) >= 4:
                        p[i] = np.median(secs) - ref
                        s16 = np.array([np.median(Rc[inb & (ang16 == a)]) if (inb & (ang16 == a)).sum() >= 2 else np.nan
                                        for a in range(16)]) - ref
                        oks = np.isfinite(s16)
                        if oks.sum() >= 10 and p[i] > 3 * med_sig.min():
                            idx_ = np.where(oks)[0]
                            resid = s16[idx_] - p[i]
                            keep_ = idx_[np.argsort(resid)[:-3]]
                            M_ = np.c_[np.cos(2 * th16[keep_]), np.sin(2 * th16[keep_])]
                            cs = np.linalg.lstsq(M_, s16[keep_] - p[i], rcond=None)[0]
                            # tour 11 : anneaux concentriques et disques ternes -> coupé (réglage « elong »)
                            if pr.get("elong", False):
                                h2[c, i] = np.clip(cs / p[i], -0.5, 0.5)
                # coeur saturé : valeur du premier anneau mesuré (zone remplie ensuite)
                first = np.where(np.isfinite(p))[0]
                if not len(first):
                    continue
                p[:first[0]] = p[first[0]]
                p = np.where(np.isfinite(p), p, 0.0)
                p[edges >= rmax] = 0.0
                p = np.minimum.accumulate(np.maximum(p, 0.0))
                prof[c] = p
            # étendue : dernier anneau au-dessus de 0.1 bruit ; une fois agrandie si le profil touche le bord
            lvl = pr["halo_level"] * med_sig.min()
            above = np.where(prof.max(0) > lvl)[0]
            reach = edges[above[-1] + 1] if len(above) and above[-1] + 1 < len(edges) else rmax
            if reach < 0.8 * rmax or rmax >= pr["halo_rmax"]:
                break
            rmax = int(min(pr["halo_rmax"], rmax * 2))
        OWNP[k] = (edges.copy(), prof.copy(), rs, rmax)
        if SHARED["P"] is not None and SHARED.get("v12") and G[k] >= 7.0:   # v13 : G < 7 = profil propre de v10
            # v12 : forme ÉTENDUE partagée par l'image (psf_etendue.py, Infante-Sainz, Trujillo & Román 2020), par unité
            # de flux ; seule l'AMPLITUDE de l'étoile est ajustée (avec un fond plat), sur les pixels d'un anneau intérieur
            # où l'étoile domine (r de max(2, rs/FWHM + 2) à +4 largeurs), voisines masquées, écrêtage 3 sigma : la lumière
            # de la nébuleuse à grande échelle va dans le fond plat, pas dans l'étoile. Le coeur (profil propre jusqu'à
            # r_core, saturé ou non) est gardé tel quel.
            import psf_etendue as PSFE
            rg, Pn = SHARED["rg"], SHARED["P"]
            r_core = (rs + 2 * fwhm) if rs > 0 else 1.5 * fwhm
            mids_own = edges + 0.5
            r_in, r_out = PSFE.ring_radii(fwhm, rs)
            ringp = vwin & (r >= r_in) & (r <= r_out)
            # voisines masquées là où leur lumière attendue (éclat Gaia x forme) dépasse 20 % de celle de l'étoile (une
            # voisine compte là où elle pèse, comme le critère de groupe de DAOPHOT) : une voisine plus faible au bord du
            # coeur ne masque pas tout l'anneau ; une étoile noyée dans une plus brillante (< 30 pixels libres) n'est pas
            # ajustée (attendu Gaia)
            nbm = np.ones_like(ringp)
            Pm = Pn.mean(0)
            iy_, ix_ = np.nonzero(ringp)
            px_, py_ = xx[iy_, ix_].astype(float), yy[iy_, ix_].astype(float)
            Pk = np.interp(r[iy_, ix_], rg, Pm, right=0.0)
            free = np.ones(len(iy_), bool)
            for j in (SHARED["tree"].query_ball_point([x, y], r_out + PSFE.R_EXT) if len(iy_) else []):
                if j == k or G[j] >= 18.5:
                    continue
                dj = float(np.hypot(xg[j] - x, yg[j] - y))
                if dj < 0.6 * fwhm:
                    continue                                  # couple fondu avec l'étoile
                q_ = 10 ** (-0.4 * (G[j] - G[k]))
                if q_ * np.interp(max(dj - r_out, 0.0), rg, Pm, right=0.0) <= 0.2 * Pk.min():
                    continue                                  # ne pèse nulle part sur l'anneau
                free &= ~(q_ * np.interp(np.hypot(px_ - xg[j], py_ - yg[j]), rg, Pm, right=0.0) > 0.2 * Pk)
            nbm[iy_, ix_] = free
            amp = np.zeros(C)
            fitinfo = []
            for c in range(C):
                okp = ringp & ~sat[c, y0:y1, x0:x1]
                use = okp & nbm
                a_, b_, n_ = PSFE.fit_amplitude(Rimg[c, y0:y1, x0:x1][use], np.interp(r[use], rg, Pn[c]))
                amp[c] = a_ if (np.isfinite(a_) and a_ > 0) else Fexp[k][c]
                amp[c] = float(np.clip(amp[c], 0.35 * Fexp[k][c], 3.0 * Fexp[k][c]))
                fitinfo.append((a_, b_, n_, int(use.sum())))
            BRIGHT_FIT[k] = dict(r_in=r_in, r_out=r_out, rs=rs, fit=fitinfo, amp=amp.copy(), Fexp=np.array(Fexp[k]))
            lvl = pr["halo_level"] * med_sig.min()
            above_ = (amp[:, None] * Pn).max(0) > lvl
            reach = rg[np.where(above_)[0].max()] if above_.any() else rmax
            ro = int(min(pr["halo_rmax"], reach) + 4)
            x0, x1 = int(max(0, x - ro)), int(min(W, x + ro + 1)); y0, y1 = int(max(0, y - ro)), int(min(H, y + ro + 1))
            x1, y1 = max(x1, x0), max(y1, y0)   # SHO 21 nov : brillante de la marge hors decoupe -> fenetre vide (garde)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            r = np.hypot(xx - x, yy - y)
            mod = np.zeros((C, y1 - y0, x1 - x0), np.float32)
            t_ = np.clip((r - r_core) / 2.0, 0, 1)                      # fondu sur 2 px
            for c in range(C):
                shared_c = amp[c] * np.interp(r, rg, Pn[c], right=0.0)
                own_c = np.interp(np.clip(r, 0, edges[-1]), mids_own, prof[c], right=0.0)
                mod[c] = (1 - t_) * own_c + t_ * shared_c
            Rimg[:, y0:y1, x0:x1] -= mod
            models[k] = ((y0, y1, x0, x1), mod)
            Fs[k] = amp
            return
        if SHARED["P"] is not None and not SHARED.get("v12"):   # v13 : G < 7 saute ce chemin (v10 : P toujours None ici) -> profil propre
            # tour 18 : profil étendu partagé (coeur + ailes mesurés sur les brillantes, une seule forme par couleur),
            # mis à l'échelle de chaque étoile par son amplitude ajustée sur ses ailes non saturées ; le profil propre ne
            # sert que dans la zone du coeur saturé. La lumière retirée à grande échelle est alors fixée par les flux et
            # ne peut pas prendre la nébuleuse (un profil propre libre en prenait sur la pente d'un halo de géante).
            rg, Pn = SHARED["rg"], SHARED["P"]
            r_core = (rs + 2 * fwhm) if rs > 0 else 1.5 * fwhm
            mids_own = edges + 0.5
            amp = np.zeros(C)
            for c in range(C):
                own_c = np.interp(rg, mids_own, prof[c], right=np.nan)
                w_ = (rg >= r_core) & (rg <= r_core + 6 * fwhm) & np.isfinite(own_c) & (Pn[c] > 0)
                if w_.sum() >= 3:
                    amp[c] = float((own_c[w_] * Pn[c][w_]).sum() / max((Pn[c][w_] ** 2).sum(), 1e-30))
                else:
                    amp[c] = Fexp[k][c]
                amp[c] = float(np.clip(amp[c], 0.35 * Fexp[k][c], 3.0 * Fexp[k][c]))
            # portée : où le profil partagé à cette amplitude tombe sous halo_level bruits
            lvl = pr["halo_level"] * med_sig.min()
            reach = rg[np.where((amp[:, None] * Pn).max(0) > lvl)[0].max()] if np.any((amp[:, None] * Pn).max(0) > lvl) else rmax
            ro = int(min(pr["halo_rmax"], reach) + 4)
            x0, x1 = int(max(0, x - ro)), int(min(W, x + ro + 1)); y0, y1 = int(max(0, y - ro)), int(min(H, y + ro + 1))
            x1, y1 = max(x1, x0), max(y1, y0)   # SHO 21 nov : brillante de la marge hors decoupe -> fenetre vide (garde)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            r = np.hypot(xx - x, yy - y)
            mod = np.zeros((C, y1 - y0, x1 - x0), np.float32)
            t_ = np.clip((r - r_core) / 2.0, 0, 1)                      # fondu sur 2 px
            for c in range(C):
                shared_c = amp[c] * np.interp(r, rg, Pn[c], right=0.0)
                own_c = np.interp(np.clip(r, 0, edges[-1]), mids_own, prof[c], right=0.0)
                mod[c] = (1 - t_) * own_c + t_ * shared_c
            Rimg[:, y0:y1, x0:x1] -= mod
            models[k] = ((y0, y1, x0, x1), mod)
            Fs[k] = amp
            return
        mod = np.zeros((C, y1 - y0, x1 - x0), np.float32)
        rc = np.clip(r, 0, edges[-1])
        mids = edges + 0.5
        th = np.arctan2(yy - y, xx - x)
        for c in range(C):
            hc = ndi.uniform_filter1d(h2[c, :, 0], 3); hs = ndi.uniform_filter1d(h2[c, :, 1], 3)
            mod[c] = np.interp(rc, mids, prof[c], right=0.0) * np.clip(
                1 + np.interp(rc, mids, hc) * np.cos(2 * th) + np.interp(rc, mids, hs) * np.sin(2 * th), 0, None)
        Rimg[:, y0:y1, x0:x1] -= mod
        models[k] = ((y0, y1, x0, x1), mod)
        # flux équivalent (pour le rapport et la garde Gaia) : somme du profil, coeur compris
        Fs[k] = mod.reshape(C, -1).sum(1)
        if Fs[k].sum() > 3.0 * Fexp[k].sum():
            # le profil prend bien plus que l'étoile attendue : c'est de la nébuleuse (noyau d'une planétaire, etc.).
            # L'étoile repasse en ajustement de PSF avec fond local (garde Gaia), le profil est rendu.
            Rimg[:, y0:y1, x0:x1] += mod
            models.pop(k, None)
            bright_mode[k] = False
            flag[k] |= 1
            Fs[k] = 0
            refit(k, free_width=False)
    def measure_halo():
        """Halo partagé par couleur (Racine 1996) : profil du résidu (coeurs soustraits) des étoiles brillantes, en
        anneaux de R à rfar, moins le fond de l'anneau juste au-delà, divisé par le flux ajusté ; médiane sur les étoiles,
        loi de puissance A r^-n (n entre 2 et 3.5). Le fond local au-delà rend la mesure juste en moyenne sur nébuleuse."""
        br = [k for k in stars_in if 0 <= Xs[k] < W and 0 <= Ys[k] < H][:40]
        rfar = int(min(pr["halo_rmax"], max(4 * R, 150), max(H, W) / 4))
        edges = np.unique(np.geomspace(R, rfar, 14).astype(int))
        rmid = np.sqrt(edges[:-1] * edges[1:])
        new = []
        for c in range(C):
            profs = []
            for k in br:
                x, y = Xs[k], Ys[k]
                ro = int(rfar * 1.25)
                x0, x1 = int(max(0, x - ro)), int(min(W, x + ro + 1)); y0, y1 = int(max(0, y - ro)), int(min(H, y + ro + 1))
                sub = Rimg[c, y0:y1:2, x0:x1:2]
                yy, xx = np.mgrid[y0:y1:2, x0:x1:2]
                r = np.hypot(xx - x, yy - y)
                ok = valid[y0:y1:2, x0:x1:2] & ~sat[c, y0:y1:2, x0:x1:2]
                ring = ok & (r > rfar) & (r <= ro)
                if ring.sum() < 30 or Fs[k][c] <= 0:
                    continue
                ref = np.median(sub[ring])
                p_ = [np.median(sub[ok & (r > a) & (r <= b)]) - ref if (ok & (r > a) & (r <= b)).sum() > 10 else np.nan
                      for a, b in zip(edges[:-1], edges[1:])]
                profs.append(np.array(p_) / Fs[k][c])
            if len(profs) < 5:
                new.append((0.0, 3.0, float(R)))
                continue
            prof = np.nanmedian(np.array(profs), 0)
            okp = np.isfinite(prof) & (prof > 0)
            if okp.sum() < 4:
                new.append((0.0, 3.0, float(R)))
                continue
            A_ = np.polyfit(np.log(rmid[okp]), np.log(prof[okp]), 1)
            n_ = float(np.clip(-A_[0], 2.0, 3.5))
            amp = float(np.exp(np.median(np.log(prof[okp]) + n_ * np.log(rmid[okp]))))
            new.append((amp, n_, float(R)))
        halo[:] = new
        E.halo = halo
        log("halo mesuré sur le résidu : " + ", ".join(f"c{c}: {h[0]:.3g} r^-{h[1]:.2f}" for c, h in enumerate(halo)))

    def core_correction():
        """Correction du coeur de la PSF par région et par couleur, mesurée sur le résidu empilé de toutes les étoiles
        moyennes ajustées (itération d'Anderson & King 2000 : la PSF effective est affinée par la moyenne des résidus
        normalisés, au suréchantillonnage de la PSF, sur la position ajustée de chaque étoile). Rayon 2.5 largeurs."""
        from scipy.spatial import cKDTree
        os2 = E.os
        n_ = E.stamps[0][0].shape[0]
        c0_ = (n_ - 1) // 2
        rcor = 2.5 * fwhm
        cand_ = [k for k in stars_in if not bright_mode[k] and not giant[k] and k in models and Fs[k].min() > 0
                 and 20 < Xs[k] < W - 20 and 20 < Ys[k] < H - 20]
        if len(cand_) < 30:
            return "correction du coeur : trop peu d'étoiles"
        tr = cKDTree(np.c_[Xs[stars_in], Ys[stars_in]])
        sel_ = []
        for k in cand_:
            pk_ = Fs[k].sum() / C * peak_frac
            if pk_ < 20 * med_sig.min():
                continue
            nb = tr.query_ball_point([Xs[k], Ys[k]], 3 * fwhm)
            if any(G[stars_in[j]] < G[k] + 2 and stars_in[j] != k for j in nb):
                continue
            sel_.append(k)
        reg = np.argmin(np.hypot(E.centres[:, 0][None] - Xs[sel_][:, None], E.centres[:, 1][None] - Ys[sel_][:, None]), 1)
        rr_ = int(np.ceil(rcor)) + 1
        gy_, gx_ = np.mgrid[-rr_:rr_ + 1, -rr_:rr_ + 1]
        changed = []
        for c in range(C):
            for g_ in range(len(E.stamps[c])):
                ks = [k for k, rg in zip(sel_, reg) if rg == g_]
                if len(ks) < 15:
                    continue
                acc = np.zeros(n_ * n_); acc2 = np.zeros(n_ * n_); cnt = np.zeros(n_ * n_)
                samples = []
                for k in ks:
                    xi, yi = int(round(Xs[k])), int(round(Ys[k]))
                    cut = Rimg[c, yi - rr_:yi + rr_ + 1, xi - rr_:xi + rr_ + 1]
                    dx_ = gx_ + xi - Xs[k]; dy_ = gy_ + yi - Ys[k]
                    r_ = np.hypot(dx_, dy_)
                    ring_ = (r_ > rcor) & (r_ <= rr_ + 0.5)
                    b_ = np.median(cut[ring_]) if ring_.any() else 0.0
                    v_ = (cut - b_) / Fs[k][c]
                    ix = np.round(dx_ * os2).astype(int) + c0_; iy = np.round(dy_ * os2).astype(int) + c0_
                    ok = (r_ <= rcor) & ~sat[c, yi - rr_:yi + rr_ + 1, xi - rr_:xi + rr_ + 1]
                    idx = iy[ok] * n_ + ix[ok]
                    samples.append((idx, v_[ok]))
                    np.add.at(acc, idx, v_[ok]); np.add.at(acc2, idx, v_[ok] ** 2); np.add.at(cnt, idx, 1)
                mean = acc / np.maximum(cnt, 1)
                sd = np.sqrt(np.maximum(acc2 / np.maximum(cnt, 1) - mean ** 2, 0))
                acc[:] = 0; cnt[:] = 0
                for idx, v_ in samples:
                    good = np.abs(v_ - mean[idx]) <= 2.5 * sd[idx] + 1e-12
                    np.add.at(acc, idx[good], v_[good]); np.add.at(cnt, idx[good], 1)
                corr = (acc / np.maximum(cnt, 1)).reshape(n_, n_)
                corr[(cnt.reshape(n_, n_) < 3)] = 0
                corr = ndi.gaussian_filter(corr, 0.5 * os2)
                rro = np.hypot(*np.mgrid[-c0_:c0_ + 1, -c0_:c0_ + 1]) / os2
                corr *= np.clip((rcor - rro) / 1.0, 0, 1)                   # fondu à zéro au bord
                E.stamps[c][g_] = E.stamps[c][g_] + corr
                changed.append(float(corr.sum() / os2 ** 2))
        stc.clear()
        return (f"correction du coeur de la PSF sur {len(sel_)} étoiles : part de flux ajoutée par région et couleur "
                f"{np.round(changed, 4).tolist()}")

    # tour 17 : ordre par classe, chaque étape sur l'image nettoyée par la précédente : (1) les brillantes (profil
    # propre, halo compris) seules, deux fois ; (2) la PSF des moyennes refaite sur cette image nettoyée (le halo des
    # brillantes biaisait leur fond et leur forme) ; (3) toutes les étoiles en passes ; (4) les faibles hors catalogue ;
    # puis le remplissage
    # v7 : avec le modèle de l'image entière, pas de PSF refaite sur la découpe (ce serait revenir au modèle de 9 étoiles) ;
    # la correction du coeur sur la découpe (passes 2 et 3, r < 2,5 largeurs) est GARDÉE : le modèle de l'image ne corrige
    # que l'aile (r >= 5 px), son coeur ne convergeait pas
    if pr.get("psf_etendue", True) and bright_mode.any():
        # v12 : forme étendue de l'image entière (psf_etendue.py), à la place du profil propre libre de chaque brillante
        import psf_etendue as PSFE
        me_ = PSFE.obtenir(name, log)
        SHARED["P"], SHARED["rg"], SHARED["v12"] = np.asarray(me_["P"], float), np.asarray(me_["rg"], float), True
        SHARED["tree"] = cKDTree(np.c_[xg, yg])
        log(f"v12 : forme étendue de l'image (portée {me_['info']['R_ext']} px, {me_['info']['etoiles_amplitude']} étoiles "
            f"empilées, raccord au modèle v7 {me_['info']['raccord']} px) ; amplitude par étoile sur l'anneau "
            f"max(2, rs/FWHM + 2) à +4 largeurs (FWHM {fwhm:.2f} px)")
    if pr.get("class_order", True) and bright_mode.any():
        tb = time.time()
        for _ in range(2):
            for k in stars_in:
                if bright_mode[k]:
                    refit(k, free_width=False)
        if pr.get("shared_profile", False):        # tour 18 : profil partagé abandonné pour l'instant (tour 19)
            log(build_shared())
            for k in stars_in:
                if bright_mode[k]:
                    refit(k, free_width=False)
        # PSF refaite : mêmes étoiles candidates, mesurées sur le résidu où seules les brillantes sont retirées
        for c in (range(C) if not hasattr(E, "rho") else []):
            for g_, (cx, cy) in enumerate(E.centres):
                d_ = np.maximum(np.abs(xg[good] - cx) / (W / E.g), np.abs(yg[good] - cy) / (H / E.g))
                sel = good[np.argsort(d_)][:max(pr["min_psf_stars"], 80)]
                st_, _, _ = build_epsf(Rimg[c], sat_lvl[c], np.c_[xg[sel], yg[sel]], R, os_)
                E.stamps[c][g_] = st_
        stc.clear()
        log(f"étape brillantes d'abord ({int(bright_mode.sum())} étoiles) puis PSF refaite sur l'image nettoyée : "
            f"{time.time() - tb:.0f} s" + (" (v7 : PSF non refaite, modèle de l'image entière)" if hasattr(E, "rho") else ""))
    for p_ in range(pr["passes"]):
        tp = time.time()
        if p_ == pr["passes"] - 1 and pr.get("shared_halo"):
            measure_halo()
        if p_ == 1 and pr.get("chroma", False) and C == 3:      # c5 : décalages minimes (< 0.08 px), et moins bon
            log(measure_chroma())
        if p_ > 0 and pr.get("core_corr", True) and CAL is None:       # v17 : correction de l'image, figée
            log(core_correction())
        for k in stars_in:
            refit(k, free_width=(p_ == pr["passes"] - 1))
        log(f"passe {p_ + 1}: {time.time() - tp:.0f} s, bloquées par Gaia {int((flag & 1).sum())}, "
            f"flux réduits par la garde « pas de trou » {hole_guard[0]}, pixels remplacés {pix_guard[0]}")
        hole_guard[0] = 0
        pix_guard[0] = 0

    # 4b. aigrettes : angles fixés par l'instrument (communs à toutes les étoiles), profil le long de l'aigrette mesuré
    # sur les étoiles brillantes et normalisé par leur flux attendu (gabarit), ajusté en amplitude par étoile (0.3-3 x le
    # gabarit à son flux) et soustrait comme partie du modèle ; ce qui reste est rempli ensuite.
    # essais 9-10 : le gabarit soustrait laissait des croix sombres fines (aigrette mesurée plus large et décalée de
    # l'angle commun d'une fraction de degré) : coupé par défaut, les aigrettes restantes sont remplies (étape 7)
    for k_, bf_ in sorted(BRIGHT_FIT.items(), key=lambda t: G[t[0]]):
        if 0 <= Xs[k_] < W and 0 <= Ys[k_] < H:
            log(f"v12 brillante ({Xs[k_]:.1f}, {Ys[k_]:.1f}) G {G[k_]:.2f} rs {bf_['rs']:.1f} anneau {bf_['r_in']:.1f}-"
                f"{bf_['r_out']:.1f} px : amplitude {np.round(bf_['amp'], 0).tolist()} (ajustée "
                f"{[round(f_[0], 0) if np.isfinite(f_[0]) else None for f_ in bf_['fit']]}, fond "
                f"{[round(f_[1], 3) if np.isfinite(f_[1]) else None for f_ in bf_['fit']]}, pixels gardés/anneau "
                f"{[f'{f_[2]}/{f_[3]}' for f_ in bf_['fit']]}) ; attendu Gaia {np.round(bf_['Fexp'], 0).tolist()} ; "
                f"rapport {np.round(bf_['amp'] / np.maximum(bf_['Fexp'], 1e-9), 2).tolist()}")
    if BRIGHT_FIT:
        np.save(out / "brillantes_v12.npy", np.array([(Xs[k_], Ys[k_], G[k_], bf_["rs"], bf_["r_in"], bf_["r_out"],
                                                         *bf_["amp"], *bf_["Fexp"], *[f_[0] for f_ in bf_["fit"]])
                                                        for k_, bf_ in BRIGHT_FIT.items()], float))
    spike_info = spike_model() if pr.get("spikes", False) else None
    if spike_info:
        log(spike_info)

    # 5. boucle fermée par étoile (les brillantes d'abord) : anneaux, correction, arrêt quand plat
    scores = np.full(N, np.nan)
    loops = np.zeros(N, int)
    bright_ix = [k for k in stars_in if Fpred[k] > 5 * med_sig.min() * fwhm ** 2]
    tl = time.time()
    for k in bright_ix:
        if bright_mode[k]:          # profil propre : plat en anneaux par construction ; score seulement
            a = annuli_residual(Rimg, valid, Xs[k], Ys[k], fwhm, footprint(k, Fs[k]), sig, ncorr, pr)
            if a is not None:
                scores[k] = a[4]
            continue
        for it in range(pr["loop_iters"]):
            fp = footprint(k, Fs[k])
            a = annuli_residual(Rimg, valid, Xs[k], Ys[k], fwhm, fp, sig, ncorr, pr)
            if a is None:
                break
            edges, res, err, ref, score = a
            scores[k] = score
            loops[k] = it
            if score <= pr["q_flat"]:
                break
            rmid = 0.5 * (edges[:-1] + edges[1:])
            far = rmid > E.R
            # halo : un résidu des anneaux lointains de même signe -> amplitude du halo de cette étoile
            if far.sum() >= 2 and Fs[k].sum() > 0:
                zf = res[:, far] / np.maximum(err[:, far], 1e-9)
                if np.abs(np.median(zf)) > pr["q_flat"]:
                    hv = np.array([E.halo_eval(c, rmid[far]) * Fs[k][c] for c in range(C)])
                    tot = (hv ** 2).sum()
                    if tot > 0:
                        dA = float((hv * res[:, far]).sum() / tot)
                        HA[k] = float(np.clip(HA[k] * (1 + dA / max(HA[k], 1e-3)), 0.3, 3.0))
            refit(k, free_width=True, loop_round=it + 1)
        else:
            a = annuli_residual(Rimg, valid, Xs[k], Ys[k], fwhm, footprint(k, Fs[k]), sig, ncorr, pr)
            if a is not None:
                scores[k] = a[4]
    log(f"boucle fermée : {len(bright_ix)} étoiles en {time.time() - tl:.0f} s ; plates (score <= {pr['q_flat']}) "
        f"{int(np.nansum(scores[bright_ix] <= pr['q_flat']))}, non plates {int(np.nansum(scores[bright_ix] > pr['q_flat']))}")

    # 6. étoiles hors catalogue : détectées dans le résidu, gardées si ponctuelles (forme de la PSF)
    # itératif (photométrie PSF itérative, IterativePSFPhotometry de photutils / DAOPHOT ALLSTAR) : trouver, ajuster,
    # soustraire, chercher encore en baissant le seuil ; les positions déjà prises (Gaia et passes précédentes) ne
    # comptent plus comme nouvelles
    ex_F = []
    taken_x, taken_y = list(xg), list(yg)
    for thr_ in pr["faint_passes"]:
      extra = faint_stars(Rimg, valid, sig, E, fwhm, np.array(taken_x), np.array(taken_y), dict(pr, faint_sigma=thr_), log)
      for (x, y) in extra:
        xi, yi = int(round(x)), int(round(y))
        rf = max(3.0, 1.5 * fwhm); rw = int(np.ceil(rf + pr["ann_w"])) + 1
        if xi - rw < 0 or yi - rw < 0 or xi + rw >= W or yi + rw >= H:
            continue
        st = [E.blend(c, x, y) for c in range(C)]
        win = Rimg[:, yi - rw:yi + rw + 1, xi - rw:xi + rw + 1]
        F, nx, ny, s, chi = fit_star(win, valid[yi - rw:yi + rw + 1, xi - rw:xi + rw + 1], sat[:, yi - rw:yi + rw + 1, xi - rw:xi + rw + 1],
                                     E, st, x - (xi - rw), y - (yi - rw), None, 1.0, rf, False, True, sig[:, yi, xi],
                                     x - (xi - rw), y - (yi - rw), dict(pr, pos_margin=1.0))
        F = np.maximum(F, 0)
        # tour 17 : couleur bornée (0.35-3 x la couleur moyenne des étoiles) : une étoile brillante absente de Gaia
        # (NGC 6543, x 1195 y 1141) prenait 5 x trop de bleu et laissait un trou sombre cerné de jaune
        Ft_ = float(F.sum())
        cb_ = pr.get("colour_bound", (0.35, 3.0))
        F = np.clip(F, cb_[0] * Ft_ * colour, cb_[1] * Ft_ * colour)
        if pr.get("debug_xy") and any(np.hypot(x - dx, y - dy) < 3 for dx, dy in pr["debug_xy"]):
            log(f"  suivi ajustement ({x:.1f}, {y:.1f}) : flux {np.round(F, 1).tolist()}, seuil de flux "
                f"{3 * float(np.mean(sig[:, yi, xi])) * np.sqrt(np.mean(ncorr)) * fwhm * np.sqrt(C):.1f}")
        # flux mesurable seulement (au moins 3 bruits sur l'aire de l'étoile, toutes couleurs) : sinon du bruit
        if F.sum() < 3 * float(np.mean(sig[:, yi, xi])) * np.sqrt(np.mean(ncorr)) * fwhm * np.sqrt(C):
            continue
        nx += xi - rw; ny += yi - rw
        x0, x1 = int(max(0, nx - E.R)), int(min(W, nx + E.R + 1)); y0, y1 = int(max(0, ny - E.R)), int(min(H, ny + E.R + 1))
        Rimg[:, y0:y1, x0:x1] -= star_model(E, st, nx, ny, F, s, x0, x1, y0, y1, with_halo=False)
        ex_F.append((nx, ny, *F))
        taken_x.append(nx); taken_y.append(ny)
      log(f"étoiles hors catalogue au seuil {thr_} bruits : {len(ex_F)} en tout")
    log(f"étoiles hors catalogue retirées : {len(ex_F)}")

    def spikes(k, r0):
        """[(angle, longueur)] des aigrettes de l'étoile k dans le résidu : 180 secteurs de 2 deg, résidu moyen par
        tranche radiale ; un secteur est une aigrette s'il dépasse de 4 erreurs la médiane des secteurs sur au moins
        3 tranches successives depuis r0."""
        x, y = Xs[k], Ys[k]
        (my0, my1, mx0, mx1), mod = models[k]
        rmax = max(r0 + 10, 0.9 * min(x - mx0, mx1 - x, y - my0, my1 - y))
        if rmax <= r0 + 4:
            return []
        y0, y1 = int(max(0, y - rmax)), int(min(H, y + rmax + 1)); x0, x1 = int(max(0, x - rmax)), int(min(W, x + rmax + 1))
        if y1 <= y0 or x1 <= x0:
            # SHO 21 nov (garde contre un plantage, sans effet sur les cas deja passes) : étoile brillante dans la marge
            # de 60 px HORS de la découpe dont le cercle de recherche ne touche pas la découpe -> rien à chercher
            return []
        yy, xx = np.mgrid[y0:y1, x0:x1]
        r = np.hypot(xx - x, yy - y)
        a = np.arctan2(yy - y, xx - x)
        nA = 180
        ia = ((a + np.pi) / (2 * np.pi) * nA).astype(int) % nA
        redges = np.geomspace(r0, rmax, 12)
        ir = np.searchsorted(redges, r) - 1
        ok = valid[y0:y1, x0:x1] & (ir >= 0) & (ir < len(redges) - 1)
        Lr = Rimg[:, y0:y1, x0:x1].mean(0)
        nb = len(redges) - 1
        idx = (ir * nA + ia)[ok]
        cnt = np.bincount(idx, minlength=nb * nA).reshape(nb, nA)
        sm = np.bincount(idx, weights=Lr[ok], minlength=nb * nA).reshape(nb, nA)
        mean = sm / np.maximum(cnt, 1)
        med = np.median(mean, 1, keepdims=True)
        mad = 1.4826 * np.median(np.abs(mean - med), 1, keepdims=True) + 1e-6
        s = float(np.median(sig[:, int(np.clip(y, 0, H - 1)), int(np.clip(x, 0, W - 1))]))
        err = np.maximum(mad, s * np.sqrt(ncorr[1 if C > 1 else 0] / np.maximum(cnt, 1)))
        hot = (mean - med > 4 * err) & (mean - med > 0.5 * s) & (cnt > 0)
        out = []
        for j in range(nA):
            col = hot[:, j]
            if col[:3].all():
                n = int(np.argmin(np.r_[col, False]))
                out.append((-np.pi + (j + 0.5) * 2 * np.pi / nA, float(redges[min(n, nb)]) + 2 * fwhm))
        return out

    # 7. remplissage : coeurs saturés et étoiles non plates
    fill = np.zeros((H, W), bool)
    ZM = {k_: np.zeros((H, W), np.uint8) for k_ in ("coeur_sature", "boucle_non_convergee", "aigrette", "coeur_ecart")}
    ZL: list = []
    # géantes : lumière rendue, zone exclue (gardée telle quelle, à écarter de l'entraînement)
    excl = np.zeros((H, W), bool)
    for k in stars_in:
        if giant[k] and k in models:
            (gy0, gy1, gx0, gx1), mod = models.pop(k)
            Rimg[:, gy0:gy1, gx0:gx1] += mod
            # zone ronde (le masque « modèle > 1 bruit » prenait le bord rectangulaire de la fenêtre du modèle :
            # marche rectangulaire visible) ; rayon = la plus grande distance où le modèle dépasse 1 bruit
            over = mod.mean(0) > med_sig.min()
            if over.any():
                yy_g, xx_g = np.mgrid[gy0:gy1, gx0:gx1]
                rg = float(np.hypot(xx_g - Xs[k], yy_g - Ys[k])[over].max())
                cv2.circle(excl.view(np.uint8), (int(round(Xs[k])), int(round(Ys[k]))), int(rg), 1, -1)
    for (cx_, cy_, rb) in giant_blobs:
        # étendue : profil médian (médiane de 8 secteurs) de la luminance jusqu'à 1 bruit au-dessus du niveau lointain
        rr_max = int(min(1500, max(H, W)))
        yy_, xx_ = np.ogrid[0:H:4, 0:W:4]
        rr = np.hypot(xx_ - cx_, yy_ - cy_)
        Ls = Rimg.mean(0)[::4, ::4]
        ed = np.geomspace(max(rb, 4), rr_max, 40)
        pv = np.array([np.median(Ls[(rr >= a) & (rr < b)]) if ((rr >= a) & (rr < b)).any() else np.nan for a, b in zip(ed[:-1], ed[1:])])
        floor_ = np.nanmin(pv)
        reach_ = ed[np.where(pv - floor_ > med_sig.min())[0].max() + 1] if np.any(pv - floor_ > med_sig.min()) else 3 * rb
        cv2.circle(excl.view(np.uint8), (int(cx_), int(cy_)), int(reach_), 1, -1)
    excl = ndi.binary_dilation(excl, iterations=3) if excl.any() else excl
    log(f"zones exclues (géantes laissées entières) : {excl.mean() * 100:.2f} % de l'image")
    for k in stars_in:
        if giant[k]:
            continue
        x, y = Xs[k], Ys[k]
        xi, yi = int(round(x)), int(round(y))
        r_fill = 0.0
        # coeur saturé : la zone saturée de l'étoile, élargie
        rs = sat_radius(x, y)
        if rs > 0:
            # c2 : coeur seul (aigrettes et halos des brillantes laissés à la piste 3)
            r_fill = max(r_fill, (rs + 0.5 * fwhm) if pr.get("core_only", False) else (rs * 1.2 + 1.5 * fwhm))
            flag[k] |= 16
        # sur un fond plus brillant que l'étoile elle-même (coeur de planétaire), le remplissage inventerait la nébuleuse :
        # jamais de remplissage pour échec de boucle là
        r_b = int(4 * fwhm) + 1
        bg_here = 0.0
        if r_b <= xi < W - r_b and r_b <= yi < H - r_b:
            gyb, gxb = np.mgrid[-r_b:r_b + 1, -r_b:r_b + 1]
            rrb = np.hypot(gxb + xi - x, gyb + yi - y)
            bg_here = float(np.median(Rimg.mean(0)[yi - r_b:yi + r_b + 1, xi - r_b:xi + r_b + 1][(rrb >= 2 * fwhm) & (rrb <= 4 * fwhm)]))
        on_bright = bg_here > 0.05 * float(Fs[k].sum()) / C * peak_frac and bg_here > 20 * med_sig.min()
        if np.isfinite(scores[k]) and scores[k] > 3 * pr["q_flat"] and not on_bright \
                and not (bright_mode[k] and not pr.get("bright_fail_fill", True)):
            # non convergé : le rayon où le résidu redevient plat
            a = annuli_residual(Rimg, valid, x, y, fwhm, footprint(k, Fs[k]), sig, ncorr, pr)
            if a is not None:
                edges, res, err, ref, sc = a
                z = (np.abs(res) / np.maximum(err, 1e-9)).max(0)
                # le modèle est en faute seulement si le résidu pèse aussi par rapport à l'étoile elle-même (> 10 % de
                # son modèle dans l'anneau) : un résidu significatif mais minime devant l'étoile est le fond, pas elle
                rmid_ = 0.5 * (edges[:-1] + edges[1:])
                Ft_ = float(Fs[k].sum())
                mk = Ft_ * np.interp(rmid_, prof_r, prof_v, right=0.0)
                A_h, n_h, _ = halo[int(np.argmax([h[0] for h in halo]))]
                mk = mk + Ft_ * A_h * np.maximum(rmid_, 1) ** (-n_h) * (rmid_ > E.R)
                big = np.abs(res).max(0) > 0.1 * mk / C + 0.5 * med_sig.min()
                bad = np.where((z > 3 * pr["q_flat"]) & big)[0]
                if len(bad):
                    # jamais au-delà de l'étendue visible de l'étoile (modèle > 1 bruit)
                    Ft = float(Fs[k].sum())
                    vis = np.where(Ft * prof_v > med_sig.min())[0]
                    r_vis = float(vis[-1] + 1.5) if len(vis) else 0.0
                    A, n_, _ = halo[int(np.argmax([h[0] for h in halo]))]
                    if A > 0 and Ft * A * E.R ** (-n_) > med_sig.min():
                        r_vis = max(r_vis, (Ft * A / med_sig.min()) ** (1 / n_))
                    r_bad = min(float(edges[bad[-1] + 1]), r_vis, sat_radius(x, y) + pr["fail_fill_fwhm"] * fwhm)
                    if r_bad > 0:
                        r_fill = max(r_fill, r_bad)
                        flag[k] |= 2
        if r_fill > 0:
            cv2.circle(fill.view(np.uint8), (xi, yi), int(np.ceil(r_fill + pr["fill_grow"])), 1, -1)
            # c9 : zones à remplir par motif (pour la piste 3) : coeur saturé et boucle non convergée séparés
            if rs > 0:
                r_s = (rs + 0.5 * fwhm) if pr.get("core_only", False) else (rs * 1.2 + 1.5 * fwhm)
                cv2.circle(ZM["coeur_sature"], (xi, yi), int(np.ceil(r_s + pr["fill_grow"])), 1, -1)
                ZL.append((x, y, G[k], "coeur_sature", r_s + pr["fill_grow"]))
            if flag[k] & 2:
                cv2.circle(ZM["boucle_non_convergee"], (xi, yi), int(np.ceil(r_fill + pr["fill_grow"])), 1, -1)
                ZL.append((x, y, G[k], "boucle_non_convergee", r_fill + pr["fill_grow"]))
        # aigrettes des étoiles brillantes : le profil radial ne les rend pas ; les angles où le résidu sort nettement du
        # fond de l'anneau sont remplis en bandes jusqu'où l'aigrette se perd dans le bruit
        if bright_mode[k] and k in models and pr.get("spike_fill", True):
            sp = spikes(k, max(r_fill, 2 * fwhm))
            for (ang_, length) in sp:
                x2 = int(round(x + length * np.cos(ang_))); y2 = int(round(y + length * np.sin(ang_)))
                cv2.line(fill.view(np.uint8), (xi, yi), (x2, y2), 1, int(max(3, round(2 * fwhm))))
                cv2.line(ZM["aigrette"], (xi, yi), (x2, y2), 1, int(max(3, round(2 * fwhm))))
                ZL.append((x, y, G[k], f"aigrette angle {np.degrees(ang_):.0f} longueur {length:.0f}", 0.0))
                flag[k] |= 32
    fill &= valid & ~excl
    np.save(out / "exclu.npy", np.packbits(excl))
    log(f"zones remplies : {fill.mean() * 100:.3f} % de l'image ; étoiles non plates remplies {int((flag[stars_in] & 2 > 0).sum())}, "
        f"coeurs saturés remplis {int((flag[stars_in] & 16 > 0).sum())}")
    import patchfill
    np.save(out / "soustraction.npy", Rimg.astype(np.float32))
    starless = Rimg.copy()
    tf = time.time()
    if fill.any() and not pr.get("no_fill", False):
        # remplissage par exemples (PatchMatch multi-échelle) ; sources : le résidu hors zones à remplir
        # sources : pixels déjà nettoyés, hors de toute empreinte d'étoile (modèle retiré > 1 bruit, élargi)
        foot = ndi.binary_dilation((img - Rimg).mean(0) > med_sig.min(), iterations=2)
        starless = patchfill.fill_mask(Rimg, fill, ~valid | foot, med_sig, log=log)
    log(f"remplissage par exemples : {time.time() - tf:.0f} s")
    # variante « remplissage seul » (sans soustraction), sur la zone demandée : disques des étoiles Gaia visibles
    zone = pr.get("fill_only_zone")
    if zone:
        zx0, zy0, zx1, zy1 = zone
        discs = np.zeros((H, W), bool)
        for k in stars_in:
            x, y = xg[k], yg[k]
            if not (zx0 - 100 < x < zx1 + 100 and zy0 - 100 < y < zy1 + 100):
                continue
            Ft = Fpred[k]
            above = np.where(Ft * prof_v > 0.5 * med_sig.min())[0]
            r = (above[-1] + 1.5) if len(above) else 0
            A, n_, _ = halo[int(np.argmax([h[0] for h in halo]))]
            if A > 0 and Ft * A * E.R ** (-n_) > 0.5 * med_sig.min():
                r = max(r, (Ft * A / (0.5 * med_sig.min())) ** (1 / n_))
            if bright_mode[k] and k in models:      # étoile brillante : l'étendue de son propre profil
                (my0, my1, mx0, mx1), mod = models[k]
                yy_, xx_ = np.mgrid[my0:my1, mx0:mx1]
                over = mod.mean(0) > 0.5 * med_sig.min()
                if over.any():
                    r = max(r, float(np.hypot(xx_ - x, yy_ - y)[over].max()) + 1)
            if r > 0:
                cv2.circle(discs.view(np.uint8), (int(round(x)), int(round(y))), int(np.ceil(r)), 1, -1)
        discs &= valid
        tf = time.time()
        only = img.copy()
        m0 = np.zeros((H, W), bool)
        m0[max(0, zy0 - 150):zy1 + 150, max(0, zx0 - 150):zx1 + 150] = True
        only = patchfill.fill_mask(img, discs & m0, ~valid | (discs & ~m0), med_sig)
        np.save(out / "remplissage_seul.npy", only.astype(np.float32))
        log(f"remplissage seul : {discs[zy0:zy1, zx0:zx1].mean() * 100:.1f} % de la tuile en disques, {time.time() - tf:.0f} s")

    # contrôle final des coeurs : pour chaque étoile retirée, coeur (r <= 1.5 largeur) contre anneau (2.5-5 largeurs), par
    # couleur ; un écart de plus de 2 bruits (puits ou reste) fait remplacer le disque du coeur par la membrane harmonique
    # tendue sur son bord (fond local) plus le grain de l'image
    if pr.get("core_check", True):
        import patchfill
        n_fix = 0
        rc_ = 1.5 * fwhm
        ra_, rb_ = 2.5 * fwhm, 5 * fwhm
        Rr = int(np.ceil(rb_)) + 1
        gy_, gx_ = np.mgrid[-Rr:Rr + 1, -Rr:Rr + 1]
        allx = np.r_[Xs[stars_in], [e[0] for e in ex_F]] if ex_F else Xs[stars_in]
        ally = np.r_[Ys[stars_in], [e[1] for e in ex_F]] if ex_F else Ys[stars_in]
        # c8 : pic attendu de chaque étoile ; sous faint_noguard bruits, pas de contrôle de coeur (soustraction PSF seule)
        allpk = np.r_[Fexp[stars_in].sum(1) * peak_frac / C, [sum(e[2:]) * peak_frac / C for e in ex_F]] if ex_F \
            else Fexp[stars_in].sum(1) * peak_frac / C
        for x, y, pk_ in zip(allx, ally, allpk):
            if pk_ < pr.get("faint_noguard", 0.0) * med_sig.min():
                continue
            xi, yi = int(round(x)), int(round(y))
            if xi < Rr or yi < Rr or xi >= W - Rr or yi >= H - Rr or excl[yi, xi]:
                continue
            rr = np.hypot(gx_ + xi - x, gy_ + yi - y)
            win = starless[:, yi - Rr:yi + Rr + 1, xi - Rr:xi + Rr + 1]
            core = rr <= rc_
            ring = (rr >= ra_) & (rr <= rb_)
            bad = False
            for c in range(C):
                s_ = float(sig[c, yi, xi])
                dlt = float(np.median(win[c][core]) - np.median(win[c][ring]))
                # tolérance : 2 bruits, plus la pente propre du fond (écart des moitiés opposées de l'anneau)
                if abs(dlt) > 2 * s_ * np.sqrt(ncorr[c] / max(core.sum(), 1)) + 2 * s_:
                    bad = True
            if bad and pr.get("no_fill", False):
                # c9 : rien n'est remplacé ; la zone est seulement notée pour la piste 3
                cv2.circle(ZM["coeur_ecart"], (xi, yi), int(np.ceil(rc_ + 1)), 1, -1)
                ZL.append((x, y, np.nan, "coeur_ecart", rc_ + 1))
                n_fix += 1
            elif bad:
                hole = rr <= rc_ + 1
                known = win.copy()
                memb = patchfill.poisson_blend(known, np.zeros_like(known), hole)    # membrane harmonique
                g_ = grain[:, yi - Rr:yi + Rr + 1, xi - Rr:xi + Rr + 1] * sig[:, yi, xi][:, None, None]
                starless[:, yi - Rr:yi + Rr + 1, xi - Rr:xi + Rr + 1] = np.where(hole[None], memb + g_, win)
                n_fix += 1
        log(f"contrôle final des coeurs : {n_fix} coeurs remplacés par le fond local + grain")

    # zones exclues (géantes) : l'image d'entrée telle quelle, aucune étoile n'y est touchée (zone écartée de
    # l'entraînement) ; le bord est fondu sur 8 px
    if excl.any():
        # fondu large (40 px) vers l'extérieur, pour qu'aucun bord ne se voie
        dout = ndi.distance_transform_edt(~excl)
        t_ = np.clip(dout / 40.0, 0, 1)
        wex = (1 - t_ * t_ * (3 - 2 * t_)).astype(np.float32)
        starless = (1 - wex) * starless + wex * img
        Rimg = (1 - wex) * Rimg + wex * img

    # 8. sorties
    np.save(out / "starless.npy", starless.astype(np.float32))
    # zones que le script aurait remplies, par motif (masques empaquetés, même cadre que l'image) et liste par étoile
    np.savez_compressed(out / "zones_a_remplir.npz", forme=np.array([H, W]),
                        **{k_: np.packbits((v_ > 0) & valid & ~excl) for k_, v_ in ZM.items()})
    import csv as _csv
    with open(out / "zones_a_remplir.csv", "w", newline="", encoding="utf-8") as f:
        wz = _csv.writer(f)
        wz.writerow(["x", "y", "G", "motif", "rayon_px"])
        for (zx, zy, zg, zr, zrad) in ZL:
            wz.writerow([f"{zx:.2f}", f"{zy:.2f}", "" if not np.isfinite(zg) else f"{zg:.3f}", zr, f"{zrad:.1f}"])
    log("zones à remplir (non remplies si no_fill) : " + ", ".join(f"{k_} {((v_ > 0) & valid).mean() * 100:.2f} %" for k_, v_ in ZM.items()))
    import csv
    with open(out / "etoiles.csv", "w", newline="", encoding="utf-8") as f:
        wri = csv.writer(f)
        wri.writerow(["x", "y", "x_gaia", "y_gaia", "G", "flux_predit", *[f"flux_{c}" for c in range(C)], "largeur", "halo",
                      "score", "tours", "drapeau"])
        for k in stars_in:
            wri.writerow([f"{Xs[k]:.2f}", f"{Ys[k]:.2f}", f"{xg[k]:.2f}", f"{yg[k]:.2f}", f"{G[k]:.3f}", f"{Fpred[k]:.4g}",
                          *[f"{v:.4g}" for v in Fs[k]], f"{Ss[k]:.3f}", f"{HA[k]:.3f}",
                          "" if not np.isfinite(scores[k]) else f"{scores[k]:.2f}", loops[k], flag[k]])
        for e in ex_F:
            wri.writerow([f"{e[0]:.2f}", f"{e[1]:.2f}", "", "", "", "", *[f"{v:.4g}" for v in e[2:]], "1", "1", "", 0, 8])
    np.save(out / "remplissage.npy", np.packbits(fill))
    with open(out / "decalages_couleur.csv", "w", newline="", encoding="utf-8") as f:
        wri = csv.writer(f)
        wri.writerow(["x", "y", "G", "fix_pos", "nsat", "Rx", "Ry", "Bx", "By"])
        for k, (fp_, ns_, co_) in CO_STAR.items():
            wri.writerow([f"{Xs[k]:.2f}", f"{Ys[k]:.2f}", f"{G[k]:.3f}", int(fp_), ns_, f"{co_[0,0]:.3f}", f"{co_[0,1]:.3f}", f"{co_[2,0]:.3f}", f"{co_[2,1]:.3f}"])
    summary = dict(name=name, tag=tag, fwhm=fwhm, R=R, regions=g, zp=zp0, zp_sd=zsd, halo=halo, n_gaia=int(len(stars_in)),
                   n_extra=len(ex_F), blocked=int((flag & 1).sum()), filled=int((flag & 2).sum()),
                   fill_percent=float(fill.mean() * 100), seconds=round(time.time() - t0),
                   flat_share=float(np.nanmean(scores[bright_ix] <= pr["q_flat"])) if bright_ix else None,
                   overrides=overrides or {}, v17_decoupe_non_utilise=DEC17, v17_couleur=np.round(colour, 4).tolist())
    (out / "resume.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log(f"fini en {time.time() - t0:.0f} s : {json.dumps(summary)}")
    return summary


def faint_stars(Rimg, valid, sig, E, fwhm, xg, yg, pr, log):
    """Étoiles absentes du catalogue (plus faibles que sa complétude) trouvées dans le résidu : détection par filtre
    adapté à la PSF, puis forme : netteté (pic / moyenne de l'anneau 1-2 largeurs) et rondeur proches de celles de la PSF,
    sans quoi c'est un noeud de nébuleuse ou du bruit."""
    import sep
    C, H, W = Rimg.shape
    L = Rimg.mean(0).astype(np.float32)
    s = np.median(sig, axis=0).astype(np.float32)
    back = cv2.blur(cv2.medianBlur(np.ascontiguousarray(L), 5), (int(6 * fwhm) | 1, int(6 * fwhm) | 1))
    sub = np.ascontiguousarray((L - back) / np.maximum(s, 1e-6))
    sub[~valid] = 0
    st = E.blend(1 if C > 1 else 0, W / 2, H / 2)
    cc = (st.shape[0] - 1) // 2
    ker = st[cc % E.os::E.os, cc % E.os::E.os]
    kr = int(np.ceil(1.5 * fwhm))
    m = ker.shape[0] // 2
    ker = np.ascontiguousarray(ker[m - kr:m + kr + 1, m - kr:m + kr + 1].astype(np.float32))
    sep.set_extract_pixstack(5000000)
    o = sep.extract(sub, pr["faint_sigma"], err=1.0, filter_kernel=ker, minarea=3, deblend_cont=0.01)
    # tour 19 : en plus de la segmentation, les maxima locaux du résidu filtré par la PSF (une étoile collée à une autre
    # source est fusionnée par la segmentation, son centre et sa taille faussés), avec leurs moments propres
    dt = np.dtype([("x", "f8"), ("y", "f8"), ("a", "f8"), ("b", "f8")])
    base = np.zeros(len(o), dtype=dt)
    for f_ in ("x", "y", "a", "b"):
        base[f_] = o[f_] if len(o) else []
    kn = ker / ker.sum()
    # essai 19 : normaliser le filtre en bruits du filtre (x4 environ) faisait passer le bruit corrélé du ciel (14 500
    # « étoiles » sur NGC 6543) ; seuil sur le pic du résidu lui-même, au maximum local du résidu lissé
    filt = cv2.filter2D(sub, -1, kn)
    mxf = ndi.maximum_filter(filt, size=int(2 * fwhm) | 1)
    py_, px_ = np.nonzero((filt >= mxf) & (sub >= pr["faint_sigma"]) & (filt >= 0.5 * pr["faint_sigma"]) & valid)
    if pr["faint_sigma"] < 5 or not pr.get("peak_finder", True):   # seuils bas : segmentation seule (bruit corrélé)
        py_, px_ = py_[:0], px_[:0]
    extra = []
    if len(py_):
        from scipy.spatial import cKDTree as _T
        newp = _T(np.c_[base["x"], base["y"]]).query(np.c_[px_, py_])[0] > 1.5 if len(base) else np.ones(len(px_), bool)
        hw = int(np.ceil(1.2 * fwhm))
        gy0, gx0 = np.mgrid[-hw:hw + 1, -hw:hw + 1]
        for xa, ya in zip(px_[newp], py_[newp]):
            if xa < hw or ya < hw or xa >= W - hw or ya >= H - hw:
                continue
            cut = np.clip(sub[ya - hw:ya + hw + 1, xa - hw:xa + hw + 1], 0, None)
            s0 = cut.sum()
            if s0 <= 0:
                continue
            mx_ = (cut * gx0).sum() / s0; my_ = (cut * gy0).sum() / s0
            vxx = (cut * (gx0 - mx_) ** 2).sum() / s0; vyy = (cut * (gy0 - my_) ** 2).sum() / s0
            vxy = (cut * (gx0 - mx_) * (gy0 - my_)).sum() / s0
            tr_, det_ = vxx + vyy, vxx * vyy - vxy ** 2
            q_ = np.sqrt(max(tr_ ** 2 / 4 - det_, 0))
            extra.append((xa + mx_, ya + my_, np.sqrt(tr_ / 2 + q_), np.sqrt(max(tr_ / 2 - q_, 1e-6))))
    o = np.concatenate([base, np.array(extra, dtype=dt)]) if extra else base
    if not len(o):
        return []
    from scipy.spatial import cKDTree
    t = cKDTree(np.c_[xg, yg])
    d, _ = t.query(np.c_[o["x"], o["y"]])
    keep = []
    # forme de référence de la PSF : part du pic dans l'anneau 1-2 largeurs
    yy, xx = np.mgrid[-cc:cc + 1, -cc:cc + 1] / E.os
    rr = np.hypot(xx, yy)
    ref_ring = st[(rr >= fwhm) & (rr < 2 * fwhm)].mean() / st[cc, cc]
    why = {"pres_d_une_etoile_prise": 0, "allongee": 0, "trop_large": 0, "bord": 0, "pic_negatif": 0, "pas_nette": 0}
    dbg_xy = pr.get("debug_xy", [])
    for i in range(len(o)):
        x, y = o["x"][i], o["y"][i]
        a_, b_ = o["a"][i], o["b"][i]
        if dbg_xy and any(np.hypot(x - dx, y - dy) < 3 for dx, dy in dbg_xy):
            xi_, yi_ = int(round(x)), int(round(y))
            log(f"  suivi ({x:.1f}, {y:.1f}) seuil {pr['faint_sigma']} : d_prise {d[i]:.1f}, a {a_:.2f}, b {b_:.2f}, "
                f"pic {float(sub[yi_, xi_]):.1f} bruits")
        # tour 15 : 1.0 largeur (1.5 écartait de vraies étoiles voisines d'une étoile Gaia ; le reste d'une étoile déjà
        # prise est à moins d'une largeur de sa position)
        if d[i] < pr.get("faint_near", 1.0) * fwhm:
            why["pres_d_une_etoile_prise"] += 1
            continue
        if b_ / max(a_, 1e-6) < pr.get("faint_round", 0.5):
            why["allongee"] += 1
            continue
        # tour 18 : taille admise relevée (1.6 -> 2.0 : de vraies étoiles un peu plus larges, ou des couples serrés,
        # étaient écartées ; Rosette 2043,968 / 2038,1036 / 1762,1159)
        if a_ > 1.3 * fwhm / 2.355 * pr.get("faint_size", 2.0):
            why["trop_large"] += 1
            continue
        xi, yi = int(round(x)), int(round(y))
        r2 = int(2 * fwhm) + 1
        if xi - r2 < 0 or yi - r2 < 0 or xi + r2 >= W or yi + r2 >= H:
            why["bord"] += 1
            continue
        cut = sub[yi - r2:yi + r2 + 1, xi - r2:xi + r2 + 1]
        gy, gx = np.mgrid[-r2:r2 + 1, -r2:r2 + 1]
        r = np.hypot(gx + xi - x, gy + yi - y)
        pk = cut[r <= 1.0].mean()
        ring = cut[(r >= fwhm) & (r < 2 * fwhm)].mean()
        if pk <= 0:
            why["pic_negatif"] += 1
            continue
        # garde « noeud de nébuleuse » : un objet étendu garde une part de son pic dans l'anneau 1-2 largeurs bien
        # au-dessus de celle de la PSF
        if ring / pk > pr.get("faint_sharp", (3.0, 0.15))[0] * ref_ring + pr.get("faint_sharp", (3.0, 0.15))[1]:
            why["pas_nette"] += 1
            continue
        keep.append((x, y))
    log(f"détections dans le résidu {len(o)}, ponctuelles hors catalogue gardées {len(keep)} ; écartées : {why}")
    return keep


def fill_zone(R, fill, valid, sig, amp, rng):
    """Fond lisse local (convolution normalisée multi-échelle autour de la zone) + grain de l'image."""
    out = R.copy()
    known = (~fill & valid).astype(np.float32)
    val = np.where(known > 0, R, 0).astype(np.float32)
    est = None
    for s in (2, 4, 8, 16, 32, 64):
        w = blur(known, s)
        e = blur(val, s) / np.maximum(w, 1e-6)
        est = np.where(w > 0.2, e, np.nan) if est is None else np.where(np.isnan(est) & (w > 0.05), e, est)
    est = np.nan_to_num(est)
    # le fond estimé est lissé à l'intérieur ; le grain remis à son niveau local
    g = synth_grain(amp, R.shape, rng)
    # niveau du grain : écart-type du passe-haut du ciel calme ~ sig
    soft = blur(fill.astype(np.float32), 1.0)
    out = (1 - soft) * R + soft * (est + g * sig)
    return np.where(valid, out, R)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--set", default="{}", help="réglages JSON")
    a = ap.parse_args()
    run(a.name, a.tag, json.loads(a.set))
