# ======================================================================
# audit_etoiles.py
# Rôle     : Étape 1 du générateur d'étoiles synthétiques (22 nov) : audit du premier générateur (6-9 nov) contre nos 4 images.
# Étape    : 4. Étoiles synthétiques réalistes (22 nov) (voir scripts/LISEZMOI.md)
# Entrées  : fonds sans étoiles (cibles de paires), banques de formes d'étoiles
# Sorties  : paires (entrée, cible), planches, test à l'aveugle
# Exemple  : Usage : python audit_etoiles.py [image ...]     (CPU, 4 fils, ~1-2 min par image)
# ======================================================================
"""Étape 1 du générateur d'étoiles synthétiques (22 nov) : audit du premier générateur (6-9 nov) contre nos 4 images.

Pour chaque image (rosette, m8_sho, m57, horsehead) :
  - 6 vraies étoiles (faible, moyenne, brillante, saturée, à aigrettes, la plus brillante), choisies isolées et sur un fond
    calme, positions et G du catalogue de l'image (astrometrie.npz de sa méthode) ; Alnitak ajoutée à la main (absente de Gaia) ;
  - pour chacune, l'étoile du premier générateur (psf.py + stars.far_kernel, lus en place, non modifiés) de même éclat
    (flux du catalogue par le point zéro de l'image) et de largeur la plus proche : un Newton tiré par draw_instrument à la
    largeur de l'image, ses aigrettes aux angles mesurés de l'image ; auréole au tirage médian (part 1,2 %, pente 2,05) et
    au tirage le plus fort (5 %, pente 1,5) ; écrêtage du premier générateur (un seul niveau par couleur et par image,
    ici le maximum de l'image) ; bruit du ciel de l'image ;
  - mesures : profils radiaux jusqu'à 330 px, profil le long de l'aigrette la plus forte, pointillés, plateau et son
    niveau, allongement et décentrement du halo, rayons fins ; à l'échelle de l'image : décalage et largeur par couleur
    sur ~300 étoiles G 11-14, niveaux de plateau des étoiles G < 8.
Sorties : A_REGARDER – 21 nov soir\\Synth_0_audit_<image>.png, audit_mesures.json (ici).
Usage : python audit_etoiles.py [image ...]     (CPU, 4 fils, ~1-2 min par image)
"""
from __future__ import annotations

import os
import sys

for k, v in dict(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", NUMBA_NUM_THREADS="2",
                 CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1").items():
    os.environ[k] = v
sys.dont_write_bytecode = True

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

cv2.setNumThreads(2)
SR = (Path(__import__("os").environ.get("RACINE_SCRIPTS") or Path(__file__).resolve().parents[1]))
HERE = Path(__file__).resolve().parent
OUT = SR / "A_REGARDER – 21 nov soir"
sys.path.insert(0, str(SR))                                            # premier générateur, lu en place
sys.path.insert(1, str(SR / "Methode v23 Horsehead – 22 nov" / "Piste1"))  # chargeur des images, lu en place
import psf as G1                                                        # noqa: E402
import stars as S1                                                      # noqa: E402
import images as IM                                                     # noqa: E402

ZP = [0.0]
V21, V22, V23 = "Methode v21 SHO – 22 nov", "Methode v22 M57 – 22 nov", "Methode v23 Horsehead – 22 nov"
IMAGES = {
    # nom : dossier de la méthode (catalogue, modèle), angles des aigrettes mesurés (journaux de l'étape A / fiches)
    "rosette": dict(dir=V21, angles=[3.0, 93.5, 183.0, 273.5], titre="Rosette (couleur, débruitée)"),
    "m8_sho": dict(dir=V21, angles=[55.5, 113.5, 172.0, 235.5, 293.5, 352.0], titre="M8 SHO (SII/Ha/OIII)"),
    "m57": dict(dir=V22, angles=[45.0, 133.0, 223.0, 313.0], titre="M57 (APN + astrographe)"),
    "horsehead": dict(dir=V23, angles=[0.0, 90.0, 180.0, 270.0], titre="Tête de Cheval (couleur)",
                      extra=[(2569.0, 1136.0, 1.8, "Alnitak")]),
}
H = 360                      # demi-côté des vignettes de mesure (profils jusqu'à 330 px)
CLASSES = ["faible", "moyenne", "brillante", "saturée", "à aigrettes", "la plus brillante"]


# ------------------------------------------------------------------------------------------------ outils
def stamp(a, x, y, h):
    """(C, 2h+1, 2h+1) centré sur le pixel le plus proche, NaN hors image ; et le décalage sous-pixel."""
    xi, yi = int(round(x)), int(round(y))
    C, Hh, W = a.shape
    out = np.full((C, 2 * h + 1, 2 * h + 1), np.nan, np.float32)
    y0, y1, x0, x1 = yi - h, yi + h + 1, xi - h, xi + h + 1
    sy0, sx0 = max(0, y0), max(0, x0)
    sy1, sx1 = min(Hh, y1), min(W, x1)
    if sy0 < sy1 and sx0 < sx1:
        out[:, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = a[:, sy0:sy1, sx0:sx1]
    return out


_RINGS = {}


def rings(h):
    if h not in _RINGS:
        yy, xx = np.mgrid[-h:h + 1, -h:h + 1]
        r = np.hypot(xx, yy)
        ri = np.rint(r).astype(int).ravel()
        order = np.argsort(ri, kind="stable")
        cuts = np.flatnonzero(np.diff(ri[order])) + 1
        _RINGS[h] = (r, np.arctan2(yy, xx), order, cuts, np.unique(ri))
    return _RINGS[h]


def ring_median(s):
    """Médiane par anneau d'un pixel de large (C, nr), r = 0 .. h."""
    h = s.shape[-1] // 2
    r, th, order, cuts, rr = rings(h)
    out = np.full((s.shape[0], h + 1), np.nan)
    for c in range(s.shape[0]):
        parts = np.split(s[c].ravel()[order], cuts)
        for k, p in zip(rr, parts):
            if k <= h:
                p = p[np.isfinite(p)]
                if p.size:
                    out[c, k] = np.median(p)
    return out


def logbin(prof, edges):
    """Moyenne des médianes d'anneau dans des intervalles logarithmiques."""
    r = np.arange(prof.shape[-1])
    out = np.full((prof.shape[0], len(edges) - 1), np.nan)
    for i in range(len(edges) - 1):
        m = (r >= edges[i]) & (r < edges[i + 1])
        if m.any():
            out[:, i] = np.nanmean(prof[:, m], axis=1)
    return out


def second_moment_fwhm(s, fw0, bg):
    """FWHM par couleur par le moment d'ordre 2 dans r <= 2 fw0 (fond soustrait) et centroïde par couleur."""
    h = s.shape[-1] // 2
    r, *_ = rings(h)
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1]
    m = r <= 2 * fw0
    fw, cx, cy = [], [], []
    for c in range(s.shape[0]):
        w = np.clip(np.nan_to_num(s[c] - bg[c]), 0, None) * m
        sw = w.sum()
        if sw <= 0:
            fw.append(np.nan); cx.append(np.nan); cy.append(np.nan); continue
        mx, my = (w * xx).sum() / sw, (w * yy).sum() / sw
        s2 = (w * ((xx - mx) ** 2 + (yy - my) ** 2)).sum() / (2 * sw)
        fw.append(2.3548 * math.sqrt(max(s2, 1e-6))); cx.append(mx); cy.append(my)
    return np.array(fw), np.array(cx), np.array(cy)


def sky_noise(a):
    """Bruit du ciel par couleur : 20e centile des MAD de blocs 64 px (les blocs calmes)."""
    C, Hh, W = a.shape
    out = []
    for c in range(C):
        mads = []
        for y in range(0, Hh - 64, 256):
            for x in range(0, W - 64, 256):
                b = a[c, y:y + 64, x:x + 64]
                if np.all(b == 0):
                    continue
                mads.append(1.4826 * np.median(np.abs(b - np.median(b))))
        out.append(float(np.percentile(mads, 20)))
    return np.array(out)


def plateau(s, peak_r, pl_r, frac=0.98):
    """Pic propre par couleur (max dans r <= peak_r), nombre de pixels >= frac x pic dans r <= pl_r."""
    h = s.shape[-1] // 2
    r, *_ = rings(h)
    pk = np.array([np.nanmax(np.where(r <= peak_r, s[c], -np.inf)) for c in range(s.shape[0])])
    n = np.array([int(np.sum((r <= pl_r) & (s[c] >= frac * pk[c]))) for c in range(s.shape[0])])
    return pk, n


# ------------------------------------------------------------------------------------------------ premier générateur
class FixedRng:
    """Pour far_kernel : rend des tirages fixés (pente, r0 / fwhm, log de la part)."""
    def __init__(self, vals):
        self.vals = list(vals)

    def uniform(self, lo, hi):
        return self.vals.pop(0)


def gen1_psf(fw_target, angles, seed, variants=(("médian", 2.05, 2.0, 0.0122), ("plus fort", 1.5, 2.0, 0.05))):
    """PSF totale (C, 2H+1, 2H+1), somme 1 par couleur, d'un Newton du premier générateur dont la largeur est la plus
    proche de fw_target ; aigrettes alignées sur les angles mesurés (4 -> 4 branches, 6 -> 3 branches)."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(40):
        inst = G1.draw_instrument(rng, fwhm_px=fw_target)
        if inst.kind != "newton":
            continue
        inst.vanes = 3 if len(angles) == 6 else 4
        inst.vane_angle = angles[0] - 90.0
        inst.vane_twist = 0.0
        inst.track_px = 0.0
        core = G1.psf_at(inst, 0.0, 0.0, S1.PSF_SIZE)
        h = core.shape[-1] // 2
        fw, *_ = second_moment_fwhm(stamp(core, h, h, 30), fw_target, np.zeros(3))
        d = abs(float(fw[1]) - fw_target)
        if best is None or d < best[0]:
            best = (d, inst, core, float(fw[1]))
        if sum(1 for _ in [0]) and best[0] < 0.1 * fw_target:
            break
    _, inst, core, fwc = best
    out = {}
    for name, gamma, r0f, share in variants:
        info = {}
        far, sh = S1.far_kernel(FixedRng([gamma, r0f, math.log(share)]), inst, core, info)
        c0 = far.shape[-1] // 2
        tot = far[:, c0 - H:c0 + H + 1, c0 - H:c0 + H + 1].copy()
        k = core.shape[-1] // 2
        tot[:, H - k:H + k + 1, H - k:H + k + 1] += (1 - sh) * core
        out[name] = (tot, info)
    return inst, fwc, out


# ------------------------------------------------------------------------------------------------ mesures d'une étoile
def spike_profile(s, angle_deg, bgprof, r_max=330, d=10.0):
    """Excès le long d'une aigrette (C, r_max) : moyenne sur 3 px en travers moins la moyenne de deux lignes
    parallèles à +-d px (le fond local : halo et nébuleuse)."""
    h = s.shape[-1] // 2
    t = math.radians(angle_deg)
    r = np.arange(r_max, dtype=float)
    ex = np.zeros((s.shape[0], r_max))
    for c in range(s.shape[0]):
        def line(o):
            xs = h + r * math.cos(t) - o * math.sin(t)
            ys = h + r * math.sin(t) + o * math.cos(t)
            return map_coordinates(np.nan_to_num(s[c]), [ys, xs], order=1)
        ex[c] = (line(-1.0) + line(0.0) + line(1.0)) / 3 - 0.5 * (line(-d) + line(d))
    return ex


def dotted(ex, sig, r0, r1):
    """Pointillés d'une aigrette par couleur : modulation (MAD de ex / lissé - 1) et période dominante (px)."""
    res = []
    for c in range(ex.shape[0]):
        r = np.arange(ex.shape[1])
        m = (r >= r0) & (r <= r1) & (ex[c] > 3 * sig[c])
        if m.sum() < 24:
            res.append(None); continue
        rr = r[m]; e = ex[c][m]
        k = 15
        sm = np.array([np.median(e[max(0, i - k // 2):i + k // 2 + 1]) for i in range(e.size)])
        q = e / np.maximum(sm, 1e-9) - 1
        mod = float(1.4826 * np.median(np.abs(q)))
        # période : spectre de q sur la partie continue
        f = np.abs(np.fft.rfft(q - q.mean()))
        fr = np.fft.rfftfreq(q.size)
        ok = (fr > 1 / 60) & (fr < 1 / 3)
        per = float(1 / fr[ok][np.argmax(f[ok])]) if ok.any() else None
        res.append(dict(modulation=round(mod, 3), periode_px=None if per is None else round(per, 1),
                        r=[int(rr[0]), int(rr[-1])]))
    return res


def halo_shape(L, sigL, angles, r_lo, r_hi):
    """Allongement (terme en 2θ) et décentrement (terme en θ) du halo, en part du niveau de l'anneau, aigrettes masquées."""
    h = L.shape[-1] // 2
    r, th, *_ = rings(h)
    thd = np.degrees(th) % 360
    edges = np.geomspace(r_lo, r_hi, 9)
    rows = []
    for i in range(len(edges) - 1):
        m = (r >= edges[i]) & (r < edges[i + 1]) & np.isfinite(L)
        bins = (thd[m] // 5).astype(int)
        v = L[m]
        med = np.array([np.median(v[bins == b]) if np.any(bins == b) else np.nan for b in range(72)])
        cen = np.arange(72) * 5 + 2.5
        keep = np.isfinite(med)
        for a in angles:
            keep &= np.abs((cen - a + 180) % 360 - 180) > 7
        if keep.sum() < 30:
            continue
        t = np.radians(cen[keep])
        A = np.c_[np.ones(t.size), np.cos(t), np.sin(t), np.cos(2 * t), np.sin(2 * t)]
        co, *_ = np.linalg.lstsq(A, med[keep], rcond=None)
        if co[0] < 5 * sigL:
            continue
        rows.append(dict(r=round(float(math.sqrt(edges[i] * edges[i + 1])), 1),
                         niveau=round(float(co[0]), 3),
                         allongement=round(float(math.hypot(co[3], co[4]) / co[0]), 3),
                         axe_deg=round(float(np.degrees(0.5 * math.atan2(co[4], co[3])) % 180), 1),
                         decentrement=round(float(math.hypot(co[1], co[2]) / co[0]), 3)))
    return rows


def fine_rays(L, sigL, angles, r_lo, r_hi):
    """Rayons fins : contraste rms en angle (pas 0,5°) après les harmoniques <= 6, aigrettes masquées, rapporté au
    contraste attendu du bruit ; nombre de pics > 3 bruits."""
    h = L.shape[-1] // 2
    r, th, *_ = rings(h)
    m = (r >= r_lo) & (r < r_hi) & np.isfinite(L)
    thd = np.degrees(th[m]) % 360
    v = L[m]
    nb = 720
    b = (thd / 0.5).astype(int) % nb
    cnt = np.bincount(b, minlength=nb)
    mean = np.bincount(b, v, minlength=nb) / np.maximum(cnt, 1)
    cen = np.arange(nb) * 0.5 + 0.25
    keep = cnt > 3
    for a in angles:
        keep &= np.abs((cen - a + 180) % 360 - 180) > 5
    t = np.radians(cen[keep])
    A = np.column_stack([np.ones(t.size)] + [f(k * t) for k in range(1, 7) for f in (np.cos, np.sin)])
    co, *_ = np.linalg.lstsq(A, mean[keep], rcond=None)
    res = mean[keep] - __import__('operator').matmul(A, co)
    exp = sigL / math.sqrt(max(float(np.median(cnt[keep])), 1))
    lvl = max(float(co[0]), 1e-9)
    n_pk = int(np.sum((res[1:-1] > res[:-2]) & (res[1:-1] > res[2:]) & (res[1:-1] > 3 * exp)))
    return dict(r=[r_lo, r_hi], contraste=round(float(res.std() / lvl), 4),
                contraste_bruit=round(float(exp / lvl), 4), pics_3bruits=n_pk)


def star_measures(s, fw, sig, imax, flux, angles, giant, local_bg=False):
    """Mesures d'une étoile sur sa vignette (C, 2H+1, 2H+1). Fond : anneau 330-350 px, ou 6-10 FWHM (local_bg)."""
    prof = ring_median(s)
    bg = np.nanmedian(prof[:, int(6 * fw):int(10 * fw)] if local_bg else prof[:, 330:350], axis=1)
    bg = np.nan_to_num(bg)
    p = prof - bg[:, None]
    pk, npl = plateau(s, 40 if giant else 15, 140 if giant else 60)
    fwc, cx, cy = second_moment_fwhm(s, fw, bg)
    # aigrette la plus forte (excès intégré 3 fw .. 150 px)
    best, ex_best = None, None
    for a in angles:
        ex = spike_profile(s, a, prof, d=max(3 * fw, 8.0))
        sc = float(np.nansum(ex[:, int(3 * fw):150]))
        if best is None or sc > best[1]:
            best, ex_best = (a, sc), ex
    L = np.nansum(s - bg[:, None, None], axis=0)
    sigL = float(np.sqrt((sig ** 2).sum()))
    out = dict(
        pic=[round(float(v), 1) for v in pk], plateau_px=npl.tolist(),
        pic_sur_max_image=[round(float(v), 3) for v in pk / imax],
        fwhm_moment=[round(float(v), 2) for v in fwc],
        centroide_RmG=[round(float(cx[0] - cx[1]), 3), round(float(cy[0] - cy[1]), 3)],
        centroide_BmG=[round(float(cx[2] - cx[1]), 3), round(float(cy[2] - cy[1]), 3)],
        aigrette_angle=best[0],
        pointilles=dotted(ex_best, sig, int(3 * fw), 300),
    )
    lum = np.nansum(p, axis=0) / flux
    for rr in (5, 10, 20, 50, 100, 200, 300):
        out[f"I_sur_F_r{rr}"] = float(f"{np.nanmean(lum[max(0, int(rr * 0.9)):int(rr * 1.1) + 1]):.3g}")
    if giant or npl[1] >= 9:
        out["halo_forme"] = halo_shape(L, sigL, angles, max(4 * fw, 3 * math.sqrt(max(npl[1], 1))), 260)
    if giant:
        out["rayons_fins"] = fine_rays(L, sigL, angles, 150, 250)
    return out, prof, bg, ex_best


# ------------------------------------------------------------------------------------------------ par image
def image_stats(a, x, y, g, fw, sig, imax, valid):
    """Décalage et largeur par couleur (G 11-14, isolées, non saturées) ; niveaux de plateau des étoiles G < 8."""
    Hh, W = a.shape[1:]
    tree = cKDTree(np.c_[x, y])
    sel = np.flatnonzero((g >= 11) & (g <= 14) & valid & (x > 40) & (y > 40) & (x < W - 40) & (y < Hh - 40))
    rows = []
    h = int(math.ceil(3 * fw)) + 3
    for i in sel:
        nb = tree.query_ball_point([x[i], y[i]], 6 * fw)
        if any(j != i and g[j] < g[i] + 4 for j in nb):
            continue
        s = stamp(a, x[i], y[i], h)
        if not np.all(np.isfinite(s)) or np.any(s == 0):
            continue
        pr = ring_median(s)
        bg = np.nanmedian(pr[:, int(2.5 * fw):int(3 * fw) + 2], axis=1)
        pk = np.nanmax(s[:, h - 2:h + 3, h - 2:h + 3], axis=(1, 2))
        if np.any(pk > 0.3 * imax) or np.any(pk - bg < 30 * sig):
            continue
        fwc, cx, cy = second_moment_fwhm(s, fw, bg)
        # centroïde vert raffiné : recentrer n'est pas nécessaire (différences entre couleurs)
        r_, *_ = rings(h)
        ap = float(np.sum(np.where(r_ <= 3 * fw, s - bg[:, None, None], 0)))
        rows.append([cx[0] - cx[1], cy[0] - cy[1], cx[2] - cx[1], cy[2] - cy[1], fwc[0] / fwc[1], fwc[2] / fwc[1],
                     float(np.min((pk - bg) / sig)), ap / 10 ** (-0.4 * (g[i] - ZP[0]))])
        if len(rows) >= 300:
            break
    R = np.array(rows) if rows else np.zeros((0, 8))
    mad = lambda v: float(1.4826 * np.median(np.abs(v - np.median(v)))) if v.size else None
    stat = dict(n=int(len(R)), k_ouverture_sur_catalogue=round(float(np.median(R[:, 7])), 3) if len(R) else 1.0)
    if len(R):
        for k, name in enumerate(["dx_RmG", "dy_RmG", "dx_BmG", "dy_BmG"]):
            stat[name] = dict(mediane=round(float(np.median(R[:, k])), 4), dispersion=round(mad(R[:, k]), 4))
        stat["largeur_R_sur_G"] = dict(mediane=round(float(np.median(R[:, 4])), 3), dispersion=round(mad(R[:, 4]), 3))
        stat["largeur_B_sur_G"] = dict(mediane=round(float(np.median(R[:, 5])), 3), dispersion=round(mad(R[:, 5]), 3))
        # erreur de centroïde due au bruit seul (approx. sigma_psf / RSB intégré)
        snr_tot = R[:, 6] * 2 * math.pi * (fw / 2.3548) ** 2 / (math.sqrt(4 * math.pi) * fw / 2.3548)
        stat["erreur_bruit_px_mediane"] = round(float(np.median((fw / 2.3548) / np.maximum(snr_tot, 1))), 4)
        stat["pic_min_bruits_mediane"] = round(float(np.median(R[:, 6])), 0)
    # plateaux des étoiles G < 8
    pl = []
    for i in np.flatnonzero((g < 8) & (x > 50) & (y > 50) & (x < W - 50) & (y < Hh - 50)):
        giant = g[i] < 4.5
        s = stamp(a, x[i], y[i], 150 if giant else 60)
        pk, n = plateau(s, 40 if giant else 15, 140 if giant else 60)
        if n[1] >= 9:
            pl.append(dict(G=round(float(g[i]), 2), xy=[int(x[i]), int(y[i])], plateau_px=n.tolist(),
                           niveau_sur_max=[round(float(v), 3) for v in pk / imax]))
    stat["plateaux_G_lt_8"] = pl
    if pl:
        lv = np.array([p["niveau_sur_max"][1] for p in pl])
        stat["niveau_plateau_V_min_max"] = [round(float(lv.min()), 3), round(float(lv.max()), 3)]
    return stat


def choose(a, x, y, g, fw, valid, sat_ok):
    """Les 6 étoiles : isolées, fond calme, assez loin du bord pour les profils."""
    Hh, W = a.shape[1:]
    tree = cKDTree(np.c_[x, y])

    def inside(i, m):
        return m < x[i] < W - m and m < y[i] < Hh - m

    def isolated(i, R, dG):
        return all(j == i or g[j] > g[i] + dG for j in tree.query_ball_point([x[i], y[i]], R))

    def calm(i, h):
        s = stamp(a[1:2], x[i], y[i], h)[0]
        r, *_ = rings(h)
        v = s[(r > 0.6 * h) & np.isfinite(s)]
        return float(1.4826 * np.median(np.abs(v - np.median(v)))) if v.size else np.inf

    gmax = float(np.nanmax(g[valid]))
    picks = {}
    spec = {"faible": ((min(15.0, gmax - 0.8), min(16.0, gmax)), 20 * fw / 4, 3, 30, H),
            "moyenne": ((12.5, 13.5), 30, 3, 40, H),
            "brillante": ((9.5, 10.5), 60, 3, 80, H)}
    for cl, ((g0, g1), R, dG, hc, m) in spec.items():
        cand = [i for i in np.flatnonzero((g >= g0) & (g <= g1) & valid) if inside(i, m) and isolated(i, R, dG)]
        cand = cand[:400]
        if cl == "brillante":
            cand = [i for i in cand if not sat_ok(i)]
        sc = [calm(i, hc) for i in cand]
        picks[cl] = cand[int(np.argmin(sc))] if cand else None
    order = np.argsort(g)
    brightest = next(i for i in order if valid[i] and inside(i, 60))
    picks["la plus brillante"] = brightest
    # saturée : la moins brillante des étoiles G 6-9,5 à plateau, isolée, loin du bord
    cand = [i for i in order[::-1] if 6 <= g[i] <= 9.5 and valid[i] and inside(i, H) and isolated(i, 80, 4)
            and i != brightest]
    cs = [i for i in cand if sat_ok(i)][:12]
    picks["saturée"] = cs[int(np.argmin([calm(i, 200) for i in cs]))] if cs else None
    # à aigrettes : la plus brillante après la plus brillante, sans plus brillante à 150 px
    cand = [i for i in order if valid[i] and i != brightest and g[i] >= 4.4 and inside(i, 200)
            and isolated(i, 150, 0)]
    cand = cand[:5]
    picks["à aigrettes"] = cand[int(np.argmin([calm(i, 200) for i in cand]))] if cand else None
    return picks


def stretch60(s, bk, wt):
    v = np.arcsinh(60 * np.clip((np.nan_to_num(s) - bk) / max(wt - bk, 1e-9), 0, None)) / np.arcsinh(60)
    return np.transpose(np.clip(v, 0, 1), (1, 2, 0))


def stretch_strong(s, sig, imax):
    """Fort : asinh à 2 bruits, blanc à 5 % du max de l'image, fond = médiane du bord de la vignette (la sienne)."""
    s = np.nan_to_num(s)
    edge = np.concatenate([s[:, :4].reshape(s.shape[0], -1), s[:, -4:].reshape(s.shape[0], -1)], 1)
    bgv = np.median(edge, axis=1)[:, None, None]
    k = 2 * sig.mean()
    v = np.arcsinh(np.clip(s - bgv + 2 * k, 0, None) / k) / np.arcsinh(0.05 * imax.mean() / k)
    return np.transpose(np.clip(v, 0, 1), (1, 2, 0))


def run(name):
    t0 = time.time()
    cfg = IMAGES[name]
    d = SR / cfg["dir"] / "Piste1" / "sorties" / name
    a = IM.load(name)
    C = a.shape[0]
    mod = json.loads((d / "modele_psf_image.json").read_text(encoding="utf-8"))
    fw, zp = float(mod["fwhm"]), float(mod["zp"])
    ZP[0] = zp
    cat = np.load(d / "astrometrie.npz", allow_pickle=True)["cat"].item()
    x, y, g = (np.asarray(cat[k], float) for k in ("x", "y", "g"))
    names = [""] * len(g)
    for (ex, ey, eg, en) in cfg.get("extra", []):
        x, y, g = np.r_[x, ex], np.r_[y, ey], np.r_[g, eg]
        names.append(en)
    Hh, W = a.shape[1:]
    valid = (x >= 0) & (y >= 0) & (x < W) & (y < Hh)
    vi = np.clip(np.rint(y).astype(int), 0, Hh - 1), np.clip(np.rint(x).astype(int), 0, W - 1)
    valid &= a[1][vi] != 0
    sig = sky_noise(a)
    imax = a.reshape(C, -1).max(1)
    bk = float(np.median(a[1, ::7, ::7])); wt = float(np.percentile(a[1, ::5, ::5], 99.8))
    print(f"[{name}] {a.shape}, fwhm {fw:.2f}, zp {zp:.2f}, bruit {sig.round(3)}, max {imax.round(0)}, "
          f"{time.time() - t0:.0f} s", flush=True)

    def sat_ok(i):
        s = stamp(a, x[i], y[i], 60)
        pk, n = plateau(s, 15, 60)
        return n[1] >= 9

    picks = choose(a, x, y, g, fw, valid, sat_ok)
    stats = image_stats(a, x, y, g, fw, sig, imax, valid)
    print(f"[{name}] choix {[(k, None if v is None else round(float(g[v]), 2)) for k, v in picks.items()]}, "
          f"stats {time.time() - t0:.0f} s", flush=True)
    res = dict(image=name, fwhm=fw, zp=zp, bruit=sig.round(4).tolist(), max_image=imax.round(1).tolist(),
               stats_image={k: v for k, v in stats.items()}, etoiles={})
    gen_cache = {}
    rows = []
    for cl in CLASSES:
        i = picks.get(cl)
        if i is None:
            res["etoiles"][cl] = None
            continue
        giant = g[i] < 4.5
        s = stamp(a, x[i], y[i], H)
        flux_cat = 10 ** (-0.4 * (g[i] - zp))
        flux = flux_cat * stats["k_ouverture_sur_catalogue"]
        loc = cl in ("faible", "moyenne")
        m_real, prof_r, bg_r, ex_r = star_measures(s, fw, sig, imax, flux, cfg["angles"], giant, loc)
        if not (m_real["plateau_px"][1] >= 9 or giant):          # non saturée : son flux mesuré (ouverture 3 FWHM)
            r_, *_ = rings(H)
            flux = float(np.nansum(np.where(r_ <= 3 * fw, s - bg_r[:, None, None], 0)))
            m_real, prof_r, bg_r, ex_r = star_measures(s, fw, sig, imax, flux, cfg["angles"], giant, loc)
        # couleur de l'étoile : ouverture 3 fw (non saturée) ou ailes 3-8 fw (saturée)
        r_, *_ = rings(H)
        if m_real["plateau_px"][1] >= 9 or giant:
            sh = np.nanmean(prof_r[:, int(3 * fw):int(8 * fw)] - bg_r[:, None], axis=1)
        else:
            sh = np.array([np.nansum(np.where(r_ <= 3 * fw, s[c] - bg_r[c], 0)) for c in range(C)])
        part = np.clip(sh, 1e-6, None) / np.clip(sh, 1e-6, None).sum()
        m_real["part_couleur"] = part.round(3).tolist()
        if not (m_real["plateau_px"][1] >= 9 or giant):
            ap = float(np.nansum(np.where(r_ <= 3 * fw, s - bg_r[:, None, None], 0)))
            m_real["flux_ouverture_sur_catalogue"] = round(ap / flux_cat, 3)
        if "gen" not in gen_cache:
            gen_cache["gen"] = gen1_psf(fw, cfg["angles"], seed=[22, 11, len(name)])
        inst, fwc, variants = gen_cache["gen"]
        rng = np.random.default_rng([7, CLASSES.index(cl)])
        synth = {}
        for vn, (T, info) in variants.items():
            S = flux * part[:, None, None] * T[:C] if C == 3 else flux * T[1:2]
            S = np.minimum(S, imax[:, None, None])                     # premier générateur : un seul niveau par couleur
            S = S + sig[:, None, None] * rng.standard_normal(S.shape).astype(np.float32)
            ms, prof_s, bg_s, ex_s = star_measures(S, fw, sig, imax, flux, cfg["angles"], giant, loc)
            synth[vn] = dict(mesures=ms, info=info, S=S, prof=prof_s, bg=bg_s, ex=ex_s)
        res["etoiles"][cl] = dict(G=round(float(g[i]), 3), xy=[round(float(x[i]), 1), round(float(y[i]), 1)],
                                  nom=names[i], flux_catalogue=float(f"{flux_cat:.4g}"),
                                  flux_normalisation=float(f"{flux:.4g}"), reel=m_real,
                                  premier_generateur={vn: v["mesures"] for vn, v in synth.items()},
                                  instrument_gen1=dict(airy_px=round(inst.airy_px, 3),
                                                       seeing_fwhm=round(inst.seeing_fwhm, 2),
                                                       obstruction=round(inst.obstruction, 3), vanes=inst.vanes,
                                                       vane_width=round(inst.vane_width, 4), fwhm_coeur=round(fwc, 2),
                                                       auréoles={vn: v["info"] for vn, v in synth.items()}))
        rows.append((cl, i, s, prof_r, bg_r, ex_r, synth, flux, m_real))
        print(f"[{name}] {cl} G {g[i]:.2f} {time.time() - t0:.0f} s", flush=True)
    sheet(name, cfg, rows, g, fw, sig, bk, wt, imax)
    print(f"[{name}] fini {time.time() - t0:.0f} s", flush=True)
    return res


# ------------------------------------------------------------------------------------------------ planche
def sheet(name, cfg, rows, g, fw, sig, bk, wt, imax):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(rows)
    fig, axs = plt.subplots(n, 8, figsize=(30, 3.9 * n), gridspec_kw=dict(width_ratios=[1] * 6 + [1.5, 1.5]))
    if n == 1:
        axs = axs[None]
    wide = {"faible": 32, "moyenne": 48, "brillante": 100, "saturée": 160, "à aigrettes": 220, "la plus brillante": 330}
    edges = np.unique(np.r_[np.arange(0, 10), np.geomspace(10, 331, 30)].astype(int))
    rc = np.sqrt(np.maximum(edges[:-1], 0.5) * np.maximum(edges[1:] - 1, 0.5))
    for k, (cl, i, s, prof_r, bg_r, ex_r, synth, flux, m_real) in enumerate(rows):
        S = synth["médian"]["S"]
        c = s.shape[-1] // 2
        z = 32
        crops = [s[:, c - z:c + z, c - z:c + z], S[:, c - z:c + z, c - z:c + z]]
        wz = wide[cl]
        wcrops = [s[:, c - wz:c + wz, c - wz:c + wz], S[:, c - wz:c + wz, c - wz:c + wz]]
        ims = [stretch60(crops[0], bk, wt), stretch60(crops[1], bk, wt), stretch_strong(crops[0], sig, imax),
               stretch_strong(crops[1], sig, imax), stretch_strong(wcrops[0], sig, imax),
               stretch_strong(wcrops[1], sig, imax)]
        tit = ["réelle ×4, arcsinh 60", "1er générateur ×4, arcsinh 60", "réelle ×4, fort (asinh)",
               "1er générateur ×4, fort",
               f"réelle {2 * wz} px, fort", f"1er générateur {2 * wz} px, fort"]
        for j in range(6):
            im = ims[j]
            if im.shape[-1] == 1:
                im = np.repeat(im, 3, -1)
            axs[k, j].imshow(im, interpolation="nearest")
            axs[k, j].set_xticks([]); axs[k, j].set_yticks([])
            axs[k, j].set_title(tit[j], fontsize=9)
        lab = f"{cl}\nG {g[i]:.2f}\nplateau V {m_real['plateau_px'][1]} px\npic/max {m_real['pic_sur_max_image'][1]:.2f}"
        axs[k, 0].set_ylabel(lab, fontsize=10)
        # profils radiaux (somme des couleurs, fond retiré, / flux du catalogue)
        ax = axs[k, 6]
        L = np.nansum(prof_r - bg_r[:, None], 0) / flux
        ax.loglog(rc, logbin(L[None], edges)[0], "k-", lw=2, label="réelle")
        for vn, st, col in (("médian", "-", "tab:blue"), ("plus fort", "--", "tab:blue")):
            Ls = np.nansum(synth[vn]["prof"] - synth[vn]["bg"][:, None], 0) / flux
            ax.loglog(rc, logbin(Ls[None], edges)[0], st, color=col, label=f"1er gén. ({vn})")
        ax.axhline(np.sqrt((sig ** 2).sum()) / flux, color="0.6", lw=0.8, ls=":", label="bruit / flux")
        ax.set_xlim(1, 330); ax.set_title("profil radial (médiane par anneau) / flux", fontsize=9)
        ax.set_ylim(max(1e-9, np.sqrt((sig ** 2).sum()) / flux / 30), None)
        ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")
        # aigrette la plus forte
        ax = axs[k, 7]
        r = np.arange(ex_r.shape[1])
        e = np.nansum(ex_r, 0) / flux
        ax.semilogy(r[3:], np.clip(e[3:], 1e-12, None), "k-", lw=1, label=f"réelle ({m_real['aigrette_angle']:.0f}°)")
        es = np.nansum(synth["médian"]["ex"], 0) / flux
        ax.semilogy(r[3:], np.clip(es[3:], 1e-12, None), color="tab:blue", lw=1, label="1er générateur")
        ax.axhline(np.sqrt((sig ** 2).sum()) / flux, color="0.6", lw=0.8, ls=":")
        ax.set_ylim(max(1e-10, np.sqrt((sig ** 2).sum()) / flux / 10), None)
        ax.set_xlim(0, 330); ax.set_title("le long de l'aigrette (excès sur 2 parallèles) / flux", fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")
    fig.suptitle(f"{cfg['titre']} — 6 vraies étoiles et l'étoile du premier générateur de même éclat et largeur "
                 f"(FWHM {fw:.2f} px, écrêtage du 1er générateur = max de l'image)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    p = OUT / f"Synth_0_audit_{name}.png"
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    assert ok
    buf.tofile(str(p))
    assert p.exists() and p.stat().st_size > 0


def main():
    names = sys.argv[1:] or list(IMAGES)
    jp = HERE / "audit_mesures.json"
    allres = json.loads(jp.read_text(encoding="utf-8")) if jp.exists() else {}
    for n in names:
        allres[n] = run(n)
        jp.write_text(json.dumps(allres, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
