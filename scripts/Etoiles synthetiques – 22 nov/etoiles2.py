# ======================================================================
# etoiles2.py
# Role     : Synthetic star generator v2 (22 Nov): a drawn star field, ADDED onto a tile of real star-free background.
# Step     : 4. Realistic synthetic stars (22 Nov) (see scripts/LISEZMOI.md)
# Inputs   : star-free backgrounds (pair targets), star shape banks
# Outputs  : pairs (input, target), sheets, blind test
# Example  : python etoiles2.py [arguments]  (see the docstring; paths through the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Synthetic star generator v2 (22 Nov): a drawn star field, ADDED onto a tile of real star-free background.

    out = ajouter_etoiles(fond, graine)          # background (C, n, n) in sky noises, background 0; C = 3 (colour or filters)
    out["entree"], out["etoiles"], out["ecretes"], out["fiche"]

Reproducible by seed (same seed, same background -> same arrays to the bit on the same machine).

What is taken over from the first generator (starRemoval, read in place, not modified): the Fourier optics (psf.py: pupil,
obstruction, field-varying Zernike: coma, astigmatism, defocus and donuts, turbulence, tracking), the real commercial
instruments (instruments.py + catalogue.json) and the field draw (stars.draw_field: power-law counts, clusters,
Alnitak-class stars).

What changes (each step has its sheet Synth_<k>_*.png; ETAPES lists the order):
  1 aureole   : per star, I/F = a50 ((r^2 + rc^2) / (50^2 + rc^2))^(-gamma/2), a50 3e-7 to 4e-6, gamma 1.6 to 2.6
                (measured on our 4 images: PLAN.md, gap 1), scatter 0.15 dex from one star to another;
  2 spikes    : analytic (the Fourier optics is computed without branches): number and shape drawn (none, 2, 4,
                4 doubled, 6, 8), independent of the instrument type, free angle; I/F along =
                s30 ((1 + (r/fw)^2) / (1 + (30/fw)^2))^(-p/2) exp(-(r / L)^2) + s11 ((r^2 + fw^2) / (121 + fw^2))^-2 :
                s30 4e-7 to 6e-6, p 1.5 to 3.2, cutoff L 40 to 300 px in 40 % of cases (mesure_aigrettes.json: our 4
                images, crossed stars discarded); steep near part s11 1.6e-5 to 1.2e-4; cross-width
                sigma 0.45 to 0.9 FWHM (measured at half height), growing with brightness; length scaled with lambda
                per colour (rainbow); dotted pattern with a pitch specific to each colour or filter;
  3 plateau   : each star clips at ITS level (0.35 to 1 of the image max, per colour), soft edge;
  4 colours   : R and B shift per star (common to the image + random) and width per colour;
  5 halo      : elongation (term in 2 theta growing with r), decentring towards the optical centre, soft reflection
                rings, off-centre ghost reflections, fine rays around the giants and, for an instrument that has
                them, around all clipped stars; the light outside the core fades out over the last quarter of the
                window (no square edge);
  6 shapes    : ellipticity, second component and lumps (smooth core texture) per star, calibrated on the measured
                ellipticity and asymmetry of the G 12-13.5 stars (mesure_formes.py), tight doubles;
  7 colour    : BP-RP drawn from the Gaia distributions of our 4 fields, converted by Planck's law at the temperature
                of Mucciarelli, Bellazzini & Massari 2021 (A&A 653, A90: dwarfs, BP-RP), white balance or filters.
"""
from __future__ import annotations

import os as _os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    _os.environ[_k] = "2"                                  # au plus 2 fils (consigne du 22 nov après la saturation)

import copy
import math
import sys
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(2)
from scipy.signal import fftconvolve

SR = (Path(__import__("os").environ.get("RACINE_SCRIPTS") or Path(__file__).resolve().parents[1]))
if str(SR) not in sys.path:
    sys.path.insert(0, str(SR))
import psf as G1                                                       # noqa: E402
import stars as S1                                                     # noqa: E402

ETAPES = ("aureole", "aigrettes", "plateau", "couleurs", "halo", "formes", "couleur_gaia", "banque")
WAVE_RGB = (640.0, 530.0, 460.0)
WAVE_NB = (672.0, 656.0, 501.0)                                        # SII, Ha, OIII
LAM0 = 530.0
PSF_SIZE = 129
MARGE = 600                                                            # portée max d'une étoile hors de la tuile
SEUIL_VIGNETTE = 20.0                                                  # pic (bruits) au-dessus duquel une étoile a sa vignette
#: BP-RP des étoiles G < 16 de nos 4 champs (catalogues Gaia profonds hors ligne), centiles 0, 1, 5, 25, 50, 75, 95, 99, 100
BPRP = {"rosette": [0.0, 0.32, 0.65, 0.97, 1.17, 1.49, 2.45, 3.12, 4.0],
        "m8_sho": [0.1, 0.37, 0.85, 1.16, 1.5, 2.25, 3.48, 5.85, 6.5],
        "m57": [0.2, 0.47, 0.7, 0.85, 0.96, 1.19, 1.5, 2.09, 3.0],
        "horsehead": [-0.2, 0.15, 0.82, 1.23, 1.52, 1.92, 2.74, 3.44, 4.5]}
CENTILES = [0, 1, 5, 25, 50, 75, 95, 99, 100]


def lu(rng, lo, hi):
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


# ------------------------------------------------------------------------------------------------ couleur
def teff(bprp):
    """Mucciarelli, Bellazzini & Massari 2021 (nains) : theta = 5040 / T = 0,4929 + 0,5092 C - 0,0353 C^2."""
    c = np.clip(bprp, -0.3, 4.5)
    return 5040.0 / (0.4929 + 0.5092 * c - 0.0353 * c * c)


def planck(lam_nm, T):
    x = 1.4388e7 / (lam_nm * np.asarray(T, float))
    return lam_nm ** -5 / np.expm1(np.clip(x, 1e-6, 700))


def parts_planck(bprp, waves, gains):
    """(C, n) parts de flux par couleur (somme 1) d'étoiles de BP-RP donnés."""
    T = teff(np.atleast_1d(bprp))
    f = np.stack([planck(w, T) * g for w, g in zip(waves, gains)])
    return (f / f.sum(0, keepdims=True)).astype(np.float32)


# ------------------------------------------------------------------------------------------------ outils
def moment_fwhm(p, rmax=None):
    """FWHM par le moment d'ordre 2 dans r <= rmax (le même calcul que la mesure sur nos images : rmax = 2 FWHM)."""
    h = p.shape[-1] // 2
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1]
    r = np.hypot(xx, yy)
    w = np.clip(p, 0, None) * (r <= (rmax or max(8, 0.25 * h)))
    mx, my = (w * xx).sum() / w.sum(), (w * yy).sum() / w.sum()
    s2 = (w * ((xx - mx) ** 2 + (yy - my) ** 2)).sum() / (2 * w.sum())
    return 2.3548 * math.sqrt(s2)


def softmin(v, L, s):
    """Écrêtage à bord doux : v loin sous L -> v ; loin au-dessus -> L ; transition d'échelle s."""
    return -s * np.logaddexp(-v / s, -L / s)


# ------------------------------------------------------------------------------------------------ paramètres d'image
def tirer_image(rng, n, imp: dict | None = None, etapes=ETAPES) -> dict:
    """Les paramètres de l'image (instrument, aigrettes, auréole, saturation, couleurs...). ``imp`` impose des valeurs
    mesurées (fwhm_moment, angles, sat, mode, couleur_fn, plateau, a50, gamma, s30, pente, L, ...)."""
    imp = imp or {}
    P = {"etapes": list(etapes)}
    from instruments import CATALOGUE, draw
    for _ in range(200):
        inst = draw(rng, n=n)[0] if CATALOGUE.exists() else G1.draw_instrument(rng)
        if S1.star_width(inst) > S1.MAX_WIDTH:
            continue
        # largeur imposée : un instrument dont la diffraction tient dans l'étoile mesurée
        if "fwhm_image" in imp and 1.03 * inst.airy_px > 0.6 * imp["fwhm_image"]:
            continue
        break
    kind = inst.kind
    inst.vanes, inst.blades, inst.vane_twist = 0, 0, 0.0              # aigrettes analytiques (étape 2)
    if "fwhm_moment" in imp:                                          # largeur mesurée imposée : la turbulence ajustée
        inst.defocus_ring_px, inst.track_px = 0.0, min(inst.track_px, 1.0)
        rmax = 2 * imp.get("fwhm_image", imp["fwhm_moment"])
        for _ in range(5):
            fw = moment_fwhm(G1.psf_at(inst, 0.0, 0.0, PSF_SIZE)[1], rmax)
            inst.seeing_fwhm = max(0.6, inst.seeing_fwhm * imp["fwhm_moment"] / fw)
    P["inst"] = inst
    fw = moment_fwhm(G1.psf_at(inst, -inst.centre[0], -inst.centre[1], PSF_SIZE)[1])
    P["fwhm"] = fw
    P["mode"] = imp.get("mode", "nb" if rng.random() < 0.2 else "rgb")
    # saturation de l'image (max, par couleur) : premier générateur (100 - 15 000 bruits, balance des blancs)
    wb = np.array([rng.uniform(1.05, 1.6), 1.0, rng.uniform(1.15, 1.75)])
    P["sat"] = np.asarray(imp.get("sat", lu(rng, 100, 15000) * wb / wb.mean()), float)
    # 1 auréole
    P["a50"] = imp.get("a50", lu(rng, 3e-7, 4e-6))
    P["gamma"] = imp.get("gamma", float(rng.uniform(1.6, 2.6)))
    P["rc"] = 3.0 * fw
    # 2 aigrettes
    r = rng.random()
    if "angles" in imp:
        P["aigrettes"] = dict(angles=list(imp["angles"]), dedoublement=0.0)
    else:
        # combinatoire des branches, indépendante du type d'instrument (galerie du 22 nov : liée au type, presque tous
        # les tirages sortaient sans aigrette) ; un réfracteur garde une chance de plus de n'en avoir aucune
        a0 = float(rng.uniform(0, 180))
        if r < (0.45 if kind in ("refractor", "smart") else 0.2):
            angs = []                                                         # réfracteur, araignée courbe : aucune
        elif r < 0.55:
            angs = [a0 + 90 * k for k in range(4)]                            # Newton, araignée à 4 branches
        elif r < 0.70:
            angs = [a0 + 60 * k for k in range(6)]                            # 3 branches : 6 aigrettes
        elif r < 0.77:
            angs = [a0, a0 + 180]                                             # une barre (câble)
        elif r < 0.85:
            angs = [a0 + 45 * k for k in range(8)]                            # deux araignées / 4 lames
        else:
            angs = [a0 + 90 * k for k in range(4)]                            # branches tordues : aigrettes doubles
        P["aigrettes"] = dict(angles=angs, dedoublement=float(rng.uniform(0.6, 3.0)) if r >= 0.85 else 0.0)
    P["s30"] = imp.get("s30", lu(rng, 4e-7, 6e-6))
    P["pente"] = imp.get("pente", float(rng.uniform(1.5, 3.2)))
    P["L"] = imp.get("L", lu(rng, 40, 300) if rng.random() < 0.4 else None)
    # partie proche, raide (r^-4 jusqu'à ~20 px) : 1,6e-5 à 1,2e-4 du flux à 11 px (mesuré à mi-hauteur en travers)
    P["s11"] = imp.get("s11", lu(rng, 1.6e-5, 1.2e-4))
    # largeur en travers : FWHM 1,1 à 2,1 FWHM de l'étoile (mesurée à mi-hauteur, r 8-40 px, sur nos 4 images)
    P["larg_sigma"] = imp.get("larg_sigma", fw * float(rng.uniform(0.45, 0.9)))
    P["larg_croissance"] = float(rng.uniform(0.0, 0.15))                       # par décade de flux
    dotted = rng.random() < (0.6 if P["mode"] == "nb" else 0.35)
    P["pointilles"] = imp.get("pointilles", float(rng.uniform(0.1, 0.5)) if dotted else 0.0)
    P["pas"] = imp.get("pas", lu(rng, 8, 20))
    # 3 plateau par étoile
    lo, hi = imp.get("plateau", (None, None))
    if lo is None:
        c = float(rng.uniform(0.4, 1.0)); sp = float(rng.uniform(0.0, 0.25))
        lo, hi = max(0.35, c - sp), min(1.0, c + sp)
    P["plateau"] = (lo, hi)
    P["plateau_couleur_p"] = imp.get("plateau_couleur_p", float(rng.uniform(0.0, 0.35)))
    P["bord"] = float(rng.uniform(0.01, 0.05))
    # 4 décalage et largeur par couleur
    com = rng.uniform(-0.25, 0.25, 4) if rng.random() < 0.3 else rng.normal(0, 0.01, 4)
    P["decal_commun"] = imp.get("decal_commun", com.tolist())                  # dxR, dyR, dxB, dyB
    P["decal_sigma"] = imp.get("decal_sigma", float(rng.uniform(0.01, 0.06)))
    P["larg_couleur"] = imp.get("larg_couleur", [float(rng.uniform(0.97, 1.03)), 1.0, float(rng.uniform(0.98, 1.05))])
    # 5 halo
    P["allongement"] = imp.get("allongement", float(rng.uniform(0, 0.25)) if rng.random() < 0.7 else 0.0)
    P["axe"] = imp.get("axe", float(rng.uniform(0, 180)))
    P["decentrement"] = imp.get("decentrement", float(rng.uniform(0, 0.3)))
    # anneaux de reflet : des marches douces (Alnitak : r ~15 et 27 px), pas des cercles nets (planche Synth_5, 1er essai :
    # amplitude 0,3-1 x l'auréole et 1,2 px de large donnaient des cercles fins qu'aucune vraie étoile ne montre)
    P["anneaux"] = [(float(rng.uniform(2, 8)), float(rng.uniform(0.05, 0.3)), float(rng.uniform(2.0, 5.0)))
                    for _ in range(int(rng.integers(1, 3)) if rng.random() < 0.3 else 0)]
    P["fantomes"] = dict(p=0.4, k=float(rng.uniform(-0.15, 0.15)))
    P["grain_halo"] = imp.get("grain_halo", float(rng.uniform(0.0, 0.2)))
    # rayons fins : autour des géantes toujours ; autour de toutes les étoiles écrêtées pour un instrument qui en fait
    # (aveugle du 22 nov : les saturées G 6-8 de la Tête de Cheval en portent des dizaines, celles de M8 aucun)
    P["rayons"] = dict(n=int(rng.integers(12, 101)),
                       contraste=imp.get("rayons_contraste", float(rng.uniform(0.03, 0.15))),
                       actif=imp.get("rayons_actif", bool(rng.random() < 0.5)), seuil=0.5)
    # 6 formes
    P["ellip_sigma"] = imp.get("ellip_sigma", float(rng.uniform(0.03, 0.18)))
    P["irreg_p"] = imp.get("irreg_p", float(rng.uniform(0.2, 1.0)))
    P["irreg_q"] = imp.get("irreg_q", float(rng.uniform(0.2, 0.6)))                                      # poids max de la 2e composante
    P["irreg_d"] = imp.get("irreg_d", float(rng.uniform(0.4, 1.2)))                                      # écart max (FWHM)
    # grumeaux : texture multiplicative lisse du cœur (échelle 0,35 FWHM), pour l'asymétrie des vraies étoiles
    # débruitées (mesure_formes.py : asymétrie médiane 0,08 à 0,45 sur nos 4 images)
    P["grumeaux"] = imp.get("grumeaux", float(rng.uniform(0.0, 0.9)))
    P["doubles_p"] = float(rng.uniform(0.02, 0.08))
    # 7 couleur
    if P["mode"] == "rgb":
        P["gains"] = imp.get("gains", (1 / wb).tolist())
    else:
        P["gains"] = imp.get("gains", [float(rng.uniform(0.5, 1.5)), 1.0, float(rng.uniform(0.5, 1.5))])
    P["champ_bprp"] = imp.get("champ_bprp", list(BPRP)[int(rng.integers(0, 4))])
    P["couleur_fn"] = imp.get("couleur_fn")                                     # fonction BP-RP -> parts (mesurée)
    P["photon_k"] = imp.get("photon_k", 0.0 if rng.random() < 0.5 else lu(rng, 1e-3, 3e-2))
    P["densite"] = imp.get("densite", inst.extra.get("density"))
    # 8 banque de formes réelles (tour 3) : une banque d'image tirée pour toute l'image (60 % des tirages libres)
    if "banque" in imp:
        P["banque"] = imp["banque"]
    else:
        bs = banques_dispo()
        P["banque"] = bs[int(rng.integers(len(bs)))] if bs and rng.random() < 0.6 else None
    P["banque_rotation"] = imp.get("banque_rotation", float(rng.uniform(0, 360)))
    P["banque_echelle"] = imp.get("banque_echelle", None)            # None : FWHM de l'image / FWHM de la banque
    P["banque_aureole"] = imp.get("banque_aureole", (0.5, 2.5))     # facteur de l'auréole (transparence)
    # les PSF du cœur : 3 x 3 places du champ (optique de Fourier, sans branches)
    grid = (-1.0, 0.0, 1.0)
    P["grille"] = {(i, j): G1.psf_at(inst, u - inst.centre[0], v - inst.centre[1], PSF_SIZE)
                   for j, v in enumerate(grid) for i, u in enumerate(grid)}
    return P


def fiche_image(P):
    keep = {k: v for k, v in P.items() if k not in ("grille", "inst", "couleur_fn", "banque")}
    bq = P.get("banque")
    keep["banque"] = None if bq is None else ([e["b"]["image"] for e in bq] if isinstance(bq, list) else bq["image"])
    keep["instrument"] = P["inst"].as_dict()
    keep["sat"] = [round(float(v), 1) for v in P["sat"]]
    return keep


# ------------------------------------------------------------------------------------------------ une étoile
def psf_place(P, u, v):
    """PSF (3, S, S) à la place (u, v) du champ : mélange bilinéaire des 4 PSF de la grille les plus proches."""
    out = 0
    for (i, j), p in P["grille"].items():
        w = max(0.0, 1 - abs(u - (i - 1))) * max(0.0, 1 - abs(v - (j - 1)))
        if w > 0:
            out = out + w * p
    return out


def vignette(P, rng, st, on):
    """La lumière d'une étoile (C, 2R+1, 2R+1) centrée sur le pixel entier le plus proche, et ses données propres."""
    C = 3
    F = st["flux"]                                          # flux total (somme des couleurs)
    parts = st["parts"]
    fw = P["fwhm"]
    fx, fy = st["x"] - round(st["x"]), st["y"] - round(st["y"])
    a50 = P["a50"] * (10 ** (0.15 * st["z_aur"]) if on("aureole") else 1.0)
    gamma, rc = P["gamma"], P["rc"]
    if not on("aureole"):                                   # premier générateur : auréole partagée faible
        a50, gamma = 3e-7, 2.05
    s30 = P["s30"] * 10 ** (0.15 * st["z_sp"])
    s11 = P["s11"] * 10 ** (0.15 * st["z_sp"])
    Fmax = F * float(parts.max())
    # fenêtre : jusqu'où la lumière dépasse 0,01 (unités de la tuile = bruits du ciel ; aveugle du 22 nov : à 0,03,
    # le bord de la fenêtre se voyait en ovale sur une image peu bruitée, Tête de Cheval)
    R_aur = 50 * (Fmax * a50 / 0.01) ** (1 / gamma) if a50 > 0 else 0
    R_sp = 30 * (Fmax * s30 / 0.01) ** (1 / P["pente"]) if P["aigrettes"]["angles"] else 0
    if P["L"]:
        R_sp = min(R_sp, 2.5 * P["L"])
    R = int(np.clip(max(4 * fw + 4, R_aur, R_sp), 8, MARGE))
    S = 2 * R + 1
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float32)
    X, Y = xx - fx, yy - fy
    rr = np.hypot(X, Y)
    th = np.arctan2(Y, X)
    # le cœur : PSF de la place, déplacée (sous-pixel + décalage de couleur), élargie par couleur, allongée
    if on("formes") and P.get("grumeaux", 0) > 0:
        nz = cv2.GaussianBlur(rng.standard_normal((S, S)).astype(np.float32), (0, 0), max(0.35 * fw, 0.6))
        nz /= max(float(nz.std()), 1e-9)
        # amorti au-dessus de la saturation (planche Synth_9 : sur une saturée, les grumeaux du cœur devenaient de
        # grosses taches autour du plateau)
        t_eff = P["grumeaux"] * min(1.0, math.sqrt(0.2 * float(np.min(P["sat"])) / max(st["peak"], 1e-9)))
        st["texture"] = np.exp(t_eff * nz).astype(np.float32)
    base = psf_place(P, st["u"], st["v"])
    h = base.shape[-1] // 2
    Rc = min(R, h)
    core = np.zeros((C, S, S), np.float32)
    e, ea = (st["ellip"], st["ellip_angle"]) if on("formes") else (0.0, 0.0)
    ca, sa = math.cos(ea), math.sin(ea)
    Rot = np.array([[ca, -sa], [sa, ca]])
    for c in range(C):
        wc = st["larg"][c] if on("couleurs") else 1.0
        A2 = __import__('operator').matmul(__import__('operator').matmul(Rot, np.diag([wc * (1 + e), wc / (1 + e)])), Rot.T)
        dx = fx + (st["dx"][c] if on("couleurs") else 0.0)
        dy = fy + (st["dy"][c] if on("couleurs") else 0.0)
        comps = [(1.0, 0.0, 0.0)]
        if on("formes") and st["irreg"]:
            q, ox, oy = st["irreg"]
            comps = [(1 - q, 0.0, 0.0), (q, ox, oy)]
        for wgt, ox, oy in comps:
            M = np.zeros((2, 3), np.float32)
            M[:, :2] = A2
            M[:, 2] = np.array([R + dx + ox, R + dy + oy]) - __import__('operator').matmul(A2, np.array([h, h]))
            core[c] += wgt * cv2.warpAffine(base[c], M, (S, S), flags=cv2.INTER_CUBIC, borderValue=0)
        core[c] = np.clip(core[c], 0, None)                 # l'interpolation cubique laisse de petits négatifs
        if on("formes") and P.get("grumeaux", 0) > 0:
            core[c] *= st["texture"][:S, :S]
        core[c] /= max(float(core[c].sum()), 1e-12)
    # l'auréole
    aur = a50 * ((rr ** 2 + rc ** 2) / (50 ** 2 + rc ** 2)) ** (-gamma / 2)
    if on("halo"):
        er = P["allongement"] * np.clip(rr / 100.0, 0, 2.5) ** 0.8
        aur = aur * (1 + np.clip(er, 0, 0.6) * np.cos(2 * (th - math.radians(P["axe"]))))
        dr = P["decentrement"] * np.clip(rr / 150.0, 0, 1)
        aur = aur * (1 + dr * np.cos(th - st["dir_centre"]))
        for (rk, amp, wd) in P["anneaux"]:
            r_k = rk * fw
            aur = aur * (1 + amp * np.exp(-((rr - r_k) / wd) ** 2))
        if st.get("avec_rayons"):
            rays = st["rayons"]                              # (angles, largeurs, amplitudes)
            tg = np.linspace(-np.pi, np.pi, 7201)
            d = (tg[:, None] - rays[0]) % (2 * np.pi)
            d = np.minimum(d, 2 * np.pi - d)
            m1 = (rays[2] * np.exp(-0.5 * (d / rays[1]) ** 2)).sum(-1)
            m1 = (m1 - m1.mean()) / max(float(m1.std()), 1e-9) * P["rayons"]["contraste"]
            m = m1[np.clip(np.rint((th + np.pi) / (2 * np.pi) * 7200).astype(int), 0, 7200)]
            r1 = 3 * math.sqrt(max(st["plateau_px"], 1))
            aur = aur * (1 + m * np.clip((rr - r1) / (2 * r1 + 1), 0, 1))
    if on("halo") and P.get("grain_halo", 0) > 0 and R >= 16:
        # grain du halo (tour 2 : halos synthétiques jugés « lisses, parfaitement radiaux ») : texture multiplicative
        # lisse à deux échelles (3 et 8 px), faible (0-0,2) : à 0,5 les halos devenaient cotonneux (planche Synth_10)
        z = cv2.GaussianBlur(rng.standard_normal((S, S)).astype(np.float32), (0, 0), 3.0)
        z = z / max(float(z.std()), 1e-9)
        z2 = cv2.GaussianBlur(rng.standard_normal((S, S)).astype(np.float32), (0, 0), 8.0)
        z = 0.6 * z + 0.8 * z2 / max(float(z2.std()), 1e-9)
        aur = aur * np.exp(P["grain_halo"] * z - 0.5 * (P["grain_halo"] * float(z.std())) ** 2)   # moyenne 1
    aur = np.clip(aur, 0, None).astype(np.float32)
    share = float(min(aur.sum(), 0.4))
    aur *= share / max(float(aur.sum()), 1e-12)
    out = np.zeros((C, S, S), np.float32)
    for c in range(C):
        out[c] = F * parts[c] * ((1 - share) * core[c] + aur)
    # les aigrettes
    ag = P["aigrettes"]
    if ag["angles"] and on("aigrettes"):
        waves = WAVE_NB if P["mode"] == "nb" else WAVE_RGB
        w = P["larg_sigma"] * (1 + P["larg_croissance"] * max(0.0, math.log10(max(F / 500.0, 1.0))))
        rs = 1.5 * fw
        for k, a in enumerate(ag["angles"]):
            # tour 7 : chaque branche son intensité (x e^N(0; 0,25)) ; largeur qui diminue vers le bout (moitié à
            # ~20 FWHM) : le juge les voyait « épaisses et trop symétriques »
            fbr = float(np.exp(rng.normal(0, 0.25))) if P.get("t7_aigrettes", False) else 1.0
            for split in ((0.0,) if not ag["dedoublement"] else (-ag["dedoublement"] / 2, ag["dedoublement"] / 2)):
                t = math.radians(a + split)
                along = X * math.cos(t) + Y * math.sin(t)
                across = -X * math.sin(t) + Y * math.cos(t)
                wr = w * (0.5 + 0.5 * np.exp(-np.maximum(along, 0) / (10 * fw))) if P.get("t7_aigrettes", False) else w
                band = np.exp(-0.5 * (across / wr) ** 2) * (along > 0)
                fac = (0.5 if ag["dedoublement"] else 1.0) * fbr
                for c in range(C):
                    # enveloppe neutre en couleur (tour 2 : l'étirer avec lambda rendait le rouge 2,5 fois plus fort que le
                    # bleu, d'où des aigrettes brunes ; mesuré : les vraies ont à peu près la couleur de l'étoile) ;
                    # l'irisation vient des perles, de pas proportionnel à lambda
                    lc = along
                    prof = s30 * ((1 + (lc / fw) ** 2) / (1 + (30 / fw) ** 2)) ** (-P["pente"] / 2)
                    if P["L"]:
                        prof = prof * np.exp(-(lc / P["L"]) ** 2)
                    prof = prof + s11 * ((lc ** 2 + fw ** 2) / (121.0 + fw ** 2)) ** -2.0
                    if P["pointilles"]:
                        prof = prof * (1 + P["pointilles"] * np.cos(2 * np.pi * along / (P["pas"] * waves[c] / LAM0)
                                                                    + st["phase"]))
                    out[c] += (fac * F * parts[c] * prof * band).astype(np.float32)
    # bord doux : la lumière hors du cœur s'éteint sur le dernier quart de la fenêtre (galerie du 22 nov : une géante
    # hors de la tuile laissait un bord carré là où sa fenêtre de 600 px coupait un halo encore brillant)
    if R >= 32:
        t = np.clip((R - rr) / (0.25 * R), 0, 1)
        taper = (t * t * (3 - 2 * t)).astype(np.float32)
        for c in range(C):
            halo = out[c] - F * parts[c] * (1 - share) * core[c]
            out[c] = F * parts[c] * (1 - share) * core[c] + halo * taper
    return out, R


_BANQUES = None


def banques_dispo():
    """Les banques construites par banque_formes.py (sous-dossier « banque »), chargées une fois."""
    global _BANQUES
    if _BANQUES is None:
        import json as _json
        _BANQUES = []
        d = Path(__file__).resolve().parent / "banque"
        fl = sorted(d.glob("banque2_*.npz")) if d.exists() else []
        for f in fl or (sorted(d.glob("banque_*.npz")) if d.exists() else []):
            z = np.load(f, allow_pickle=True)
            m = _json.loads(str(z["meta"]))
            m["formes"] = z["formes"]                                 # float16, converti à l'usage
            _BANQUES.append(m)
    return _BANQUES


def profil_rond(img, R):
    """Profil radial robuste (moyenne par anneau sans les pixels > 2 x la moyenne : aigrettes, rayons)."""
    h = img.shape[-1] // 2
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1]
    ri = np.rint(np.hypot(xx, yy)).astype(int).ravel()
    v = img.ravel()
    m1 = np.bincount(ri, v, minlength=h * 2) / np.maximum(np.bincount(ri, minlength=h * 2), 1)
    ok = v <= 2 * np.maximum(m1[ri], 0) + 1e-12
    m2 = np.bincount(ri[ok], v[ok], minlength=h * 2) / np.maximum(np.bincount(ri[ok], minlength=h * 2), 1)
    rr = np.hypot(xx, yy)
    return np.interp(rr, np.arange(len(m2)), m2).astype(np.float32)        # au rayon vrai (pas d'escalier)


def rond_et_reste(fb, Rb, r0, r_garde=None, fondu=None):
    """Tour 7 : profil rond LISSÉ (moyenne par anneau sans les pixels > 2 x la moyenne, puis moyenne glissante en
    log-log sur +-0,08 décade au-delà de r0, redécroissant) et reste NON ROND (le reste moins sa moyenne par anneau).
    Le juge du tour 6 voyait des anneaux : les marches du profil des formes restaient dans le rond et dans le reste.
    Rend (rond (3,S,S), reste (3,S,S), profil de luminance 1D)."""
    h = fb.shape[-1] // 2
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1]
    ri = np.rint(np.hypot(xx, yy)).astype(int)
    rf = ri.ravel()
    rrf = np.hypot(xx, yy)
    nb = int(rf.max()) + 1                                   # jusqu'aux coins (h x racine de 2)
    rf2 = np.rint(rrf * 2).astype(int).ravel()               # anneaux d'un demi-pixel
    nb2 = int(rf2.max()) + 1
    cnt2 = np.bincount(rf2, minlength=nb2)
    r2 = np.arange(nb2) / 2.0
    cnt = np.maximum(np.bincount(rf, minlength=nb), 1)
    ronds, restes = [], []
    r = np.arange(nb)
    i0 = int(math.ceil(r0))
    lr = np.log(np.maximum(r, 1.0))
    raw = []
    for c in range(3):
        v = fb[c].ravel()
        m1 = np.bincount(rf, v, minlength=nb) / cnt
        ok = v <= 2 * np.maximum(m1[rf], 0) + 1e-30
        raw.append(np.bincount(rf[ok], v[ok], minlength=nb) / np.maximum(np.bincount(rf[ok], minlength=nb), 1))
    raw = np.stack(raw)
    lum = raw.sum(0)
    # luminance lissée UNE fois (log-log, +-0,08 décade), redécroissante ; part de chaque couleur lissée sur +-0,25
    # décade (tour 7, 1er essai : lissée couleur par couleur, les ondulations tombaient à des rayons différents en R,
    # V, B : anneaux violets sur M57)
    def lisse(prof, dex):
        pos = prof[i0:] > 0
        out = prof.copy()
        if pos.sum() > 5:
            lp = np.log(np.where(pos, prof[i0:], np.nan))
            ll = lr[i0:]
            sm = np.array([np.nanmean(lp[np.abs(ll - ll[k]) <= dex * math.log(10)]) for k in range(len(lp))])
            out[i0:] = np.where(pos & np.isfinite(sm), np.exp(sm), 0.0)
        return out
    L = lisse(lum, 0.08)
    L[i0:] = np.minimum.accumulate(np.minimum(L[i0:], L[i0]))
    fr = np.clip(raw / np.maximum(lum, 1e-30), 0, 1)
    frs = np.stack([fr[c].copy() for c in range(3)])
    for c in range(3):
        q = frs[c, i0:]
        frs[c, i0:] = np.array([q[np.abs(lr[i0:] - lr[i0 + k]) <= 0.25 * math.log(10)].mean() for k in range(len(q))])
    frs[:, i0:] /= np.maximum(frs[:, i0:].sum(0, keepdims=True), 1e-30)
    # forme brute gardée seulement sur le plateau + 1 FWHM (r_garde), fondu sur « fondu » px (tour 7 : gardée jusqu'à
    # 2 r0, les marches couleur par couleur des formes de la banque y faisaient des anneaux irisés sur M57)
    rg = r0 if r_garde is None else r_garde
    fd = max(rg, 1.0) if fondu is None else fondu
    wb = np.clip((np.hypot(xx, yy) - rg) / max(fd, 1.0), 0, 1).astype(np.float32)
    for c in range(3):
        p = raw[c].copy()
        p[i0:] = L[i0:] * frs[c, i0:]
        rond = np.interp(rrf, r, p).astype(np.float32)                    # au rayon vrai (pas d'escalier)
        # reste = la forme moins SON profil radial fin (anneaux d'un demi-pixel, interpolé) : la partie non ronde
        # seule (aigrettes, rayons). Tour 7 : retirer la moyenne par anneau entier puis l'interpoler laissait, près
        # d'un cœur raide, une dent de scie radiale vue en anneaux irisés (M57)
        # profil fin ROBUSTE (sans les pixels > 2 x la moyenne de l'anneau : aigrettes) : une moyenne qui garde les
        # aigrettes perlées hérite de leur période le long du rayon et la soustraction dessine des anneaux irisés
        v = fb[c].ravel()
        m_1 = np.bincount(rf2, v, minlength=nb2) / np.maximum(cnt2, 1)
        okp = v <= 2 * np.maximum(m_1[rf2], 0) + 1e-30
        c_2 = np.bincount(rf2[okp], minlength=nb2)
        m_f = np.bincount(rf2[okp], v[okp], minlength=nb2) / np.maximum(c_2, 1)
        okf = c_2 > 0
        radial_f = np.interp(rrf, r2[okf], m_f[okf]).astype(np.float32)
        res = fb[c] - radial_f
        # le cœur gardé tel quel (planche Synth_22, 1er essai : la moyenne robuste écartait le plateau et creusait un
        # anneau sombre autour du cœur) ; le nettoyage entre en fondu de r0 à 2 r0
        rond = (1 - wb) * fb[c] + wb * rond
        res = wb * res
        ronds.append(rond); restes.append(res.astype(np.float32))
    lum = L
    return np.stack(ronds), np.stack(restes), lum


def vignette_banque2(P, rng, st, L1):
    """Tour 4 : brillante tirée d'une banque d'étoiles INDIVIDUELLES nettoyées : 1 à 3 étoiles de la banque mêlées
    (poids tirés), tournées, légèrement allongées, mises à la largeur voulue ; halo lointain selon la loi mesurée de
    l'image (niveau à 50 px, pente), un peu décentré ; aigrettes prolongées loin aux angles de l'image, seulement là où
    l'aigrette de la forme est plus faible que la loi. None si aucune forme ne garde son cœur écrêté."""
    b = P["banque"]
    rot_b, ech_b = 0.0, None
    if isinstance(b, list):
        # tour 6 : plusieurs banques (l'image et d'autres images de même configuration) : une par étoile, au poids ;
        # chaque entrée porte sa rotation vers les angles de l'image cible et son échelle de largeur
        w = np.array([e_["poids"] for e_ in b], float)
        e_ = b[int(rng.choice(len(b), p=w / w.sum()))]
        b, rot_b, ech_b = e_["b"], e_["rot"], e_["echelle"]
        st["banque_image"] = b["image"]
    F, parts = st["flux"], st["parts"]
    niv = np.array(b["niveau"])
    q = F * float(parts[1]) * niv[:, 1] / max(L1, 1e-9)
    # forme gardée seulement si son cœur reste écrêté (q >= 1) et sans la grossir plus de 8 fois (Synth_16 : Alnitak
    # faite d'une G 4,9 grossie 16 fois : ses restes aussi) ; sinon le modèle paramétrique
    okg = np.flatnonzero((q >= 1.0) & (q <= 8.0))
    if not okg.size:
        return None
    # tour 5 : MÉDIANE d'un sous-ensemble tiré de 5 à 8 étoiles (tour 4 : 1 à 3 étoiles mêlées gardaient les restes
    # propres à chaque étoile, repérés par le juge) ; le sous-ensemble change à chaque tirage
    pool = okg[np.argsort(q[okg])][:12]
    m = int(min(len(pool), rng.integers(5, 9)))
    pick = rng.choice(pool, m, replace=False)
    forme = np.median(np.stack([b["formes"][int(k)].astype(np.float32) for k in pick]), axis=0)
    s = ech_b or P["banque_echelle"] or float(np.clip(P["fwhm"] / b["fwhm"], 0.6, 1.8))
    s *= float(rng.uniform(0.95, 1.05))
    e = abs(float(rng.normal(0, 0.04)))
    ph = float(rng.uniform(0, np.pi))
    rot = P["banque_rotation"] + rot_b + float(rng.uniform(-2, 2))
    th = math.radians(rot)
    Rt = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    Rp = np.array([[math.cos(ph), -math.sin(ph)], [math.sin(ph), math.cos(ph)]])
    A2 = __import__('operator').matmul(__import__('operator').matmul(__import__('operator').matmul(s * Rt, Rp), np.diag([1 + e, 1 / (1 + e)])), Rp.T)
    h0 = forme.shape[-1] // 2
    Rb = int(min(MARGE, round(h0 * s)))
    Sb = 2 * Rb + 1
    fx, fy = st["x"] - round(st["x"]), st["y"] - round(st["y"])
    M = np.zeros((2, 3), np.float32)
    M[:, :2] = A2
    M[:, 2] = np.array([Rb + fx, Rb + fy]) - __import__('operator').matmul(A2, np.array([h0, h0]))
    fb = np.stack([cv2.warpAffine(forme[c], M, (Sb, Sb), flags=cv2.INTER_LINEAR, borderValue=0) / (s * s)
                   for c in range(3)])
    fwb = P["fwhm"]
    npl = int(np.sum(fb[1] >= 0.98 * fb[1].max()))
    r0 = max(2 * fwb, 1.5 * math.sqrt(npl / math.pi))
    rg = math.sqrt(npl / math.pi) + fwb
    # interrupteurs du tour 8 : les trois changements du tour 7 (rond lissé, raccord adapté, aigrettes amincies) font
    # chacun RECULER les très brillantes au juge (tous : 5,2 % ; sans aigrettes 19 ; sans raccord 31 ; sans rond 33 ;
    # sans les trois 62 %) : éteints par défaut, gardés pour mémoire
    t7r, t7j, t7a = P.get("t7_rond", False), P.get("t7_raccord", False), P.get("t7_aigrettes", False)
    if t7r:
        rond, res, plum = rond_et_reste(fb, Rb, min(r0, rg), r_garde=rg, fondu=2 * fwb)
    else:                                                     # tour 6 : profil rond robuste, reste = forme - rond
        rond = np.stack([profil_rond(fb[c], Rb) for c in range(3)])
        res = fb - rond
        plum = rond.sum(0)[Rb, Rb:].astype(np.float64)
    lo, hi = P["banque_aureole"]
    t = lu(rng, lo, hi)
    gam = float(b.get("gamma", 2.2))
    # raccord adapté à la forme (tour 7 : à 60 px fixes, plusieurs formes étaient déjà retombées à zéro, le halo
    # prolongé partait de zéro : « bord net » du juge) : 0,6 x le rayon où la forme tombe à 1e-3 de son niveau à r0
    i0 = int(math.ceil(r0))
    tail_ = plum[i0:]
    zr = np.flatnonzero(tail_ <= 1e-3 * max(tail_[0], 1e-30))
    r_zero = float(i0 + zr[0]) if zr.size else float(Rb)
    rj = float(np.clip(0.6 * r_zero, r0 + 5, min(60.0, 0.6 * Rb))) if t7j else float(min(0.3 * Rb, 60.0))
    Fm = F * float(np.max(parts))
    rbb = np.hypot(*np.mgrid[-Rb:Rb + 1, -Rb:Rb + 1])
    sel_j = (rbb >= 0.9 * rj) & (rbb <= rj)
    lev = np.array([max(float(np.mean(rond[c][sel_j])), 0.0) for c in range(3)]) * t
    a50 = float(b.get("a50", 0.0)) * t
    if t7j:
        # fenêtre fixée par la queue RÉELLE (tour 7 : par la loi de l'image, la queue était coupée encore visible)
        R = int(np.clip(rj * (max(Fm * float(lev.max()) / 0.01, 1.0)) ** (1 / gam), Rb, MARGE))
    else:                                                     # tour 6 : fenêtre par la loi de l'image
        R = int(np.clip(50 * (max(Fm * max(a50, 1e-30) / 0.01, 1e-9)) ** (1 / gam), Rb, MARGE))
    S = 2 * R + 1
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float32)
    d0 = float(rng.uniform(0, 0.05 * rj)); a0 = float(rng.uniform(0, 2 * np.pi))
    rr = np.hypot(xx - fx, yy - fy)
    rh = np.hypot(xx - fx - d0 * math.cos(a0), yy - fy - d0 * math.sin(a0))     # halo un peu décentré
    wj = np.clip((rr - 0.6 * rj) / (0.4 * rj), 0, 1) if t7j else np.clip((rr - 0.5 * rj) / (0.5 * rj), 0, 1)
    tw = np.clip((R - rr) / (0.15 * R), 0, 1)
    o = R - Rb
    yb, xb = np.mgrid[-Rb:Rb + 1, -Rb:Rb + 1].astype(np.float32)
    fadeb = np.clip((0.95 * Rb - np.hypot(xb, yb)) / (0.15 * Rb), 0, 1)
    # aigrettes : angles de l'image tournés comme la forme ; loi en r^-p depuis r_j, niveau pris sur la forme ;
    # amincies vers le bout, inégales d'une branche à l'autre (tour 7 : « épaisses et symétriques » pour le juge)
    angs = [a + rot for a in b.get("angles", [])]
    wsp0 = 0.6 * fwb
    out = np.zeros((3, S, S), np.float32)
    if t7a:
        branches = [(math.radians(ag), float(rng.uniform(1.8, 2.8)), float(np.exp(rng.normal(0, 0.25)))) for ag in angs]
    else:                                                     # tour 6 : même pente, mêmes branches
        p1 = float(rng.uniform(1.8, 2.8))
        branches = [(math.radians(ag), p1, 1.0) for ag in angs]
    for c in range(3):
        near = np.zeros((S, S), np.float32); near[o:o + Sb, o:o + Sb] = rond[c]
        rs = np.zeros((S, S), np.float32); rs[o:o + Sb, o:o + Sb] = res[c] * fadeb
        if t7j:
            tail_c = lev[c] * (np.maximum(rh, 1.0) / rj) ** (-gam)      # continue au raccord
        else:                                                 # tour 6 : niveau plafonné par la loi de l'image
            law = a50 * (rj / 50.0) ** (-gam)
            tail_c = min(lev[c] / max(t, 1e-9), law) * (np.maximum(rh, 1.0) / rj) ** (-gam)
        halo = (1 - wj) * near + wj * tail_c
        ext = np.zeros((S, S), np.float32)
        for tt, p_sp, fbr in branches:
            along = (xx - fx) * math.cos(tt) + (yy - fy) * math.sin(tt)
            across = -(xx - fx) * math.sin(tt) + (yy - fy) * math.cos(tt)
            wsp = wsp0 * (0.5 + 0.5 * np.exp(-np.maximum(along - rj, 0) / (3 * rj))) if t7a else wsp0
            band = np.exp(-0.5 * (across / wsp) ** 2) * (along > rj)
            sel = (np.abs(across) <= 1.0) & (along > 0.7 * rj) & (along < rj)
            lvl = float(np.mean(rs[sel])) if sel.any() else 0.0
            if lvl <= 0:
                continue
            ext += fbr * lvl * (np.maximum(along, rj) / (0.85 * rj)) ** (-p_sp) * band
        ext = np.clip(ext - np.clip(rs, 0, None), 0, None)
        out[c] = F * float(parts[c]) * (halo + rs + ext) * tw
    st["banque_groupe"] = [int(k) for k in pick]
    return out, R


def vignette_banque(P, rng, st, L1):
    if isinstance(P["banque"], list) or "a50" in P["banque"]:
        return vignette_banque2(P, rng, st, L1)
    """Une brillante tirée de la banque de formes réelles : forme du groupe dont le cœur reste écrêté au nouveau
    niveau, tournée, mise à la largeur voulue, auréole changée (transparence), colorée. None si aucun groupe ne va."""
    b = P["banque"]
    F, parts = st["flux"], st["parts"]
    niv = np.array(b["niveau"])                                   # (groupes, 3) plateau / flux de la couleur
    q = F * float(parts[1]) * niv[:, 1] / max(L1, 1e-9)
    okg = np.flatnonzero(q >= 1.0)
    if not okg.size:
        return None
    k = int(okg[np.argmin(q[okg])])
    forme = b["formes"][k]
    s = P["banque_echelle"] or float(np.clip(P["fwhm"] / b["fwhm"], 0.6, 1.8))
    s *= float(rng.uniform(0.95, 1.05))
    h0 = forme.shape[-1] // 2
    R = int(min(MARGE, round(h0 * s)))
    S = 2 * R + 1
    fx, fy = st["x"] - round(st["x"]), st["y"] - round(st["y"])
    th = math.radians(P["banque_rotation"] + float(rng.uniform(-2, 2)))
    A2 = s * np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    M = np.zeros((2, 3), np.float32)
    M[:, :2] = A2
    M[:, 2] = np.array([R + fx, R + fy]) - __import__('operator').matmul(A2, np.array([h0, h0]))
    lo, hi = P["banque_aureole"]
    t = lu(rng, lo, hi)
    Rb = R
    fb = np.stack([cv2.warpAffine(forme[c], M, (S, S), flags=cv2.INTER_LINEAR, borderValue=0) / (s * s)
                   for c in range(3)])
    # halo lointain (planche Synth_14 : au fort étirement, anneaux et bord de disque : le fond de la banque était pris
    # dans le halo à 280-300 px) : au-delà de r_j, le profil rond de la forme prolongé en loi de puissance de pente
    # ajustée sur la forme elle-même (même pente pour les 3 couleurs) ; aigrettes et rayons de la forme gardés
    rj = float(min(0.3 * Rb, 60.0))        # Synth_14 2e essai : à 90 px la lueur d'IC 434 restait dans la pile
    rond = np.stack([profil_rond(fb[c], Rb) for c in range(3)])
    res = fb - rond
    yb, xb = np.mgrid[-Rb:Rb + 1, -Rb:Rb + 1].astype(np.float32)
    rb = np.hypot(xb, yb)
    lum = rond.sum(0)
    sel1 = (rb >= 0.5 * rj) & (rb < 0.6 * rj); sel2 = (rb >= 0.9 * rj) & (rb < rj)
    l1, l2 = float(lum[sel1].mean()), float(lum[sel2].mean())
    gam = float(np.clip(math.log(max(l1, 1e-30) / max(l2, 1e-30)) / math.log(0.95 / 0.55), 1.9, 3.0))         if l1 > 0 and l2 > 0 else 2.0
    amp = np.array([float(rond[c][sel2].mean()) for c in range(3)])
    Fm = F * float(np.max(parts))
    R = int(np.clip(rj * (max(Fm * max(float(amp.max()), 1e-30) / 0.01, 1.0)) ** (1 / gam), Rb, MARGE))
    S = 2 * R + 1
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float32)
    rr = np.hypot(xx - fx, yy - fy)
    rc = max(4 * P["fwhm"], 1.5 * math.sqrt(max(st.get("plateau_px", 0), 1)))
    ramp = np.clip((rr - rc) / (0.5 * rc), 0, 1)
    wj = np.clip((rr - 0.8 * rj) / (0.2 * rj), 0, 1)                    # 0 avant 0,8 r_j, 1 après r_j
    tw = np.clip((R - rr) / (0.15 * R), 0, 1)
    o = R - Rb
    out = np.zeros((3, S, S), np.float32)
    for c in range(3):
        near = np.zeros((S, S), np.float32)
        near[o:o + 2 * Rb + 1, o:o + 2 * Rb + 1] = rond[c]
        rs = np.zeros((S, S), np.float32)
        rs[o:o + 2 * Rb + 1, o:o + 2 * Rb + 1] = res[c] * np.clip((0.95 * Rb - rb) / (0.15 * Rb), 0, 1)
        tail = amp[c] * (np.maximum(rr, 1.0) / rj) ** (-gam)
        rondc = (1 - wj) * near + wj * tail
        if abs(t - 1) > 1e-3:
            rondc = rondc * (1 + (t - 1) * ramp)
        out[c] = F * float(parts[c]) * (rondc + rs) * tw
    st["banque_groupe"] = k
    return out, R


def tirer_etoiles(P, rng, n, champ=None):
    """Les étoiles : du champ tiré (stars.draw_field du premier générateur) ou d'une liste imposée ``champ``
    (x, y en px de la tuile, flux total, bp_rp)."""
    psf_peak = float(P["grille"][(1, 1)][1].max())
    info = {}
    if champ is None:
        f = S1.draw_field(rng, n, MARGE, psf_peak, info, P["densite"])
        x, y = f["x"] - MARGE, f["y"] - MARGE
        flux = f["flux"] * 3.0                                   # stars.py : flux par couleur, moyenne 1 -> total
        nst = len(x)
        q = rng.random(nst)
        bp = np.interp(q * 100, CENTILES, BPRP[P["champ_bprp"]])
    else:
        x, y, flux, bp = (np.asarray(champ[k], float) for k in ("x", "y", "flux", "bp_rp"))
        nst = len(x)
    if "couleur_gaia" in P["etapes"]:
        if P["couleur_fn"] is not None:
            parts = P["couleur_fn"](bp)
        else:
            parts = parts_planck(bp, WAVE_NB if P["mode"] == "nb" else WAVE_RGB, P["gains"])
    else:
        bv = np.clip(0.7 + 0.35 * rng.standard_normal(nst), -0.3, 2.0)       # premier générateur
        col = S1.bv_colour(bv)
        parts = col / col.sum(0, keepdims=True)
    # doubles serrées (étape 6)
    if "formes" in P["etapes"] and champ is None:
        k = rng.random(nst) < P["doubles_p"]
        m = int(k.sum())
        if m:
            sep = P["fwhm"] * rng.uniform(0.5, 3.0, m)
            ang = rng.uniform(0, 2 * np.pi, m)
            dm = rng.uniform(0, 3, m)
            x = np.r_[x, x[k] + sep * np.cos(ang)]
            y = np.r_[y, y[k] + sep * np.sin(ang)]
            flux = np.r_[flux, flux[k] * 10 ** (-0.4 * dm)]
            bp = np.r_[bp, bp[k] + rng.normal(0, 0.3, m)]
            parts = np.c_[parts, parts[:, k]]
            info["doubles"] = m
            nst = len(x)
    peak = flux / 3.0 * psf_peak
    return dict(x=x, y=y, flux=flux, bp=bp, parts=parts, peak=peak), info


def ajouter_etoiles(fond, graine, imp: dict | None = None, etapes=ETAPES, champ=None, P=None) -> dict:
    """Ajoute un champ d'étoiles tiré à une tuile de vrai fond sans étoiles (C, n, n) (bruits du ciel, fond 0).
    Rend entree (fond + étoiles, écrêté), etoiles (= entree - fond), ecretes (pixels écrêtés, bool), fiche."""
    rng = np.random.default_rng(graine)
    C, n, n2 = fond.shape
    assert n == n2 and C == 3
    if P is None:
        P = tirer_image(rng, n, imp, etapes)
    else:
        P = dict(P, etapes=list(etapes))
    on = lambda e: e in etapes
    fld, finfo = tirer_etoiles(P, rng, n, champ)
    x, y, flux, parts, peak = fld["x"], fld["y"], fld["flux"], fld["parts"], fld["peak"]
    nst = len(x)
    fw = P["fwhm"]
    inst = P["inst"]
    # par étoile : décalages et largeurs par couleur, forme, auréole, plateau
    com = np.array(P["decal_commun"])
    ds = P["decal_sigma"]
    dxR = com[0] + ds * rng.standard_normal(nst); dyR = com[1] + ds * rng.standard_normal(nst)
    dxB = com[2] + ds * rng.standard_normal(nst); dyB = com[3] + ds * rng.standard_normal(nst)
    lc = np.array(P["larg_couleur"])[:, None] * (1 + 0.015 * rng.standard_normal((3, nst)))
    lc[1] = 1.0
    ellip = np.abs(P["ellip_sigma"] * rng.standard_normal(nst))
    ellip_a = rng.uniform(0, np.pi, nst)
    z_aur, z_sp = rng.standard_normal(nst), rng.standard_normal(nst)
    lo, hi = P["plateau"]
    lvl = rng.uniform(lo, hi, nst)
    lvl_c = np.repeat(lvl[None], 3, 0)
    drop = rng.random(nst) < P["plateau_couleur_p"]
    if drop.any():
        cc = rng.integers(0, 3, nst)
        lvl_c[cc[drop], np.flatnonzero(drop)] *= rng.uniform(0.55, 1.0, int(drop.sum()))
    phase = rng.uniform(0, 2 * np.pi, nst)
    irreg = rng.random(nst) < P["irreg_p"]
    u = np.clip(x / n * 2 - 1, -1, 1)
    v = np.clip(y / n * 2 - 1, -1, 1)
    cxo, cyo = n / 2 * (1 + inst.centre[0]), n / 2 * (1 + inst.centre[1])
    # 1) les faibles : points convolués par la PSF de la grille (bilinéaire), position propre à chaque couleur
    m2 = PSF_SIZE
    C2 = n + 2 * m2
    faint = (peak <= SEUIL_VIGNETTE) & (x > -m2) & (x < n + m2) & (y > -m2) & (y < n + m2)
    stars = np.zeros((C, n, n), np.float32)
    if faint.any():
        grid = (-1.0, 0.0, 1.0)
        acc = np.zeros((C, C2, C2), np.float32)
        offs = [(dxR, dyR), (np.zeros(nst), np.zeros(nst)), (dxB, dyB)] if on("couleurs") else [(0, 0)] * 3
        for (i, j), p in P["grille"].items():
            wu = np.clip(1 - np.abs(u[faint] - grid[i]), 0, 1)
            wv = np.clip(1 - np.abs(v[faint] - grid[j]), 0, 1)
            w = wu * wv
            if not w.any():
                continue
            for c in range(C):
                ox = offs[c][0][faint] if on("couleurs") else 0.0
                oy = offs[c][1][faint] if on("couleurs") else 0.0
                pts = S1.splat(C2, x[faint] + m2 + ox, y[faint] + m2 + oy, flux[faint] * parts[c, faint] * w)
                ker = p[c]
                if on("couleurs") and c != 1:
                    s = P["larg_couleur"][c]
                    hh = ker.shape[-1] // 2
                    M = np.float32([[s, 0, hh * (1 - s)], [0, s, hh * (1 - s)]])
                    ker = cv2.warpAffine(ker, M, ker.shape[::-1], flags=cv2.INTER_CUBIC)
                    ker = ker / ker.sum()
                acc[c] += fftconvolve(pts, ker, mode="same").astype(np.float32)
        stars += acc[:, m2:m2 + n, m2:m2 + n]
    # 2) les autres : une vignette chacune (auréole, aigrettes, halo...)
    big = np.flatnonzero(peak > SEUIL_VIGNETTE)
    Lmap = np.repeat(np.asarray(P["sat"], np.float32)[:, None, None], n, 1).repeat(n, 2)   # max de l'image
    sat_px = np.zeros((n, n), bool)
    giants = []
    n_banque = 0
    for k in big[np.argsort(-peak[big])]:
        geante = peak[k] > 2e4 and on("halo")
        st = dict(x=x[k], y=y[k], flux=flux[k], peak=float(peak[k]), parts=parts[:, k], u=u[k], v=v[k], z_aur=z_aur[k], z_sp=z_sp[k],
                  dx=(dxR[k], 0.0, dxB[k]), dy=(dyR[k], 0.0, dyB[k]), larg=lc[:, k], ellip=ellip[k],
                  ellip_angle=ellip_a[k], phase=phase[k], geante=geante,
                  dir_centre=math.atan2(cyo - y[k], cxo - x[k]), plateau_px=0,
                  irreg=None)
        if irreg[k] and on("formes"):
            a = rng.uniform(0, 2 * np.pi)
            dd = fw * rng.uniform(0.3, 1.0) * P["irreg_d"]
            st["irreg"] = (float(rng.uniform(0.25, 1.0) * P["irreg_q"]), dd * math.cos(a), dd * math.sin(a))
        ray_on = on("halo") and (geante or (P["rayons"]["actif"] and peak[k] > P["rayons"]["seuil"] * float(np.min(P["sat"]))))
        if ray_on:
            nr = P["rayons"]["n"]
            st["rayons"] = (rng.uniform(0, 2 * np.pi, nr), np.radians(rng.uniform(0.5, 3.0, nr)), rng.random(nr))
            st["avec_rayons"] = True
            st["plateau_px"] = 100 if geante else (2 * fw / 3) ** 2
        if geante:
            giants.append([round(float(x[k]), 1), round(float(y[k]), 1), float(f"{peak[k]:.3g}")])
        R0 = MARGE
        if x[k] < -R0 or x[k] > n + R0 or y[k] < -R0 or y[k] > n + R0:
            continue
        res_b = None
        if on("banque") and P.get("banque") is not None and peak[k] > 0.3 * float(np.min(P["sat"])):
            res_b = vignette_banque(P, rng, st, float(lvl_c[1, k] * P["sat"][1]))
        if res_b is not None:
            light, R = res_b
            n_banque += 1
        else:
            light, R = vignette(P, rng, st, on)
        xi, yi = int(round(x[k])), int(round(y[k]))
        x0, y0 = xi - R, yi - R
        sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(n, x0 + 2 * R + 1), min(n, y0 + 2 * R + 1)
        if sx0 >= sx1 or sy0 >= sy1:
            continue
        part = light[:, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
        stars[:, sy0:sy1, sx0:sx1] += part
        # 3) plateau propre : là où l'étoile dépasse la moitié de son niveau
        if on("plateau"):
            Lk = (lvl_c[:, k] * np.asarray(P["sat"]))[:, None, None].astype(np.float32)
            foot = part > 0.5 * Lk
            if foot.any():
                sub = Lmap[:, sy0:sy1, sx0:sx1]
                sub[foot] = np.minimum(sub, np.broadcast_to(Lk, sub.shape))[foot]
        # reflet fantôme décentré (étoiles très brillantes)
        if on("halo") and peak[k] > 3e3 and rng.random() < P["fantomes"]["p"]:
            kk = P["fantomes"]["k"]
            gx, gy = cxo + (x[k] - cxo) * (1 + kk), cyo + (y[k] - cyo) * (1 + kk)
            rad = lu(rng, 10, 150)
            gst = S1.ghost_stamp(rng, rad, rng.random() < 0.6)
            frac = lu(rng, 1e-4, 3e-3)
            s2 = gst.shape[0] // 2
            gx0, gy0 = int(gx) - s2, int(gy) - s2
            ys_, xs_ = slice(max(0, gy0), min(n, gy0 + gst.shape[0])), slice(max(0, gx0), min(n, gx0 + gst.shape[1]))
            if ys_.start < ys_.stop and xs_.start < xs_.stop:
                g = gst[ys_.start - gy0:ys_.stop - gy0, xs_.start - gx0:xs_.stop - gx0]
                for c in range(C):
                    stars[c, ys_, xs_] += frac * flux[k] * parts[c, k] * g
    # bruit de photons des étoiles (corrélé comme un empilement débruité, faible)
    if P["photon_k"] > 0:
        nz = rng.standard_normal(stars.shape).astype(np.float32)
        nz = np.stack([cv2.GaussianBlur(z, (0, 0), 0.7) * 1.9 for z in nz])
        stars = stars + np.sqrt(np.clip(stars, 0, None) * P["photon_k"]) * nz
    tot = fond + stars
    s = (P["bord"] if on("plateau") else 1e-3) * Lmap
    entree = softmin(tot, Lmap, s).astype(np.float32)
    ecretes = np.any(tot > 0.98 * Lmap, axis=0)
    fiche = dict(graine=str(graine), etoiles=int(nst), vignettes=int(len(big)), geantes=giants,
                 banque=fiche_image(P)["banque"], de_la_banque=n_banque, **finfo,
                 image=fiche_image(P))
    return dict(entree=entree, etoiles=(entree - fond).astype(np.float32), ecretes=ecretes, fiche=fiche, P=P)
