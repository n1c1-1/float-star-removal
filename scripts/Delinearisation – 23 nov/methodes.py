# ======================================================================
# methodes.py
# Role     : The de-linearisation methods compared on the bench.
# Step     : 3. De-linearisation of web images (23 Nov) (see scripts/LISEZMOI.md)
# Inputs   : stretched image
# Outputs  : image made linear
# Example  : python methodes.py [arguments]  (see the docstring; paths via the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""The de-linearisation methods compared on the bench. Each one: y (3, H, W) stretched 0-1 -> (linear, info).
A method that refuses the image (as FLOAT does: “the stretch cannot be estimated”) returns (None, reason)."""
from __future__ import annotations

import math

import numpy as np

import float_unstretch_copie as fu


def aucune(y):
    """Référence : l'image étirée prise telle quelle pour linéaire."""
    return y.copy(), {}


def _mono(y):
    return bool(np.allclose(y[0], y[1]) and np.allclose(y[1], y[2]))


def float_174(y):
    """La fonction FLOAT (#174) : estimate() de astro_gpu/stages/unstretch.py (copie mot pour mot, float/main a94fec34).
    Son refus (ValueError) est rendu tel quel : dans FLOAT le bloc Image s'arrête alors sur une erreur."""
    try:
        out, info = fu.estimate(y, mono=_mono(y))
    except ValueError as e:
        return None, f"refus FLOAT : {e}"
    return out, [(round(m, 5), round(u, 6)) for m, u in info]


def m_force(c):
    """Le m de la fonction FLOAT quand son équation n'a pas de racine : celui qui l'approche le mieux (|g(u)| minimal
    sur une grille de u). Diagnostic seulement, ce n'est pas ce que FLOAT fait."""
    v = c[np.isfinite(c)]
    s = float(np.median(v))
    sn = 1.4826 * float(np.median(np.abs(v - s)))
    best, mb = math.inf, 0.5
    for u in np.geomspace(1e-7, 0.3, 400):
        x = 2.8 * u
        m = x * (s - 1.0) / (2.0 * s * x - s - x)
        if not 0 < m < 1:
            continue
        d = (2.0 * m - 1.0) * x - m
        gval = abs(sn - u * m * (1.0 - m) / (d * d))
        if gval < best:
            best, mb = gval, m
    return mb


def float_force(y):
    """Diagnostic : la formule FLOAT (inverse de MTF par canal) avec le m le plus proche quand FLOAT refuse."""
    n = 1 if _mono(y) else 3
    out, info = np.empty_like(y), []
    for i in range(n):
        try:
            out[i], m, _u = fu.estimate_channel(y[i])
        except ValueError:
            m = m_force(y[i])
            out[i] = fu.mtf_inverse(np.clip(y[i], 0, 1), m, 0.0, 1.0)
        info.append(round(float(m), 5))
    if n == 1:
        out[1:] = out[0]
    return out, info


def etoiles(y):
    """Amélioration 1 : courbe estimée depuis les profils des étoiles (courbe_etoiles.py), même courbe sur chaque canal."""
    import courbe_etoiles as ce
    g, info = ce.estimer(y)
    return ce.appliquer(y, g), info


METHODES = {"aucune": aucune, "float": float_174, "float_force": float_force, "etoiles": etoiles}


def etoiles2(y):
    """Amélioration 1, version 2 : profil commun non paramétrique (courbe_etoiles2.py), même courbe sur chaque canal."""
    import courbe_etoiles2 as ce
    g, info = ce.estimer(y)
    return ce.appliquer(y, g), info


METHODES["etoiles2"] = etoiles2


def etoiles3(y):
    """Amélioration 2 : la courbe des étoiles (version 2) estimée canal par canal (étirements déliés, STF par canal)."""
    import courbe_etoiles2 as ce
    if _mono(y):
        g, info = ce.estimer(y)
        return ce.appliquer(y, g), info
    out, infos = np.empty_like(y), []
    for c in range(3):
        g, info = ce.estimer(y, canal=c)
        out[c] = ce.appliquer(y[c], g)
        infos.append({k: info.get(k) for k in ("etoiles_utilisees", "m_depart", "cout_depart", "cout_fin")})
    return out, infos


METHODES["etoiles3"] = etoiles3


def etoiles2_lum(y):
    """Amélioration 2 : courbe des étoiles (version 2) appliquée à la luminance, rapports de couleur de l'image étirée
    gardés (lin_c = g(Y) y_c / Y), comme l'arcsinh de [logiciel tiers] les garde (filters/asinh.c)."""
    import courbe_etoiles2 as ce
    g, info = ce.estimer(y)
    Y = y.mean(0)
    G = ce.appliquer(Y, g)
    with np.errstate(all="ignore"):
        k = np.where(Y > 1e-4, G / np.maximum(Y, 1e-4), np.interp(1e-4, ce.GRILLE, g) / 1e-4)
    return y * k[None], info


METHODES["etoiles2_lum"] = etoiles2_lum


def _ecart_couleur_coeur_ailes(D, ys, xs):
    """Écart de couleur entre le cœur (r < 1,5 px) et les ailes (2,5 à 5 px) des étoiles d'une image linéaire : la
    fonction d'étalement est la même dans les trois canaux, donc une bonne délinéarisation donne à une étoile la même
    couleur au cœur et dans les ailes. Médiane sur les étoiles de |log(R/G)| + |log(B/G)| (cœur moins ailes)."""
    R = 8
    g = np.mgrid[-R:R + 1, -R:R + 1]
    r = np.hypot(g[0], g[1])
    ec = []
    for cy, cx in zip(ys, xs):
        a, b = int(round(cy)), int(round(cx))
        c = D[:, a - R:a + R + 1, b - R:b + R + 1]
        if c.shape[1:] != r.shape:
            continue
        bg = np.median(c[:, r >= 6.5], axis=1)
        core = (c[:, r < 1.5] - bg[:, None]).sum(1)
        wing = (c[:, (r >= 2.5) & (r < 5)] - bg[:, None]).sum(1)
        if np.all(core > 0) and np.all(wing > 0):
            ec.append(abs(np.log(core[0] / core[1]) - np.log(wing[0] / wing[1]))
                      + abs(np.log(core[2] / core[1]) - np.log(wing[2] / wing[1])))
    return float(np.median(ec)) if len(ec) > 10 else np.inf


def etoiles_auto(y):
    """Amélioration 3 : courbe des étoiles (version 2) ; la façon d'appliquer la courbe aux couleurs (canal par canal,
    ou luminance avec rapports gardés) est choisie image par image : celle qui donne aux étoiles la même couleur au cœur
    et dans les ailes."""
    import courbe_etoiles2 as ce
    from commun import bruit_laplacien, mad_sigma
    from etoiles_outils import detecter
    g, info = ce.estimer(y)
    if _mono(y):
        return ce.appliquer(y, g), info
    A = ce.appliquer(y, g)
    Y = y.mean(0)
    G = ce.appliquer(Y, g)
    k = np.where(Y > 1e-4, G / np.maximum(Y, 1e-4), np.interp(1e-4, ce.GRILLE, g) / 1e-4)
    B = y * k[None]
    s = max(mad_sigma(bruit_laplacien(Y)[::2, ::2]), 1.0 / 255 / np.sqrt(12))
    ys, xs = detecter(Y, s, k=20.0, isol=14, bord=20, max_n=300, plafond=0.97)
    ea, eb = _ecart_couleur_coeur_ailes(A, ys, xs), _ecart_couleur_coeur_ailes(B, ys, xs)
    info = dict(info) if isinstance(info, dict) else {"info": info}
    info.update(couleur="canaux" if ea <= eb else "luminance", ecart_canaux=round(ea, 4), ecart_lum=round(eb, 4))
    return (A if ea <= eb else B), info


METHODES["etoiles_auto"] = etoiles_auto
