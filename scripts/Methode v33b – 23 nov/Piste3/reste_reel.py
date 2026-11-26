# ======================================================================
# reste_reel.py
# Rôle     : v18 (22 nov) : dans une zone recreee d'une etoile brillante (disque du coeur, bandes d'aigrettes), un pixel n'est recree que si la piste 1 y a laisse un vrai reste : |piste 1 - fond local| > K x bruit local dans au moins une coule
# Étape    : 2. Chaîne classique de soustraction guidée par Gaia (v33b) (voir scripts/LISEZMOI.md)
# Entrées  : image empilée (unités de bruit), astrométrie, extrait Gaia profond hors ligne
# Sorties  : image sans étoiles, cartes de classes
# Exemple  : python reste_reel.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""v18 (22 nov) : dans une zone recreee d'une etoile brillante (disque du coeur, bandes d'aigrettes), un pixel n'est recree
que si la piste 1 y a laisse un vrai reste : |piste 1 - fond local| > K x bruit local dans au moins une couleur.

  fond local  : mediane (robuste) par couleur dans un carre de rayon 3 x FWHM centre sur le pixel, pixels exclus (voir
                exclusions) ; si moins de MIN_VALID du carre est utilisable, rayon double (jusqu'a 2 fois).
  bruit local : par couleur et par morceau connexe de la zone, 1,4826 MAD du MEME detail (piste 1 - fond local a 3 FWHM)
                dans l'anneau 4-12 px autour du morceau, hors exclusions ; plancher = bruit de la fenetre de cette couleur.
                (Le bruit pixel seul ne convient pas : sur le coeur de M8 la structure de la nebuleuse a 3 FWHM est 10 a 30
                fois au-dessus ; mesure dans diag_v18/proto_reste.py et proto_texture.py.)
  exclusions  : les autres etoiles et les disques des coeurs ; PAS les bandes d'aigrettes (un reste d'aigrette est fin, la
                mediane l'ignore ; exclure les bandes larges fait suivre au fond le bord de la bande, pas la nebuleuse :
                27,8 % de faux restes sur un faux trou de ciel sans etoile contre 2,2 %).
  masque      : reste reel dilate de DIL px, garde dans la zone.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from numpy.lib.stride_tricks import sliding_window_view

K = 3.0             # seuil, en bruits locaux
SCALE_FWHM = 3.0    # rayon du fond local, en largeurs a mi-hauteur
DIL = 2             # dilatation du masque des restes reels (px)
MIN_VALID = 0.25    # part minimale de pixels utilisables dans le carre du fond


def fond_local(P, zone, excl, r):
    """Mediane par couleur de P (C, H, W) aux pixels de ``zone``, sur un carre de rayon r (puis 2r, 4r) sans ``excl``."""
    C, H, W = P.shape
    ys, xs = np.nonzero(zone)
    bg = np.full((C, ys.size), np.nan, np.float32)
    todo = np.ones(ys.size, bool)
    for rr in (r, 2 * r, 4 * r):
        if not todo.any():
            break
        pad = rr
        valid = np.pad(~excl, pad, constant_values=False)
        vw = sliding_window_view(valid, (2 * rr + 1, 2 * rr + 1))
        idx = np.nonzero(todo)[0]
        for c in range(C):
            Pc = np.pad(P[c].astype(np.float32), pad)
            pw = sliding_window_view(Pc, (2 * rr + 1, 2 * rr + 1))
            for a in range(0, idx.size, 2000):
                j = idx[a:a + 2000]
                v = pw[ys[j], xs[j]].reshape(j.size, -1).copy()
                ok = vw[ys[j], xs[j]].reshape(j.size, -1)
                frac = ok.mean(1)
                v[~ok] = np.nan
                with np.errstate(all="ignore"):
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m = np.nanmedian(v, 1)
                good = frac >= MIN_VALID
                bg[c, j[good]] = m[good]
        todo = np.isnan(bg).any(0)
    if todo.any():      # carre 4r encore trop occupe : mediane de toute la fenetre hors exclusions
        for c in range(C):
            bg[c, todo] = np.median(P[c][~excl]) if (~excl).any() else np.median(P[c])
    out = np.full(P.shape, np.nan, np.float32)
    out[:, ys, xs] = bg
    return out


def bruit_local(P, zone, excl, sig_floor, r):
    """Bruit par couleur, par morceau connexe de la zone : MAD de P - fond local (rayon r) dans l'anneau 4-12 px autour."""
    C = P.shape[0]
    out = np.zeros(P.shape, np.float32)
    lab, n = ndi.label(zone)
    rings = []
    allr = np.zeros(zone.shape, bool)
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        y0, y1 = max(0, sl[0].start - 13), min(zone.shape[0], sl[0].stop + 13)
        x0, x1 = max(0, sl[1].start - 13), min(zone.shape[1], sl[1].stop + 13)
        comp = lab[y0:y1, x0:x1] == k
        ring = ndi.binary_dilation(comp, iterations=12) & ~ndi.binary_dilation(comp, iterations=4) & ~excl[y0:y1, x0:x1] & ~zone[y0:y1, x0:x1]
        rings.append((k, (y0, y1, x0, x1), comp, ring))
        allr[y0:y1, x0:x1] |= ring
    bg = fond_local(P, allr, excl, r) if allr.any() else None
    for k, (y0, y1, x0, x1), comp, ring in rings:
        for c in range(C):
            s = sig_floor[c]
            if ring.sum() >= 30:
                v = (P[c, y0:y1, x0:x1] - bg[c, y0:y1, x0:x1])[ring]
                s = max(s, float(np.median(np.abs(v - np.median(v))) * 1.4826))
            out[c, y0:y1, x0:x1][comp] = s
    return out


def restes_reels(P, zone, excl, fwhm, sig_floor, k=K, dil=DIL):
    """Masque (H, W) des pixels de ``zone`` a recreer : reste reel de la piste 1 (P), dilate de dil px, dans la zone.
    Renvoie aussi l'ecart max en bruits (pour le journal)."""
    if not zone.any():
        return np.zeros(zone.shape, bool), np.zeros(zone.shape, np.float32), np.zeros((P.shape[0],), np.float32)
    r = int(round(SCALE_FWHM * fwhm))
    bg = fond_local(P, zone, excl, r)
    sg = bruit_local(P, zone, excl, sig_floor, r)
    z = np.zeros(zone.shape, np.float32)
    with np.errstate(all="ignore"):
        dev = np.abs(P - bg) / np.where(sg > 0, sg, np.inf)
    z[zone] = np.nanmax(dev[:, zone], 0)
    left = zone & (z > k)
    m = ndi.binary_dilation(left, iterations=dil) & zone if dil else left
    nz = np.array([float(np.median(sg[c][zone])) for c in range(P.shape[0])], np.float32)
    return m, z, nz
