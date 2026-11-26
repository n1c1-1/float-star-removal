# ======================================================================
# astrometrie.py
# Rôle     : Astrométrie de l'image et étoiles Gaia du champ, hors ligne seulement.
# Étape    : 2. Chaîne classique de soustraction guidée par Gaia (v33b) (voir scripts/LISEZMOI.md)
# Entrées  : image empilée (unités de bruit), astrométrie, extrait Gaia profond hors ligne
# Sorties  : image sans étoiles, cartes de classes
# Exemple  : python astrometrie.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Astrométrie de l'image et étoiles Gaia du champ, hors ligne seulement.

Solution de départ : le solveur local de FLOAT (core/asterion, index Tycho-2 installé), centre approché par le nom de
l'objet (catalogues livrés avec FLOAT, sans réseau). Affinage : ajustement TAN-SIP (ordre 3) sur les étoiles détectées
appariées aux étoiles Gaia de l'extrait astrométrique hors ligne, comme le bloc Astrométrie de FLOAT.
Catalogue d'étoiles à retirer : l'extrait astrométrique (127 plus brillantes par pixel HEALPix de 14') réuni à l'extrait
spectrophotométrique (toutes les étoiles G < 17.65 avec spectre XP), lus en place par astro_gpu.core.catalogue_local ; jamais de
téléchargement (on vérifie que les fichiers lus sont ceux de ASTRO_GPU_GAIA_EXTRACTS).
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, r"<FLOAT>")
EXTRACTS = "<CACHE_FLOAT>/cache/gaia_extracts"
os.environ["ASTRO_GPU_GAIA_EXTRACTS"] = EXTRACTS

from astro_gpu.core import asterion, catalogue_local  # noqa: E402
from astro_gpu.stages.astrometry import detect, find_offline  # noqa: E402


def _check_offline(log):
    f = catalogue_local.locate()
    a = f["astro"]
    assert a is not None and str(a.path).replace("\\", "/").startswith(EXTRACTS), f"extrait astrométrique introuvable hors ligne: {a}"
    assert f["xp"], "extrait XP introuvable"
    x = next(iter(f["xp"].values()))
    assert str(x.path).replace("\\", "/").startswith(EXTRACTS), x.path
    log(f"catalogue hors ligne: {a.path} ; XP: {len(f['xp'])} morceaux dans {x.path.parent}")


def solve(img: np.ndarray, target: str, work, log=print):
    """WCS astropy (TAN-SIP) de l'image (C, H, W)."""
    from astropy.wcs.utils import fit_wcs_from_points
    from scipy.spatial import cKDTree
    _check_offline(log)
    ra0, dec0 = find_offline(target)
    H, W = img.shape[-2:]
    work.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = os.environ["TEMP"] = str(work)
    import tempfile
    tempfile.tempdir = str(work)
    info: dict = {}
    sol = None
    # 22 nov soir (jeu 1, M97) : sans solution a echelle libre (champ pauvre en etoiles Tycho-2), reessai a l'echelle des cameras LCO
    # (0,58 puis 0,74 "/px) ; une image deja resolue a echelle libre est inchangee
    for scale in (None, 0.58, 0.74):
        sol = asterion.solve(img, work, ra=ra0, dec=dec0, radius=8.0, scale=scale, timeout=60.0, info=info)
        if sol is not None:
            break
    if sol is None:
        raise RuntimeError(f"pas de solution: {info}")
    log(f"solution de départ: {sol.scale:.3f}\"/px, {sol.matched} étoiles, centre {sol.ra:.4f} {sol.dec:.4f}, "
        f"retourné {sol.flipped}, {sol.seconds:.1f} s")
    g = img.sum(0).astype(np.float32)
    o = detect(g)
    o = np.sort(o[o["flag"] == 0], order="flux")[::-1]
    det = np.c_[o["x"], o["y"]][:8000]
    w1 = sol.wcs
    c = w1.pixel_to_world(W / 2, H / 2)
    rad = 0.5 * np.hypot(W, H) * sol.scale / 3600 * 1.05
    cat = catalogue_local.astro_cone(float(c.ra.deg), float(c.dec.deg), rad, 17.0)
    from astropy.coordinates import SkyCoord
    gs = SkyCoord(cat["RA_ICRS"], cat["DE_ICRS"], unit="deg")
    sip = 3 if len(det) > 200 else None
    res = None
    for tol in (3.0, 1.5, 1.0):
        gx, gy = w1.world_to_pixel(gs)
        ins = (gx > 5) & (gx < W - 5) & (gy > 5) & (gy < H - 5)
        tr = cKDTree(det)
        dd, j = tr.query(np.c_[gx[ins], gy[ins]], distance_upper_bound=tol)
        m = np.isfinite(dd)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w1 = fit_wcs_from_points((det[j[m], 0], det[j[m], 1]), gs[ins][m], projection="TAN", sip_degree=sip)
        gx2, gy2 = w1.world_to_pixel(gs[ins][m])
        res = np.hypot(gx2 - det[j[m], 0], gy2 - det[j[m], 1])
    log(f"affinage Gaia: résidu médian {np.median(res):.3f} px sur {len(res)} étoiles (SIP {sip})")
    return w1, float(sol.scale), float(np.median(res))


def gaia_stars(wcs, shape, scale, margin=60, log=print):
    """Étoiles Gaia dans l'image : dict x, y, G, teff (0 si inconnu), xp (bool), et la source de chaque liste."""
    from astropy.coordinates import SkyCoord
    H, W = shape
    c = wcs.pixel_to_world(W / 2, H / 2)
    ra, dec = float(c.ra.deg), float(c.dec.deg)
    rad = 0.5 * np.hypot(W + 2 * margin, H + 2 * margin) * scale / 3600 * 1.02
    # extrait astrométrique
    f = catalogue_local.locate()
    a = catalogue_local.astro_cone(ra, dec, rad, 25.0)
    # Teff : relu des mêmes enregistrements (astro_cone ne le rend pas)
    from astro_gpu.core.skycache import tiles_in_cone
    cf = f["astro"]
    rec, _ = cf.read(tiles_in_cone(cf.level, ra, dec, rad))
    r_ra = rec["ra"].astype(np.float64) * catalogue_local.SCALE
    r_dec = rec["dec"].astype(np.float64) * catalogue_local.SCALE
    keep = catalogue_local._sep_deg(ra, dec, r_ra, r_dec) <= rad
    A = dict(ra=r_ra[keep], dec=r_dec[keep], g=rec["g"][keep] / 1000.0, teff=rec["teff"][keep].astype(float))
    log(f"extrait astrométrique: {len(A['ra'])} étoiles, complet jusqu'à G {a['complete_to']:.2f}")
    # extrait XP : toutes ses étoiles (G < 17.65) dans les pixels du cône
    from astropy_healpix import HEALPix
    import astropy.units as u
    xs = []
    for cfx in f["xp"].values():
        lvl = cfx.level
        pix = tiles_in_cone(lvl, ra, dec, rad)
        if cfx.chunk_level:
            pix = np.asarray(pix)[(np.asarray(pix) >> (2 * (lvl - cfx.chunk_level))) == cfx.chunk]
        if not len(pix):
            continue
        r, _ = cfx.read(pix)
        if len(r):
            xs.append((r["ra"].astype(np.float64) * catalogue_local.SCALE, r["dec"].astype(np.float64) * catalogue_local.SCALE,
                       r["g"] / 1000.0, cfx.spectra(r)))
    if xs:
        X_ra = np.concatenate([x[0] for x in xs]); X_dec = np.concatenate([x[1] for x in xs])
        X_g = np.concatenate([x[2] for x in xs]); X_sp = np.concatenate([x[3] for x in xs])
        k = catalogue_local._sep_deg(ra, dec, X_ra, X_dec) <= rad
        X_ra, X_dec, X_g, X_sp = X_ra[k], X_dec[k], X_g[k], X_sp[k]
    else:
        X_ra = X_dec = X_g = np.zeros(0); X_sp = np.zeros((0, 343))
    log(f"extrait XP: {len(X_ra)} étoiles (G < 17.65)")
    # réunion : une étoile XP déjà dans l'extrait astrométrique (< 0.3", même G à 0.02) n'est pas doublée
    from scipy.spatial import cKDTree
    def xyz(r, d):
        r, d = np.radians(r), np.radians(d)
        return np.c_[np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)]
    ra_all = np.r_[A["ra"], X_ra]; dec_all = np.r_[A["dec"], X_dec]; g_all = np.r_[A["g"], X_g]
    teff = np.r_[A["teff"], np.zeros(len(X_ra))]
    src = np.r_[np.zeros(len(A["ra"]), int), np.ones(len(X_ra), int)]
    if len(X_ra) and len(A["ra"]):
        t = cKDTree(xyz(A["ra"], A["dec"]))
        d, j = t.query(xyz(X_ra, X_dec), distance_upper_bound=np.radians(0.5 / 3600))
        dup = np.isfinite(d)
        dup[dup] &= np.abs(A["g"][j[dup]] - X_g[dup]) < 0.02
        keepx = np.r_[np.ones(len(A["ra"]), bool), ~dup]
        # l'étoile de l'extrait astrométrique garde son spectre XP quand il existe
        xp_of_astro = np.full(len(A["ra"]), -1)
        xp_of_astro[j[dup]] = np.where(dup)[0]
    else:
        keepx = np.ones(len(ra_all), bool)
        xp_of_astro = np.full(len(A["ra"]), -1)
    sp_index = np.r_[xp_of_astro, np.arange(len(X_ra))]
    ra_all, dec_all, g_all, teff, src, sp_index = (v[keepx] for v in (ra_all, dec_all, g_all, teff, src, sp_index))
    x, y = wcs.world_to_pixel(SkyCoord(ra_all, dec_all, unit="deg"))
    ins = (x > -margin) & (x < W + margin) & (y > -margin) & (y < H + margin)
    out = dict(x=x[ins], y=y[ins], g=g_all[ins], teff=teff[ins], src=src[ins], sp=sp_index[ins])
    # couleur B-R grossière du spectre XP (bandes 400-500 / 600-700 nm), pour le rapport et la couleur de départ
    col = np.full(ins.sum(), np.nan)
    wl = catalogue_local.XP_WAVELENGTHS
    b = (wl >= 400) & (wl < 500); rr = (wl >= 600) & (wl < 700); gg = (wl >= 500) & (wl < 600)
    has = out["sp"] >= 0
    if has.any():
        S = X_sp[out["sp"][has]]
        out["rgb_xp"] = np.zeros((ins.sum(), 3))
        out["rgb_xp"][has] = np.c_[S[:, rr].sum(1), S[:, gg].sum(1), S[:, b].sum(1)]
    else:
        out["rgb_xp"] = np.zeros((ins.sum(), 3))
    o = np.argsort(out["g"])
    out = {k: v[o] for k, v in out.items()}
    log(f"étoiles Gaia dans l'image (marge {margin} px): {len(out['x'])}, dont {int((out['src'] == 1).sum())} venues de l'extrait XP seul")
    return out
