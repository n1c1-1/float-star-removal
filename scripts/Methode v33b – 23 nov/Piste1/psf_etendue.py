# ======================================================================
# psf_etendue.py
# Rôle     : Méthode v12 (21 nov) : PSF ÉTENDUE des étoiles brillantes, bâtie sur l'IMAGE ENTIÈRE, une fois par image, gardée sur disque.
# Étape    : 2. Chaîne classique de soustraction guidée par Gaia (v33b) (voir scripts/LISEZMOI.md)
# Entrées  : image empilée (unités de bruit), astrométrie, extrait Gaia profond hors ligne
# Sorties  : image sans étoiles, cartes de classes
# Exemple  : python psf_etendue.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Méthode v12 (21 nov) : PSF ÉTENDUE des étoiles brillantes, bâtie sur l'IMAGE ENTIÈRE, une fois par image, gardée sur disque.

Pourquoi (JOURNAL, décision v11) : en v10 chaque étoile brillante est retirée avec son PROPRE profil radial (médiane par anneau
de 16 secteurs, moins l'anneau au-delà) : sur la nébuleuse, ce profil prend la lumière de la nébuleuse autour de l'étoile, et
ce disque est soustrait aussi sur une bande sombre voisine -> trous noirs (bord intérieur, étoile G 8,2 à aigrettes).

Méthode (Infante-Sainz, Trujillo & Román 2020, MNRAS 491, 5317 ; arXiv:1911.01430) :
  (a) UNE forme par couleur pour toute l'image : profils radiaux des étoiles brillantes de l'image entière (G 6-10,5),
      chacun = médiane, par anneau, des médianes de 16 secteurs (aigrettes, voisines, nœuds de nébuleuse écartés), moins le
      fond lointain (anneau 165-200 px, même médiane de secteurs), divisé par l'amplitude de l'étoile ; voisines masquées
      (disque selon leur éclat), pixels saturés écartés ; empilement = médiane sur les étoiles à chaque rayon (au moins
      5 étoiles) ; raccord au modèle v7 (coeur, par unité de flux) entre 10 et 14 px ; décroissante et >= 0 au-delà de 10 px ;
      fondu à zéro de 125 à 150 px (portée 150 px).
  (b) amplitude d'une étoile = ajustement robuste (moindres carrés, écrêtage 3 sigma MAD, 5 tours) de amplitude x forme
      + fond plat, sur les PIXELS d'un anneau intérieur où l'étoile domine : r de max(2, rs/FWHM + 2) à +4 largeurs
      (rs = rayon saturé), voisines masquées ; le fond plat absorbe la lumière de la nébuleuse à cette échelle, elle n'entre
      pas dans l'étoile. La même fonction sert ici (normer les profils empilés) et dans la piste 1 (chaque étoile).
  Itérations : tour 0 avec le profil du modèle v7 (étoiles dont l'anneau tient dans 24 px), tours 1-2 avec la forme raccordée.
Cache : Piste1/sorties/<image>/psf_etendue.npz (+ .json : réglages, étoiles, profils, amplitudes).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
import images  # noqa: E402
import retrait_modele as RM  # noqa: E402
import modele_psf_image as MPI  # noqa: E402

VERSION = "v12-2"
R_EXT = 150.0                  # portée de la forme (px)
FADE = (125.0, 150.0)          # fondu à zéro
BG_FAR = (165.0, 200.0)        # fond lointain d'un profil empilé
JOIN = (10.0, 14.0)            # raccord modèle v7 (coeur) -> profil empilé (16-24 px : la forme v7 y est 2 x sous les ailes des brillantes, essai 2)
G_RANGE = (6.0, 10.5)          # étoiles empilées
RING_FW = (2.0, 4.0)           # anneau d'amplitude : début max(2 FWHM, rs + 2 FWHM), largeur 4 FWHM
MIN_STARS = 5
NSEC = 16
RG = np.arange(0.0, R_EXT + 0.01, 0.25)        # grille radiale de la forme


def neighbour_radius(Gj):
    """Disque masqué autour d'une voisine (px) : règle du modèle v7 (3 + 2,5 (16 - G), 3-25 px), élargi pour les
    brillantes (G < 11 : + 12 px par magnitude, au plus 80 px)."""
    Gj = np.asarray(Gj, float)
    r = np.clip(3 + 2.5 * (16 - Gj), 3, 25)
    return np.where(Gj < 11, np.minimum(25 + 12 * (11 - Gj), 80), r)


def ring_radii(fwhm, rs):
    r_in = max(RING_FW[0] * fwhm, rs + 2.0 * fwhm)
    return r_in, r_in + RING_FW[1] * fwhm


def fit_amplitude(vals, shp, iters=5, k=3.0, min_n=30):
    """amplitude x forme + fond plat, moindres carrés, écrêtage k sigma (MAD) ; (amplitude, fond, pixels gardés)."""
    vals = np.asarray(vals, float); shp = np.asarray(shp, float)
    ok = np.isfinite(vals) & np.isfinite(shp)
    if ok.sum() < min_n or np.ptp(shp[ok]) <= 0:
        return np.nan, np.nan, 0
    keep = ok.copy()
    sol = np.array([np.nan, np.nan])
    for _ in range(iters):
        A = np.c_[shp[keep], np.ones(keep.sum())]
        sol = np.linalg.lstsq(A, vals[keep], rcond=None)[0]
        res = vals - (sol[0] * shp + sol[1])
        sd = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-12
        new = ok & (np.abs(res) < k * sd)
        if new.sum() < min_n or np.array_equal(new, keep):
            break
        keep = new
    return float(sol[0]), float(sol[1]), int(keep.sum())


def sector_profile(v, r, sec, edges, ok):
    """Médiane, par anneau, des médianes de NSEC secteurs (>= 3 px par secteur, >= 8 secteurs)."""
    nb = len(edges) - 1
    b = np.searchsorted(edges, r, side="right") - 1
    sel = ok & (b >= 0) & (b < nb)
    lab = b[sel] * NSEC + sec[sel]
    vv = v[sel]
    o = np.lexsort((vv, lab))
    lab, vv = lab[o], vv[o]
    cnt = np.bincount(lab, minlength=nb * NSEC)
    st = np.r_[0, np.cumsum(cnt)[:-1]]
    med = np.full(nb * NSEC, np.nan)
    g = cnt >= 3
    med[g] = 0.5 * (vv[st[g] + (cnt[g] - 1) // 2] + vv[st[g] + cnt[g] // 2])
    med = med.reshape(nb, NSEC)
    n_ok = np.isfinite(med).sum(1)
    with np.errstate(all="ignore"):
        p = np.nanmedian(np.where(n_ok[:, None] >= 8, med, np.nan), 1)
    return p


def _v7_profile(m):
    """Profil radial par unité de flux du modèle v7 pour la tranche la plus brillante (G < 11), par couleur, sur RG."""
    S, rho, os_ = m["S"], m["rho"], int(m["os"])
    C = S.shape[0]
    n = S.shape[-1]
    c0 = (n - 1) // 2
    rr = np.hypot(*np.mgrid[-c0:c0 + 1, -c0:c0 + 1]) / os_
    lab = np.digitize(rr, RG)
    out = np.zeros((C, len(RG)))
    for c in range(C):
        st = S[c, 0].astype(np.float64) + rho[c, 0].astype(np.float64)
        prof = np.asarray(ndi.mean(st, labels=lab, index=np.arange(1, len(RG) + 1)), float)
        cnt = np.bincount(lab.ravel(), minlength=len(RG) + 1)[1:]
        mids = RG + 0.125
        okp = (cnt > 0) & np.isfinite(prof)
        out[c] = np.interp(RG, mids[okp], prof[okp], right=0.0)
        out[c][RG > int(m["R"])] = 0.0
    return out


def construire(name, log=print):
    t0 = time.time()
    m = MPI.obtenir(name, log)
    info7 = m["info"]
    fwhm = float(info7["fwhm"])
    SIG = np.asarray(info7["bruit"], float)
    img = images.load(name)
    C, H, W = img.shape
    valid = ndi.binary_erosion(np.any(img != 0, axis=0), iterations=3)
    xg, yg, G = RM.catalogue_image(name, log)       # SHO 21 nov : profond, sinon extrait hors ligne
    # saturation : même règle que la piste 1 et le modèle v7
    inb = (xg > 3) & (xg < W - 3) & (yg > 3) & (yg < H - 3)
    brightest = np.argsort(np.where(inb, G, 99))[:25]
    sat_lvl = np.zeros(C)
    for c in range(C):
        pk_ = np.array([img[c, int(yg[k]) - 2:int(yg[k]) + 3, int(xg[k]) - 2:int(xg[k]) + 3].max() for k in brightest])
        top_ = np.sort(pk_)[::-1][:12]
        plateau = len(top_) >= 5 and np.std(top_[1:]) / max(np.median(top_), 1e-9) < 0.12
        sat_lvl[c] = float(np.median(top_[1:])) * RM.P["sat_frac"] if plateau else np.inf
    sat = np.stack([ndi.binary_dilation(img[c] >= sat_lvl[c], iterations=2) for c in range(C)])
    satany = sat.any(0)

    def sat_radius(x, y):
        xi, yi = int(round(x)), int(round(y))
        if not satany[max(0, yi - 1):yi + 2, max(0, xi - 1):xi + 2].any():
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

    # ---- étoiles empilées
    RW = int(BG_FAR[1])
    tree = cKDTree(np.c_[xg, yg])
    cand = np.where((G >= G_RANGE[0]) & (G <= G_RANGE[1]) & (xg > RW + 2) & (xg < W - RW - 3)
                    & (yg > RW + 2) & (yg < H - RW - 3))[0]
    rej = dict(voisine_plus_brillante=0, voisine_semblable=0, bord=0, anneau=0)
    stars = []
    for k in cand:
        nb_ = [j for j in tree.query_ball_point([xg[k], yg[k]], 1.5 * R_EXT) if j != k]
        d_ = np.hypot(xg[nb_] - xg[k], yg[nb_] - yg[k]) if nb_ else np.zeros(0)
        Gn = G[nb_] if nb_ else np.zeros(0)
        if np.any(Gn < G[k] - 1.0):
            rej["voisine_plus_brillante"] += 1          # son halo domine le nôtre sur la portée
            continue
        if np.any((Gn < G[k] + 1.0) & (d_ < 60)):
            rej["voisine_semblable"] += 1
            continue
        xi, yi = int(round(xg[k])), int(round(yg[k]))
        if valid[yi - RW:yi + RW + 1, xi - RW:xi + RW + 1].mean() < 0.9:
            rej["bord"] += 1
            continue
        stars.append(k)
    stars = np.array(stars, int)
    log(f"[PSF étendue] candidates G {G_RANGE[0]}-{G_RANGE[1]} : {len(cand)}, gardées {len(stars)}, écartées {rej}, "
        f"largeur {fwhm:.2f} px, {time.time() - t0:.0f} s")

    yy, xx = np.mgrid[-RW:RW + 1, -RW:RW + 1]
    edges = np.unique(np.r_[np.arange(4.0, 20.0, 1.0), np.geomspace(20.0, R_EXT, 36)])
    rb = np.sqrt(edges[:-1] * edges[1:]) if edges[0] > 0 else 0.5 * (edges[:-1] + edges[1:])
    rb[edges[:-1] < 20] = 0.5 * (edges[:-1] + edges[1:])[edges[:-1] < 20]
    meas = {}
    for k in stars:
        x, y = xg[k], yg[k]
        xi, yi = int(round(x)), int(round(y))
        dx = xx + xi - x; dy = yy + yi - y
        r = np.hypot(dx, dy)
        sec = ((np.arctan2(dy, dx) + np.pi) / (2 * np.pi) * NSEC).astype(int) % NSEC
        msk = valid[yi - RW:yi + RW + 1, xi - RW:xi + RW + 1].copy()
        for j in tree.query_ball_point([x, y], RW * 1.42 + 80):
            if j == k or G[j] >= 18.5:
                continue
            rm_ = float(neighbour_radius(G[j]))
            d = np.hypot(xg[j] - x, yg[j] - y)
            if d - rm_ > RW * 1.42:
                continue
            msk &= np.hypot(dx - (xg[j] - x), dy - (yg[j] - y)) > rm_
        rs = sat_radius(x, y)
        cut = img[:, yi - RW:yi + RW + 1, xi - RW:xi + RW + 1]
        prof = np.full((C, len(rb)), np.nan)
        bg = np.zeros(C)
        okcs = []
        for c in range(C):
            okc = msk & ~sat[c, yi - RW:yi + RW + 1, xi - RW:xi + RW + 1]
            okcs.append(okc)
            far = sector_profile(cut[c], r, sec, np.array(BG_FAR), okc)
            bg[c] = float(far[0]) if np.isfinite(far[0]) else np.nan
            prof[c] = sector_profile(cut[c], r, sec, edges, okc) - bg[c]
            prof[c][rb < rs + 2.0 * fwhm] = np.nan                    # coeur saturé : pas une mesure de la forme
        r_in, r_out = ring_radii(fwhm, rs)
        meas[k] = dict(r=r, okc=okcs, cut=cut, prof=prof, rs=rs, r_in=r_in, r_out=r_out)
    log(f"[PSF étendue] profils mesurés sur {len(meas)} étoiles, {time.time() - t0:.0f} s")

    P7 = _v7_profile(m)
    P = P7.copy()
    hist = []
    amps = {}
    for it in range(3):
        amps = {}
        for k, d in meas.items():
            if it == 0 and d["r_out"] > 24.0:
                continue                                    # tour 0 : l'anneau doit tenir dans le modèle v7
            ring = (d["r"] >= d["r_in"]) & (d["r"] <= d["r_out"])
            a_ = np.full(C, np.nan)
            for c in range(C):
                sel = ring & d["okc"][c]
                amp, b_, n_ = fit_amplitude(d["cut"][c][sel], np.interp(d["r"][sel], RG, P[c]))
                if np.isfinite(amp) and amp > 0:
                    a_[c] = amp
            amps[k] = a_
        stack = np.full((C, len(rb)), np.nan)
        nst = np.zeros((C, len(rb)), int)
        for c in range(C):
            rows = [meas[k]["prof"][c] / amps[k][c] for k in amps if np.isfinite(amps[k][c])]
            if not rows:
                continue
            A_ = np.array(rows)
            nst[c] = np.isfinite(A_).sum(0)
            with np.errstate(all="ignore"):
                md = np.nanmedian(A_, 0)
            md[nst[c] < MIN_STARS] = np.nan
            stack[c] = md
        newP = np.zeros_like(P)
        ratio = []
        for c in range(C):
            okb = np.isfinite(stack[c])
            if okb.sum() < 4:
                newP[c] = P7[c]
                continue
            sm = stack[c].copy()
            ix = np.where(okb)[0]
            sm[ix] = ndi.median_filter(sm[ix], size=3, mode="nearest")      # 3 anneaux, en log r au-delà de 20 px
            s_rg = np.interp(RG, rb[okb], sm[okb], left=np.nan, right=0.0)
            t = np.clip((RG - JOIN[0]) / (JOIN[1] - JOIN[0]), 0, 1)
            p = np.where(RG < JOIN[0], P7[c], (1 - t) * P7[c] + t * np.nan_to_num(s_rg))
            jo = RG >= JOIN[0]
            p[jo] = np.minimum.accumulate(np.maximum(p[jo], 0.0))
            fade = np.clip((FADE[1] - RG) / (FADE[1] - FADE[0]), 0, 1)
            p = p * (0.5 - 0.5 * np.cos(np.pi * fade))
            newP[c] = p
            jm = (RG >= JOIN[0]) & (RG <= JOIN[1]) & np.isfinite(s_rg) & (P7[c] > 0)
            ratio.append(float(np.median(s_rg[jm] / P7[c][jm])) if jm.any() else np.nan)
        P = newP
        n_amp = int(sum(np.isfinite(a).any() for a in amps.values()))
        hist.append(dict(tour=it, etoiles=n_amp, raccord_empile_sur_v7=[round(v, 3) for v in ratio],
                         etoiles_par_rayon={f"{rb[i]:.0f}": nst[:, i].tolist() for i in range(0, len(rb), 4)}))
        log(f"[PSF étendue] tour {it} : {n_amp} étoiles ; raccord (empilé / v7, 16-24 px) {np.round(ratio, 3).tolist()} ; "
            f"étoiles à r ~{rb[len(rb) // 2]:.0f} px {nst[:, len(rb) // 2].tolist()}, à r ~{rb[-1]:.0f} px {nst[:, -1].tolist()}")
    secs = time.time() - t0
    prof_out = {f"{r_:.0f}": [float(P[c][np.searchsorted(RG, r_)]) for c in range(C)]
                for r_ in (5, 10, 16, 20, 24, 30, 40, 60, 80, 100, 125, 140)}
    info = dict(version=VERSION, image=name, R_ext=R_EXT, fondu=FADE, fond_lointain=BG_FAR, raccord=JOIN, G=G_RANGE,
                anneau_amplitude_largeurs=RING_FW, largeur=fwhm, candidates=int(len(cand)), ecartees=rej,
                etoiles=int(len(meas)), etoiles_amplitude=int(sum(np.isfinite(a).any() for a in amps.values())),
                secondes=round(secs), tours=hist, profil_par_unite_de_flux=prof_out,
                somme_jusqu_a_R=[float((P[c] * 2 * np.pi * RG).sum() * 0.25) for c in range(C)],
                etoiles_liste=[dict(x=round(float(xg[k]), 1), y=round(float(yg[k]), 1), G=round(float(G[k]), 2),
                                    rs=meas[k]["rs"], amp=[None if not np.isfinite(v) else round(float(v), 1) for v in amps.get(k, [np.nan] * C)])
                               for k in meas])
    return dict(rg=RG, P=P.astype(np.float64), P7=P7, stack=stack, rb=rb, info=info)


def obtenir(name, log=print):
    d = HERE / "sorties" / name
    f = d / "psf_etendue.npz"
    if f.exists():
        info = json.loads((d / "psf_etendue.json").read_text(encoding="utf-8"))
        if info.get("version") == VERSION:
            z = np.load(f)
            log(f"[PSF étendue] relue du cache : {info['etoiles_amplitude']} étoiles, portée {info['R_ext']} px, "
                f"bâtie en {info['secondes']} s")
            return dict(rg=z["rg"], P=z["P"], info=info)
    m = construire(name, log)
    np.savez(f, rg=m["rg"], P=m["P"], P7=m["P7"], stack=m["stack"], rb=m["rb"])
    (d / "psf_etendue.json").write_text(json.dumps(m["info"], ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[PSF étendue] bâtie en {m['info']['secondes']} s, gardée : {f}")
    return m


if __name__ == "__main__":
    import sys
    nm = sys.argv[1] if len(sys.argv) > 1 else "rosette"
    if "--refaire" in sys.argv:
        (HERE / "sorties" / nm / "psf_etendue.npz").unlink(missing_ok=True)
    obtenir(nm)
