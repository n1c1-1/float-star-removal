# ======================================================================
# etalonnage_image.py
# Role     : Method v17 (22 Nov): settings of the WHOLE IMAGE, measured once and frozen; each crop re-reads them and only adjusts the brightness (and the position, as before) of each star.
# Step     : 2. Classic Gaia-guided subtraction chain (v33b) (see scripts/LISEZMOI.md)
# Inputs   : stacked image (noise units), astrometry, offline deep Gaia extract
# Outputs  : star-free image, class maps
# Example  : python etalonnage_image.py [arguments]  (see the docstring; paths through the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Method v17 (22 Nov): settings of the WHOLE IMAGE, measured once and frozen; each crop re-reads them and only
adjusts the brightness (and the position, as before) of each star.

Why (Recul M8 – 22 nov\\RECUL.md, Q1 step 1): track 1 redid on each crop its width, its zero point
(G -> flux), its share per colour, its core correction and its saturation. Its calibration stars were chosen by
« pixel > 10 noises », which the nebula fills on its own: on the core of M8 (tile a) width 7.48 px instead of 4.37,
zero point 25.57 instead of 22.98 (11 times too much flux), S/H/O shares 0.12/0.37/0.52 = the colour of the nebula.

Method (source): forced photometry, Lang, Hogg & Schlegel 2016 (« The Tractor », arXiv:1410.7397): the shape of
the star and the calibration belong to the image; over a given zone only the brightness of each source is free.

What is measured here, once, on the whole image:
  - width: that of the image's v7 star model (modele_psf_image.json), already measured on the whole image;
  - shape: the v7 model + the core correction (same computation as core_correction of track 1, two rounds, like
    passes 2 and 3), stacked on the calibration stars below instead of the crop's stars;
  - zero point, dispersion, share per colour (median F_c / F_total), and for colour images the flux per
    colour / BP - RP relation: on stars chosen by STAR criteria (not by « pixel > 10 noises »):
      Gaia catalogue, G 10-17, in the image and far from the edge, unsaturated (no v14 plateau pixel within 3 px, peak under
      0.4 x the saturation of each colour), isolated (no neighbour brighter than G+1 within 24 px, than G+3 within 2.5
      widths, than G+2 within 3 widths), position fitted to within 1 px of the Gaia position, point-like (peak / fitted
      flux between 0.7 and 1.4 times that of the model), quiet background (standard deviation of the medians of 8 sectors of the ring
      29-35 px under 3 sky noises of L);
  - saturation: the v14 detection per colour (plateaus >= (1 - eps) x maximum of the whole image); the « plateau
    of the 12 brightest of the crop » rule is no longer used.
Cache: Piste1/sorties/<image>/etalonnage_image.npz (+ .json), next to modele_psf_image.npz.
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
import modele_psf_image as MPI  # noqa: E402

VERSION = "v17-1"
G_RANGE = (10.0, 17.0)
RB = 36
BG_RING = (29.0, 35.0)


def _ci_image(name, xg, yg):
    """BP - RP du catalogue profond (aligné sur catalogue_image) ; NaN sans catalogue profond."""
    deep = RM.DEEP_DIR / f"gaia_dr3_profond_{name}.npz"
    if not deep.exists():
        return np.full(len(xg), np.nan)
    z = np.load(deep)
    ok_ = np.isfinite(z["Gmag"]) & np.isfinite(z["x_pix"]) & np.isfinite(z["y_pix"])
    return (z["BPmag"] - z["RPmag"])[ok_].astype(float)


def construire(name, log=print):
    t0 = time.time()
    img = images.load(name)
    C, H, W = img.shape
    valid = ndi.binary_erosion(np.any(img != 0, axis=0), iterations=3)
    xg, yg, G = RM.catalogue_image(name, log)
    ci_all = _ci_image(name, xg, yg)
    pr = dict(RM.P)

    # ---- saturation (v14, image entière) : niveaux et masque des plateaux
    sat_lvl, sat_m, sat_info = RM.plateaux_satures(img, pr["sat_eps"], pr["sat_min_px"])
    sat = np.stack([ndi.binary_dilation(sat_m[c], iterations=2) for c in range(C)])
    inb = (xg > 3) & (xg < W - 3) & (yg > 3) & (yg < H - 3)
    brightest = np.argsort(np.where(inb, G, 99))[:25]
    ref_lvl = np.zeros(C)
    for c in range(C):
        pk_ = np.array([img[c, int(yg[k]) - 2:int(yg[k]) + 3, int(xg[k]) - 2:int(xg[k]) + 3].max() for k in brightest if inb[k]])
        top_ = np.sort(pk_)[::-1][:12]
        ref_lvl[c] = sat_lvl[c] if np.isfinite(sat_lvl[c]) else (float(top_[1]) if len(top_) > 1 else float(img[c].max()))
    log(f"[étalonnage image] saturation v14 par couleur {np.round(sat_lvl, 1)} (max, plateaux, px : {sat_info}) ; "
        f"niveau de référence {np.round(ref_lvl, 1)}")

    # ---- bruit (même mesure que la piste 1, sur l'image entière)
    star_mask = np.zeros((H, W), bool)
    for x, y, g in zip(xg, yg, G):
        if g > 19:
            continue
        rr = int(np.clip(3 + 2.5 * (16 - g), 3, 60))
        cv2.circle(star_mask.view(np.uint8), (int(round(x)), int(round(y))), rr, 1, -1)
    sig, ncorr = RM.noise_stats(img, star_mask | ~valid)
    SIG = np.array([float(np.median(s)) for s in sig])
    log(f"[étalonnage image] bruit du ciel médian {np.round(SIG, 4)}, {time.time() - t0:.0f} s")

    # ---- largeur et forme : le modèle v7 de l'image
    m = MPI.obtenir(name, log)
    fwhm = float(m["info"]["fwhm"])
    R_anc = int(np.clip(np.ceil(5 * fwhm), 10, 40))
    E, _ = MPI.epsf_pour_decoupe(name, (0, 0), R_anc, log)
    os2 = E.os
    n_ = E.stamps[0][0].shape[0]
    c0_ = (n_ - 1) // 2
    ann_w = max(4.0, 1.5 * fwhm)
    P_fit = dict(pr, ann_w=ann_w)
    # pic du modèle par unité de flux (couleur du milieu, G 15) : test « ponctuelle »
    st_c = E.blend(1 if C > 1 else 0, W / 2, H / 2, 15.0)
    pf_model = float(st_c[c0_, c0_])

    # ---- étoiles d'étalonnage : critères d'étoile
    tree = cKDTree(np.c_[xg, yg])
    L = img.mean(0)
    yy, xx = np.mgrid[-RB:RB + 1, -RB:RB + 1]
    rr0 = np.hypot(xx, yy)
    ang = ((np.arctan2(yy, xx) + np.pi) / (2 * np.pi) * 8).astype(int) % 8
    ringm = (rr0 >= BG_RING[0]) & (rr0 <= BG_RING[1])
    rej = dict(bord=0, saturee=0, voisine=0, fond_agite=0, trop_forte=0, faible=0)
    cand = np.where((G >= G_RANGE[0]) & (G < G_RANGE[1]) & (xg > RB + 2) & (xg < W - RB - 3) & (yg > RB + 2) & (yg < H - RB - 3))[0]
    pre = []
    SIG_L = float(np.median(SIG)) / np.sqrt(C)
    for k in cand:
        xi, yi = int(round(xg[k])), int(round(yg[k]))
        if not valid[yi - RB:yi + RB + 1, xi - RB:xi + RB + 1].all():
            rej["bord"] += 1
            continue
        if sat[:, yi - 3:yi + 4, xi - 3:xi + 4].any():
            rej["saturee"] += 1
            continue
        bad = False
        for j in tree.query_ball_point([xg[k], yg[k]], 24.0):
            if j == k:
                continue
            d = np.hypot(xg[j] - xg[k], yg[j] - yg[k])
            if G[j] < G[k] + 1.0 or (d < 2.5 * fwhm and G[j] < G[k] + 3.0) or (d < 3 * fwhm and G[j] < G[k] + 2.0):
                bad = True
                break
        if bad:
            rej["voisine"] += 1
            continue
        cut = L[yi - RB:yi + RB + 1, xi - RB:xi + RB + 1]
        secs = np.array([np.median(cut[ringm & (ang == a)]) for a in range(8)])
        bgl = float(np.median(secs))
        pkL = float(L[yi - 1:yi + 2, xi - 1:xi + 2].max()) - bgl
        if pkL <= 5 * SIG_L:
            rej["faible"] += 1
            continue
        if float(np.std(secs)) > 3 * SIG_L:          # seuil indépendant de l'éclat mesuré (pas de biais vers les étoiles trop brillantes)
            rej["fond_agite"] += 1
            continue
        pkc = np.array([float(img[c, yi - 1:yi + 2, xi - 1:xi + 2].max()) for c in range(C)])
        if np.any(pkc >= 0.4 * ref_lvl):
            rej["trop_forte"] += 1
            continue
        pre.append(k)
    pre = np.array(pre, int)
    log(f"[étalonnage image] candidates G {G_RANGE[0]}-{G_RANGE[1]} : {len(cand)} ; gardées avant ajustement {len(pre)} ; "
        f"écartées {rej}, {time.time() - t0:.0f} s")

    def window(k, x, y, rf):
        xi, yi = int(round(x)), int(round(y))
        rw = int(np.ceil(rf + ann_w)) + 1
        sl = (slice(None), slice(yi - rw, yi + rw + 1), slice(xi - rw, xi + rw + 1))
        return xi, yi, rw, sl

    def fit(k, fix_pos, rf):
        """Ajustement de la piste 1 (fit_star, positions communes aux couleurs) : (F, x, y)."""
        xi, yi, rw, sl = window(k, xg[k], yg[k], rf)
        st = [E.blend(c, xg[k], yg[k], G[k]) for c in range(C)]
        F, nx, ny, s, chi = RM.fit_star(img[sl], valid[sl[1:]], sat[sl], E, st, xg[k] - (xi - rw), yg[k] - (yi - rw), None, 1.0,
                                        rf, fix_pos, False, sig[:, yi, xi], xg[k] - (xi - rw), yg[k] - (yi - rw), P_fit)
        return np.asarray(F, float), nx + xi - rw, ny + yi - rw

    # ponctuelle + position Gaia : ajustement à position libre (rayon des passes de la piste 1)
    rf_p = max(3.0, 2.0 * fwhm)
    sel, Fsel, Xsel, Ysel = [], [], [], []
    rej2 = dict(position=0, pas_ponctuelle=0, flux=0)
    QDIAG = []
    REJ_Q = []
    for k in pre:
        F, nx, ny = fit(k, False, rf_p)
        if not np.all(np.isfinite(F)) or F.min() <= 0:
            rej2["flux"] += 1
            continue
        if np.hypot(nx - xg[k], ny - yg[k]) > 1.0:
            rej2["position"] += 1
            continue
        xi, yi = int(round(nx)), int(round(ny))
        cut = L[yi - RB:yi + RB + 1, xi - RB:xi + RB + 1]
        bgl = float(np.median(cut[ringm]))
        pkL = float(L[yi, xi]) - bgl
        # pic attendu du modèle au pixel le plus proche du centre
        st = E.blend(1 if C > 1 else 0, nx, ny, G[k])
        pm = float(E.eval(st, np.array([xi - nx]), np.array([yi - ny]))[0]) * F.mean()
        q = pkL / max(pm, 1e-9)
        QDIAG.append((float(q), float(G[k]), float(F.min()), float(F.max())))
        if not (0.7 <= q <= 1.4):
            rej2["pas_ponctuelle"] += 1
            REJ_Q.append((k, F, nx, ny, q))
            continue
        sel.append(k); Fsel.append(F); Xsel.append(nx); Ysel.append(ny)
    if not sel and len(REJ_Q) >= 20:
        # v33b : aucune étoile ne passe le test « ponctuelle » à l'échelle absolue (modèle dont le pic n'est pas à l'échelle du flux
        # ajusté, ex. ic1318_stellina : q ~ 0,18 partout) ; le test se fait alors sur le rapport q / médiane de q (forme, pas échelle).
        qm = float(np.median([r[4] for r in REJ_Q]))
        for k, F, nx, ny, q in REJ_Q:
            if 0.7 <= q / qm <= 1.4:
                sel.append(k); Fsel.append(F); Xsel.append(nx); Ysel.append(ny)
                rej2["pas_ponctuelle"] -= 1
        log(f"[étalonnage image] aucune étoile au test absolu (q médian {qm:.3f}) : test sur q / médiane, {len(sel)} étoiles")
    sel = np.array(sel, int); Fsel = np.array(Fsel); Xsel = np.array(Xsel); Ysel = np.array(Ysel)
    if len(sel) == 0:
        raise RuntimeError("aucune étoile d'étalonnage")
    if QDIAG:
        qa = np.array(QDIAG)
        log(f"[étalonnage image] DIAG q quantiles 5/25/50/75/95 : {np.round(np.percentile(qa[:, 0], [5, 25, 50, 75, 95]), 3).tolist()} "
            f"; par G : " + " ".join(f"{g0}:{np.median(qa[(qa[:, 1] >= g0) & (qa[:, 1] < g0 + 1), 0]):.2f}"
                                    for g0 in range(10, 17) if ((qa[:, 1] >= g0) & (qa[:, 1] < g0 + 1)).any()))
    log(f"[étalonnage image] étoiles d'étalonnage : {len(sel)} (écartées après ajustement {rej2}), {time.time() - t0:.0f} s")

    # ---- correction du coeur (calcul de core_correction de la piste 1, sur ces étoiles), deux tours
    rcor = 2.5 * fwhm
    rr_ = int(np.ceil(rcor)) + 1
    gy_, gx_ = np.mgrid[-rr_:rr_ + 1, -rr_:rr_ + 1]
    tr_s = cKDTree(np.c_[xg, yg])
    corr_tot = np.zeros((C, n_, n_))
    shares = []
    for tour in range(2):
        cc_sel = []
        for i, k in enumerate(sel):
            if Fsel[i].sum() / C * pf_model < 20 * SIG.min():
                continue
            if any(G[j] < G[k] + 2 and j != k for j in tr_s.query_ball_point([Xsel[i], Ysel[i]], 3 * fwhm)):
                continue
            cc_sel.append(i)
        ch = []
        for c in range(C):
            acc = np.zeros(n_ * n_); acc2 = np.zeros(n_ * n_); cnt = np.zeros(n_ * n_)
            samples = []
            for i in cc_sel:
                k = sel[i]
                x, y = Xsel[i], Ysel[i]
                xi, yi = int(round(x)), int(round(y))
                cut = img[c, yi - rr_:yi + rr_ + 1, xi - rr_:xi + rr_ + 1].astype(np.float64)
                dx_ = gx_ + xi - x; dy_ = gy_ + yi - y
                r_ = np.hypot(dx_, dy_)
                st = E.blend(c, x, y, G[k])
                mod = Fsel[i][c] * E.eval(st, dx_, dy_)
                res = cut - mod
                ring_ = (r_ > rcor) & (r_ <= rr_ + 0.5)
                b_ = np.median(res[ring_])
                v_ = (res - b_) / Fsel[i][c]
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
            corr *= np.clip((rcor - rro) / 1.0, 0, 1)
            E.stamps[c][0] = E.stamps[c][0] + corr
            corr_tot[c] += corr
            ch.append(float(corr.sum() / os2 ** 2))
        shares.append(dict(tour=tour + 1, etoiles=len(cc_sel), part_flux=np.round(ch, 4).tolist()))
        log(f"[étalonnage image] correction du coeur, tour {tour + 1} : {len(cc_sel)} étoiles, part de flux ajoutée par couleur "
            f"{np.round(ch, 4).tolist()}")
        # flux réajustés avec le modèle corrigé (positions libres, comme la passe suivante de la piste 1)
        for i, k in enumerate(sel):
            F, nx, ny = fit(k, False, rf_p)
            if np.all(np.isfinite(F)):
                Fsel[i] = F; Xsel[i] = nx; Ysel[i] = ny

    # ---- point zéro et couleur : ajustement rapide de l'étalonnage de la piste 1 (position Gaia, rayon 1,5 largeur),
    # modèle corrigé
    rf_q = max(3.0, 1.5 * fwhm)
    Fq = np.array([fit(k, True, rf_q)[0] for k in sel])
    Ftot = Fq.sum(1)
    ok = np.all(np.isfinite(Fq), 1) & (Fq.min(1) > 0)
    zp = G[sel] + 2.5 * np.log10(np.where(Ftot > 0, Ftot, np.nan))
    keep = ok.copy()
    for _ in range(4):
        zp0 = float(np.median(zp[keep])); zsd = float(1.4826 * np.median(np.abs(zp[keep] - zp0)))
        keep = ok & (np.abs(zp - zp0) < 3 * zsd)
    zp0 = float(np.median(zp[keep])); zsd = float(1.4826 * np.median(np.abs(zp[keep] - zp0)))
    share = Fq[keep] / Ftot[keep, None]
    colour = np.median(share, 0)
    colour_sd = 1.4826 * np.median(np.abs(share - colour), 0)
    # largeur mesurée sur ces étoiles (même estimateur que la piste 1), contrôle seulement
    fw = []
    for i in np.where(keep)[0][:400]:
        xi, yi = int(round(Xsel[i])), int(round(Ysel[i]))
        cut = L[yi - 8:yi + 9, xi - 8:xi + 9]
        b = np.median(np.r_[cut[0], cut[-1], cut[:, 0], cut[:, -1]])
        v = cut - b
        if v.max() > 0:
            fw.append(2 * np.sqrt((v >= v.max() / 2).sum() / np.pi))
    fw_chk = float(np.median(fw)) if fw else float("nan")
    log(f"[étalonnage image] point zéro {zp0:.3f} ± {zsd:.3f} mag sur {int(keep.sum())} étoiles (modèle v7 : {m['info'].get('zp')}) ; "
        f"part par couleur {np.round(colour, 3)} ± {np.round(colour_sd, 3)} ; largeur figée {fwhm:.2f} px (sur ces étoiles {fw_chk:.2f})")

    # ---- flux attendu par couleur selon BP - RP (images couleur ; ancien FLOAT stars.py:predict_flux), sur ces étoiles
    cf_all = []
    sd_all = []
    ci = ci_all[sel]
    for c in range(C):
        with np.errstate(all="ignore"):
            yv = np.log10(Fq[:, c]) + 0.4 * G[sel]
        m_ = np.isfinite(yv) & (Fq[:, c] > 0) & np.isfinite(ci) & keep
        if m_.sum() < 20:
            cf_all.append([np.nan] * 4); sd_all.append(np.nan)
            continue
        kp = m_.copy()
        for _ in range(4):
            deg = 3 if kp.sum() > 60 else 1
            cf = np.polyfit(ci[kp], yv[kp], deg)
            rr_c = yv - np.polyval(cf, ci)
            sd = 1.4826 * np.median(np.abs(rr_c[kp]))
            kp = m_ & (np.abs(rr_c) < 3 * sd)
        cf_all.append(np.r_[np.zeros(4 - len(cf)), cf].tolist()); sd_all.append(float(sd / 0.4))
    info = dict(version=VERSION, image=name, fwhm=fwhm, fwhm_controle=fw_chk, zp=zp0, zp_sd=zsd, couleur=colour.tolist(),
                couleur_sd=colour_sd.tolist(), cf_couleur=cf_all, cf_dispersion_mag=sd_all, sat_lvl=sat_lvl.tolist(),
                sat_info=sat_info, ref_lvl=ref_lvl.tolist(), bruit=SIG.tolist(), etoiles=int(keep.sum()),
                ecartees_avant=rej, ecartees_apres=rej2, correction_coeur=shares, zp_modele_v7=m["info"].get("zp"),
                secondes=round(time.time() - t0))
    return dict(corr=corr_tot.astype(np.float32), xy=np.c_[Xsel[keep], Ysel[keep], G[sel][keep]].astype(np.float32),
                F=Fq[keep].astype(np.float32), info=info)


def obtenir(name, log=print):
    d = HERE / "sorties" / name
    d.mkdir(parents=True, exist_ok=True)
    f = d / "etalonnage_image.npz"
    fj = d / "etalonnage_image.json"
    if f.exists() and fj.exists():
        info = json.loads(fj.read_text(encoding="utf-8"))
        if info.get("version") == VERSION:
            z = np.load(f)
            log(f"[étalonnage image] relu du cache : point zéro {info['zp']:.3f} ± {info['zp_sd']:.3f}, part par couleur "
                f"{np.round(info['couleur'], 3)}, largeur {info['fwhm']:.2f} px, {info['etoiles']} étoiles")
            return dict(corr=z["corr"], info=info)
    m = construire(name, log)
    np.savez(f, corr=m["corr"], xy=m["xy"], F=m["F"])
    fj.write_text(json.dumps(m["info"], ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[étalonnage image] bâti en {m['info']['secondes']} s, gardé : {f}")
    return m
