# ======================================================================
# soustraire.py
# Rôle     : Piste 3, tour 4 : moins de pixels inventés.
# Étape    : 2. Chaîne classique de soustraction guidée par Gaia (v33b) (voir scripts/LISEZMOI.md)
# Entrées  : image empilée (unités de bruit), astrométrie, extrait Gaia profond hors ligne
# Sorties  : image sans étoiles, cartes de classes
# Exemple  : python soustraire.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Piste 3, tour 4 : moins de pixels inventés.
- halo d'une étoile brillante SOUSTRAIT : modèle radial mesuré sur l'étoile elle-même (médiane par anneau de la base,
  voisins masqués, fond pris juste au-delà du halo), lissé, ramené à zéro au bord ; c'est le principe de la soustraction
  des étoiles brillantes par profil radial étendu (Infante-Sainz, Trujillo & Román 2020, MNRAS 491, arXiv:1911.01430),
  ici avec le profil de chaque étoile plutôt qu'un profil commun empilé ;
- aigrettes SOUSTRAITES : gabarit empirique de chaque aigrette (profil le long, signé, et profil en travers moyen),
  retranché ; ce qui reste au-delà du seuil seulement est recréé ;
- étoiles moyennes laissées par la piste 1 avec un défaut : correction radiale (le défaut symétrique), puis le surcroît
  de bruit de photons de l'étoile ramené au bruit du ciel ; rien n'est inventé ;
- petits trous : réalisation gaussienne contrainte par l'anneau (Hoffman & Ribak 1991, ApJ 380 L5 ; Bucher & Louis 2012,
  arXiv:1109.0286 ; Benoit-Lévy et al. 2013, arXiv:1301.4145), sur la couche de détail ;
- variante : réseau E3 de FLOAT appliqué localement, sortie gardée seulement dans le trou, niveau et grain recalés.
"""
from __future__ import annotations

import numpy as np
import cv2
from scipy import ndimage as ndi

import mesure as M


def _ring_profile(img2d, x, y, rmax, bad):
    return M.radial_profile(img2d, x, y, rmax, bad)


def subtract_halo(B, s, bad, r_in, R):
    """Retire de B (C, H, W) le halo radial de l'étoile s entre r_in et R (voisins ``bad`` exclus de la mesure).
    Fond : médiane de l'anneau R..R+20. Profil lissé sur 5 px, nul à R (passage progressif de 0.8 R à R)."""
    C, H, W = B.shape
    x, y = s["x"], s["y"]
    Rm = int(R + 20)
    out = B.copy()
    x0, x1 = max(0, int(x) - Rm), min(W, int(x) + Rm + 1)
    y0, y1 = max(0, int(y) - Rm), min(H, int(y) + Rm + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.hypot(xx - x, yy - y)
    taper = np.clip((R - rr) / (0.2 * R), 0, 1) * (rr >= r_in - 2)
    model = np.zeros((C, y1 - y0, x1 - x0), np.float32)
    for c in range(C):
        p = _ring_profile(B[c], x, y, Rm, bad)
        bg = np.nanmedian(p[int(R):Rm + 1])
        h = np.nan_to_num(p - bg)
        h = ndi.uniform_filter1d(h, 5)
        h[:max(0, int(r_in) - 2)] = h[max(0, int(r_in) - 2)]
        # un halo d'étoile décroît avec le rayon et n'est pas négatif : profil rendu décroissant (minimum cumulé depuis le
        # centre) et >= 0 ; une volute de nébuleuse qui traverse un anneau ne peut plus entrer dans le modèle
        h = np.minimum.accumulate(np.clip(h, 0, None))
        model[c] = np.interp(rr, np.arange(len(h)), h, right=0) * taper
    out[:, y0:y1, x0:x1] -= model
    return out, model, (y0, y1, x0, x1)


def subtract_halo_secteurs(B, s, bad, r_in, R, n=8):
    """g4 (22 nov), géantes seulement : comme subtract_halo, mais un profil par secteur de 360/n degrés (halo non rond :
    reflets plus forts d'un côté), interpolé linéairement en angle entre les centres de secteurs."""
    C, H, W = B.shape
    x, y = s["x"], s["y"]
    Rm = int(R + 20)
    out = B.copy()
    x0, x1 = max(0, int(x) - Rm), min(W, int(x) + Rm + 1)
    y0, y1 = max(0, int(y) - Rm), min(H, int(y) + Rm + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.hypot(xx - x, yy - y)
    taper = np.clip((R - rr) / (0.2 * R), 0, 1) * (rr >= r_in - 2)
    YY, XX = np.mgrid[0:H, 0:W]
    TH = np.degrees(np.arctan2(YY - y, XX - x)) % 360
    step = 360.0 / n
    u = TH[y0:y1, x0:x1] / step - 0.5
    k0 = np.floor(u).astype(int) % n; k1 = (k0 + 1) % n; w = (u - np.floor(u)).astype(np.float32)
    model = np.zeros((C, y1 - y0, x1 - x0), np.float32)
    for c in range(C):
        hs = []
        for k in range(n):
            out_sect = ~((TH >= k * step) & (TH < (k + 1) * step))
            p = _ring_profile(B[c], x, y, Rm, bad | out_sect)
            bg = np.nanmedian(p[int(R):Rm + 1])
            h = np.nan_to_num(p - bg)
            h = ndi.uniform_filter1d(h, 5)
            h[:max(0, int(r_in) - 2)] = h[max(0, int(r_in) - 2)]
            h = np.minimum.accumulate(np.clip(h, 0, None))
            hs.append(h)
        ri = np.arange(len(hs[0]))
        m0 = np.stack([np.interp(rr, ri, hk, right=0) for hk in hs])                  # (n, h, w)
        a = np.take_along_axis(m0, k0[None], 0)[0]; b = np.take_along_axis(m0, k1[None], 0)[0]
        model[c] = ((1 - w) * a + w * b) * taper
    out[:, y0:y1, x0:x1] -= model
    return out, model, (y0, y1, x0, x1)


def subtract_quadrupole(B, s, bad, r_in, R, lissage=15):
    """g5 (22 nov), géantes seulement : après le halo radial, le reste garde un allongement en « 2 lobes » (reflets plus
    forts sur un axe). Par anneau de 1 px entre r_in et R, moindres carrés de a + c cos 2t + d sin 2t sur les pixels hors
    ``bad`` ; c et d lissés sur ``lissage`` px en rayon ; seul le terme en 2t est retranché (le niveau a et une pente de
    nébuleuse, terme en t, ne sont pas touchés), avec le même passage progressif vers R que le halo."""
    C, H, W = B.shape
    x, y = s["x"], s["y"]
    Rm = int(R)
    out = B.copy()
    x0, x1 = max(0, int(x) - Rm), min(W, int(x) + Rm + 1)
    y0, y1 = max(0, int(y) - Rm), min(H, int(y) + Rm + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.hypot(xx - x, yy - y); t2 = 2 * np.arctan2(yy - y, xx - x)
    cs, sn = np.cos(t2), np.sin(t2)
    ri = np.floor(rr).astype(int)
    ok = ~bad[y0:y1, x0:x1]
    taper = np.clip((R - rr) / (0.2 * R), 0, 1) * (rr >= r_in)
    for c in range(C):
        sub = B[c, y0:y1, x0:x1]
        co = np.zeros((Rm + 1, 2))
        for r in range(int(r_in), Rm + 1):
            m = ok & (ri == r)
            if m.sum() < 12:
                continue
            X = np.stack([np.ones(m.sum()), cs[m], sn[m]], 1)
            co[r] = np.linalg.lstsq(X, sub[m], rcond=None)[0][1:]
        co = ndi.uniform_filter1d(co, lissage, axis=0)
        rc = np.clip(ri, 0, Rm)
        out[c, y0:y1, x0:x1] = sub - (co[rc, 0] * cs + co[rc, 1] * sn) * taper
    return out


def subtract_spikes(B, s, sig, start_r):
    """Gabarit empirique de chaque aigrette de s, retranché de B. Renvoie B corrigé et le masque des tronçons où un reste
    dépasse encore le seuil (à recréer)."""
    C, H, W = B.shape
    out = B.copy()
    left = np.zeros((H, W), bool)
    for sp in s["spikes"]:
        a = np.deg2rad(sp["angle"]); L = int(sp["length"]) + 6; hw = int(sp["hw"]) + 2
        ux, uy = np.cos(a), np.sin(a); nx, ny = -uy, ux
        ds = np.arange(0, L, 1.0); ts = np.arange(-hw, hw + 1, 1.0)
        D_, T_ = np.meshgrid(ds, ts, indexing="ij")
        px = s["x"] + D_ * ux + T_ * nx; py = s["y"] + D_ * uy + T_ * ny
        side_off = [hw + 3, hw + 6, -(hw + 3), -(hw + 6)]
        for c in range(C):
            strip = ndi.map_coordinates(B[c], [py, px], order=1, mode="nearest")
            side = np.median([ndi.map_coordinates(B[c], [s["y"] + ds * uy + o * ny, s["x"] + ds * ux + o * nx], order=1, mode="nearest")
                              for o in side_off], 0)
            ex = strip - side[:, None]
            prof_along = ndi.uniform_filter1d(ex.mean(1), 7)                    # signé
            cross = ex[start_r:].mean(0) if L > start_r + 5 else ex.mean(0)
            # profil en travers : signe pris au centre (aigrette claire, ou sombre si sur-soustraite), normé à 1 au centre
            c0 = cross[np.abs(ts) <= 1].mean()
            if abs(c0) < 1e-6 or abs(c0) < 0.3 * np.abs(cross).max():
                cross = np.exp(-0.5 * (ts / max(1.0, sp["hw"] / 2)) ** 2)        # profil mal mesuré : gaussienne
            else:
                cross = np.clip(cross / c0, 0, 1.5)
            cross = cross / max(cross[np.abs(ts) <= 1].mean(), 1e-6)
            amp = ndi.uniform_filter1d(ex[:, np.abs(ts) <= 1].mean(1), 7)
            mod = amp[:, None] * cross[None, :]
            mod[:max(0, start_r - 2)] = 0
            # dépôt du gabarit dans l'image (plus proche pixel, chaque pixel une fois)
            xi = np.round(px).astype(int); yi = np.round(py).astype(int)
            ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            acc = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
            np.add.at(acc, (yi[ok], xi[ok]), mod[ok]); np.add.at(cnt, (yi[ok], xi[ok]), 1)
            m = cnt > 0
            out[c][m] -= acc[m] / cnt[m]
        # reste après soustraction, sur la luminance
        Lo = out.mean(0)
        cen = np.mean([ndi.map_coordinates(Lo, [s["y"] + ds * uy + o * ny, s["x"] + ds * ux + o * nx], order=1) for o in (-1, 0, 1)], 0)
        sd = np.median([ndi.map_coordinates(Lo, [s["y"] + ds * uy + o * ny, s["x"] + ds * ux + o * nx], order=1) for o in side_off], 0)
        e = np.abs(ndi.uniform_filter1d(cen - sd, 7))
        bad = (e > 2.5 * sig) & (ds >= start_r)
        for d in np.nonzero(bad)[0]:
            cv2.circle(left.view(np.uint8), (int(round(s["x"] + d * ux)), int(round(s["y"] + d * uy))), hw, 1, -1)
    return out, left


def medium_fix(B, x, y, bad, sky_mad, r_fix=12):
    """Défaut laissé par la soustraction sous une étoile moyenne : la partie radiale (médiane par anneau moins l'anneau
    r_fix..r_fix+5) est retranchée ; puis, dans le disque, l'excès de bruit (photons de l'étoile) est ramené au grain du
    ciel autour en réduisant le détail fin. Aucun pixel n'est remplacé."""
    C, H, W = B.shape
    out = B.copy()
    Rm = r_fix + 6
    x0, x1 = max(0, int(x) - Rm), min(W, int(x) + Rm + 1)
    y0, y1 = max(0, int(y) - Rm), min(H, int(y) + Rm + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.hypot(xx - x, yy - y)
    taper = np.clip((r_fix - rr) / 3.0, 0, 1)
    model = np.zeros((C,) + rr.shape, np.float32)
    for c in range(C):
        p = _ring_profile(B[c], x, y, Rm, bad)
        bg = np.nanmedian(p[r_fix:Rm + 1])
        h = np.nan_to_num(p - bg, nan=0.0)
        model[c] = np.interp(rr, np.arange(len(h)), h, right=0) * taper
    # la correction ne doit pas passer de l'autre côté : coeur moins anneau (0-4 / 8-12 px) ramené vers 0, pas au-delà
    core_, ann_ = rr <= 4, (rr >= 8) & (rr <= 12)
    if core_.any() and ann_.any():
        L0 = B[:, y0:y1, x0:x1].mean(0); Mm = model.mean(0)
        c0 = np.median(L0[core_]) - np.median(L0[ann_])
        dm = np.median(Mm[core_]) - np.median(Mm[ann_])
        if abs(dm) > 1e-9 and c0 / dm < 1.0:
            model *= float(np.clip(c0 / dm, 0.0, 1.0))
    out[:, y0:y1, x0:x1] -= model
    # garde-fou : la correction doit rapprocher le disque de son anneau (sinon, voisin mal masqué : on n'y touche pas)
    Lb0 = B[:, y0:y1, x0:x1].mean(0); Lb1 = out[:, y0:y1, x0:x1].mean(0)
    din_ = rr <= r_fix - 1; ring_ = (rr > r_fix) & (rr <= Rm)
    if din_.sum() > 5 and ring_.sum() > 5:
        e0 = abs(np.median(Lb0[din_]) - np.median(Lb0[ring_])) + np.abs(Lb0[din_] - np.median(Lb0[ring_])).mean()
        e1 = abs(np.median(Lb1[din_]) - np.median(Lb1[ring_])) + np.abs(Lb1[din_] - np.median(Lb1[ring_])).mean()
        if e1 >= e0:
            return B
    # bruit de photons : détail fin du disque ramené au grain de l'anneau s'il le dépasse
    Lb = out[:, y0:y1, x0:x1].mean(0)
    hp = Lb - cv2.GaussianBlur(Lb.astype(np.float32), (0, 0), 1.5)
    mad = lambda v: float(np.median(np.abs(v - np.median(v))) * 1.4826)
    din = rr <= r_fix - 2; ring = (rr > r_fix) & (rr <= Rm)
    if din.sum() > 10 and ring.sum() > 10:
        f = mad(hp[ring]) / max(mad(hp[din]), 1e-9)
        if f < 1.0:
            w = np.clip((r_fix - rr) / 3.0, 0, 1) * (1 - f)
            for c in range(C):
                blk = out[c, y0:y1, x0:x1]
                hpc = blk - cv2.GaussianBlur(blk.astype(np.float32), (0, 0), 1.5)
                out[c, y0:y1, x0:x1] = blk - w * hpc
    return out


def autocov(det, valid, maxlag=8):
    """Autocovariance radiale de chaque couleur du détail (pixels valides seulement, par FFT masquée)."""
    C = det.shape[0]
    w = valid.astype(np.float64)
    Fw = np.fft.rfft2(w)
    nw = np.fft.irfft2(Fw * np.conj(Fw), s=w.shape)
    out = []
    yy, xx = np.mgrid[-maxlag:maxlag + 1, -maxlag:maxlag + 1]
    rr = np.hypot(yy, xx)
    for c in range(C):
        d = np.where(valid, det[c] - det[c][valid].mean(), 0).astype(np.float64)
        Fd = np.fft.rfft2(d)
        ac = np.fft.irfft2(Fd * np.conj(Fd), s=d.shape) / np.maximum(nw, 1)
        ac = np.fft.fftshift(ac)
        cy, cx = np.array(ac.shape) // 2
        A = ac[cy - maxlag:cy + maxlag + 1, cx - maxlag:cx + maxlag + 1]
        prof = np.array([A[(rr >= k - 0.5) & (rr < k + 0.5)].mean() for k in range(maxlag + 1)])
        prof = prof * np.clip(1 - np.arange(maxlag + 1) / (maxlag + 1), 0, 1)      # fenêtre : matrice définie positive
        out.append(prof)
    return np.array(out)


def chan_corr(det, valid):
    v = det[:, valid]
    return np.corrcoef(v)


def gauss_fill(det, comp, known, acov, corr, rng, ring_w=6, nk_max=500):
    """Réalisation gaussienne contrainte du détail dans ``comp`` à partir des pixels ``known`` de l'anneau."""
    C, H, W = det.shape
    ys, xs = np.nonzero(comp)
    ring = ndi.binary_dilation(comp, iterations=ring_w) & ~comp & known
    ky, kx = np.nonzero(ring)
    if len(ky) > nk_max:
        sel = rng.choice(len(ky), nk_max, replace=False); ky, kx = ky[sel], kx[sel]
    if len(ky) < 10:
        return None
    maxlag = acov.shape[1] - 1
    def cov(ay, ax, by, bx, prof):
        d = np.hypot(ay[:, None] - by[None, :], ax[:, None] - bx[None, :])
        return np.interp(d, np.arange(maxlag + 1), prof, right=0.0)
    Lc = np.linalg.cholesky(corr + 1e-6 * np.eye(C))
    z = __import__('operator').matmul(Lc, rng.standard_normal((C, len(ys))))
    out = np.zeros((C, len(ys)))
    for c in range(C):
        # variance locale : celle de l'anneau (le grain recopié a la force du grain d'à côté)
        xk = det[c, ky, kx].astype(np.float64)
        prof = acov[c] * (max(np.var(xk), 1e-12) / max(acov[c][0], 1e-12))
        Skk = cov(ky, kx, ky, kx, prof) + 1e-6 * prof[0] * np.eye(len(ky))
        Shk = cov(ys, xs, ky, kx, prof)
        Shh = cov(ys, xs, ys, xs, prof)
        Kinv = np.linalg.solve(Skk, np.eye(len(ky)))
        mean = __import__('operator').matmul(Shk, (__import__('operator').matmul(Kinv, (xk - xk.mean())))) + xk.mean()
        post = Shh - __import__('operator').matmul(__import__('operator').matmul(Shk, Kinv), Shk.T)
        post = 0.5 * (post + post.T) + 1e-6 * prof[0] * np.eye(len(ys))
        try:
            Lp = np.linalg.cholesky(post)
        except np.linalg.LinAlgError:
            w_, V_ = np.linalg.eigh(post); Lp = V_ * np.sqrt(np.clip(w_, 0, None))
        out[c] = mean + __import__('operator').matmul(Lp, z[c])
    res = det.copy()
    res[:, ys, xs] = out
    return res
