# ======================================================================
# patchfill.py
# Rôle     : Remplissage par exemples (comme le remplissage « d'après le contenu » des éditeurs d'images).
# Étape    : 2. Chaîne classique de soustraction guidée par Gaia (v33b) (voir scripts/LISEZMOI.md)
# Entrées  : image empilée (unités de bruit), astrométrie, extrait Gaia profond hors ligne
# Sorties  : image sans étoiles, cartes de classes
# Exemple  : python patchfill.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Remplissage par exemples (comme le remplissage « d'après le contenu » des éditeurs d'images).

Complétion multi-échelle de Wexler, Shechtman & Irani 2007 (« Space-Time Completion of Video », PAMI 29(3)) avec la
recherche approchée du plus proche voisin de patch PatchMatch (Barnes, Shechtman, Finkelstein & Goldman 2009, SIGGRAPH),
la méthode du remplissage « d'après le contenu » de l'éditeur d'images de référence :
- pyramide de l'image et du trou ; au niveau le plus grossier, le trou part d'une interpolation lisse ;
- à chaque niveau, itérations E-M : PatchMatch (initialisation aléatoire ou montée du niveau d'avant, propagation,
  recherche aléatoire à rayon décroissant) trouve pour chaque patch qui touche le trou le patch connu le plus proche,
  puis chaque pixel du trou est le vote pondéré des patches qui le couvrent ;
- au niveau final, dernière passe sans moyenne (chaque pixel copie le centre de son patch voisin) : le grain copié est
  le vrai grain de l'image, pas une moyenne lissée.
La correspondance est commune aux couleurs (distance sur les couleurs étirées en asinh, en unités de bruit), les valeurs
copiées sont les valeurs linéaires de chaque couleur. Les sources : les pixels connus de la fenêtre autour du trou, hors
de toute zone interdite (autres trous, étoiles non retirées).
"""
from __future__ import annotations

import cv2
import numpy as np
from numba import njit
from scipy import ndimage as ndi

PATCH_R = 3          # patches 7 x 7


def _dist(F, ty, tx, sy, sx, r, best):
    C, H, W = F.shape
    d = 0.0
    for c in range(C):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                a = F[c, ty + dy, tx + dx] - F[c, sy + dy, sx + dx]
                d += a * a
        if d > best:
            return d
    return d
_dist = njit(cache=False)(_dist)


def _patchmatch(F, tgt_y, tgt_x, tmap, src_ok, nnf_y, nnf_x, nnd, r, iters, seed):
    """NNF des patches cibles (listés) vers les centres sources permis (src_ok)."""
    np.random.seed(seed)
    C, H, W = F.shape
    n = tgt_y.shape[0]
    # sources permises en liste pour les tirages
    cnt = 0
    for y in range(H):
        for x in range(W):
            if src_ok[y, x]:
                cnt += 1
    sy_l = np.empty(cnt, np.int32); sx_l = np.empty(cnt, np.int32)
    cnt = 0
    for y in range(H):
        for x in range(W):
            if src_ok[y, x]:
                sy_l[cnt] = y; sx_l[cnt] = x; cnt += 1
    if cnt == 0:
        return
    for i in range(n):
        ty, tx = tgt_y[i], tgt_x[i]
        if nnf_y[i] < 0 or not src_ok[nnf_y[i], nnf_x[i]]:
            j = np.random.randint(cnt)
            nnf_y[i] = sy_l[j]; nnf_x[i] = sx_l[j]
        nnd[i] = _dist(F, ty, tx, nnf_y[i], nnf_x[i], r, 1e30)
    maxR = max(H, W)
    for it in range(iters):
        if it % 2 == 0:
            start, stop, step = 0, n, 1
        else:
            start, stop, step = n - 1, -1, -1
        for ii in range(start, stop, step):
            ty, tx = tgt_y[ii], tgt_x[ii]
            # propagation depuis les voisins déjà vus (gauche/haut ou droite/bas)
            for k in range(2):
                if k == 0:
                    ny, nx = ty, tx - step
                else:
                    ny, nx = ty - step, tx
                if ny < 0 or nx < 0 or ny >= H or nx >= W:
                    continue
                j = tmap[ny, nx]
                if j < 0:
                    continue
                cy, cx = nnf_y[j] + (ty - ny), nnf_x[j] + (tx - nx)
                if cy < 0 or cx < 0 or cy >= H or cx >= W or not src_ok[cy, cx]:
                    continue
                d = _dist(F, ty, tx, cy, cx, r, nnd[ii])
                if d < nnd[ii]:
                    nnd[ii] = d; nnf_y[ii] = cy; nnf_x[ii] = cx
            # recherche aléatoire, rayon divisé par 2
            rad = maxR
            while rad >= 1:
                cy = nnf_y[ii] + np.random.randint(-rad, rad + 1)
                cx = nnf_x[ii] + np.random.randint(-rad, rad + 1)
                if 0 <= cy < H and 0 <= cx < W and src_ok[cy, cx]:
                    d = _dist(F, ty, tx, cy, cx, r, nnd[ii])
                    if d < nnd[ii]:
                        nnd[ii] = d; nnf_y[ii] = cy; nnf_x[ii] = cx
                rad //= 2
_patchmatch = njit(cache=False)(_patchmatch)


def _vote(V, hole, tmap, nnf_y, nnf_x, nnd, r, sig2, center_only):
    """Chaque pixel du trou : vote pondéré des patches qui le couvrent (ou copie du centre de son propre patch)."""
    C, H, W = V.shape
    out = V.copy()
    for y in range(H):
        for x in range(W):
            if not hole[y, x]:
                continue
            if center_only:
                j = tmap[y, x]
                if j >= 0:
                    for c in range(C):
                        out[c, y, x] = V[c, nnf_y[j], nnf_x[j]]
                continue
            acc = np.zeros(C); wsum = 0.0
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    qy, qx = y - dy, x - dx
                    if qy < 0 or qx < 0 or qy >= H or qx >= W:
                        continue
                    j = tmap[qy, qx]
                    if j < 0:
                        continue
                    sy, sx = nnf_y[j] + dy, nnf_x[j] + dx
                    w = np.exp(-nnd[j] / sig2)
                    for c in range(C):
                        acc[c] += w * V[c, sy, sx]
                    wsum += w
            if wsum > 0:
                for c in range(C):
                    out[c, y, x] = acc[c] / wsum
    return out
_vote = njit(cache=False)(_vote)


def _down(a):
    return np.stack([cv2.pyrDown(c) for c in a])


def _smooth_init(V, hole):
    out = V.copy()
    known = (~hole).astype(np.float32)
    for c in range(V.shape[0]):
        val = np.where(hole, 0, V[c]).astype(np.float32)
        est = None
        for s in (1, 2, 4, 8, 16, 32):
            w = cv2.GaussianBlur(known, (0, 0), s)
            e = cv2.GaussianBlur(val, (0, 0), s) / np.maximum(w, 1e-6)
            est = np.where(w > 0.2, e, np.nan) if est is None else np.where(np.isnan(est) & (w > 0.02), e, est)
        out[c][hole] = np.nan_to_num(est)[hole]
    return out


def complete(V, hole, forbid, feat_scale, iters_em=6, pm_iters=5, seed=0):
    """Complète le trou de V (C, h, w) linéaire. ``forbid`` : pixels qui ne doivent pas servir de source.
    ``feat_scale`` : unités du bruit par couleur (la distance se fait sur asinh(V / (3 x bruit)))."""
    r = PATCH_R
    C, H, W = V.shape
    # pyramide : jusqu'à ce que le trou fasse peu de patches de large
    levels = [(V.astype(np.float32), hole, forbid)]
    while True:
        v, h, f = levels[-1]
        dist = ndi.distance_transform_edt(h).max() if h.any() else 0
        if dist <= 2 * r or min(v.shape[1:]) < 8 * (2 * r + 1):
            break
        vd = _down(v)
        hd = cv2.resize(h.astype(np.uint8), (vd.shape[2], vd.shape[1]), interpolation=cv2.INTER_AREA) > 0
        hd |= ndi.maximum_filter(h, 2)[::2, ::2][:vd.shape[1], :vd.shape[2]]
        fd = ndi.maximum_filter(f, 2)[::2, ::2][:vd.shape[1], :vd.shape[2]]
        levels.append((vd, hd, fd))
    nnf = None
    cur = None
    for li in range(len(levels) - 1, -1, -1):
        v, h, f = levels[li]
        Cc, Hh, Ww = v.shape
        if cur is None:
            cur = _smooth_init(v, h)
        else:
            up = np.stack([cv2.resize(c, (Ww, Hh), interpolation=cv2.INTER_LINEAR) for c in cur])
            cur = np.where(h[None], up, v)
        pad = r
        # cibles : pixels à moins de r du trou ; sources : patch entier connu, hors interdits, hors bords
        tgt = ndi.binary_dilation(h, iterations=r)
        bad = ndi.binary_dilation(h | f, structure=np.ones((2 * r + 1, 2 * r + 1), bool))
        src_ok = ~bad
        src_ok[:pad] = src_ok[-pad:] = False; src_ok[:, :pad] = src_ok[:, -pad:] = False
        tgt[:pad] = tgt[-pad:] = False; tgt[:, :pad] = tgt[:, -pad:] = False
        if not src_ok.any():            # niveau sans source possible : on garde l'interpolation et on descend
            nnf = None
            continue
        ty, tx = np.nonzero(tgt)
        ty = ty.astype(np.int32); tx = tx.astype(np.int32)
        tmap = -np.ones((Hh, Ww), np.int32)
        tmap[ty, tx] = np.arange(len(ty), dtype=np.int32)
        nny = -np.ones(len(ty), np.int32); nnx = -np.ones(len(ty), np.int32)
        if nnf is not None:
            py, px, pmap = nnf
            # montée du NNF du niveau grossier : coordonnées x 2
            for i in range(len(ty)):
                yy, xx = ty[i] // 2, tx[i] // 2
                if yy < pmap.shape[0] and xx < pmap.shape[1] and pmap[yy, xx] >= 0:
                    j = pmap[yy, xx]
                    nny[i] = min(Hh - 1, 2 * py[j] + ty[i] % 2); nnx[i] = min(Ww - 1, 2 * px[j] + tx[i] % 2)
        nnd = np.zeros(len(ty), np.float64)
        sc = np.asarray(feat_scale, np.float32)[:, None, None]
        n_em = iters_em if li > 0 else iters_em + 1
        for it in range(n_em):
            Fe = np.arcsinh(cur / (3 * sc)).astype(np.float32)
            _patchmatch(Fe, ty, tx, tmap, src_ok, nny, nnx, nnd, r, pm_iters, seed + it + 17 * li)
            last = (li == 0 and it == n_em - 1)
            s2 = float(np.median(nnd)) + 1e-6
            cur = _vote(cur.astype(np.float64), h, tmap, nny, nnx, nnd, r, s2, last).astype(np.float32)
        nnf = (nny, nnx, tmap)
    return np.where(hole[None], cur, V)


def fill_mask(img, mask, forbid, noise, margin=None, log=None):
    """Remplit chaque zone connexe du masque dans une fenêtre autour d'elle (sources : la fenêtre hors interdits)."""
    out = img.copy()
    lab, n = ndi.label(mask)
    objs = ndi.find_objects(lab)
    C, H, W = img.shape
    for i, s in enumerate(objs):
        h_ = s[0].stop - s[0].start; w_ = s[1].stop - s[1].start
        m = margin or int(max(40, 2.5 * max(h_, w_)))
        y0, y1 = max(0, s[0].start - m), min(H, s[0].stop + m)
        x0, x1 = max(0, s[1].start - m), min(W, s[1].stop + m)
        lw = lab[y0:y1, x0:x1]
        hole = lw == i + 1
        # les zones pas encore remplies ne servent pas de source ; celles déjà remplies si
        fb = (forbid[y0:y1, x0:x1] | (lw > i + 1)) & ~hole
        win = out[:, y0:y1, x0:x1]
        filled = complete(win, hole, fb, noise, seed=i)
        out[:, y0:y1, x0:x1] = poisson_blend(win, filled, hole)
    return out


def poisson_blend(known, filled, hole):
    """Fondu dans le domaine du gradient (Pérez, Gangnet & Blake 2003, « Poisson Image Editing ») : dans le trou, on
    garde le gradient du remplissage et on impose au bord les valeurs connues autour ; revient à ajouter au remplissage
    la membrane harmonique qui annule l'écart au bord (Laplace, conditions de Dirichlet)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    H, W = hole.shape
    ys, xs = np.nonzero(hole)
    n = len(ys)
    if not n:
        return filled
    idx = -np.ones((H, W), np.int64)
    idx[ys, xs] = np.arange(n)
    rows, cols, vals = [], [], []
    C = filled.shape[0]
    bvec = np.zeros((C, n))
    deg = np.zeros(n)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ny, nx = ys + dy, xs + dx
        inside = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        nyc, nxc = np.clip(ny, 0, H - 1), np.clip(nx, 0, W - 1)
        nb_hole = inside & hole[nyc, nxc]
        nb_known = inside & ~hole[nyc, nxc]
        deg += nb_hole | nb_known
        rows.append(np.arange(n)[nb_hole]); cols.append(idx[nyc, nxc][nb_hole]); vals.append(-np.ones(int(nb_hole.sum())))
        for c in range(C):
            d = known[c, nyc, nxc] - filled[c, ys, xs]
            bvec[c, nb_known] += d[nb_known]
    rows.append(np.arange(n)); cols.append(np.arange(n)); vals.append(deg)
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    out = filled.copy()
    for c in range(C):
        out[c, ys, xs] = filled[c, ys, xs] + spsolve(A, bvec[c])
    return out
