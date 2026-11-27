# ======================================================================
# remplir.py
# Role     : Example-based filling (like the « content-aware » fill of image editors).
# Step     : 2. Classic Gaia-guided subtraction chain (v33b) (see scripts/LISEZMOI.md)
# Inputs   : stacked image (noise units), astrometry, offline deep Gaia extract
# Outputs  : star-free image, class maps
# Example  : python remplir.py [arguments]  (see the docstring; paths through the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Example-based filling (like the « content-aware » fill of image editors).

Multi-scale completion of Wexler, Shechtman & Irani 2007 (« Space-Time Completion of Video », PAMI 29(3)) with the
approximate nearest-neighbour patch search PatchMatch (Barnes, Shechtman, Finkelstein & Goldman 2009, SIGGRAPH),
the method of the « content-aware » fill of the reference image editor:
- pyramid of the image and of the hole; at the coarsest level, the hole starts from a smooth interpolation;
- at each level, E-M iterations: PatchMatch (random initialisation or upscaled from the previous level, propagation,
  random search with decreasing radius) finds, for each patch touching the hole, the nearest known patch,
  then each pixel of the hole is the weighted vote of the patches covering it;
- at the final level, a last pass without averaging (each pixel copies the centre of its neighbouring patch): the copied grain is
  the image's real grain, not a smoothed average.
The matching is common to the colours (distance on the asinh-stretched colours, in units of noise), the copied values
are the linear values of each colour. The sources: the known pixels of the window around the hole, outside
any forbidden zone (other holes, stars not removed).
"""
from __future__ import annotations

import cv2
import numpy as np
from numba import njit
from scipy import ndimage as ndi

PATCH_R = 3          # patches 7 x 7
FEAT_BLUR = 1.2      # flou des traits de comparaison (px)
HP_SIGMA = 3.0       # tour 4 : coupure de la texture copiée dans les trous durs (px)
BIHARMONIC = False   # trous durs du tour 4 : la membrane harmonique (principe du maximum : jamais plus bas que son
                     # pourtour) ; le biharmonique creusait des cuvettes sous les coeurs (creux -29 % à G 7.20)
GAUSS_MAX = 400      # trous jusqu'à 400 px : réalisation gaussienne contrainte (variante)
GRAIN_W = 10.0       # 10 % d'écart de grain pèse comme 1 bruit d'écart de niveau
GUIDE_W = 0.25       # poids du guidage par la structure (écart de niveau en bruits, au carré)
# Diagnostic 23 nov (défaut 1, taches saturées) : l'échelle du bruit synthétisé était mesurée sur les seuls pixels de la
# zone ; sur une zone de 1-19 px la dispersion vaut 0 et l'échelle cible / 1e-9 envoie un pixel à ±1e7. Le champ de bruit
# est stationnaire : son écart type se mesure sur toute la fenêtre synthétisée. DIAG_GRAIN_FENETRE=0 rend l'ancien calcul.
import os as _os
GRAIN_ECHELLE_FENETRE = _os.environ.get("DIAG_GRAIN_FENETRE", "1") == "1"


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
            # distance sur une version adoucie (1 px) : le grain n'entre pas dans la comparaison, sinon un patch lisse
            # « ressemble » plus à la cible lissée par les votes qu'un vrai patch bruité (biais vers le lisse) ;
            # les pixels copiés restent les pixels bruts
            Fe = np.arcsinh(np.stack([cv2.GaussianBlur(c, (0, 0), FEAT_BLUR) for c in cur]) / (3 * sc)).astype(np.float32)
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


def _laplace_fill(V, U):
    """Membrane harmonique : dans U, solution de Laplace avec les valeurs de V autour (Dirichlet), par couleur."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import splu
    H, W = U.shape
    ys, xs = np.nonzero(U)
    n = len(ys)
    out = V.copy()
    if not n:
        return out
    idx = -np.ones((H, W), np.int64); idx[ys, xs] = np.arange(n)
    rows, cols, vals = [], [], []
    deg = np.zeros(n); bvec = np.zeros((V.shape[0], n))
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ny, nx = ys + dy, xs + dx
        inside = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        nyc, nxc = np.clip(ny, 0, H - 1), np.clip(nx, 0, W - 1)
        nb_u = inside & U[nyc, nxc]; nb_k = inside & ~U[nyc, nxc]
        deg += inside
        rows.append(np.arange(n)[nb_u]); cols.append(idx[nyc, nxc][nb_u]); vals.append(-np.ones(int(nb_u.sum())))
        for c in range(V.shape[0]):
            bvec[c, nb_k] += V[c, nyc, nxc][nb_k]
    rows.append(np.arange(n)); cols.append(np.arange(n)); vals.append(deg)
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsc()
    lu = splu(A)
    for c in range(V.shape[0]):
        out[c, ys, xs] = lu.solve(bvec[c])
    return out


def lowpass_masked(V, bad, s):
    """Flou gaussien normalisé qui ignore les pixels ``bad``."""
    w = cv2.GaussianBlur((~bad).astype(np.float32), (0, 0), s)
    out = np.empty_like(V)
    for c in range(V.shape[0]):
        out[c] = cv2.GaussianBlur(np.where(bad, 0, V[c]).astype(np.float32), (0, 0), s) / np.maximum(w, 1e-6)
    return out


def structure_texture(V, hard, soft, forbid, noise, low_sigma=6.0, log=None):
    """Remplissage en deux couches (structure / texture, Bertalmio, Vese, Sapiro & Osher 2003, « Simultaneous
    structure and texture image inpainting », IEEE TIP 12(8)) :
    - structure (basse fréquence, flou de ``low_sigma`` px) : refaite par membrane harmonique sur dur + doux, à partir
      des valeurs autour ; c'est là qu'est le halo de l'étoile ;
    - texture (détail + grain) : gardée telle quelle dans le doux (elle y est vraie), recréée dans le dur par le
      remplissage d'après le contenu (PatchMatch multi-échelle, copie du grain, fondu de Poisson)."""
    U = hard | soft
    low = lowpass_masked(V, hard | soft, low_sigma)
    low_in = lowpass_masked(V, hard, low_sigma)     # basse fréquence du doux (avec halo) pour en tirer le détail
    new_low = _laplace_fill(low, U)
    out = V.copy()
    det = V - low_in
    out[:, soft] = (new_low + det)[:, soft]
    out[:, hard] = new_low[:, hard]
    if log:
        log(f"  structure refaite sur {int(U.sum())} px (dont dur {int(hard.sum())} px)")
    return fill_mask(out, hard, forbid & ~hard, noise)


def _criminisi(F, V, Gd, wg, hole, src_ok, r, n_rand, seed):
    """Remplissage par exemples dans l'ordre de priorité de Criminisi, Pérez & Toyama 2004 (« Region filling and object
    removal by exemplar-based image inpainting », IEEE TIP 13(9)), terme de confiance seul (le détail n'a pas
    d'isophote utile) : on remplit d'abord le point du front le mieux entouré ; le patch source est cherché parmi des
    tirages au hasard et les candidats cohérents des voisins déjà copiés (propagation de PatchMatch, Barnes 2009) ;
    on COPIE les pixels (aucune moyenne) : le grain du remplissage est le vrai grain de l'image."""
    np.random.seed(seed)
    C, H, W = F.shape
    known = ~hole
    conf = np.where(hole, 0.0, 1.0)
    offy = np.zeros((H, W), np.int32); offx = np.zeros((H, W), np.int32); hasoff = np.zeros((H, W), np.bool_)
    cnt = 0
    for y in range(H):
        for x in range(W):
            if src_ok[y, x]:
                cnt += 1
    sy = np.empty(cnt, np.int32); sx = np.empty(cnt, np.int32)
    cnt = 0
    for y in range(H):
        for x in range(W):
            if src_ok[y, x]:
                sy[cnt] = y; sx[cnt] = x; cnt += 1
    if cnt == 0:
        return V
    out = V.copy()
    Fo = F.copy()
    y0, y1, x0, x1 = H, 0, W, 0
    left = 0
    for y in range(H):
        for x in range(W):
            if hole[y, x]:
                left += 1
                y0 = min(y0, y); y1 = max(y1, y); x0 = min(x0, x); x1 = max(x1, x)
    while left > 0:
        # point du front le plus sûr
        best = -1.0; py = -1; px = -1
        for y in range(max(r, y0), min(H - r, y1 + 1)):
            for x in range(max(r, x0), min(W - r, x1 + 1)):
                if known[y, x]:
                    continue
                front = False
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if known[y + dy, x + dx]:
                            front = True
                if not front:
                    continue
                c = 0.0
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        c += conf[y + dy, x + dx]
                if c > best:
                    best = c; py = y; px = x
        if py < 0:
            # trou collé au bord de la fenêtre : on garde la valeur de départ
            for y in range(H):
                for x in range(W):
                    if not known[y, x]:
                        known[y, x] = True
            break
        # recherche : candidats cohérents des voisins + tirages
        bd = 1e30; by = -1; bx = -1
        for k in range(n_rand + (2 * r + 1) * (2 * r + 1)):
            if k < (2 * r + 1) * (2 * r + 1):
                dy = k // (2 * r + 1) - r; dx = k % (2 * r + 1) - r
                qy = py + dy; qx = px + dx
                if not hasoff[qy, qx]:
                    continue
                cy = py + offy[qy, qx]; cx = px + offx[qy, qx]
                if cy < 0 or cx < 0 or cy >= H or cx >= W or not src_ok[cy, cx]:
                    continue
            else:
                j = np.random.randint(cnt)
                cy = sy[j]; cx = sx[j]
            d = 0.0; n = 0
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if known[py + dy, px + dx]:
                        for c in range(C):
                            a = Fo[c, py + dy, px + dx] - F[c, cy + dy, cx + dx]
                            d += a * a
                        n += 1
                if d > bd * max(n, 1) * 4:
                    break
            if n > 0:
                d /= n
                # guidage par la structure : le patch source doit avoir le même niveau (basse fréquence) que la cible
                g = 0.0
                for c in range(C):
                    a = Gd[c, py, px] - Gd[c, cy, cx]
                    g += a * a
                d += wg * g
                if d < bd:
                    bd = d; by = cy; bx = cx
        cc = best / ((2 * r + 1) * (2 * r + 1))
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                yy = py + dy; xx = px + dx
                if not known[yy, xx]:
                    for c in range(C):
                        out[c, yy, xx] = V[c, by + dy, bx + dx]
                        Fo[c, yy, xx] = F[c, by + dy, bx + dx]
                    known[yy, xx] = True; conf[yy, xx] = cc
                    offy[yy, xx] = by - py; offx[yy, xx] = bx - px; hasoff[yy, xx] = True
                    left -= 1
    return out
_criminisi = njit(cache=False)(_criminisi)


def fill_detail(Dv, hole, forbid, noise, guide=None, wg=1.0, r=4, n_rand=4000, margin=None, center_forbid=None):
    """Remplit le DÉTAIL (couche texture) de chaque zone connexe du trou par _criminisi, dans une fenêtre autour."""
    out = Dv.copy()
    lab, n = ndi.label(hole)
    objs = ndi.find_objects(lab)
    C, H, W = Dv.shape
    sc = np.asarray(noise, np.float32)[:, None, None]
    for i, s in enumerate(objs):
        h_ = s[0].stop - s[0].start; w_ = s[1].stop - s[1].start
        m = margin or int(max(60, 1.5 * max(h_, w_)))
        y0, y1 = max(0, s[0].start - m), min(H, s[0].stop + m)
        x0, x1 = max(0, s[1].start - m), min(W, s[1].stop + m)
        lw = lab[y0:y1, x0:x1]
        hw = lw == i + 1
        bad = ndi.binary_dilation(hole[y0:y1, x0:x1] | forbid[y0:y1, x0:x1], structure=np.ones((2 * r + 1, 2 * r + 1), bool))
        src = ~bad
        if center_forbid is not None:      # centres de patch interdits (sans élargissement : catalogue profond)
            src &= ~center_forbid[y0:y1, x0:x1]
        src[:r] = src[-r:] = False; src[:, :r] = src[:, -r:] = False
        win = out[:, y0:y1, x0:x1].astype(np.float64)
        Fw = (win / sc).astype(np.float64)
        Gw = guide[:, y0:y1, x0:x1].astype(np.float64) if guide is not None else np.zeros((1,) + Fw.shape[1:])
        res = _criminisi(Fw, win, Gw, wg, hw, src, r, n_rand, i)
        out[:, y0:y1, x0:x1] = np.where(hw[None], res, out[:, y0:y1, x0:x1])
    return out


def structure_texture2(V, hard, soft, forbid, noise, low_sigma=6.0, log=None, giants=(), filler="patch", ring_excl=None, center_forbid=None):
    """Structure (basse fréquence) : membrane harmonique sur dur + doux. Texture (détail) : gardée dans le doux,
    copiée par exemples (ordre de Criminisi, recherche PatchMatch) dans le dur. Somme des deux."""
    U = hard | soft
    low = lowpass_masked(V, U, low_sigma)
    new_low = _laplace_fill(low, U)
    if not soft.any() and BIHARMONIC:
        # sans halo (tour 4) : chaque trou dur moyen est refait en biharmonique, qui prolonge pente et courbure du
        # pourtour (Damelin & Hoang 2018, « On surface completion and image inpainting by biharmonic functions ») :
        # pas de disque plat sur une lueur bombée (coeur de nébuleuse)
        from skimage.restoration import inpaint_biharmonic
        lab_b, n_b = ndi.label(hard)
        for k, sl in enumerate(ndi.find_objects(lab_b), 1):
            comp = lab_b[sl] == k
            if comp.sum() > 3000:
                continue
            y0, y1 = max(0, sl[0].start - 12), min(hard.shape[0], sl[0].stop + 12)
            x0, x1 = max(0, sl[1].start - 12), min(hard.shape[1], sl[1].stop + 12)
            cm = lab_b[y0:y1, x0:x1] == k
            for c in range(V.shape[0]):
                blk = low[c, y0:y1, x0:x1].astype(np.float64)
                new_low[c, y0:y1, x0:x1][cm] = inpaint_biharmonic(blk, cm)[cm]
    low_in = lowpass_masked(V, hard, low_sigma)
    det = V - low_in
    det0 = det.copy()
    taper = np.zeros(hard.shape, np.float32)
    for (gcx, gcy, rmin, rmax) in giants:
        Hh, Ww = hard.shape
        yy, xx = np.mgrid[0:Hh, 0:Ww]
        rr = np.hypot(xx - gcx, yy - gcy)
        rmax_d = 1.6 * rmax                      # les rayons dépassent le halo mesuré : on les suit plus loin
        disc = (rr <= rmax_d) & (rr >= rmin)
        det = polar_destripe(det, disc, gcx, gcy, rmin, rmax_d)
        taper = np.maximum(taper, np.clip((rmax_d - rr) / (0.2 * rmax), 0, 1) * disc)
        if log:
            log(f"  rayons de la géante retirés du détail (polaire) sur {int(disc.sum())} px")
    ray = det0 - det
    det_keep = det.copy()
    det[:, hard] = 0
    # guidage : niveau (basse fréquence refaite, en bruits) et énergie locale du grain (log de l'écart type local du
    # détail, interpolée dans le trou comme la structure) : le grain recopié a la force du grain d'à côté
    sc = np.asarray(noise, np.float32)[:, None, None]
    E = 0.5 * np.log(cv2.GaussianBlur((det ** 2).mean(0).astype(np.float32), (0, 0), 4.0) + 1e-12)
    Eb = cv2.GaussianBlur((~hard).astype(np.float32), (0, 0), 4.0)
    E = np.where(Eb > 0.5, E - 0.5 * np.log(np.maximum(Eb, 1e-6)), np.nan)      # bord du trou : correction du flou
    Ef = _laplace_fill(np.nan_to_num(E, nan=0.0)[None], hard | np.isnan(E))[0]
    guide = np.concatenate([new_low / sc, GRAIN_W * Ef[None]], 0)
    if filler == "gauss":
        # petits trous (<= GAUSS_MAX px) : réalisation gaussienne contrainte ; grands : copie par exemples
        import soustraire as S4
        lab_, n_ = ndi.label(hard)
        sizes_ = ndi.sum(hard, lab_, np.arange(1, n_ + 1)) if n_ else np.zeros(0)
        small = np.isin(lab_, 1 + np.nonzero(sizes_ <= GAUSS_MAX)[0]) & hard
        big = hard & ~small
        det_f = fill_detail(det, big, forbid & ~hard, noise, guide=guide, wg=GUIDE_W) if big.any() else det.copy()
        valid = ~ndi.binary_dilation(hard | forbid | soft, iterations=3)
        acov = S4.autocov(det, valid); corr = S4.chan_corr(det, valid)
        rng = np.random.default_rng(7)
        for k, sl in enumerate(ndi.find_objects(lab_), 1):
            if sizes_[k - 1] > GAUSS_MAX:
                continue
            y0, y1 = max(0, sl[0].start - 10), min(hard.shape[0], sl[0].stop + 10)
            x0, x1 = max(0, sl[1].start - 10), min(hard.shape[1], sl[1].stop + 10)
            comp = lab_[y0:y1, x0:x1] == k
            known = ~(hard | forbid)[y0:y1, x0:x1]
            r_ = S4.gauss_fill(det_f[:, y0:y1, x0:x1], comp, known, acov, corr, rng)
            if r_ is not None:
                det_f[:, y0:y1, x0:x1] = r_
        if log:
            log(f"  réalisation gaussienne contrainte : {int(small.sum())} px en {int((sizes_ <= GAUSS_MAX).sum())} trous ; copie par exemples : {int(big.sum())} px")
    else:
        det_f = fill_detail(det, hard, forbid & ~hard, noise, guide=guide, wg=GUIDE_W, center_forbid=center_forbid)
    if not soft.any():
        # tour 4 : dans un trou dur, le niveau vient du pourtour (surface harmonique) ; des patches copiés on ne garde que
        # la texture fine (au-dessous de ~3 largeurs à mi-hauteur : passe-haut, flou de 3 px), pour que le résultat ne
        # dépende plus des patches tirés ; l'accord du grain se fait ensuite, sur cette texture
        hpf = det_f - np.stack([cv2.GaussianBlur(c.astype(np.float32), (0, 0), HP_SIGMA) for c in det_f])
        det_f = np.where(hard[None], hpf, det_f)
    # accord du grain par zone (accord des statistiques de la texture, Heeger & Bergen 1995 ; Kaspar et al. 2015 contre
    # l'affadissement) : l'écart type robuste du grain fin recopié est ramené à celui de l'anneau autour (4-16 px)
    hpD = det_f.mean(0) - cv2.GaussianBlur(det_f.mean(0).astype(np.float32), (0, 0), 2.0)
    hpV = V.mean(0) - cv2.GaussianBlur(V.mean(0).astype(np.float32), (0, 0), 2.0)
    mad = lambda v: float(np.median(np.abs(v - np.median(v))) * 1.4826)
    lab, n = ndi.label(hard)
    scales = []
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        y0, y1 = max(0, sl[0].start - 16), min(hard.shape[0], sl[0].stop + 16)
        x0, x1 = max(0, sl[1].start - 16), min(hard.shape[1], sl[1].stop + 16)
        comp = lab[y0:y1, x0:x1] == k
        inner = ndi.binary_erosion(comp, iterations=2)
        if inner.sum() < 20:
            continue
        ring = ndi.binary_dilation(comp, iterations=16) & ~ndi.binary_dilation(comp, iterations=4) & ~(U | forbid)[y0:y1, x0:x1]
        if ring.sum() < 40:
            ring = ndi.binary_dilation(comp, iterations=16) & ~ndi.binary_dilation(comp, iterations=4) & ~hard[y0:y1, x0:x1]
        if ring.sum() < 40:
            continue
        f = mad(hpV[y0:y1, x0:x1][ring]) / max(mad(hpD[y0:y1, x0:x1][inner]), 1e-9)
        f = float(np.clip(f, 0.5, 2.2 if soft.any() else 1.2))   # tour 4 : au-delà, le manque est ajouté en bruit
        scales.append(f)
        blk = det_f[:, y0:y1, x0:x1]
        blk[:, comp] = blk[:, comp] * f
    if log and scales:
        log(f"  accord du grain : {len(scales)} zones, facteur médian {np.median(scales):.2f} (min {min(scales):.2f}, max {max(scales):.2f})")
    if not soft.any():
        det_f = add_missing_grain(det_f, V, hard, (U | ring_excl) if ring_excl is not None else (U | forbid), log)
    out = V - taper[None] * ray                  # hors halo : rayons retirés (nul sans géante)
    out[:, U] = (new_low + det_f)[:, U]
    # bord extérieur du halo : passage progressif sur 8 px (pas de marche entre la membrane et le flou local)
    dist = ndi.distance_transform_edt(U)
    rw = max([8.0] + [0.15 * g[3] for g in giants])     # géante : halo encore en pente au bord, passage plus long
    w = np.clip(dist / rw, 0, 1)
    ramp = soft & (w < 1)
    out[:, ramp] = (low_in + w[None] * (new_low - low_in) + det_keep)[:, ramp]
    if log:
        log(f"  structure refaite sur {int(U.sum())} px, détail copié sur {int(hard.sum())} px")
    return out


def polar_destripe(det, zone, cx, cy, rmin, rmax, win=31):
    """Rayons en éventail d'une géante (traits radiaux) retirés de la couche de détail : en coordonnées polaires centrées
    sur l'étoile, un rayon est une colonne à angle fixe ; sa trace est la médiane glissante le long du rayon (fenêtre
    ``win`` px), qu'on retranche (correction de rayures en polaire, comme pour les anneaux en tomographie : Sijbers &
    Postnov 2004, Phys. Med. Biol. 49). Le détail de la nébuleuse, sans direction radiale, a une médiane nulle."""
    from scipy.ndimage import median_filter
    C, H, W = det.shape
    nth = int(2 * np.pi * rmax)
    th = np.linspace(0, 2 * np.pi, nth, endpoint=False)
    rs = np.arange(rmin, rmax + 1, 1.0)
    R_, T_ = np.meshgrid(rs, th, indexing="ij")
    yy = cy + R_ * np.sin(T_); xx = cx + R_ * np.cos(T_)
    out = det.copy()
    ys, xs = np.nonzero(zone)
    rp = np.hypot(xs - cx, ys - cy); tp = np.mod(np.arctan2(ys - cy, xs - cx), 2 * np.pi)
    ir = np.clip(rp - rmin, 0, len(rs) - 1); it = tp / (2 * np.pi) * nth
    for c in range(C):
        P = ndi.map_coordinates(det[c], [yy, xx], order=1, mode="nearest")
        ray = median_filter(P, size=(win, 1), mode="nearest")
        ray = ndi.uniform_filter1d(ray, 3, axis=1, mode="wrap")
        v = ndi.map_coordinates(ray, [ir, it], order=1, mode="wrap")
        out[c, ys, xs] = det[c, ys, xs] - v
    return out



def rim(comp, excl, hard):
    """Anneau de référence d'une zone recréée (4-16 px autour), commun à la correction du grain et à sa mesure :
    pixels hors ``excl`` (étoiles, masques, zones recréées) ; zone serrée : hors zones dures seulement."""
    ring = ndi.binary_dilation(comp, iterations=16) & ~ndi.binary_dilation(comp, iterations=4)
    r1 = ring & ~excl
    return r1 if r1.sum() >= 40 else ring & ~hard


def add_missing_grain(det_f, V, hard, notring, log=None, hp_s=2.0, lo=0.95, hi=1.05):
    """Grain manquant d'une zone recréée (même bande passe-haut que la mesure, flou 2 px) : la part qui manque,
    d'écart type s_anneau * sqrt(1 - r^2), est ajoutée en bruit de même corrélation que l'anneau (bruit blanc filtré par
    la racine du spectre de puissance du passe-haut de l'anneau, couleurs corrélées comme dans l'anneau)."""
    import soustraire as S4
    C, H, W = det_f.shape
    hpV = np.stack([c - cv2.GaussianBlur(c.astype(np.float32), (0, 0), hp_s) for c in V.astype(np.float32)])
    mad = lambda v: float(np.median(np.abs(v - np.median(v))) * 1.4826)
    rng = np.random.default_rng(11)
    lab, n = ndi.label(hard)
    added = []
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        y0, y1 = max(0, sl[0].start - 20), min(H, sl[0].stop + 20)
        x0, x1 = max(0, sl[1].start - 20), min(W, sl[1].stop + 20)
        comp = lab[y0:y1, x0:x1] == k
        inner = ndi.binary_erosion(comp, iterations=2)
        if inner.sum() < 20:
            inner = comp
        ring = rim(comp, notring[y0:y1, x0:x1], hard[y0:y1, x0:x1])
        if ring.sum() < 40:
            continue
        blk = det_f[:, y0:y1, x0:x1]
        hpD = np.stack([c - cv2.GaussianBlur(c.astype(np.float32), (0, 0), hp_s) for c in blk.astype(np.float32)])
        sr = mad(hpV[:, y0:y1, x0:x1].mean(0)[ring]); sd = mad(hpD.mean(0)[inner])
        r = sd / max(sr, 1e-9)
        if r >= lo:
            continue
        # spectre de l'anneau : autocovariance (radiale, 8 px) du passe-haut, par couleur ; corrélation entre couleurs
        ac = S4.autocov(hpV[:, y0:y1, x0:x1], ring, maxlag=8)
        corr = S4.chan_corr(hpV[:, y0:y1, x0:x1], ring)
        hh, ww = blk.shape[1:]
        yy, xx = np.meshgrid(np.fft.fftfreq(hh) * hh, np.fft.fftfreq(ww) * ww, indexing="ij")
        dist = np.hypot(yy, xx)
        Lc = np.linalg.cholesky(corr + 1e-6 * np.eye(C))
        wn = __import__('operator').matmul(Lc, rng.standard_normal((C, hh * ww)))
        wn = wn.reshape(C, hh, ww)
        noise = np.empty_like(wn)
        for c in range(C):
            A = np.interp(dist, np.arange(ac.shape[1]), ac[c], right=0.0)       # autocovariance sur la grille
            P = np.clip(np.real(np.fft.fft2(A)), 0, None)
            noise[c] = np.real(np.fft.ifft2(np.fft.fft2(wn[c]) * np.sqrt(P)))
        nh = noise.mean(0) - cv2.GaussianBlur(noise.mean(0).astype(np.float32), (0, 0), hp_s)
        target = sr * np.sqrt(max(0.0, 1 - r * r))
        noise *= target / max(mad(nh if GRAIN_ECHELLE_FENETRE else nh[inner]), 1e-9)
        blk[:, comp] = blk[:, comp] + noise[:, comp]
        added.append(r)
    if log and added:
        log(f"  grain manquant ajouté en bruit corrélé : {len(added)} zones (rapport avant : médiane {np.median(added):.2f}, min {min(added):.2f})")
    return det_f


def relevel_lowfreq(V, zone, low_sigma=6.0, ramp=4.0):
    """Creux large laissé par une soustraction (sur-soustraction d'un coeur voisin) : seule la basse fréquence de la
    zone est refaite (membrane harmonique depuis le pourtour) ; le détail et le grain de la zone restent les vrais."""
    low = lowpass_masked(V, zone, low_sigma)
    new_low = _laplace_fill(low, zone)
    low_in = lowpass_masked(V, np.zeros_like(zone), low_sigma)
    w = np.clip(ndi.distance_transform_edt(zone) / ramp, 0, 1)[None]
    out = V.copy()
    out[:, zone] = (V + w * (new_low - low_in))[:, zone]
    return out
