# ======================================================================
# passe_finale_t3_p.py
# Rôle     : Tour 3 de la passe finale (21 nov) : base = passe_finale_c9_t2\sans_etoiles_tuile.npy.
# Étape    : 2. Chaîne classique de soustraction guidée par Gaia (v33b) (voir scripts/LISEZMOI.md)
# Entrées  : image empilée (unités de bruit), astrométrie, extrait Gaia profond hors ligne
# Sorties  : image sans étoiles, cartes de classes
# Exemple  : python passe_finale_t3_p.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Tour 3 de la passe finale (21 nov) : base = passe_finale_c9_t2\\sans_etoiles_tuile.npy. Défauts relevés par le juge :
 A. points sombres (5 bruits) et clairs près des étoiles, taches colorées, nœud (84, 96) : guéris (remplir.fill_mask) ;
 B. bords nets des grands polygones recréés : fondus sur ~6 px (lissage 1.5 -> 4 px dans la bande) ;
 C. chroma grossier des pixels recréés ramené au chroma de l'entrée (le grain devient cohérent avec la luminance) ;
 D. grain 2-4 px : texture prise dans le ciel sans étoile de l'entrée (patches 16 px tirés au hasard, jamais deux fois),
    ajoutée SEULEMENT aux pixels changés, à l'amplitude du déficit local (par couleur).
Écrit seulement dans sorties/passe_finale_c9_t3. CPU seulement."""
import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"; os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, cv2
from scipy import ndimage as ndi
import passe_finale_p as pf
from passe_finale_p import mad, F

OUT = pf.CFG.B_ROOT / "t3"
T2 = pf.CFG.B_ROOT / "t2"
JUGE = (Path(__import__("os").environ.get("RACINE_SCRIPTS") or Path(__file__).resolve().parents[2]) / "Juge – essais 20 nov" / "juge_passe_finale_t2")
FORCED = dict(magenta=(196, 320), vert=(165, 182), bleu=(62, 113), noeud_trou=(84, 96), noeud_point=(83, 90),
              sombre1=(272, 154), sombre2=(332, 207)) if pf.CFG.NAMED_FIXES else dict()   # positions nommees de decoupe6 seulement


def band(a, s1=1.0, s2=4.0):
    return cv2.GaussianBlur(a.astype(np.float32), (0, 0), s1) - cv2.GaussianBlur(a.astype(np.float32), (0, 0), s2)


def nconv(v, w, s):
    return cv2.GaussianBlur((v * w).astype(np.float32), (0, 0), s), cv2.GaussianBlur(w.astype(np.float32), (0, 0), s)


def judge_free_mask(I):
    """Masque de ciel libre du juge (j2_mesures.py), recalculé depuis l'entrée."""
    I = I.astype(np.float64); H, W = I.shape[1:]
    s = []
    for c in range(3):
        hp = I[c] - ndi.gaussian_filter(I[c], 1.0)
        v = sorted(mad(hp[y:y + 32, x:x + 32]) for y in range(0, H, 32) for x in range(0, W, 32))
        s.append(float(np.median(v[:len(v) // 4])))
    nI = np.array(s)
    L = I.sum(0); nL = float(np.sqrt((nI ** 2).sum()))
    f = ndi.gaussian_filter(L, 1.0) - ndi.median_filter(L, 31)
    src = f > 3 * nL * 0.45
    lab, n = ndi.label(src)
    peak = ndi.maximum(f, lab, np.arange(1, n + 1))
    star = np.zeros((H, W), bool)
    yy, xx = np.mgrid[:H, :W]
    for (cy, cx), p in zip(ndi.center_of_mass(f, lab, np.arange(1, n + 1)), peak):
        r = 4 + 3.0 * np.log1p(max(p, 0) / (5 * nL))
        y0, y1 = int(max(cy - r - 1, 0)), int(min(cy + r + 2, H)); x0, x1 = int(max(cx - r - 1, 0)), int(min(cx + r + 2, W))
        star[y0:y1, x0:x1] |= (yy[y0:y1, x0:x1] - cy) ** 2 + (xx[y0:y1, x0:x1] - cx) ** 2 <= r * r
    return ~ndi.binary_dilation(star | src, iterations=2)


def judge_grain(I, R, free):
    """Méthode du juge (j2 / j3) : rapport du grain 2-4 px du résultat à celui de l'entrée en ciel libre."""
    bI, bR = band(I.sum(0)), band(R.sum(0))
    ref = mad(bI[free])
    H, W = bI.shape
    g = np.array([[mad(bR[y:y + 32, x:x + 32]) / ref for x in range(0, W, 32)] for y in range(0, H, 32)])
    rat = []
    for c in range(3):
        bi, br = band(I[c]), band(R[c])
        rc = []
        for y in range(0, H, 64):
            for x in range(0, W, 64):
                m = free[y:y + 64, x:x + 64]
                rc.append(np.nan if m.sum() < 300 else mad(br[y:y + 64, x:x + 64][m]) / max(mad(bi[y:y + 64, x:x + 64][m]), 1e-9))
        rat.append(rc)
    rat = np.array(rat)
    return dict(blocs32_lum=dict(mediane=round(float(np.median(g)), 3), p10=round(float(np.percentile(g, 10)), 3),
                                 sous_0_7=int((g < 0.7).sum()), sous_0_5=int((g < 0.5).sum()), pire=round(float(g.min()), 3), total=int(g.size)),
                blocs64_par_canal_mediane=np.nanmedian(rat, 1).round(3).tolist(),
                blocs64_sous_0_7=int((np.nanmean(rat, 0) < 0.7).sum())), g


def judge_points(R):
    LR = R.sum(0)
    fR = ndi.gaussian_filter(LR, 1.2) - ndi.gaussian_filter(ndi.median_filter(LR, 25), 2)
    sf = mad(fR)
    return fR, sf


def chroma_spots(R):
    chroma = R - R.mean(0, keepdims=True)
    cb = np.stack([ndi.median_filter(chroma[c], 9) for c in range(3)])
    cd = np.sqrt(((ndi.gaussian_filter(chroma - cb, (0, 1, 1))) ** 2).sum(0))
    sc = mad(cd)
    mm = (cd > np.median(cd) + 6 * sc) & (cd == ndi.maximum_filter(cd, 5))
    return np.argwhere(mm), cd


def disk_mask(H, W, pts, r):
    m = np.zeros((H, W), np.uint8)
    for x, y in pts:
        cv2.circle(m, (int(x), int(y)), r, 1, -1)
    return m.astype(bool)


def build_texture(I, src_ok, shape, rng, P=10, step=7):
    """Champ de texture (3, H, W), bande 0.7-4 px de l'entrée, patches P x P tirés au hasard dans le ciel sans étoile."""
    H, W = shape
    Bi = np.stack([band(I[c], 0.7, 4.0) for c in range(3)])
    sd0 = np.array([mad(Bi[c][src_ok]) for c in range(3)])
    bad = (np.abs(Bi) > 3.5 * sd0[:, None, None]).any(0)          # étoile faible ou reste dans la source : écarté
    src_ok = src_ok & ~ndi.binary_dilation(bad, iterations=2)
    # origines valides : patch entièrement dans le ciel sans étoile
    ok = ndi.minimum_filter(src_ok.astype(np.uint8), size=P) > 0
    oy, ox = np.nonzero(ok[:H - P + 1, :W - P + 1])
    # le filtre minimum est centré : décalage de P//2
    cand = [(y - P // 2, x - P // 2) for y, x in zip(oy, ox) if 0 <= y - P // 2 <= H - P and 0 <= x - P // 2 <= W - P]
    cand = np.array(cand)
    if len(cand) == 0 and os.environ.get("DIAG_T3_TEXTURE", "1") == "1":
        # Diagnostic 23 nov : pas de ciel sans étoile -> pas de texture copiée (le repli ci-dessous recopiait des
        # morceaux d'étoiles dans le grain)
        return np.zeros((3, H, W), np.float32), dict(patchs=0, note="aucun ciel sans étoile : pas de grain ajouté (diagnostic 23 nov)")
    if len(cand) == 0:
        # 22 nov (NGC 869, amas dense) : aucun patch de ciel sans étoile ; on prend les patchs les moins touchés (au moins
        # 80 % de pixels sûrs), sinon aucun grain n'est ajouté (texture nulle)
        okf = ndi.uniform_filter(src_ok.astype(np.float32), size=P) >= 0.8
        oy, ox = np.nonzero(okf[:H - P + 1, :W - P + 1])
        cand = np.array([(y - P // 2, x - P // 2) for y, x in zip(oy, ox) if 0 <= y - P // 2 <= H - P and 0 <= x - P // 2 <= W - P])
        if len(cand) == 0:
            return np.zeros((3, H, W), np.float32), dict(patchs=0, note="aucun ciel sans étoile : pas de grain ajouté")
    sd = np.array([mad(Bi[c][src_ok]) for c in range(3)])
    win1 = np.hanning(P + 2)[1:-1]; win = np.outer(win1, win1).astype(np.float32)
    acc = np.zeros((3, H, W), np.float32); wsum = np.zeros((H, W), np.float32)
    used = set()
    gy = list(range(0, H - P + 1, step)) + [H - P]
    gx = list(range(0, W - P + 1, step)) + [W - P]
    n_rep = 0
    for y in gy:
        for x in gx:
            for _ in range(50):
                k = int(rng.integers(len(cand)))
                if k not in used:
                    break
            else:
                n_rep += 1
            used.add(k)
            sy, sx = cand[k]
            patch = Bi[:, sy:sy + P, sx:sx + P]
            acc[:, y:y + P, x:x + P] += patch * win[None]
            wsum[y:y + P, x:x + P] += win ** 2
    T = acc / np.sqrt(np.maximum(wsum, 1e-6))[None]
    T /= sd[:, None, None]
    return T.astype(np.float32), dict(patchs=len(gy) * len(gx), origines_valides=int(len(cand)), reutilises=n_rep)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    logf = OUT / "journal.txt"
    def log(s):
        print(s, flush=True)
        with open(logf, "a", encoding="utf-8") as f:
            f.write(s + "\n")
    full, I, crop, tile, gaia, deep, p1 = pf.load_all()
    I = I.astype(np.float32)
    H, W = I.shape[1:]
    R2 = np.load(T2 / "sans_etoiles_tuile.npy").astype(np.float32)
    carte2 = np.load(T2 / "carte_remplissage_tuile.npy")
    ign = np.load(T2 / "masque_ignorer.npy")
    c9 = np.load(pf.BASE / "sans_etoiles_tuile.npy").astype(np.float32)
    free_j = judge_free_mask(I)
    stars = pf.star_list(H, W, gaia, deep, p1)
    zone, core = pf.star_zone(H, W, stars)
    sm = np.zeros((H, W), np.uint8)
    for x, y, g_ in stars:
        cv2.circle(sm, (int(round(x)), int(round(y))), int(4 + 3 * max(0, 12 - g_)), 1, -1)
    # Diagnostic 23 nov (défaut 4, puits sombres) : la texture D est la bande 0,7-4 px de l'ENTRÉE ; la réponse de cette
    # bande à une étoile (cœur positif, anneau négatif) s'étend à ~3 x 4 px au-delà du disque exclu (4 px pour une
    # faible) : les patches « sans étoile » portaient les anneaux négatifs des étoiles voisines, grain ajouté biaisé
    # vers le sombre (centile 0,1 % -0,88 contre +0,27 au 99,9 % sur NGC 7000), d'où des taches sombres. L'exclusion
    # couvre maintenant le support du filtre (12 px). DIAG_T3_TEXTURE=0 rend l'ancien calcul.
    _t3_tex = os.environ.get("DIAG_T3_TEXTURE", "1") == "1"
    src_ok = ~ndi.binary_dilation(sm.astype(bool), iterations=2)
    tex_ok = ~ndi.binary_dilation(sm.astype(bool), iterations=2 + 12) if _t3_tex else src_ok   # source de la texture D seulement
    _unused = ~ndi.binary_dilation(zone, iterations=4)              # ciel sans étoile de l'entrée (source de texture et cible d'amplitude)
    log(f"ciel sans étoile (source de texture) : {100 * src_ok.mean():.1f} % de la tuile ; ciel libre du juge : {100 * free_j.mean():.1f} %")
    meas = dict()
    gb, gmap_b = judge_grain(I, R2, free_j)
    meas["grain_avant"] = gb
    invented = np.isin(carte2, [2, 4, 5, 6, 7])
    R = R2.copy()
    forbid = (carte2 > 0) | core | ign

    # A. guérisons
    fR, sf = judge_points(R)
    pk_dark = (-fR > 5 * sf) & (-fR == ndi.maximum_filter(-fR, 5))
    pk_bright = (fR > 5 * sf) & (fR == ndi.maximum_filter(fR, 5)) & zone
    dark_pts = [(x, y) for y, x in np.argwhere(pk_dark)]
    bright_pts = [(x, y) for y, x in np.argwhere(pk_bright)]
    csp, cd = chroma_spots(R)
    chroma_pts = [(x, y) for y, x in csp]
    forced = list(FORCED.values())
    # v8 : un point n'est gueri que s'il est a moins de max(2 px, 0,7 FWHM) d'une source (composante = pic isole, rond par construction)
    from scipy.spatial import cKDTree
    _tr = cKDTree(stars[:, :2]); _tol = max(2.0, 0.7 * pf.FWHM)
    _D = pf.dog_stack(R)
    _near = lambda pts: [p_ for p_ in pts if pf.point_is_star_leftover(_D, p_, _tr)]      # v9 : blob a l'echelle de l'etoile
    _n0 = (len(dark_pts), len(bright_pts), len(chroma_pts))
    dark_pts, bright_pts, chroma_pts = _near(dark_pts), _near(bright_pts), _near(chroma_pts)
    log(f"v8 t3 : pics gueris pres d'une source (<= {_tol:.1f} px) : sombres {_n0[0]}->{len(dark_pts)}, clairs {_n0[1]}->{len(bright_pts)}, colores {_n0[2]}->{len(chroma_pts)}")
    # zone nœud : restes des deux signes à 3 bruits dans 12 px de (84,96)
    dog = band(R.mean(0), 1.0, 4.0); sd = mad(dog[::2, ::2])
    yy, xx = np.mgrid[:H, :W]
    knot = ((np.hypot(xx - 84, yy - 96) < 14) & (np.abs(dog) > 3 * sd)) if pf.CFG.NAMED_FIXES else np.zeros((H, W), bool)   # noeud nomme de decoupe6
    heal = disk_mask(H, W, dark_pts, 4) | disk_mask(H, W, bright_pts, 4) | disk_mask(H, W, chroma_pts, 3) | disk_mask(H, W, forced, 4) | knot
    heal_grown = ndi.binary_dilation(heal, iterations=1) & ~ign
    nz = int(ndi.label(heal_grown)[1])
    filled = F.fill_mask(R, heal_grown, forbid | ndi.binary_dilation(heal_grown, iterations=3),
                         [mad(c - cv2.GaussianBlur(c, (0, 0), 6)) for c in R])
    w = np.clip(1 - ndi.distance_transform_edt(~heal_grown) / 2.0, 0, 1).astype(np.float32)
    R = (R + w[None] * (filled - R)).astype(np.float32)
    far = int(sum(1 for (x, y) in dark_pts + chroma_pts if (0 <= x < W and 0 <= y < H and zone[int(y), int(x)] == False)))
    log(f"A. guérisons : {len(dark_pts)} sombres 5 bruits, {len(bright_pts)} clairs 5 bruits près d'étoiles, {len(chroma_pts)} taches colorées, "
        f"{len(forced)} points nommés + nœud ; {nz} zones remplies ; hors zone d'étoile : {far}")
    meas["guerisons"] = dict(sombres_5sigma=len(dark_pts), clairs_5sigma_pres_etoiles=len(bright_pts), taches_colorees_6sigma=len(chroma_pts),
                             points_nommes=len(forced), zones_remplies=nz, px_masque=int(heal_grown.sum()), sombres_et_colores_hors_zone_etoile=far)
    changed = (carte2 > 0) | (R2 != c9).any(0) | heal_grown

    # B. bords nets des grands polygones recréés : fondus sur ~6 px
    lab, n = ndi.label(invented)
    big = np.zeros((H, W), bool)
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        if (lab[sl] == k).sum() >= 150:
            big |= (lab == k)
    edge = ndi.binary_dilation(big, iterations=3) & ~ndi.binary_erosion(big, iterations=3) & ~ign
    wb = np.clip(cv2.GaussianBlur(edge.astype(np.float32), (0, 0), 1.5) * 1.8, 0, 1) * edge
    R = (R + wb[None] * np.stack([cv2.GaussianBlur(c, (0, 0), 4.0) - cv2.GaussianBlur(c, (0, 0), 1.5) for c in R])).astype(np.float32)
    changed |= edge
    log(f"B. bords fondus : {int(ndi.label(big)[1])} grandes zones recréées (>= 150 px), bande {int(edge.sum())} px")
    meas["bords_fondus"] = dict(grandes_zones=int(ndi.label(big)[1]), px_bande=int(edge.sum()))

    # C. chroma grossier des pixels changés ramené à celui de l'entrée
    bandC = lambda a: cv2.GaussianBlur(a, (0, 0), 0.7) - cv2.GaussianBlur(a, (0, 0), 8.0)
    def chroma_band(A):
        B = np.stack([bandC(c) for c in A]); Lb = B.mean(0)
        return B - Lb[None], B
    Ci, _ = chroma_band(I)
    Cr, Br = chroma_band(R)
    wch = cv2.GaussianBlur(changed.astype(np.float32), (0, 0), 1.0) * changed
    fr_c = []
    for c in range(3):
        tgt = mad(Ci[c][src_ok])
        num, den = nconv(Cr[c] ** 2, changed.astype(np.float32), 8)
        cur = np.sqrt(num / np.maximum(den, 1e-6))
        cur = np.where(den > 0.05, cur, tgt)
        f = np.clip(tgt / np.maximum(cur, 1e-9), 0, 1)
        fr_c.append(float(np.median(f[changed]))); frac_c = float((f[changed] < 0.9).mean()); log(f'   chroma couleur {c} : cible {tgt:.4f}, part des pixels changés ramenés de plus de 10 % : {100 * frac_c:.1f} %')
        R[c] -= (wch * (1 - f) * Cr[c]).astype(np.float32)
    log(f"C. chroma : facteur médian sur les pixels changés par couleur {np.round(fr_c, 3).tolist()}")
    meas["chroma_facteur_median"] = np.round(fr_c, 3).tolist()

    # D. grain 2-4 px aux pixels changés
    rng = np.random.default_rng(321)
    T, tinfo = build_texture(I, tex_ok, (H, W), rng)
    log(f"D. texture : {tinfo}")
    cur_amp = []
    out = R.copy()
    wchg = cv2.GaussianBlur(changed.astype(np.float32), (0, 0), 0.8)
    wchg = np.where(changed, np.maximum(wchg, 0.0), wchg * 0)           # jamais hors des pixels changés
    amp_log = []
    for c in range(3):
        if _t3_tex and not T[c].any():                    # diagnostic 23 nov : pas de texture, pas d'ajout
            amp_log.append(0.0)
            continue
        bI_, bR_ = band(I[c]), band(R[c])
        ref_c = mad(bI_[free_j])                                         # cible = grain du ciel libre de l'entrée (celui du juge)
        tgt = np.full((H, W), ref_c, np.float32)
        brc = np.clip(bR_, -4 * ref_c, 4 * ref_c)
        nc, dc = nconv(brc ** 2, changed.astype(np.float32), 10)
        cur = np.sqrt(nc / np.maximum(dc, 1e-6)); cur = np.where(dc > 0.03, cur, np.sqrt(np.mean(brc[changed] ** 2)))
        need = np.sqrt(np.maximum(tgt ** 2 - cur ** 2, 0))
        # texture normalisée sur la bande 1-4 : son écart type propre dans cette bande
        sT = mad(band(T[c])[::2, ::2])
        out[c] = R[c] + wchg * need * T[c] / max(sT, 1e-9)
        amp_log.append(float(np.median((need * wchg)[changed])))
    log(f"D. amplitude ajoutée médiane par couleur {np.round(amp_log, 4).tolist()}")
    R3 = out.astype(np.float32)
    unchanged_touched = int(((R3 != R2).any(0) & ~changed).sum())
    meas["pixels_non_changes_touches"] = unchanged_touched
    ga, gmap_a = judge_grain(I, R3, free_j)
    meas["grain_apres"] = ga
    meas["grain_carte_blocs32_avant"] = np.round(gmap_b, 2).tolist()
    meas["grain_carte_blocs32_apres"] = np.round(gmap_a, 2).tolist()
    log(f"grain 2-4 px (méthode du juge) avant {gb['blocs32_lum']}, après {ga['blocs32_lum']}")
    # points restants selon le juge
    fR2, sf2 = judge_points(R3)
    pm = lambda f_, s_, sg: int(((sg * f_ > 5 * s_) & (sg * f_ == ndi.maximum_filter(sg * f_, 5))).sum())
    meas["points_5sigma"] = dict(avant=dict(sombres=pm(judge_points(R2)[0], judge_points(R2)[1], -1), clairs=pm(judge_points(R2)[0], judge_points(R2)[1], 1)),
                                 apres=dict(sombres=pm(fR2, sf2, -1), clairs=pm(fR2, sf2, 1)))
    ch_new = (R3 != R2).any(0)
    carte3 = carte2.copy()
    carte3[ch_new & (carte2 == 0)] = 8                     # 8 = repris au tour 3
    share = lambda c, cl: round(100 * float(np.isin(c, cl).mean()), 2)
    classes = {int(k): round(100 * float((carte3 == k).mean()), 2) for k in np.unique(carte3)}
    meas["part_inventee"] = dict(
        tour2_classes_2_4_5_6_7=share(carte2, [2, 4, 5, 6, 7]), tour3_classes_2_4_5_6_7_8=share(carte3, [2, 4, 5, 6, 7, 8]),
        par_classe_tour3_pct=classes, note="1 halo soustrait, 2 coeurs recréés, 3 défauts corrigés sur place, 4 défauts recréés, 5 tour 1, 7 tour 2, 8 tour 3 (pixels changés en ciel non marqué avant)")
    meas["part_changee_vs_tour2_pct"] = round(100 * float(ch_new.mean()), 2)
    # couleur du ciel libre (mesure)
    free2 = src_ok
    meas["couleur_ciel_sans_etoile"] = {nm: dict(R_sur_G=round(float(a[0][free2].mean() / a[1][free2].mean()), 4), B_sur_G=round(float(a[2][free2].mean() / a[1][free2].mean()), 4))
                                        for nm, a in (("entree", I), ("tour2", R2), ("tour3", R3))}
    np.save(OUT / "sans_etoiles_tuile.npy", R3)
    np.save(OUT / "carte_remplissage_tuile.npy", carte3)
    np.save(OUT / "masque_ignorer.npy", ign)
    (OUT / "mesures.json").write_text(json.dumps(meas, ensure_ascii=False, indent=1), encoding="utf-8")
    log(json.dumps({k: v for k, v in meas.items() if not k.startswith("grain_carte")}, ensure_ascii=False))
    # vues
    pf.OUT = OUT; pf.SIX = []
    pf.views(full, I, crop, tile, R2, R3, log)
    g = full[1]; bk = float(np.median(g[::7, ::7])); t998 = float(np.percentile(g[::5, ::5], 99.8))
    def post(a):
        x = np.clip((a - bk) / max(t998 - bk, 1e-9), 0, None); v = np.arcsinh(40 * x) / np.arcsinh(40)
        return (np.transpose(np.clip(v, 0, 1), (1, 2, 0)) * 255 + 0.5).astype(np.uint8)
    hw = 25
    for nm, (cx, cy) in (dict(x287_y200=(287, 200), x128_y416=(128, 416), x305_y300=(305, 300), x84_y96=(84, 96), x196_y320=(196, 320)) if pf.CFG.NAMED_FIXES else {}).items():
        ya, yb = max(0, cy - hw), min(H, cy + hw); xa, xb = max(0, cx - hw), min(W, cx + hw)
        up = lambda a: cv2.resize(post(a[:, ya:yb, xa:xb]), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
        sp = np.full(((yb - ya) * 6, 6, 3), 255, np.uint8)
        pf.save_img(OUT / f"zoom_x6_{nm}_entree_tour2_tour3.png", np.concatenate([up(I), sp, up(R2), sp, up(R3)], 1))
    log(f"fini en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
