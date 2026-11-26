# ======================================================================
# passe_finale_t2_p.py
# Rôle     : Tour 2 de la passe finale (21 nov) : base = passe_finale_c9\sans_etoiles_tuile.npy.
# Étape    : 2. Chaîne classique de soustraction guidée par Gaia (v33b) (voir scripts/LISEZMOI.md)
# Entrées  : image empilée (unités de bruit), astrométrie, extrait Gaia profond hors ligne
# Sorties  : image sans étoiles, cartes de classes
# Exemple  : python passe_finale_t2_p.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Tour 2 de la passe finale (21 nov) : base = passe_finale_c9\\sans_etoiles_tuile.npy.
 1. pixels noirs isolés (1-2 px) : très en dessous de la médiane 5x5 sans le centre, seuil sur le bruit local, guéris par médiane des voisins sains ;
 2. restes de 3-8 px (arcs, taches rouges, reste d'étoile) : différence de gaussiennes, deux signes, couleurs et chroma, près des étoiles, remplis comme au tour 1 ;
 3. couleur du ciel bas, mesurée seulement (entrée / résultat, hors zones d'étoiles).
Écrit seulement dans sorties/passe_finale_c9_t2. CPU seulement."""
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

OUT = pf.CFG.B_ROOT / "t2"
R1DIR = pf.CFG.B_ROOT / "t1"
K_PIX = float(os.environ.get("T2_KPIX", 8.0))
HI, LO = float(os.environ.get("T2_HI", 5.0)), float(os.environ.get("T2_LO", 3.0))
MAXA = 400


def isolated_dark(R, stars):
    C, H, W = R.shape
    fp = np.ones((5, 5), bool); fp[2, 2] = False
    flag = np.zeros((H, W), bool)
    for c in range(C):
        a = R[c]
        m = ndi.median_filter(a, footprint=fp, mode="nearest")
        d = m - a                                   # > 0 : plus sombre que le voisinage
        s = ndi.median_filter(np.abs(a - m), size=21, mode="nearest") * 1.4826 + 1e-6
        flag |= d > K_PIX * s
    # un point isolé : 1-2 px (composantes de 8 voisins)
    lab, n = ndi.label(flag, structure=np.ones((3, 3)))
    sz = ndi.sum(flag, lab, np.arange(1, n + 1))
    ok = np.zeros(n + 1, bool); ok[1:] = sz <= 3
    flag = ok[lab]
    return flag


def heal_points(R, flag):
    out = R.copy()
    ring = ndi.binary_dilation(flag, iterations=1) & ~flag
    ys, xs = np.nonzero(flag)
    C, H, W = R.shape
    good = ~ndi.binary_dilation(flag, iterations=1)
    for y, x in zip(ys, xs):
        y0, y1, x0, x1 = max(0, y - 2), min(H, y + 3), max(0, x - 2), min(W, x + 3)
        g = good[y0:y1, x0:x1]
        if g.sum() < 4:
            continue
        for c in range(C):
            out[c, y, x] = np.median(R[c, y0:y1, x0:x1][g])
    ry, rx = np.nonzero(ring)
    out[:, ry, rx] = 0.5 * R[:, ry, rx] + 0.5 * np.stack([ndi.median_filter(out[c], size=3, mode="nearest") for c in range(C)])[:, ry, rx]
    return out


def dog_detect(R, zone, exclude, hi_thr=None, maxa=None):
    C = R.shape[0]
    chans = [R[0], R[1], R[2], R.mean(0), R[0] - R[1], R[2] - R[1]]
    Z = []
    for a in chans:
        a = a.astype(np.float32)
        d = cv2.GaussianBlur(a, (0, 0), 1.0) - cv2.GaussianBlur(a, (0, 0), 4.0)     # échelle 3-8 px
        Z.append(d / mad(d[::2, ::2]))
    A = np.abs(np.stack(Z)).max(0)
    hi = A > (HI if hi_thr is None else hi_thr)
    lo = (A > LO) & zone & ~exclude
    lab, n = ndi.label(lo)
    keep = np.zeros(n + 1, bool)
    if n:
        keep[1:] = ndi.maximum(hi & zone, lab, np.arange(1, n + 1)) > 0
    m = keep[lab]
    lab2, n2 = ndi.label(m)
    if n2:
        sz = ndi.sum(m, lab2, np.arange(1, n2 + 1))
        ok = np.zeros(n2 + 1, bool); ok[1:] = (sz >= pf.MIN_AREA) & (sz <= (MAXA if maxa is None else maxa))
        m = ok[lab2]
    return m, A


def sky_colour(I, R, zone):
    free = ~ndi.binary_dilation(zone, iterations=4)
    H = I.shape[1]
    out = {}
    for nm, sl in (("tuile", slice(0, H)), ("tiers_bas", slice(2 * H // 3, H))):
        fm = free[sl]
        d = {}
        for lab, a in (("entree", I), ("resultat", R)):
            mu = [float(a[c][sl][fm].mean()) for c in range(3)]
            d[lab] = dict(R_sur_G=round(mu[0] / mu[1], 4), B_sur_G=round(mu[2] / mu[1], 4), moyennes_RGB=[round(v, 4) for v in mu])
        d["px_ciel_libre"] = int(fm.sum())
        out[nm] = d
    return out


def zoom6(post, I, R1, R2, box, name):
    vx0, vy0, vx1, vy1 = box
    x0, x1, y0, y1 = (vx0 - 8) // 2, (vx1 - 8) // 2, vy0 // 2, vy1 // 2
    sp = np.full(((y1 - y0) * 6, 6, 3), 255, np.uint8)
    up = lambda a: cv2.resize(post(a[:, y0:y1, x0:x1]), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
    pf.save_img(OUT / name, np.concatenate([up(I), sp, up(R1), sp, up(R2)], 1))


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    logf = OUT / "journal.txt"
    def log(s):
        print(s, flush=True)
        with open(logf, "a", encoding="utf-8") as f:
            f.write(s + "\n")
    full, I, crop, tile, gaia, deep, p1 = pf.load_all()
    H, W = I.shape[1:]
    R1 = np.load(R1DIR / "sans_etoiles_tuile.npy").astype(np.float32)
    carte1 = np.load(R1DIR / "carte_remplissage_tuile.npy")
    ign1 = np.load(R1DIR / "masque_ignorer.npy")
    stars = pf.star_list(H, W, gaia, deep, p1)
    zone, core = pf.star_zone(H, W, stars)
    # distance à la source la plus proche
    seed = np.ones((H, W), bool)
    for x, y, g in stars:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H:
            seed[yi, xi] = False
    dist = ndi.distance_transform_edt(seed)
    # 1. points noirs isolés
    flag = isolated_dark(R1, stars)
    # v10 : une composante de <= 3 px a plus de K_PIX bruits locaux (5x5 sans centre) est TOUJOURS guerie (un filament n'est jamais un point de 1-3 px) : plus de test de reste d'etoile ici
    _c = dict(detectees=int(ndi.label(flag, structure=np.ones((3, 3)))[1]), gardees='toutes (v10)')
    R2 = heal_points(R1, flag)
    ly, lx = ndi.label(flag, structure=np.ones((3, 3)))
    npts = int(ly.max())
    far = int(sum(dist[ly == k].min() > 8 for k in range(1, npts + 1))) if npts else 0
    log(f"points noirs isolés : {npts} ({int(flag.sum())} px), loin (> 8 px) de toute source : {far}")
    # 2. restes 3-8 px
    noise = [mad(c - cv2.GaussianBlur(c, (0, 0), 6)) for c in R2]
    zone2 = zone | ndi.binary_dilation(~seed, iterations=10)
    m1, A = dog_detect(R2, zone2, ign1)
    # taches compactes fortes (> 8 bruits, <= 80 px) même loin des étoiles listées : défauts ponctuels
    m2, _ = dog_detect(R2, np.ones((H, W), bool), ign1, hi_thr=8.0, maxa=80)
    m1, _c1 = pf.star_leftovers_only(m1, stars, log, 'restes 3-8 px (zone)', img=R2)      # v9
    m2, _c2 = pf.star_leftovers_only(m2, stars, log, 'taches compactes fortes', img=R2)      # v9
    m = m1 | m2
    lab_, nn_ = ndi.label(m2 & ~m1)
    far2 = int(sum(dist[lab_ == k].min() > 10 for k in range(1, nn_ + 1))) if nn_ else 0
    log(f"taches compactes fortes hors zone ajoutées : {nn_} dont loin (> 10 px) de toute source : {far2}")
    lab, n = ndi.label(m)
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)).astype(bool)
    grown = ndi.binary_dilation(m, structure=disk)
    nfill = int(ndi.label(grown)[1])
    forbid = (carte1 > 0) | ndi.binary_dilation(m, iterations=3) | core | ign1
    if nfill:
        filled = F.fill_mask(R2.astype(np.float32), grown, forbid, noise)
        w = np.clip(1 - ndi.distance_transform_edt(~grown) / 2.0, 0, 1).astype(np.float32)
        R2 = (R2 + w[None] * (filled - R2)).astype(np.float32)
    log(f"restes 3-8 px : {n} détectés ({int(m.sum())} px), {nfill} zones remplies")
    # zones cibles du coordinateur
    cible = {}
    for nm, (x0, x1, y0, y1) in dict(arcs=(330, 361, 30, 231), reste_haut_gauche=(80, 101, 82, 103)).items():
        cible[nm] = dict(restes_detectes=int(len(np.unique(lab[y0:y1, x0:x1])) - (0 in lab[y0:y1, x0:x1])),
                         px_changes_vs_tour1=int((R2 != R1)[:, y0:y1, x0:x1].any(0).sum()))
    ch = (R2 != R1).any(0)
    ch0 = (R2 != np.load(pf.BASE / "sans_etoiles_tuile.npy")).any(0)
    carte2 = carte1.copy()
    carte2[ch & (carte1 == 0)] = 7                     # 7 = repris au tour 2
    c0 = np.load(pf.BASE / "carte_remplissage_tuile.npy")
    share = lambda c, cl: round(100 * float(np.isin(c, cl).mean()), 2)
    meas = dict(
        points_noirs=dict(seuil_k=K_PIX, nombre=npts, pixels=int(flag.sum()), loin_de_toute_source_8px=far),
        restes_3_8px=dict(taches_fortes_hors_zone=nn_, dont_loin_10px=far2, seuil_haut=HI, seuil_bas=LO, detectes=int(n), zones_remplies=nfill, px_masque=int(grown.sum())),
        zones_cibles=cible,
        part_changee_vs_tour1_pct=round(100 * float(ch.mean()), 2),
        part_changee_vs_c9_pct=round(100 * float(ch0.mean()), 2),
        part_a_ignorer_pct=round(100 * float(ign1.mean()), 2),
        part_inventee=dict(
            c9_classes_2_4_5_6=share(c0, [2, 4, 5, 6]),
            tour1_classes_2_4_5_6=share(carte1, [2, 4, 5, 6]),
            tour2_classes_2_4_5_6_7=share(carte2, [2, 4, 5, 6, 7]),
            c9_toute_modification_classes_1_a_4=share(c0, [1, 2, 3, 4]),
            tour2_toute_modification=share(carte2, [1, 2, 3, 4, 5, 6, 7]),
            note="c9 : classe 1 = halo soustrait (modification, pas invention), 2 = coeurs recréés, 3 = défauts corrigés sur place, "
                 "4 = défauts recréés ; l'ancien « total_invente » de c9 = classes 2,4,5,6 = 16.38 %. Le 53 % du tour 1 était "
                 "toute modification (classes > 0) ; classe 5 = tour 1, classe 7 = tour 2."),
        couleur_ciel_libre=sky_colour(I, R2, zone),
        couleur_ciel_libre_tour1=sky_colour(I, R1, zone))
    np.save(OUT / "sans_etoiles_tuile.npy", R2)
    np.save(OUT / "carte_remplissage_tuile.npy", carte2)
    np.save(OUT / "masque_ignorer.npy", ign1)
    (OUT / "mesures.json").write_text(json.dumps(meas, ensure_ascii=False, indent=1), encoding="utf-8")
    log(json.dumps(meas, ensure_ascii=False))
    # vues : même étirement et cadrage
    pf.OUT = OUT; pf.SIX = []
    pf.views(full, I, crop, tile, R1, R2, log)
    g = full[1]; bk = float(np.median(g[::7, ::7])); t998 = float(np.percentile(g[::5, ::5], 99.8))
    def post(a):
        x = np.clip((a - bk) / max(t998 - bk, 1e-9), 0, None); v = np.arcsinh(40 * x) / np.arcsinh(40)
        return (np.transpose(np.clip(v, 0, 1), (1, 2, 0)) * 255 + 0.5).astype(np.uint8)
    zoom6(post, I, R1, R2, (640, 60, 940, 360), "zoom_x6_vue640_60_entree_tour1_tour2.png")
    zoom6(post, I, R1, R2, (560, 640, 860, 940), "zoom_x6_vue560_640_entree_tour1_tour2.png")
    log(f"fini en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
