# ======================================================================
# passe_finale_p.py
# Role     : Final pass of 21 Nov (hybrid c9, crop of the 6 stars): go over the star remnants (bright or hollow) again, like an operator with the healing tool, once the bulk of the removal is done.
# Step     : 2. Classic Gaia-guided subtraction chain (v33b) (see scripts/LISEZMOI.md)
# Inputs   : stacked image (noise units), astrometry, offline deep Gaia extract
# Outputs  : star-free image, class maps
# Example  : python passe_finale_p.py [arguments]  (see the docstring; paths through the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Final pass of 21 Nov (hybrid c9, crop of the 6 stars): go over the star remnants (bright or hollow) again,
like an operator with the healing tool, once the bulk of the removal is done. A single change on the c9 base:
  1. residual = result - robust local background (normalised gaussian, outliers rejected), in noises, per colour and in luminance;
  2. remnants = connected zones where |residual| exceeds a threshold (hysteresis), of both signs, only near a star
     (radius growing with brightness around each Gaia position / deep catalogue / track 1 source,
     plus bands along the spikes of the bright stars);
  3. mask enlarged by 2 px, feathered 2 px; filled from the sky already cleaned around (remplir.complete = PatchMatch
     + Poisson), level and colour matched to the ring;
  4. guard: zone both large AND on structured nebula (high 16-48 px structure) = not filled, marked « to ignore »;
  5. two passes (the second on the output of the first).
Read-only on the c9 base; writes only to sorties/passe_finale_c9. CPU only."""
import os, sys, json, csv, time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, cv2
from scipy import ndimage as ndi
cv2.setNumThreads(4)
import mv_cfg as CFG
import mesure as M
import remplir as F
from juger import save_img, to_rgb8

BASE = CFG.A_OUT
OUT = CFG.B_ROOT / "t1"
CASE = CFG.CASE
SIX = CFG.SIX
ANGLES = CFG.ANGLES          # aigrettes de l'instrument (journal c9) ; + perpendiculaires
Z_HI, Z_LO = float(os.environ.get("PF_HI", 5.0)), float(os.environ.get("PF_LO", 3.0))
ZK = float(os.environ.get("PF_ZK", 4.0))
MIN_AREA = int(os.environ.get("PF_MINA", 6))
GROW = 2
BIG = int(os.environ.get("PF_BIG", 350))
STRUCT_RATIO = float(os.environ.get("PF_STRUCT", 2.5))
FWHM = float(CFG.CFG.get("fwhm", 3.0))   # v5 : FWHM de la piste 1 de la decoupe (resume.json) ; disques de sauvetage <= 2 x FWHM
DISC_CAP = max(2, int(round(2 * FWHM)))
STARS = None                              # v5 : liste des sources (x, y, G) de t1, fixee dans main


SIG_PETIT, SIG_GRAND = 0.5, 2.0     # v9 : echelle de l'etoile, DoG sigma 0,5 FWHM - sigma 2 FWHM
MATCH_PX = 6.0                      # v9 : centroide du blob a l'echelle de l'etoile a <= 6 px d'une source


def dog_stack(img):
    """v9 : residu passe-bande a l'echelle de l'etoile, par canal R, G, B, L, en bruits (MAD) -> (4, H, W)."""
    chans = [img[0], img[1], img[2], img.mean(0)]
    out = []
    for a in chans:
        a = a.astype(np.float32)
        d = cv2.GaussianBlur(a, (0, 0), SIG_PETIT * FWHM) - cv2.GaussianBlur(a, (0, 0), SIG_GRAND * FWHM)
        out.append(d / max(mad(d[::2, ::2]), 1e-9))
    return np.stack(out)


def star_blob(D, cmask, tree):
    """v9 : blob a l'echelle de l'etoile dans la zone cmask (composante agrandie de 2 px) : pic de |DoG| sur les 4 canaux, region
    connexe ou signe*DoG > moitie du pic. Rend (rond, rayon equivalent, distance centroide-source) ou None."""
    ys, xs = np.nonzero(cmask)
    sub = D[:, ys, xs]
    c, i = np.unravel_index(np.argmax(np.abs(sub)), sub.shape)
    pk = float(sub[c, i]); sg = np.sign(pk)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    loc = (sg * D[c, y0:y1, x0:x1] > 0.5 * abs(pk)) & cmask[y0:y1, x0:x1]
    lab, _ = ndi.label(loc)
    l0 = lab[ys[i] - y0, xs[i] - x0]
    by, bx = np.nonzero(lab == l0)
    area = len(bx)
    cov = (np.cov(np.stack([bx, by]), bias=True) if area > 1 else np.zeros((2, 2))) + np.eye(2) / 12.0
    ev = np.linalg.eigvalsh(cov)
    elong = float(np.sqrt(ev[1] / ev[0]))
    cx, cy = bx.mean() + x0, by.mean() + y0
    d = float(tree.query([cx, cy])[0])
    return elong, float(np.sqrt(area / np.pi)), d


def star_leftovers_only(m, stars, log=None, tag="", img=None):
    """v9 : une composante detectee est un reste d'etoile si son blob A L'ECHELLE DE L'ETOILE (DoG sigma 0,5 FWHM - sigma 2 FWHM de
    img, pic sur R, G, B, L, region a plus de la moitie du pic dans la composante agrandie de 2 px) est RONDE (rapport des axes < 2),
    de rayon equivalent <= 2 FWHM, de centroide a <= 6 px d'une source (Gaia, Gaia profond, piste 1). Les pixels gueris sont ceux de la
    composante dans le disque de zone de l'etoile (rayon croissant avec la brillance, comme star_zone). Rend (masque garde, comptes)."""
    from scipy.spatial import cKDTree
    lab, n = ndi.label(m)
    out = np.zeros(m.shape, bool)
    cnt = dict(detectees=int(n), gardees=0, gardees_aigrette=0, rejet_allongee=0, rejet_decentree=0, rejet_trop_grande=0)
    if n == 0 or len(stars) == 0:
        cnt["rejet_decentree"] = int(n)
        return out, cnt
    tree = cKDTree(stars[:, :2])
    D = dog_stack(img)
    H, W = m.shape
    BANDES = spike_bands(H, W)
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        comp = lab[sl] == k
        # v21 : un reste dont au moins 60 % des pixels sont dans la bande d'une aigrette connue d'une brillante est un reste
        # d'aigrette : rebouché (pixels de la bande), quelle que soit sa forme (allongée par nature) ; ailleurs, règle v9.
        bk = BANDES[sl]
        if bk[comp].mean() >= 0.6:
            out[sl] |= comp & bk
            cnt["gardees_aigrette"] += 1
            continue
        pad = 3
        y0, x0 = max(sl[0].start - pad, 0), max(sl[1].start - pad, 0)
        y1, x1 = min(sl[0].stop + pad, H), min(sl[1].stop + pad, W)
        cm = np.zeros((y1 - y0, x1 - x0), bool)
        cm[sl[0].start - y0:sl[0].stop - y0, sl[1].start - x0:sl[1].stop - x0] = comp
        cmg = ndi.binary_dilation(cm, iterations=2)
        Dl = D[:, y0:y1, x0:x1]
        class _T:  # arbre en coordonnees locales
            def query(self, p): return tree.query([p[0] + x0, p[1] + y0])
        elong, req, d = star_blob(Dl, cmg, _T())
        if elong >= 2.0:
            cnt["rejet_allongee"] += 1; continue
        if req > 2 * FWHM:
            cnt["rejet_trop_grande"] += 1; continue
        if d > MATCH_PX:
            cnt["rejet_decentree"] += 1; continue
        # pixels gueris : la composante restreinte au disque de zone des sources voisines
        ys, xs = np.nonzero(comp); ys = ys + sl[0].start; xs = xs + sl[1].start
        cx, cy = xs.mean(), ys.mean()
        idx = tree.query_ball_point([cx, cy], MATCH_PX + 2 * FWHM)
        zone = np.zeros(comp.shape, bool)
        yy, xx = np.mgrid[sl[0].start:sl[0].stop, sl[1].start:sl[1].stop]
        for i in idx:
            g = stars[i, 2]
            rz = int(np.clip(ZK * (14 - g), 5, 70)) if g < 14 else 5
            zone |= (xx - stars[i, 0]) ** 2 + (yy - stars[i, 1]) ** 2 <= rz * rz
        keep = comp & zone
        if not keep.any():
            cnt["rejet_decentree"] += 1; continue
        out[sl] |= keep
        cnt["gardees"] += 1
    if log:
        log(f"v9 {tag} : filtre restes d'etoile (blob DoG {SIG_PETIT}-{SIG_GRAND} FWHM, rond < 2, rayon <= 2 FWHM, centroide <= {MATCH_PX:.0f} px d'une source) : {cnt}")
    return out, cnt


def point_is_star_leftover(D, pt, stars_tree):
    """v9 : un pic isole (t3) est un reste d'etoile si le blob a l'echelle de l'etoile autour de lui est rond, <= 2 FWHM, a <= 6 px d'une source."""
    x, y = int(round(pt[0])), int(round(pt[1]))
    H, W = D.shape[1:]
    r = int(np.ceil(2 * FWHM)) + 1
    y0, y1, x0, x1 = max(y - r, 0), min(y + r + 1, H), max(x - r, 0), min(x + r + 1, W)
    cm = np.zeros((y1 - y0, x1 - x0), bool); cm[y - y0, x - x0] = True
    cm = ndi.binary_dilation(cm, iterations=r)
    class _T:
        def query(self, p): return stars_tree.query([p[0] + x0, p[1] + y0])
    elong, req, d = star_blob(D[:, y0:y1, x0:x1], cm, _T())
    return elong < 2.0 and req <= 2 * FWHM and d <= MATCH_PX


def mad(v):
    return float(np.median(np.abs(v - np.median(v))) * 1.4826)


def load_all():
    name, crop, tile = M.CASES[CASE]
    full = M.load_image(name)
    tx0, ty0, tx1, ty1 = tile
    I = full[:, crop[1] + ty0:crop[1] + ty1, crop[0] + tx0:crop[0] + tx1].copy()
    ox, oy = crop[0] + tx0, crop[1] + ty0                      # image entière = tuile + (ox, oy)
    z = np.load(M.P1 / "sorties" / name / "astrometrie.npz", allow_pickle=True)["cat"].item()
    gaia = np.stack([z["x"] - ox, z["y"] - oy, z["g"]], 1)
    if CFG.DEEP.exists():                                      # SHO 21 nov : chemin du cfg (defaut : rosette, inchange)
        d = np.load(CFG.DEEP)
        deep = np.stack([d["x_pix"] - ox, d["y_pix"] - oy, d["Gmag"]], 1)
    else:                                                      # pas de catalogue profond : l'extrait hors ligne seul
        deep = np.zeros((0, 3))
    cs = []
    with open(CFG.BASE_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                g = float(row["G"])
            except ValueError:
                g = np.nan
            cs.append((float(row["x"]) - tile[0], float(row["y"]) - tile[1], g))
    p1 = np.array(cs)
    return full, I, crop, tile, gaia, deep, p1


def star_list(H, W, gaia, deep, p1):
    """Toutes les sources à surveiller : (x, y, G) dans la tuile ; G NaN -> 15."""
    allv = [gaia[gaia[:, 2] < 18], deep[deep[:, 2] < 17.5], p1]
    s = np.concatenate(allv, 0)
    s[:, 2] = np.where(np.isfinite(s[:, 2]), s[:, 2], 15.0)
    keep = (s[:, 0] > -60) & (s[:, 0] < W + 60) & (s[:, 1] > -60) & (s[:, 1] < H + 60)
    return s[keep]


_BANDES = {}


def spike_bands(H, W):
    """v21 : bandes des aigrettes CONNUES des brillantes traitées par l'étape A (mesures.json, « etoiles » : position, angle,
    longueur et demi-largeur mesurés par étoile), aux seuls angles de l'instrument (« angles », ±4°) : longueur x 1,2 + 20 px,
    largeur 2 x max(demi-largeur, 4) + 1. Coordonnées de la tuile. Vide si l'étape A n'a pas de mesures."""
    if (H, W) in _BANDES:
        return _BANDES[(H, W)]
    z = np.zeros((H, W), np.uint8)
    try:
        mes = json.loads((CFG.A_OUT / "mesures.json").read_text(encoding="utf-8"))
    except Exception:
        mes = {}
    inst = [float(a) for a in mes.get("angles", ANGLES)]
    for e in mes.get("etoiles", []):
        x, y = float(e["x_tuile"]), float(e["y_tuile"])
        for s in e.get("aigrettes", []):
            a = float(s["angle"])
            da = [abs((a - b + 180) % 360 - 180) for b in inst]
            if not da or min(da) > 4:
                continue
            t = np.deg2rad(inst[int(np.argmin(da))])        # angle de l'instrument, fixe
            L = 1.2 * float(s["length"]) + 20
            w = 2 * max(int(s.get("hw", 4)), 4) + 1
            cv2.line(z, (int(round(x)), int(round(y))), (int(round(x + L * np.cos(t))), int(round(y + L * np.sin(t)))), 1, w)
    _BANDES[(H, W)] = z.astype(bool)
    return _BANDES[(H, W)]


def star_zone(H, W, stars):
    """Zone où un reste d'étoile est admis : disque de rayon croissant avec la brillance + bandes d'aigrettes."""
    z = np.zeros((H, W), np.uint8)
    core = np.zeros((H, W), np.uint8)
    for x, y, g in stars:
        r = int(np.clip(ZK * (14 - g), 5, 70)) if g < 14 else 5
        cv2.circle(z, (int(round(x)), int(round(y))), r, 1, -1)
        cv2.circle(core, (int(round(x)), int(round(y))), 4, 1, -1)
        if g < 11:
            L = int(np.clip(14 * (11 - g), 10, 110))
            for a in ANGLES + [a + 90 for a in ANGLES]:
                t = np.deg2rad(a)
                p0 = (int(round(x)), int(round(y)))
                p1 = (int(round(x + L * np.cos(t))), int(round(y + L * np.sin(t))))
                p2 = (int(round(x - L * np.cos(t))), int(round(y - L * np.sin(t))))
                cv2.line(z, p1, p2, 1, 9)
    # v21 : la zone couvre aussi les aigrettes connues sur leur longueur mesurée (G 5,23 de M8 : jusqu'à 245 px, au lieu
    # de 81 px par la formule ci-dessus), sinon leurs restes lointains ne sont même pas détectés
    return z.astype(bool) | spike_bands(H, W), core.astype(bool)


def robust_bg(a, sigma=9.0, iters=3, hp_sig=None):
    """Fond local : gaussienne normalisée sur les pixels non aberrants (rejet > 2.5 bruits, dilaté de 2 px)."""
    w = np.ones(a.shape, np.float32)
    bg = a
    for _ in range(iters + 1):
        num = cv2.GaussianBlur(a * w, (0, 0), sigma)
        den = cv2.GaussianBlur(w, (0, 0), sigma)
        bg = num / np.maximum(den, 1e-4)
        r = cv2.GaussianBlur(a - bg, (0, 0), 1.2)
        s = mad(r)
        out = np.abs(r) > 2.5 * s
        out = ndi.binary_dilation(out, iterations=2)
        w = (~out).astype(np.float32)
    return bg.astype(np.float32)


def residual_z(R):
    """Résidu lissé 1.2 px en bruits ; canaux R, G, B, L -> (4, H, W)."""
    chans = [R[0], R[1], R[2], R.mean(0)]
    Z = []
    for a in chans:
        a = a.astype(np.float32)
        bg = robust_bg(a)
        r = cv2.GaussianBlur(a - bg, (0, 0), 1.2)
        Z.append(r / mad(r[::2, ::2]))
    return np.stack(Z)


def structure_map(R):
    """Structure à 16-48 px : écart type local de G(8) - G(24) de la luminance."""
    L = R.mean(0).astype(np.float32)
    d = cv2.GaussianBlur(L, (0, 0), 8) - cv2.GaussianBlur(L, (0, 0), 24)
    return np.sqrt(cv2.GaussianBlur(d * d, (0, 0), 20))


def detect(R, zone, ignore_prev):
    Z = residual_z(R)
    A = np.abs(Z).max(0)
    hi = (A > Z_HI)
    lo = (A > Z_LO) & zone
    lab, n = ndi.label(lo)
    if n == 0:
        return np.zeros(R.shape[1:], bool), Z
    keep = np.zeros(n + 1, bool)
    hm = ndi.maximum(hi & zone, lab, np.arange(1, n + 1))
    keep[1:] = hm > 0
    m = keep[lab]
    # petits bouts
    lab2, n2 = ndi.label(m)
    if n2:
        sz = ndi.sum(m, lab2, np.arange(1, n2 + 1))
        ok = np.zeros(n2 + 1, bool); ok[1:] = sz >= MIN_AREA
        m = ok[lab2]
    return m & ~ignore_prev, Z


def rescue_discs(comp, A, ignore, H, W):
    """v5 : dans une zone grande ET structuree (non remplie en v4), disques autour des etoiles du catalogue dont le pic
    de residu (|Z| max des 4 canaux) dans le disque, restreint a la zone, depasse le seuil haut. Disque restreint a la zone. Rayon = rayon de zone de t1
    plafonne a 2 x FWHM. Rend (masque des disques, nombre de disques)."""
    out = np.zeros((H, W), bool)
    n = 0
    ys, xs = np.nonzero(comp)
    x0, x1, y0, y1 = xs.min() - DISC_CAP - 1, xs.max() + DISC_CAP + 1, ys.min() - DISC_CAP - 1, ys.max() + DISC_CAP + 1
    for x, y, g in STARS:
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            continue
        r = int(np.clip(ZK * (14 - g), 5, 70)) if g < 14 else 5
        r = min(r, DISC_CAP)
        d = np.zeros((H, W), np.uint8)
        cv2.circle(d, (int(round(x)), int(round(y))), r, 1, -1)
        d = d.astype(bool)
        if not (d & comp).any() or A[d & comp].max() <= Z_HI:
            continue
        out |= d & comp & ~ignore
        n += 1
    return out, n


def run_pass(R, carte_prev, zone, core, ignore, noise, log, tag):
    H, W = R.shape[1:]
    m, Z = detect(R, zone, ignore)
    m, _cnt = star_leftovers_only(m, STARS, log, tag, img=R)      # v9
    lab, n = ndi.label(m)
    S = structure_map(R)
    Smed = float(np.median(S))
    ign_new = np.zeros((H, W), bool)
    fill = np.zeros((H, W), bool)
    info = []
    A = np.abs(Z).max(0)
    n_discs = 0
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        comp = lab == k
        area = int(comp.sum())
        ring = ndi.binary_dilation(comp, iterations=14) & ~ndi.binary_dilation(comp, iterations=4)
        st = float(np.median(S[ring])) / Smed if ring.any() else 0.0
        big_struct = area >= BIG and st > STRUCT_RATIO
        info.append(dict(aire=area, structure=round(st, 2), ignoree=bool(big_struct)))
        if big_struct:
            dsc, nd = rescue_discs(comp, A, ignore, *comp.shape)
            info[-1]["disques"] = nd; info[-1]["px_disques"] = int(dsc.sum())
            n_discs += nd
            fill |= dsc
            ign_new |= comp & ~ndi.binary_dilation(dsc, iterations=GROW)
        else:
            fill |= comp
    log(f"{tag} v5 : {n_discs} disques de sauvetage, {int(sum(i.get('px_disques', 0) for i in info))} px (rayon max {DISC_CAP} px = 2 x FWHM {FWHM:.2f})")
    log(f"{tag} : {n} restes détectés ({int(m.sum())} px), {int(fill.any() and ndi.label(fill)[1])} à remplir, {len(info) and sum(i['ignoree'] for i in info)} à ignorer ; structure médiane du ciel {Smed:.3f}")
    # masque agrandi de 2 px, fondu 2 px
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * GROW + 1, 2 * GROW + 1)).astype(bool)
    grown = ndi.binary_dilation(fill, structure=disk) & ~ign_new
    n_fill = int(ndi.label(grown)[1])
    if n_fill == 0:
        return R, 0, ign_new, info, np.zeros((H, W), bool)
    forbid = carte_prev | ndi.binary_dilation(m, iterations=3) | core | ign_new | ndi.binary_dilation(ignore, iterations=2)
    filled = F.fill_mask(R.astype(np.float32), grown, forbid, noise)
    # fondu : poids 1 dans le masque agrandi, rampe de 2 px au-dehors
    dist = ndi.distance_transform_edt(~grown)
    w = np.clip(1 - dist / 2.0, 0, 1).astype(np.float32)
    w[ign_new] = 0
    out = R + w[None] * (filled - R)
    return out.astype(np.float32), n_fill, ign_new, info, grown


def main(view_only=False):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    logf = OUT / "journal.txt"
    def log(s):
        print(s, flush=True)
        with open(logf, "a", encoding="utf-8") as f:
            f.write(s + "\n")
    full, I, crop, tile, gaia, deep, p1 = load_all()
    H, W = I.shape[1:]
    R0 = np.load(BASE / "sans_etoiles_tuile.npy").astype(np.float32)
    carte0 = np.load(BASE / "carte_remplissage_tuile.npy")
    stars = star_list(H, W, gaia, deep, p1)
    global STARS
    STARS = stars
    zone, core = star_zone(H, W, stars)
    log(f"sources surveillées {len(stars)}, zone admise {100 * zone.mean():.1f} % de la tuile")
    noise = [mad(c - cv2.GaussianBlur(c, (0, 0), 6)) for c in R0]
    log(f"bruit du résultat c9 par couleur {np.round(noise, 3).tolist()}")
    R = R0.copy()
    ignore = np.zeros((H, W), bool)
    carte = carte0 > 0
    meas = dict(seuils=dict(haut=Z_HI, bas=Z_LO, aire_min=MIN_AREA, grande_aire=BIG, rapport_structure=STRUCT_RATIO, agrandissement_px=GROW))
    passes = []
    for p in (1, 2):
        R, n_fill, ign, info, grown = run_pass(R, carte, zone, core, ignore, noise, log, f"passe {p}")
        ignore |= ign
        carte = carte | grown
        passes.append(dict(passe=p, zones_remplies=n_fill, zones_detectees=len(info), zones_ignorees=int(sum(i['ignoree'] for i in info)), disques_ajoutes=int(sum(i.get('disques', 0) for i in info)), px_disques=int(sum(i.get('px_disques', 0) for i in info)),
                           px_masque_rempli=int(grown.sum()), px_changes_cumules=int((R != R0).any(0).sum())))
    ch = (R != R0).any(0)
    carte_new = carte0.copy()
    carte_new[ch & (carte0 == 0)] = 5                         # 5 = repris par la passe finale
    meas["passes"] = passes
    meas["part_inventee_c9_pct"] = round(100 * float((carte0 > 0).mean()), 2)
    meas["part_inventee_finale_pct"] = round(100 * float((carte_new > 0).mean()), 2)
    meas["part_changee_pct"] = round(100 * float(ch.mean()), 2)
    meas["part_a_ignorer_pct"] = round(100 * float(ignore.mean()), 2)
    meas["classe_5"] = "pixel repris par la passe finale (résultat != base c9) ; classes 1-4 : celles de c9"
    meas["deviations"] = "aigrettes : angles mesurés par c9 (journal), pas remesurés ; fond : gaussienne normalisée 9 px à rejet itératif"
    np.save(OUT / "sans_etoiles_tuile.npy", R)
    np.save(OUT / "carte_remplissage_tuile.npy", carte_new)
    np.save(OUT / "masque_ignorer.npy", ignore)
    (OUT / "mesures.json").write_text(json.dumps(meas, ensure_ascii=False, indent=1), encoding="utf-8")
    log(json.dumps(meas, ensure_ascii=False))
    views(full, I, crop, tile, R0, R, log)
    log(f"fini en {time.time() - t0:.0f} s")


def views(full, I, crop, tile, R0, R, log):
    g = full[1]; bk = float(np.median(g[::7, ::7])); t998 = float(np.percentile(g[::5, ::5], 99.8))
    def post(a):
        x = np.clip((a - bk) / max(t998 - bk, 1e-9), 0, None); v = np.arcsinh(40 * x) / np.arcsinh(40)
        return (np.transpose(np.clip(v, 0, 1), (1, 2, 0)) * 255 + 0.5).astype(np.uint8)
    L = I.mean(0); sig = M.noise_sigma(L)
    hard = lambda a: to_rgb8(a, -2 * sig, float(np.percentile(L, 60)) + 15 * sig)
    x2 = lambda im: cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # vue du résultat : même cadrage que celle de c9 (bande blanche de 8 px à gauche, 1008 colonnes de l'image x2)
    v = x2(post(R)); v = np.concatenate([np.full((v.shape[0], 8, 3), 255, np.uint8), v[:, :1008]], 1)
    save_img(OUT / "vue_resultat_x2.png", v)
    sep = np.full((1024, 8, 3), 255, np.uint8)
    save_img(OUT / "cote_a_cote_x2.png", np.concatenate([x2(post(R0)), sep, x2(post(R))], 1))
    save_img(OUT / "cote_a_cote_etire_fort_x2.png", np.concatenate([x2(hard(R0)), sep, x2(hard(R))], 1))
    ox, oy = crop[0], crop[1]
    for fx, fy in SIX:
        cx, cy = fx - ox - tile[0], fy - oy - tile[1]
        ya, yb, xa, xb = max(0, cy - 40), min(R.shape[1], cy + 40), max(0, cx - 40), min(R.shape[2], cx + 40)
        a = post(R0[:, ya:yb, xa:xb]); b = post(R[:, ya:yb, xa:xb])
        sp = np.full((yb - ya, 4, 3), 255, np.uint8)
        row = np.concatenate([a, sp, b], 1)
        save_img(OUT / f"zoom_{fx}_{fy}_x3_c9_nouveau.png", cv2.resize(row, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))


if __name__ == "__main__":
    main()
