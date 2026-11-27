# ======================================================================
# passe_finale.py
# Role     : Final pass of 21 Nov (hybrid c9, crop of the 6 stars): go over the star remnants (bright or hollow) again, like an operator with the healing tool, once the bulk of the removal is done.
# Step     : 2. Classic Gaia-guided subtraction chain (v33b) (see scripts/LISEZMOI.md)
# Inputs   : stacked image (noise units), astrometry, offline deep Gaia extract
# Outputs  : star-free image, class maps
# Example  : python passe_finale.py [arguments]  (see the docstring; paths through the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
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
import mesure as M
import remplir as F
from juger import save_img, to_rgb8

BASE = HERE / "sorties" / "hybride_c9" / "rosette_decoupe6"
OUT = HERE / "sorties" / "passe_finale_c9"
CASE = "rosette_decoupe6"
SIX = [(1706, 1009), (2043, 968), (2038, 1036), (1762, 1159), (2078, 1220), (1999, 1327)]
ANGLES = [3.0, 11.5, 183.0, 191.5]          # aigrettes de l'instrument (journal c9) ; + perpendiculaires
Z_HI, Z_LO = float(os.environ.get("PF_HI", 5.0)), float(os.environ.get("PF_LO", 3.0))
ZK = float(os.environ.get("PF_ZK", 4.0))
MIN_AREA = int(os.environ.get("PF_MINA", 6))
GROW = 2
BIG = int(os.environ.get("PF_BIG", 350))
STRUCT_RATIO = float(os.environ.get("PF_STRUCT", 2.5))


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
    d = np.load(r"<SCRIPTS>\Catalogue Gaia profond – 20 nov\gaia_dr3_profond_rosette.npz")
    deep = np.stack([d["x_pix"] - ox, d["y_pix"] - oy, d["Gmag"]], 1)
    cs = []
    with open(HERE / "sorties" / "base_piste1" / CASE / "c9_decoupe6_m200" / "etoiles.csv", encoding="utf-8") as f:
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
    return z.astype(bool), core.astype(bool)


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


def run_pass(R, carte_prev, zone, core, ignore, noise, log, tag):
    H, W = R.shape[1:]
    m, Z = detect(R, zone, ignore)
    lab, n = ndi.label(m)
    S = structure_map(R)
    Smed = float(np.median(S))
    ign_new = np.zeros((H, W), bool)
    fill = np.zeros((H, W), bool)
    info = []
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        comp = lab == k
        area = int(comp.sum())
        ring = ndi.binary_dilation(comp, iterations=14) & ~ndi.binary_dilation(comp, iterations=4)
        st = float(np.median(S[ring])) / Smed if ring.any() else 0.0
        big_struct = area >= BIG and st > STRUCT_RATIO
        info.append(dict(aire=area, structure=round(st, 2), ignoree=bool(big_struct)))
        if big_struct:
            ign_new |= comp
        else:
            fill |= comp
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
        passes.append(dict(passe=p, zones_remplies=n_fill, zones_detectees=len(info), zones_ignorees=int(sum(i['ignoree'] for i in info)),
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
