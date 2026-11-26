# ======================================================================
# chercher_galaxies.py
# Rôle     : Cherche les petites galaxies compactes d'images JAMAIS vues a l'entrainement (jeu de controle, ne sert qu'a juger).
# Étape    : 13. Jeu de contrôle des petites galaxies (26 nov) (voir scripts/LISEZMOI.md)
# Entrées  : images web et extraits Gaia (non publiés)
# Sorties  : controle.json, mesures par réseau
# Exemple  : chercher_galaxies.py [image ...]       (par defaut : les 7 images APOD sans aucune tuile d'entrainement)
# ======================================================================
r"""Cherche les petites galaxies compactes d'images JAMAIS vues a l'entrainement (jeu de controle, ne sert qu'a juger).

    chercher_galaxies.py [image ...]       (par defaut : les 7 images APOD sans aucune tuile d'entrainement)

Entree : <DONNEES>\Paires étoiles APOD – 24 nov\entrees\<image>\input.npy (unites de bruit, fond 0) + extrait Gaia deja prepare
(<DONNEES>\Catalogue Gaia profond APOD – 24 nov). Aucun telechargement.
Critères : source detectee ABSENTE de Gaia (aucune source a moins de 1,2 FWHM d'etoile), FWHM >= 1,2 x FWHM des etoiles voisines, crete >= 20 bruits,
loin des etoiles brillantes, pas dans une grande galaxie (couronne faible), pas de bord, pas de trainee.
Sortie : <DONNEES>\Controle galaxies compactes – 26 nov\candidats\<image>.json
"""
import os, sys, json, glob, time
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np, cv2
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

APOD = Path(glob.glob(r"<DONNEES>\Paires ?toiles APOD*")[0])
GAIA = Path(glob.glob(r"<DONNEES>\Catalogue Gaia profond APOD*")[0])
OUT = Path(glob.glob(r"<DONNEES>\Controle galaxies compactes*")[0]) if glob.glob(r"<DONNEES>\Controle galaxies compactes*") \
    else (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Controle galaxies compactes – 26 nov")
SANS_TUILE = ["apod_017", "apod_023", "apod_025", "apod_042", "apod_046", "apod_059", "apod_071"]
MARGE = 192
TUILE = 512
MARGE_TUILE = 64


def tuiles_apod():
    """{image: [(x, y)]} : coins haut gauche des tuiles de TOUTES les paires APOD (toutes versions) ; l'entrainement n'y a vu que ces zones."""
    import re
    t = {}
    for f in os.listdir(APOD):
        m = re.match(r"(apod_\d+)_g(\d+)_(\d+)_", f)
        if m and f.endswith(".npz"):
            t.setdefault(m.group(1), set()).add((int(m.group(2)), int(m.group(3))))
    return {k: sorted(v) for k, v in t.items()}


def dist_tuile(tuiles, x, y):
    """distance (px) du point au rectangle de tuile le plus proche (0 s'il est dedans) ; None s'il n'y a aucune tuile."""
    if not tuiles:
        return None
    d = [np.hypot(max(tx - x, 0, x - (tx + TUILE)), max(ty - y, 0, y - (ty + TUILE))) for tx, ty in tuiles]
    return float(min(d))
R_PROF = 24


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


def profil(L, x, y, bg_r=(16, 24)):
    """FWHM par profil radial moyen autour du centroide sous-pixel ; renvoie (fwhm, crete, fond, cx, cy, flux3, flux12, ell)."""
    H, W = L.shape
    xi, yi = int(round(x)), int(round(y))
    if xi < R_PROF + 2 or yi < R_PROF + 2 or xi >= W - R_PROF - 2 or yi >= H - R_PROF - 2:
        return None
    s = L[yi - R_PROF:yi + R_PROF + 1, xi - R_PROF:xi + R_PROF + 1].astype(np.float64)
    yy, xx = np.mgrid[-R_PROF:R_PROF + 1, -R_PROF:R_PROF + 1].astype(np.float64)
    r0 = np.hypot(yy, xx)
    fond = float(np.median(s[(r0 >= bg_r[0]) & (r0 <= bg_r[1])]))
    t = s - fond
    # centroide pondere (rayon 3 px)
    m = r0 <= 3
    w = np.clip(t[m], 0, None)
    if w.sum() <= 0:
        return None
    cx = float((xx[m] * w).sum() / w.sum()); cy = float((yy[m] * w).sum() / w.sum())
    r = np.hypot(yy - cy, xx - cx)
    crete = float(ndi.gaussian_filter(t, 0.7)[R_PROF - 1:R_PROF + 2, R_PROF - 1:R_PROF + 2].max())
    if crete <= 0:
        return None
    nb = np.minimum((r * 2).astype(int), 2 * R_PROF)
    pm = np.bincount(nb.ravel(), t.ravel(), 2 * R_PROF + 1) / np.maximum(np.bincount(nb.ravel(), minlength=2 * R_PROF + 1), 1)
    rr = (np.arange(2 * R_PROF + 1) + 0.5) / 2
    p0 = max(pm[0], pm[1], crete * 0.9)
    half = p0 / 2
    k = np.flatnonzero(pm[1:] < half)
    if len(k) == 0:
        return None
    k = k[0] + 1
    a, b = pm[k - 1], pm[k]
    rh = rr[k - 1] + (a - half) / max(a - b, 1e-9) * (rr[k] - rr[k - 1])
    fwhm = 2 * float(rh)
    f3 = float(t[r <= 3].sum()); f12 = float(t[r <= 12].sum())
    # ellipticite : moments d'ordre 2 dans un rayon de 1,5 FWHM
    mm = (r <= max(1.5 * fwhm, 3)) & (t > 0)
    ww = t[mm]
    if ww.sum() <= 0:
        return None
    xs, ys = xx[mm] - cx, yy[mm] - cy
    cxx = (ww * xs * xs).sum() / ww.sum(); cyy = (ww * ys * ys).sum() / ww.sum(); cxy = (ww * xs * ys).sum() / ww.sum()
    tr, det = cxx + cyy, cxx * cyy - cxy ** 2
    ev = np.sort(np.linalg.eigvalsh(np.array([[cxx, cxy], [cxy, cyy]])))
    ell = float(np.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9)))
    ring = float(np.median(t[(r >= 10) & (r <= 16)]))
    return dict(fwhm=fwhm, crete=crete, fond=fond, cx=xi + cx, cy=yi + cy, f3=f3, f12=f12, ell=ell, couronne=ring / crete,
                p3f=float(np.interp(3 * fwhm, rr, pm) / p0) if 3 * fwhm < rr[-1] else 0.0)


def bruit_local(L):
    hp = L - cv2.GaussianBlur(L, (0, 0), 3)
    H, W = L.shape
    b = 64
    h2, w2 = H // b, W // b
    a = np.abs(hp[:h2 * b, :w2 * b]).reshape(h2, b, w2, b).transpose(0, 2, 1, 3).reshape(h2, w2, -1)
    med = np.median(a, axis=2) * 1.4826
    return cv2.resize(med.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)


def traiter(nom):
    t0 = time.time()
    img = np.array(np.load(APOD / "entrees" / nom / "input.npy", mmap_mode="r"), np.float32)
    C, H, W = img.shape
    L = img.mean(0)
    z = np.load(GAIA / f"gaia_dr3_profond_{nom}.npz")
    gx, gy, gg = z["x_pix"].astype(float), z["y_pix"].astype(float), z["Gmag"].astype(float)
    ok = np.isfinite(gg) & (gx > -50) & (gx < W + 50) & (gy > -50) & (gy < H + 50)
    gx, gy, gg = gx[ok], gy[ok], gg[ok]
    sig = bruit_local(L)
    Ls = cv2.GaussianBlur(L, (0, 0), 1.0)
    gt = cKDTree(np.c_[gx, gy])
    # ---- FWHM des etoiles : Gaia 12,5 < G < 17,5, isolees, ni faibles ni saturees
    sat = float(np.percentile(L, 99.99))
    ref = []
    for gmin, gmax, cmin, cmax in ((12.5, 17.5, 30, 0.5), (11.0, 19.5, 15, 0.9)):     # 2e essai si peu d'etoiles de reference
        idx = np.flatnonzero((gg > gmin) & (gg < gmax) & (gx > 40) & (gx < W - 40) & (gy > 40) & (gy < H - 40))
        rng = np.random.default_rng(0)
        idx = idx[rng.permutation(len(idx))[:2500]]
        for i in idx:
            d, _ = gt.query([gx[i], gy[i]], k=2)
            if d[1] < 10:
                continue
            p = profil(L, gx[i], gy[i])
            if p is None or p["crete"] < cmin * sig[int(gy[i]), int(gx[i])] or p["crete"] > cmax * sat or np.hypot(p["cx"] - gx[i], p["cy"] - gy[i]) > 2.5:
                continue
            if 1.0 < p["fwhm"] < 40:
                ref.append((p["cx"], p["cy"], p["fwhm"], p["crete"]))
        if len(ref) >= 15:
            break
    ref = np.array(ref)
    if len(ref) < 15:
        log(f"{nom} : seulement {len(ref)} etoiles de reference ; ignoree")
        return None
    rt = cKDTree(ref[:, :2])
    fw_glob = float(np.median(ref[:, 2]))
    log(f"{nom} {W}x{H} : {len(ref)} etoiles de reference, FWHM mediane {fw_glob:.2f} px, {len(gx)} sources Gaia")

    def fw_local(x, y):
        d, j = rt.query([x, y], k=min(15, len(ref)))
        return float(np.median(ref[np.atleast_1d(j), 2]))

    # ---- pics
    mx = ndi.maximum_filter(Ls, size=7)
    pk = np.argwhere((Ls == mx) & (Ls > 20 * sig * 0.6) & (np.arange(H)[:, None] > MARGE) & (np.arange(H)[:, None] < H - MARGE)
                     & (np.arange(W)[None, :] > MARGE) & (np.arange(W)[None, :] < W - MARGE))
    log(f"  {len(pk)} pics bruts")
    # pics detectes (pour isolement) : tous ceux > 8 bruits
    pk8 = np.argwhere((Ls == mx) & (Ls > 8 * sig))
    pt = cKDTree(pk8[:, ::-1].astype(float)); pv = Ls[pk8[:, 0], pk8[:, 1]]
    bright = gg < 15
    bt = cKDTree(np.c_[gx[bright], gy[bright]]) if bright.any() else None
    res = []
    TUI = tuiles_apod().get(nom, [])
    rej = dict(tuile=0, gaia=0, profil=0, crete=0, fwhm=0, ell=0, couronne=0, brillante=0, voisin=0, trop_large=0)
    for yp, xp in pk:
        dt = dist_tuile(TUI, xp, yp)
        if dt is not None and dt < MARGE_TUILE:
            rej["tuile"] += 1; continue
        fs = fw_local(xp, yp)
        dg, _ = gt.query([xp, yp])
        if dg < max(1.2 * fs, 3.0):
            rej["gaia"] += 1; continue
        p = profil(L, xp, yp)
        if p is None:
            rej["profil"] += 1; continue
        if p["crete"] < 20 * sig[yp, xp]:
            rej["crete"] += 1; continue
        ratio = p["fwhm"] / fs
        if ratio < 1.2:
            rej["fwhm"] += 1; continue
        if ratio > 3.5:
            rej["trop_large"] += 1; continue
        if p["ell"] > 2.5:
            rej["ell"] += 1; continue
        if p["couronne"] > 0.25 or p["p3f"] > 0.3:
            rej["couronne"] += 1; continue
        dg2, _ = gt.query([p["cx"], p["cy"]])
        if dg2 < max(1.2 * fs, 3.0):
            rej["gaia"] += 1; continue
        if bt is not None and bt.query([xp, yp])[0] < 10 * fs:
            rej["brillante"] += 1; continue
        nb = pt.query_ball_point([xp, yp], 4 * max(p["fwhm"], fs))
        if any(pv[j] > 0.3 * Ls[yp, xp] and (abs(pk8[j, 1] - xp) + abs(pk8[j, 0] - yp)) > 0 for j in nb):
            rej["voisin"] += 1; continue
        res.append(dict(image=nom, x=round(p["cx"], 2), y=round(p["cy"], 2), fwhm=round(p["fwhm"], 2), fwhm_etoiles=round(fs, 2),
                        rapport_fwhm=round(ratio, 2), crete_bruits=round(p["crete"] / float(sig[yp, xp]), 1), part_3px=round(p["f3"] / max(p["f12"], 1e-9), 3),
                        flux3=round(p["f3"], 1), flux12=round(p["f12"], 1), ellipticite=round(p["ell"], 2), couronne=round(p["couronne"], 3),
                        dist_gaia=round(float(dg2), 1), nb_tuiles_image=len(TUI), dist_tuile_px=None if dt is None else round(dt, 1), W=W, H=H))
    log(f"  {len(res)} candidats ; rejets {rej} ; {time.time() - t0:.0f} s")
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "candidats").mkdir(exist_ok=True)
    (OUT / "candidats" / f"{nom}.json").write_text(json.dumps(dict(image=nom, fwhm_etoiles_mediane=fw_glob, candidats=res, rejets=rej), indent=1, ensure_ascii=False), encoding="utf-8")
    return res


if __name__ == "__main__":
    if sys.argv[1:] == ["--toutes"]:
        noms = sorted(Path(f).stem.replace("gaia_dr3_profond_", "") for f in glob.glob(str(GAIA / "gaia_dr3_profond_apod_*.npz")))
    else:
        noms = sys.argv[1:] or SANS_TUILE
    for nom in noms:
        traiter(nom)
