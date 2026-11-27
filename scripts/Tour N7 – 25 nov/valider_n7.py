# ======================================================================
# valider_n7.py
# Role     : Validation of a star network (or of starless images from another tool) on the never-seen images of the validation list.
# Stage    : 9. N7: synthetic galaxy cores (25 Nov) (see scripts/LISEZMOI.md)
# Inputs   : v33g pairs + synthetic cores
# Outputs  : N7 snapshots, tableau_n7.md
# Example  : python valider_reseau.py <checkpoint.pt | starless folder> [--images name ...] [--tag name] [--device cuda:0|cpu]
# ======================================================================
"""Validation of a star network (or of starless images from another tool) on the never-seen images of the validation list.

    python valider_reseau.py <checkpoint.pt | starless folder> [--images name ...] [--tag name] [--device cuda:0|cpu]

Per image (whole, tiled inference with overlap, network = chain_nafnet from models.py, asinh 3/10 forward/inverse as in train_jeu1):
 (a) remaining stars: remaining flux / original flux at the Gaia positions (2 FWHM aperture, background = median of the 3-5 FWHM ring
     taken in the same image), median and 90th percentile per G band (<8, 8-11, 11-14, 14-17, >17);
 (b) damage to the background: |output - input| (mean of the 3 colours, input noise units) on the pixels far from any detectable
     Gaia star (radius = max(4 FWHM, magnitude-specific radius)), median and 99th percentile, then the same on the
     "bright structure" (input smoothed above its 90th percentile);
 (c) exact error under synthetic stars: 20 tiles of 256 far from bright stars, paire_synthetique(background, seed)
     (target = background), L1 on the asinh stretch over the footprint of the stars, per star class (component crest);
 (d) contact sheet: input | output | difference (same stretch), whole image and two enlargements.
A folder of already-made starless images (<nom>.npy or <nom>/starless.npy, same units as the input) is scored the same way, except (c).
Outputs: <DONNEES>/Validation étoiles – 22 nov/Resultats/<tag>/<nom>.json, <nom>_planche.jpg, resume.json, resume.md.
Machine rules: 4 threads, one computation at a time; the GPU (4060 Ti) only if no produire_jeu1 / train_jeu1 is running.
"""
import os, sys, json, time, argparse, zlib, subprocess
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"                      # la 4060 Ti seule, jamais la 3070
os.environ["ASTRO_GPU_OFFLINE"] = "1"
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np
import cv2
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

SR = (Path(__import__("os").environ.get("RACINE_SCRIPTS") or Path(__file__).resolve().parents[1]))
ICI = (Path(__import__("os").environ.get("RACINE_SCRIPTS") or Path(__file__).resolve().parents[1]) / "Validation – 22 nov")
DATA = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Validation étoiles – 22 nov")
sys.path.insert(0, str(SR)); sys.path.insert(0, str(ICI)); sys.path.insert(0, str(SR / "Etoiles synthetiques – 22 nov"))
sys.path.insert(0, r"<FLOAT>")
from liste import LISTE

BANDES = [("<8", -9, 8), ("8-11", 8, 11), ("11-14", 11, 14), ("14-17", 14, 17), (">17", 17, 99)]
CLASSES = [("<30", 0, 30), ("30-300", 30, 300), ("300-3000", 300, 3000), (">3000", 3000, 1e12)]
PATCH = 256


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


def production_en_cours():
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
                        "(Get-CimInstance Win32_Process | ? { $_.CommandLine -match '[p]roduire_jeu1|[t]rain_jeu1' } | measure).Count"],
                       capture_output=True, text=True)
    return int((r.stdout or "0").strip() or 0) > 0


# ----------------------------------------------------------------------------------------------------- reseau
def charger_reseau(chemin, dev):
    import torch
    from models import chain_nafnet
    ck = torch.load(chemin, map_location=dev, weights_only=False)
    net = chain_nafnet().to(dev)
    net.load_state_dict({k: v.float() for k, v in ck["net"].items()})
    net.eval()
    return net, {"step": ck.get("step"), "model": ck.get("model", "chain_nafnet (reseau livre)")}


def inference(net, img, dev):
    from astro_gpu.stages import star_removal as SRm
    return SRm.apply(net, img, dev, None, tile=1024, overlap=256)


def inference_tuiles(net, X, dev):
    """X : (n, 3, 256, 256) -> sorties, meme etirement."""
    import torch
    from models import fwd, inv
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 10):
            b = torch.from_numpy(np.ascontiguousarray(X[i:i + 10])).to(dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                r = net(fwd(b)).float()
            out.append(inv(r).cpu().numpy())
    return np.concatenate(out)


# ----------------------------------------------------------------------------------------------------- donnees
def entree_de(nom):
    return Path(LISTE[nom][0]) / "input.npy"


def charger_gaia(nom, shape):
    z = np.load(DATA / nom / "gaia_dr3_profond.npz")
    x, y, g = z["x_pix"].astype(np.float64), z["y_pix"].astype(np.float64), z["Gmag"].astype(np.float64)
    ok = np.isfinite(g)
    return x[ok], y[ok], g[ok]


def ouverture(L, x, y, fw, R=None):
    """Flux dans 2 FWHM moins le fond de la couronne 3-5 FWHM, pour chaque etoile (vecteur) ; NaN si sur le bord."""
    H, W = L.shape
    r_out = int(np.ceil(5 * fw)) + 1
    yy, xx = np.mgrid[-r_out:r_out + 1, -r_out:r_out + 1]
    r2 = (yy ** 2 + xx ** 2).astype(np.float32)
    ap = r2 <= (2 * fw) ** 2
    an = (r2 >= (3 * fw) ** 2) & (r2 <= (5 * fw) ** 2)
    res = np.full(len(x), np.nan, np.float64)
    for i, (a, b) in enumerate(zip(x, y)):
        xi, yi = int(round(a)), int(round(b))
        if xi - r_out < 0 or yi - r_out < 0 or xi + r_out >= W or yi + r_out >= H:
            continue
        s = L[yi - r_out:yi + r_out + 1, xi - r_out:xi + r_out + 1]
        bg = np.median(s[an])
        res[i] = float((s[ap] - bg).sum())
    return res


def mesures_entree(nom, L, gx, gy, gg):
    """FWHM, bruit, magnitude limite : mesures sur l'entree, gardees pour que tous les reseaux soient notes pareil."""
    f = DATA / nom / "mesures_entree.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    H, W = L.shape
    valid = L != 0
    # bruit : MAD de L - mediane 5x5 sur un sous-echantillon de bandes
    sub = L[::2, ::2]
    res = sub - cv2.medianBlur(np.clip(sub, -50, 1e4).astype(np.float32), 5)
    bruit = float(1.4826 * np.median(np.abs(res[sub != 0]))) if (sub != 0).any() else 1.0
    # FWHM : etoiles G 11-14.5 isolees ; nb de pixels au-dessus de la moitie du pic -> diametre equivalent
    _s = gg < 18.5
    tree = cKDTree(np.c_[gx[_s], gy[_s]])
    cand = np.flatnonzero((gg > 10) & (gg < 16.5) & (gx > 30) & (gx < W - 30) & (gy > 30) & (gy < H - 30))
    rng = np.random.default_rng(0)
    cand = rng.permutation(cand)[:1500]
    fws = []
    for i in cand:
        if len(tree.query_ball_point([gx[i], gy[i]], 12)) > 1:
            continue
        xi, yi = int(round(gx[i])), int(round(gy[i]))
        s = L[yi - 15:yi + 16, xi - 15:xi + 16].astype(np.float32)
        s = s - np.median(np.r_[s[0], s[-1], s[:, 0], s[:, -1]])
        c = cv2.GaussianBlur(s, (3, 3), 0)
        p = c[11:20, 11:20].max()
        if p < 8 * bruit:
            continue
        lab, n = ndi.label(c > p / 2)
        k = lab[15, 15] if lab[15, 15] else lab[11:20, 11:20][np.unravel_index(np.argmax(c[11:20, 11:20]), (9, 9))]
        if k == 0:
            continue
        fws.append(2 * np.sqrt((lab == k).sum() / np.pi))
        if len(fws) >= 80:
            break
    fw = float(np.median(fws)) if len(fws) >= 5 else 4.0
    # magnitude limite : zero-point par les G 12-15, puis G ou le flux d'ouverture vaut 5 sigma
    sel = np.flatnonzero((gg > 12) & (gg < 15) & (gx > 40) & (gx < W - 40) & (gy > 40) & (gy < H - 40))[:600]
    fl = ouverture(L, gx[sel], gy[sel], fw)
    m = np.isfinite(fl) & (fl > 0)
    nap = np.pi * (2 * fw) ** 2
    if m.sum() >= 10:
        zp = float(np.median(gg[sel][m] + 2.5 * np.log10(fl[m])))
        g_det = zp - 2.5 * np.log10(5 * bruit * np.sqrt(nap))
    else:
        zp, g_det = float("nan"), 17.0
    out = {"fwhm_px": round(fw, 2), "bruit_pixel": round(bruit, 4), "nb_fwhm": len(fws), "zero_point": zp, "g_detectable": round(float(g_det), 2)}
    f.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def rayon_etoile(g, fw, fac):
    return np.maximum(fac * fw, fw * (fac + 2.5 * np.maximum(0.0, 14.0 - g)))


def masque_loin(shape, gx, gy, gg, fw, g_det, fac_init=4.0, ds=4):
    """True = pixel loin de toute etoile detectable (rayon max(fac FWHM, rayon de la magnitude)). fac baisse si il ne reste rien."""
    H, W = shape
    sel = gg <= g_det + 0.5
    x, y, g = gx[sel], gy[sel], gg[sel]
    for fac in (fac_init, 3.0, 2.0):
        far = np.ones((H // ds + 1, W // ds + 1), bool)
        r = rayon_etoile(g, fw, fac)
        niveaux = np.unique(np.round(np.log2(np.maximum(r, 1)) * 2) / 2)
        for lv in niveaux:
            k = np.round(np.log2(np.maximum(r, 1)) * 2) / 2 == lv
            m = np.ones_like(far)
            xi = np.clip((x[k] / ds).astype(int), 0, far.shape[1] - 1); yi = np.clip((y[k] / ds).astype(int), 0, far.shape[0] - 1)
            m[yi, xi] = False
            d = ndi.distance_transform_edt(m)
            far &= d * ds > 2 ** lv                      # rayon du niveau (quantifie au demi-octave)
        up = np.kron(far, np.ones((ds, ds), bool))[:H, :W]
        if up.mean() > 0.03 or fac == 2.0:
            return up, fac
    return up, 2.0


# ----------------------------------------------------------------------------------------------------- mesures
def etoiles_restantes(L_in, L_out, gx, gy, gg, fw, bruit, plafond=1500):
    H, W = L_in.shape
    rng = np.random.default_rng(1)
    res = {}
    nap = np.pi * (2 * fw) ** 2
    for nom, lo, hi in BANDES:
        idx = np.flatnonzero((gg >= lo) & (gg < hi) & (gg < 21))
        if len(idx) > plafond:
            idx = np.sort(rng.choice(idx, plafond, replace=False))
        fi = ouverture(L_in, gx[idx], gy[idx], fw)
        fo = ouverture(L_out, gx[idx], gy[idx], fw)
        m = np.isfinite(fi) & np.isfinite(fo) & (fi > 10 * bruit * np.sqrt(nap))
        r = fo[m] / fi[m]
        res[nom] = ({"n": int(m.sum()), "mediane": round(float(np.median(r)), 4), "p90": round(float(np.percentile(r, 90)), 4),
                     "part_gt_0.1": round(float((r > 0.1).mean()), 4)} if m.sum() >= 3 else {"n": int(m.sum())})
    return res


def degats_fond(D, far, struct):
    out = {}
    for nom, m in (("fond", far), ("structure_brillante", far & struct)):
        v = D[m]
        out[nom] = ({"pixels": int(m.sum()), "mediane": round(float(np.median(v)), 4), "p99": round(float(np.percentile(v, 99)), 4)}
                    if m.sum() > 1000 else {"pixels": int(m.sum())})
    return out


def tuiles_loin(nom, far, valid, n=20):
    """Positions de 20 tuiles 256 (graine fixe par image) : celles dont la part de pixels « loin » est la plus grande parmi 400 tirages."""
    H, W = far.shape
    rng = np.random.default_rng(zlib.crc32(nom.encode()))
    ys = rng.integers(0, H - PATCH + 1, 400); xs = rng.integers(0, W - PATCH + 1, 400)
    ii = cv2.integral((far & valid).astype(np.uint8)).astype(np.int64)
    sc = ii[ys + PATCH, xs + PATCH] - ii[ys, xs + PATCH] - ii[ys + PATCH, xs] + ii[ys, xs]
    ordre = np.argsort(-sc, kind="stable")
    chosen, pos = [], []
    for o in ordre:
        if all(abs(ys[o] - p[0]) > PATCH or abs(xs[o] - p[1]) > PATCH for p in pos):     # tuiles disjointes
            pos.append((int(ys[o]), int(xs[o]))); chosen.append(float(sc[o]) / PATCH ** 2)
        if len(pos) == n:
            break
    return pos, chosen


def synthetique(nom, net, dev, L_in3, far, valid):
    from paires_synthetiques import paire_synthetique
    from models import fwd
    pos, parts = tuiles_loin(nom, far, valid)
    ent, fon = [], []
    for k, (y, x) in enumerate(pos):
        fond = np.ascontiguousarray(L_in3[:, y:y + PATCH, x:x + PATCH]).astype(np.float32)
        e, c = paire_synthetique(fond, 1000 + k)
        ent.append(e); fon.append(c)
    ent, fon = np.stack(ent), np.stack(fon)
    sor = inference_tuiles(net, ent, dev)
    acc = {c[0]: {"l1": [], "l1_rien": [], "n": 0} for c in CLASSES}
    tous = {"l1": [], "l1_rien": []}
    for e, f, s in zip(ent, fon, sor):
        d = (e - f).mean(0)
        lab, n = ndi.label(ndi.binary_dilation(d > 0.5, iterations=1))
        if n == 0:
            continue
        pic = ndi.maximum(d, lab, np.arange(1, n + 1))
        el = np.abs(fwd(s) - fwd(f)).mean(0); er = np.abs(fwd(e) - fwd(f)).mean(0)
        for j in range(n):
            m = lab == j + 1
            for cn, lo, hi in CLASSES:
                if lo <= pic[j] < hi:
                    acc[cn]["l1"].append(float(el[m].sum())); acc[cn]["l1_rien"].append(float(er[m].sum()))
                    acc[cn]["pix"] = acc[cn].get("pix", 0) + int(m.sum()); acc[cn]["n"] += 1
        m = ndi.binary_dilation(lab > 0, iterations=2)
        tous["l1"].append((el[m].sum(), m.sum())); tous["l1_rien"].append((er[m].sum(), m.sum()))
    res = {"tuiles": len(pos), "part_loin_des_tuiles_mediane": round(float(np.median(parts)), 3), "classes": {}}
    for cn, a in acc.items():
        if a["n"]:
            res["classes"][cn] = {"etoiles": a["n"], "l1": round(sum(a["l1"]) / a["pix"], 6), "l1_sans_retrait": round(sum(a["l1_rien"]) / a["pix"], 6)}
    px = sum(p for _, p in tous["l1"])
    if px:
        res["toutes"] = {"l1": round(sum(a for a, _ in tous["l1"]) / px, 6), "l1_sans_retrait": round(sum(a for a, _ in tous["l1_rien"]) / px, 6)}
    return res, pos


# ----------------------------------------------------------------------------------------------------- planche
def vue(a, s=3.0, top=None):
    v = np.arcsinh(np.clip(a, 0, None) / s)
    return v


def planche(nom, img, out, gx, gy, gg, ratios_xy, dest, fw):
    H, W = img.shape[1:]
    top = float(np.percentile(np.arcsinh(np.clip(img.mean(0), 0, None) / 3.0), 99.97)) + 1e-6

    def rgb(a):      # (3,h,w) -> BGR uint8 sur l'etirement commun
        v = np.clip(np.arcsinh(np.clip(a, 0, None) / 3.0) / top, 0, 1)
        return (v.transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)

    def diff(a, b):  # entree - sortie : positif gris, negatif rouge
        d = a - b
        p = np.clip(np.arcsinh(np.clip(d, 0, None) / 3.0) / top, 0, 1).mean(0)
        n = np.clip(np.arcsinh(np.clip(-d, 0, None) / 3.0) / top, 0, 1).mean(0)
        o = np.stack([p, p, p], -1) * 255
        o[:, :, 2] = np.clip(o[:, :, 2] + n * 255, 0, 255)      # BGR : negatif en rouge
        return o.astype(np.uint8)

    w = 700
    sc = w / W
    rows = []
    pe = [cv2.resize(rgb(img), (w, int(H * sc)), interpolation=cv2.INTER_AREA)]
    ps = [cv2.resize(rgb(out), (w, int(H * sc)), interpolation=cv2.INTER_AREA)]
    pd = [cv2.resize(diff(img, out), (w, int(H * sc)), interpolation=cv2.INTER_AREA)]
    rows.append(np.hstack([pe[0], ps[0], pd[0]]))
    cs = 3 * int(np.ceil(fw * 40 / 3))
    for (cx, cy) in ratios_xy:
        x0 = int(np.clip(cx - cs // 2, 0, W - cs)); y0 = int(np.clip(cy - cs // 2, 0, H - cs))
        a, b = img[:, y0:y0 + cs, x0:x0 + cs], out[:, y0:y0 + cs, x0:x0 + cs]
        z = lambda m: cv2.resize(m, (w, w), interpolation=cv2.INTER_NEAREST)
        rows.append(np.hstack([z(rgb(a)), z(rgb(b)), z(diff(a, b))]))
    hmax = max(r.shape[0] for r in rows)
    sheet = np.vstack([np.pad(r, ((0, 0), (0, 0), (0, 0))) for r in rows])
    cv2.putText(sheet, f"{nom} : entree | sortie | entree-sortie (rouge = sortie plus claire)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    buf.tofile(str(dest))


# ----------------------------------------------------------------------------------------------------- principal
def starless_de(dossier, nom):
    for p in (dossier / f"{nom}.npy", dossier / nom / "starless.npy"):
        if p.exists():
            return np.load(p).astype(np.float32)
    npz = dossier / f"{nom}.npz"
    if npz.exists():
        return np.load(npz)["starless"].astype(np.float32)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--images", nargs="*")
    ap.add_argument("--tag")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--attendre", action="store_true", help="attendre qu'aucune production/entrainement ne tourne (GPU)")
    ap.add_argument("--sans-synthetique", action="store_true")
    a = ap.parse_args()
    src = Path(a.source)
    dossier = src.is_dir()
    tag = a.tag or (src.name if dossier else src.stem)
    dest = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles N7 – 25 nov" / "Resultats") / tag
    dest.mkdir(parents=True, exist_ok=True)
    noms = a.images or [n for n in LISTE if (DATA / n / "gaia_dr3_profond.npz").exists()]
    net = dev = None
    info = {"source": str(src), "type": "dossier de starless" if dossier else "checkpoint"}
    if not dossier:
        import torch
        torch.set_num_threads(4)
        if a.device.startswith("cuda"):
            while production_en_cours():
                if not a.attendre:
                    sys.exit("une production / un entrainement tourne : le GPU est pris (--attendre, ou --device cpu)")
                time.sleep(60)
            assert "4060" in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
        dev = torch.device(a.device)
        net, i2 = charger_reseau(src, dev)
        info.update(i2)
    log(f"validation {tag} : {len(noms)} images, {info}")
    resume = {}
    for nom in noms:
        t0 = time.time()
        fj = dest / f"{nom}.json"
        img = np.array(np.load(entree_de(nom), mmap_mode="r"), np.float32)
        C, H, W = img.shape
        L_in = img.mean(0)
        valid = L_in != 0
        gx, gy, gg = charger_gaia(nom, (H, W))
        me = mesures_entree(nom, L_in, gx, gy, gg)
        fw, bruit = me["fwhm_px"], me["bruit_pixel"]
        g_det = min(me["g_detectable"], 19.0)          # le bruit apres debruitage est minuscule : on plafonne a G 19
        log(f"{nom} {img.shape} FWHM {fw} px, bruit {bruit}, G detectable {g_det}, {len(gx)} etoiles Gaia")
        if dossier:
            out = starless_de(src, nom)
            if out is None:
                log(f"  pas de starless pour {nom}"); continue
            if out.shape != img.shape:
                log(f"  forme {out.shape} != {img.shape} : ignore"); continue
        else:
            out = inference(net, img, dev)
        L_out = out.mean(0)
        r = {"image": nom, "type": LISTE[nom][2], "instrument": LISTE[nom][3], "forme": [C, H, W], **me}
        r["etoiles_restantes"] = etoiles_restantes(L_in, L_out, gx, gy, gg, fw, bruit)
        far, fac = masque_loin((H, W), gx, gy, gg, fw, g_det)
        far &= valid
        lis = cv2.GaussianBlur(L_in, (0, 0), 6)
        seuil = float(np.percentile(lis[valid], 90))
        D = np.abs(out - img).mean(0)
        r["fond"] = {"facteur_fwhm_rayon": fac, "part_loin": round(float(far.mean()), 4), **degats_fond(D, far, lis > seuil)}
        # etoile la plus abimee et la plus brillante : agrandissements de la planche
        zoom = []
        ib = np.flatnonzero((gg < 9) & (gx > 60) & (gx < W - 60) & (gy > 60) & (gy < H - 60))
        if len(ib):
            j = ib[np.argmin(gg[ib])]; zoom.append((gx[j], gy[j]))
        im = np.flatnonzero((gg > 9) & (gg < 14) & (gx > 60) & (gx < W - 60) & (gy > 60) & (gy < H - 60))
        if len(im):
            im = im[np.random.default_rng(2).permutation(len(im))[:60]]
            fi = ouverture(L_in, gx[im], gy[im], fw); fo = ouverture(L_out, gx[im], gy[im], fw)
            rr = np.where(np.isfinite(fi) & (fi > 0), fo / np.where(fi > 0, fi, 1), -9)
            j = im[np.argmax(rr)]; zoom.append((gx[j], gy[j]))
        if net is not None and not a.sans_synthetique:
            r["synthetique"], pos = synthetique(nom, net, dev, img, far, valid)
        planche(nom, img, out, gx, gy, gg, zoom, dest / f"{nom}_planche.jpg", fw)
        r["secondes"] = round(time.time() - t0)
        fj.write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
        resume[nom] = r
        log(f"  fait {nom} en {r['secondes']} s ; restant 8-11 : {r['etoiles_restantes'].get('8-11')} ; fond {r['fond'].get('fond')}")
        del img, out, L_in, L_out, D, lis
    # resume : toutes les images deja notees sous ce tag (un lancement partiel complete le precedent)
    resume = {n: json.loads((dest / f"{n}.json").read_text(encoding="utf-8")) for n in LISTE if (dest / f"{n}.json").exists()}
    (dest / "resume.json").write_text(json.dumps({"info": info, "images": resume}, indent=1, ensure_ascii=False), encoding="utf-8")
    md = [f"# Validation {tag}", "", f"source : {src} ; {info}", "",
          "## Etoiles restantes : mediane du flux restant / flux d'origine (p90), par bande de G", "",
          "| image | " + " | ".join(b[0] for b in BANDES) + " |", "|---|" + "---|" * len(BANDES)]
    for n, r in resume.items():
        cells = []
        for b, *_ in BANDES:
            e = r["etoiles_restantes"].get(b, {})
            cells.append(f"{e['mediane']:.3f} ({e['p90']:.3f}) n={e['n']}" if "mediane" in e else f"n={e.get('n', 0)}")
        md.append(f"| {n} | " + " | ".join(cells) + " |")
    md += ["", "## Degats sur le fond (|sortie-entree|, unites de bruit) : mediane / p99", "",
           "| image | fond | structure brillante | part loin |", "|---|---|---|---|"]
    for n, r in resume.items():
        f, s = r["fond"]["fond"], r["fond"]["structure_brillante"]
        fmt = lambda e: f"{e['mediane']:.3f} / {e['p99']:.2f}" if "mediane" in e else "-"
        md.append(f"| {n} | {fmt(f)} | {fmt(s)} | {r['fond']['part_loin']:.2f} |")
    if any("synthetique" in r for r in resume.values()):
        md += ["", "## Etoiles synthetiques : L1 asinh sur l'emprise (sans retrait entre parentheses)", "",
               "| image | " + " | ".join(c[0] for c in CLASSES) + " | toutes |", "|---|" + "---|" * (len(CLASSES) + 1)]
        for n, r in resume.items():
            s = r.get("synthetique")
            if not s:
                continue
            cells = [(f"{s['classes'][c]['l1']:.4f} ({s['classes'][c]['l1_sans_retrait']:.4f}) n={s['classes'][c]['etoiles']}" if c in s["classes"] else "-") for c, *_ in CLASSES]
            t = s.get("toutes")
            md.append(f"| {n} | " + " | ".join(cells) + f" | {t['l1']:.4f} ({t['l1_sans_retrait']:.4f}) |" if t else "| - |")
    (dest / "resume.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
