# ======================================================================
# banque_formes.py
# Role     : Bank of REAL shapes of the bright stars (round 3): per image, stack (median) of the real bright stars.
# Step     : 4. Realistic synthetic stars (22 Nov) (see scripts/LISEZMOI.md)
# Inputs   : star-free backgrounds (pair targets), star shape banks
# Outputs  : pairs (input, target), sheets, blind test
# Example  : python banque_formes.py [image ...]  -> banque\banque_<image>.npz, banque\banque.json ; planche Synth_13_banque.png
# ======================================================================
"""Bank of REAL shapes of the bright stars (round 3): per image, stack (median) of the real bright stars.

Per image: width (2nd-order moment of the isolated G 12-14 stars) and zero point (3 FWHM aperture of the G 11-14 stars)
measured here, in the same way for all images; stars G 4.5-9.5 more than 310 px from the edge, with no brighter star
within 150 px nor a star less than 1.5 mag apart within 40 px; 601 px thumbnail, background = median of the 280-300 px
ring per colour, recentred to the sub-pixel, each colour divided by ITS flux (catalogue flux x the colour's share read on
the 3-8 FWHM wing): the shape keeps the iridescence and the clipped plateau specific to the image; catalogue neighbours
masked (3 FWHM + 10 px per magnitude above G 12). Brightness groups of >= 5 stars (at most 4); per-pixel median: the
instrument's features (spikes, rays, non-round halo) stay, the nebula and the neighbours go away. Far part
(r > 40 px) smoothed (sigma 2 px) against residual noise.

    construire(nom, exclure=[(x, y), ...]) -> dict(formes (g, 3, 601, 601) float32, G, niveau (plateau normalised per
        colour), fwhm, zp, n, image)
    python banque_formes.py [image ...]  -> banque\\banque_<image>.npz, banque\\banque.json ; sheet Synth_13_banque.png
"""
from __future__ import annotations

import os as _os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    _os.environ[_k] = "2"                                  # au plus 2 fils (consigne du 22 nov après la saturation)

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_etoiles as A                                            # noqa: E402

V32 = A.SR / "Methode v32 production – 22 nov" / "Piste1" / "sorties"
IMAGES = ["rosette", "m8_sho", "m57", "horsehead", "ngc6543", "ngc869", "58_hoo"]
H = 300
OUTB = HERE / "banque"


def catalogue(nom):
    p = V32 / nom / "astrometrie.npz"
    if not p.exists():
        p = A.SR / A.IMAGES[nom]["dir"] / "Piste1" / "sorties" / nom / "astrometrie.npz"
    cat = np.load(p, allow_pickle=True)["cat"].item()
    x, y, g = (np.asarray(cat[k], float) for k in ("x", "y", "g"))
    return x, y, g


def largeur_zp(a, x, y, g, sig):
    """FWHM (moment, r <= 2 FWHM, deux passes) et point zéro (ouverture 3 FWHM) des G 11-14 isolées non saturées."""
    tree = cKDTree(np.c_[x, y])
    Hh, W = a.shape[1:]
    imax = a.reshape(3, -1).max(1)
    fw = 4.0
    for _ in range(2):
        fws, zps = [], []
        h = int(math.ceil(3 * fw)) + 3
        for i in np.flatnonzero((g >= 11) & (g <= 14)):
            if not (h < x[i] < W - h and h < y[i] < Hh - h):
                continue
            if any(j != i and g[j] < g[i] + 4 for j in tree.query_ball_point([x[i], y[i]], 6 * fw)):
                continue
            s = A.stamp(a, x[i], y[i], h)
            if np.isnan(s).any() or np.any(s == 0):
                continue
            pr = A.ring_median(s)
            bg = np.nanmedian(pr[:, int(2.5 * fw):int(3 * fw) + 2], axis=1)
            if np.any(np.nanmax(s, axis=(1, 2)) > 0.3 * imax):
                continue
            f, *_ = A.second_moment_fwhm(s, fw, bg)
            r, *_ = A.rings(h)
            ap = float(np.sum(np.where(r <= 3 * fw, s - bg[:, None, None], 0)))
            if ap <= 0:
                continue
            fws.append(f[1]); zps.append(g[i] + 2.5 * math.log10(ap))
            if len(fws) >= 300:
                break
        fw = float(np.median(fws))
    return fw, float(np.median(zps)), len(fws)


def construire(nom, exclure=(), a=None, verbose=True):
    if a is None:
        a = A.IM.load(nom)
    if a.shape[0] == 1:
        a = np.repeat(a, 3, 0)
    x, y, g = catalogue(nom)
    Hh, W = a.shape[1:]
    sig = A.sky_noise(a)
    imax = a.reshape(3, -1).max(1)
    fw, zp, nfw = largeur_zp(a, x, y, g, sig)
    if nfw < 50:
        if verbose:
            print(nom, "trop peu d'étoiles G 11-14 pour la largeur et le point zéro :", nfw, flush=True)
        return None
    tree = cKDTree(np.c_[x, y])
    ex = np.array(exclure, float).reshape(-1, 2)
    cand = []
    for i in np.argsort(g):
        if g[i] > 9.5:
            break
        if g[i] < 4.5 or not (H + 10 < x[i] < W - H - 10 and H + 10 < y[i] < Hh - H - 10):
            continue
        if len(ex) and np.min(np.hypot(ex[:, 0] - x[i], ex[:, 1] - y[i])) < 3:
            continue
        nb = tree.query_ball_point([x[i], y[i]], 150)
        if any(j != i and g[j] < g[i] for j in nb):
            continue
        if any(j != i and g[j] < g[i] + 1.5 for j in tree.query_ball_point([x[i], y[i]], 40)):
            continue
        cand.append(i)
    yy, xx = np.mgrid[-H:H + 1, -H:H + 1].astype(np.float32)
    rr = np.hypot(xx, yy)
    stamps, Gs, lv = [], [], []
    for i in cand:
        s = A.stamp(a, x[i], y[i], H)
        if np.isnan(s).mean() > 0.01 or np.mean(s == 0) > 0.01:
            continue
        s = np.nan_to_num(s)
        ring = (rr >= 280) & (rr <= 300)
        s = s - np.median(s[:, ring], axis=1)[:, None, None]
        fx, fy = x[i] - round(x[i]), y[i] - round(y[i])
        M = np.float32([[1, 0, -fx], [0, 1, -fy]])
        s = np.stack([cv2.warpAffine(c, M, (2 * H + 1, 2 * H + 1), flags=cv2.INTER_LINEAR) for c in s])
        pr = A.ring_median(s[:, H - 60:H + 61, H - 60:H + 61])
        sh = np.clip(np.nanmean(pr[:, int(3 * fw):int(8 * fw)], axis=1), 1e-9, None)
        parts = sh / sh.sum()
        F = 10 ** (-0.4 * (g[i] - zp))
        shape = s / (F * parts[:, None, None])
        # voisines masquées
        for j in tree.query_ball_point([x[i], y[i]], H * 1.42):
            if j == i:
                continue
            rad = 3 * fw + 10 * max(0.0, 12 - g[j])
            m = np.hypot(xx - (x[j] - round(x[i])), yy - (y[j] - round(y[i]))) < rad
            shape[:, m] = np.nan
        pk = np.nanmax(s[:, H - 3:H + 4, H - 3:H + 4], axis=(1, 2))
        stamps.append(shape.astype(np.float32)); Gs.append(g[i]); lv.append(pk / (F * parts))
    n = len(stamps)
    if n < 5:
        if verbose:
            print(nom, "trop peu d'étoiles pour la banque :", n, flush=True)
        return None
    ng = int(min(4, n // 5))
    order = np.argsort(Gs)
    groups = np.array_split(order, ng)
    formes, GG, NIV, NN = [], [], [], []
    far = rr > 40
    import etoiles2 as E
    for grp in groups:
        st = np.stack([stamps[k] for k in grp])
        med = np.nanmedian(st, axis=0)
        med = np.where(np.isfinite(med), med, 0).astype(np.float32)
        # nettoyage (planche Synth_13, 1er essai : fond marbré, reste de nébuleuse d'une médiane de 8-9 étoiles) :
        # hors du cœur (r > 3 FWHM), profil rond + seulement les structures non rondes nettes (aigrettes, rayons :
        # reste lissé > 4 x son bruit, mesuré à 150-280 px), le reste du fond à 0
        for c in range(3):
            rond = E.profil_rond(med[c], H)
            res = med[c] - rond
            sm = cv2.GaussianBlur(res, (0, 0), 1.5)
            ann = (rr > 150) & (rr < 280)
            nz = 1.4826 * float(np.median(np.abs(sm[ann] - np.median(sm[ann])))) + 1e-12
            keep = ((np.abs(sm) > 4 * nz) | (rr <= 3 * fw)).astype(np.uint8)
            # seulement ce qui touche le cœur (aigrettes et rayons en partent ; les taches isolées de fond, non)
            nlab, lab = cv2.connectedComponents(keep, connectivity=8)
            touch = np.unique(lab[rr <= 4 * fw])
            keep = np.isin(lab, touch[touch > 0]).astype(np.float32)
            keep = cv2.GaussianBlur(keep, (0, 0), 1.0)
            med[c] = rond + res * keep
            bl = cv2.GaussianBlur(med[c], (0, 0), 2.0)
            med[c] = np.where(far, bl, med[c])
        # le cœur doit être le maximum de la forme (Synth_13 : un groupe de la Rosette n'avait pas de cœur)
        if any(float(med[c][rr <= 2 * fw].max()) < 0.9 * float(np.abs(med[c]).max()) for c in range(3)):
            if verbose:
                print(nom, "groupe écarté (cœur absent), G", round(float(np.mean([Gs[k] for k in grp])), 2), flush=True)
            continue
        formes.append(med); GG.append(float(np.mean([Gs[k] for k in grp]))); NN.append(len(grp))
        NIV.append(np.median(np.array([lv[k] for k in grp]), axis=0).tolist())
    if not formes:
        return None
    res = dict(image=nom, formes=np.stack(formes), G=GG, niveau=NIV, n=NN, fwhm=fw, zp=zp, etoiles_largeur=nfw,
               bruit=sig.tolist(), max_image=imax.tolist())
    if verbose:
        print(nom, f"fwhm {fw:.2f} zp {zp:.2f} ({nfw} étoiles) ; brillantes {n} ; groupes G {[round(v, 2) for v in GG]} "
                   f"n {NN}", flush=True)
    return res


def planche(banques):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [(b, k) for b in banques for k in range(len(b["G"]))]
    fig, axs = plt.subplots(len(rows), 3, figsize=(12, 3.6 * len(rows)))
    for i, (b, k) in enumerate(rows):
        f = b["formes"][k]
        F = 10 ** (-0.4 * (b["G"][k] - b["zp"]))
        img = f * F / 3
        sig = np.array(b["bruit"])
        c = H
        for j, (z, fn) in enumerate(((32, "x4 arcsinh"), (100, "200 px fort"), (300, "600 px fort"))):
            cr = img[:, c - z:c + z, c - z:c + z]
            if j == 0:
                v = np.arcsinh(60 * np.clip(cr / max(np.percentile(img, 99.99), 1e-9), 0, None)) / np.arcsinh(60)
                v = np.transpose(np.clip(v, 0, 1), (1, 2, 0))
            else:
                v = A.stretch_strong(cr, sig, np.array(b["max_image"]))
            axs[i, j].imshow(v, interpolation="nearest"); axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
            axs[i, j].set_title(fn, fontsize=8)
        axs[i, 0].set_ylabel(f"{b['image']}\nG {b['G'][k]:.2f}\n{b['n'][k]} étoiles\nfwhm {b['fwhm']:.2f}", fontsize=8)
    fig.suptitle("Banque de formes réelles : médiane des vraies brillantes par groupe d'éclat", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.canvas.draw()
    out = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR)); assert ok
    buf.tofile(str(A.OUT / "Synth_13_banque.png"))


def charger(nom):
    p = OUTB / f"banque_{nom}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    d = json.loads(str(z["meta"]))
    d["formes"] = z["formes"].astype(np.float32)
    return d


def main():
    noms = sys.argv[1:] or IMAGES
    OUTB.mkdir(exist_ok=True)
    out, meta = [], {}
    for nom in noms:
        try:
            b = construire(nom)
        except Exception as e:                                       # image absente ou illisible : on le dit
            print(nom, "ÉCHEC", repr(e)[:200], flush=True)
            continue
        if b is None:
            continue
        m = {k: v for k, v in b.items() if k != "formes"}
        np.savez_compressed(OUTB / f"banque_{nom}.npz", formes=b["formes"].astype(np.float16), meta=json.dumps(m))
        meta[nom] = m
        out.append(b)
    (OUTB / "banque.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    if out:
        planche(out)


if __name__ == "__main__":
    main()
