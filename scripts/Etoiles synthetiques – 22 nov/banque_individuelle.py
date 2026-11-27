# ======================================================================
# banque_individuelle.py
# Role     : Bank of REAL shapes, round 4: cleaned INDIVIDUAL bright stars (instead of a few medians).
# Step     : 4. Realistic synthetic stars (22 Nov) (see scripts/LISEZMOI.md)
# Inputs   : star-free backgrounds (pair targets), star shape banks
# Outputs  : pairs (input, target), sheets, blind test
# Example  : python banque_individuelle.py [image ...]   (the method's images, catalogue from their astrometry)
# ======================================================================
"""Bank of REAL shapes, round 4: cleaned INDIVIDUAL bright stars (instead of a few medians).

For each isolated bright G 4.5-9.5 star (no brighter star within 150 px, no star less than 1.5 mag apart within 40 px,
more than 210 px from the edge): 401 px thumbnail, background = median of the 185-200 px ring, recentred to the
sub-pixel, each colour divided by ITS flux (catalogue x share read on the 3-8 FWHM wing); catalogue neighbours masked;
CLEANING: round profile (median per ring, neighbours excluded) + only the non-round structures > 4 noises LINKED TO
THE CORE (spikes, rays, irregular core); everything else (nebula, neighbours, blobs) replaced by the round profile.
Image measurements, on the median of the cleaned stars: law of the far halo (level at 50 px, slope 30-80 px) and
spike angles (peaks of the angular remainder at 3 FWHM-60 px), to extend halo and spikes at render time.

    construire(nom ou None, a=image (C,H,W), cat=(x, y, g), exclure=[(x, y)]) -> dict
    sauver(b), charger(nom)   banque\\banque2_<nom>.npz (formes float16 (n, 3, 401, 401) + json meta)
    python banque_individuelle.py [image ...]   (the method's images, catalogue from their astrometry)
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
import banque_formes as BF                                           # noqa: E402

H = 200
OUTB = HERE / "banque"
MAXI = 40                                                            # étoiles gardées par image au plus


def nettoyer(shape, fw):
    """Profil rond (médiane par anneau, NaN exclus) + structures non rondes > 4 bruits reliées au cœur."""
    h = shape.shape[-1] // 2
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1].astype(np.float32)
    rr = np.hypot(xx, yy)
    ri = np.rint(rr).astype(int)
    prof = A.ring_median(shape)                                       # (3, h+1), NaN ignorés
    out = np.zeros_like(shape)
    r0 = int(math.ceil(2 * fw))
    for c in range(3):
        p = np.nan_to_num(prof[c])
        # profil forcé décroissant au-delà de 2 FWHM (planche Synth_15 : anneaux concentriques = nébuleuse sous l'étoile)
        p[r0:] = np.minimum.accumulate(p[r0:])
        # puis lissé en log sur une fenêtre qui croît avec r, et redécroissant (tour 5 : les marches du profil
        # monotone dessinaient de faibles anneaux au fort étirement, planche Synth_16)
        q = p[r0:].copy()
        pos = q > 0
        if pos.sum() > 5:
            lq = np.log(np.where(pos, q, q[pos].min()))
            sm = np.array([lq[max(0, k - int(1 + 0.05 * (k + r0))):k + int(1 + 0.05 * (k + r0)) + 1].mean()
                           for k in range(len(lq))])
            q = np.where(pos, np.exp(sm), q)
            p[r0:] = np.minimum.accumulate(np.minimum(q, p[r0:].max()))
        # au rayon VRAI (interpolé) : p[rint(r)] faisait un escalier d'anneaux de 1 px, vus en anneaux réguliers
        # au fort étirement (juge des tours 6-7)
        rond = np.interp(rr, np.arange(len(p)), p).astype(np.float32)
        res = np.nan_to_num(shape[c] - rond)
        sm = cv2.GaussianBlur(res, (0, 0), 1.5)
        ann = (rr > 120) & (rr < 190)
        nz = 1.4826 * float(np.median(np.abs(sm[ann] - np.median(sm[ann])))) + 1e-12
        keep = ((np.abs(sm) > 4 * nz) | (rr <= 3 * fw)).astype(np.uint8)
        nlab, lab = cv2.connectedComponents(keep, connectivity=8)
        touch = np.unique(lab[rr <= 4 * fw])
        keep = cv2.GaussianBlur(np.isin(lab, touch[touch > 0]).astype(np.float32), (0, 0), 1.0)
        out[c] = rond + res * keep
    return out.astype(np.float32)


def angles_aigrettes(formes, fw):
    """Angles (deg, 0 = +x, sens image) des aigrettes : pics du reste angulaire moyen (vert) à 3 FWHM - 60 px."""
    med = np.median(formes[:, 1], axis=0)
    h = med.shape[-1] // 2
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1].astype(np.float32)
    rr = np.hypot(xx, yy)
    th = (np.degrees(np.arctan2(yy, xx)) % 360).astype(int)
    m = (rr >= max(3 * fw, 10)) & (rr <= 60)
    ri = np.rint(rr).astype(int)
    prof = np.bincount(ri[m], med[m], minlength=h + 1) / np.maximum(np.bincount(ri[m], minlength=h + 1), 1)
    rel = (med - prof[np.clip(ri, 0, h)]) / np.maximum(prof[np.clip(ri, 0, h)], 1e-30)
    ang = np.bincount(th[m], rel[m], minlength=360) / np.maximum(np.bincount(th[m], minlength=360), 1)
    ang = np.convolve(np.r_[ang[-2:], ang, ang[:2]], np.ones(3) / 3, mode="same")[2:-2]
    base = np.median(ang); mad = 1.4826 * np.median(np.abs(ang - base)) + 1e-9
    pk = [i for i in range(360) if ang[i] > base + 6 * mad and ang[i] == ang[max(0, i - 5):i + 6].max()]
    out = []
    for p_ in sorted(pk, key=lambda i: -ang[i]):                     # doublons autour de 0/360 écartés
        if all(abs((p_ - q + 180) % 360 - 180) > 5 for q in out):
            out.append(p_)
    # une vraie aigrette a sa jumelle à 180° (tour 5 : pics de bruit pris pour des aigrettes sur des réfracteurs)
    out = [p_ for p_ in out if any(abs((q - p_ - 180 + 180) % 360 - 180) <= 4 for q in out)]
    return sorted(float(p_) + 0.5 for p_ in out[:8])


def construire(nom, a=None, cat=None, exclure=(), verbose=True):
    if a is None:
        a = A.IM.load(nom)
    if a.shape[0] == 1:
        a = np.repeat(a, 3, 0)
    x, y, g = cat if cat is not None else BF.catalogue(nom)
    Hh, W = a.shape[1:]
    sig = A.sky_noise(a)
    fw, zp, nfw = BF.largeur_zp(a, x, y, g, sig)
    if not nfw or nfw < 50 or not np.isfinite(fw):
        if verbose:
            print(nom, "trop peu d'étoiles G 11-14 pour la largeur et le point zéro :", nfw, flush=True)
        return None
    tree = cKDTree(np.c_[x, y])
    ex = np.array(exclure, float).reshape(-1, 2)
    yy, xx = np.mgrid[-H:H + 1, -H:H + 1].astype(np.float32)
    rr = np.hypot(xx, yy)
    formes, Gs, niv = [], [], []
    for i in np.argsort(g):
        if g[i] > 9.5 or len(formes) >= MAXI:
            break
        if g[i] < 4.5 or not (H + 10 < x[i] < W - H - 10 and H + 10 < y[i] < Hh - H - 10):
            continue
        if len(ex) and np.min(np.hypot(ex[:, 0] - x[i], ex[:, 1] - y[i])) < 3:
            continue
        if any(j != i and g[j] < g[i] for j in tree.query_ball_point([x[i], y[i]], 150)):
            continue
        if any(j != i and g[j] < g[i] + 1.5 for j in tree.query_ball_point([x[i], y[i]], 40)):
            continue
        s = A.stamp(a, x[i], y[i], H)
        if np.isnan(s).mean() > 0.01 or np.mean(s == 0) > 0.01:
            continue
        s = np.nan_to_num(s)
        ring = (rr >= 185) & (rr <= 200)
        s = s - np.median(s[:, ring], axis=1)[:, None, None]
        fx, fy = x[i] - round(x[i]), y[i] - round(y[i])
        M = np.float32([[1, 0, -fx], [0, 1, -fy]])
        s = np.stack([cv2.warpAffine(c, M, (2 * H + 1, 2 * H + 1), flags=cv2.INTER_LINEAR) for c in s])
        pr = A.ring_median(s[:, H - 60:H + 61, H - 60:H + 61])
        sh = np.nanmean(pr[:, int(3 * fw):int(8 * fw)], axis=1)
        if not np.all(np.isfinite(sh)) or np.any(sh <= 0):
            continue
        parts = sh / sh.sum()
        F = 10 ** (-0.4 * (g[i] - zp))
        shape = s / (F * parts[:, None, None])
        for j in tree.query_ball_point([x[i], y[i]], H * 1.42):
            if j == i or math.hypot(x[j] - x[i], y[j] - y[i]) < 3:          # doublon du catalogue : c'est l'étoile
                continue
            rad = 3 * fw + 10 * max(0.0, 12 - g[j])
            shape[:, np.hypot(xx - (x[j] - round(x[i])), yy - (y[j] - round(y[i]))) < rad] = np.nan
        if np.isnan(shape[:, rr <= 2 * fw]).any():
            continue                                                  # cœur masqué
        cl = nettoyer(shape, fw)
        if any(float(cl[c][rr <= 1.5].max()) < 0.9 * float(np.abs(cl[c]).max()) for c in range(3)):
            continue                                                  # cœur qui n'est pas le maximum : écartée
        pr1 = [float(cl[1][(rr >= a_) & (rr < a_ + 3)].mean()) for a_ in (15, 30, 60)]
        if not (pr1[1] < 0.5 * pr1[0] and pr1[2] < 0.5 * pr1[1]) or min(
                float(cl[c][rr < 60].min()) for c in range(3)) < -0.1 * pr1[0]:
            continue                                                  # halo qui ne décroît pas : nébuleuse dessous
        pk = np.nanmax(s[:, H - 3:H + 4, H - 3:H + 4], axis=(1, 2))
        formes.append(cl.astype(np.float16)); Gs.append(float(g[i])); niv.append((pk / (F * parts)).tolist())
    if len(formes) < 3:
        if verbose:
            print(nom, "trop peu de brillantes pour la banque :", len(formes), flush=True)
        return None
    F32 = np.stack(formes).astype(np.float32)
    # loi du halo lointain de l'image : 30e centile des étoiles (vert) ; la médiane était tirée vers le haut par une
    # lueur de nébuleuse sous une partie des étoiles (Tête de Cheval : 3e-6 à 50 px contre 1e-6 mesuré en direct)
    med = np.percentile(F32[:, 1], 30, axis=0)
    ri = np.rint(rr).astype(int)
    prof = np.bincount(ri.ravel(), med.ravel()) / np.maximum(np.bincount(ri.ravel()), 1)
    p30, p50, p80 = float(np.mean(prof[27:34])), float(np.mean(prof[45:56])), float(np.mean(prof[72:89]))
    gam = float(np.clip(math.log(max(p30, 1e-30) / max(p80, 1e-30)) / math.log(80 / 30), 1.6, 3.0)) \
        if p30 > 0 and p80 > 0 else 2.2
    res = dict(image=nom, formes=np.stack(formes), G=Gs, niveau=niv, fwhm=fw, zp=zp, etoiles_largeur=nfw,
               a50=max(p50, 0.0), gamma=gam, angles=angles_aigrettes(F32, fw), bruit=sig.tolist(),
               max_image=a.reshape(3, -1).max(1).tolist())
    if verbose:
        print(nom, f"fwhm {fw:.2f} zp {zp:.2f} : {len(formes)} étoiles G {min(Gs):.2f}-{max(Gs):.2f} ; halo a50 "
                   f"{res['a50']:.2e} pente {gam:.2f} ; aigrettes {[round(v) for v in res['angles']]}", flush=True)
    return res


def sauver(b):
    OUTB.mkdir(exist_ok=True)
    m = {k: v for k, v in b.items() if k != "formes"}
    np.savez_compressed(OUTB / f"banque2_{b['image']}.npz", formes=b["formes"].astype(np.float16), meta=json.dumps(m))


def charger(nom):
    p = OUTB / f"banque2_{nom}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    d = json.loads(str(z["meta"]))
    d["formes"] = z["formes"]                                         # float16, converti à l'usage
    return d


def planche(banques, nom_planche=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(banques), 8, figsize=(24, 3.2 * len(banques)))
    axs = np.atleast_2d(axs)
    for i, b in enumerate(banques):
        sig, imax = np.array(b["bruit"]), np.array(b["max_image"])
        idx = np.linspace(0, len(b["G"]) - 1, 8).astype(int)
        for j, k in enumerate(idx):
            f = b["formes"][k].astype(np.float32)
            F = 10 ** (-0.4 * (b["G"][k] - b["zp"])) if b.get("zp") else 300 * float(np.mean(sig)) / max(float(f[1].max()), 1e-30)
            v = A.stretch_strong(f[:, H - 100:H + 101, H - 100:H + 101] * F / 3, sig, imax)
            axs[i, j].imshow(v); axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
            axs[i, j].set_title(f"{b['image']} G {b['G'][k]:.2f}", fontsize=8)
    fig.suptitle("Banque d'étoiles individuelles nettoyées (200 px, fort étirement)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.canvas.draw()
    out = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR)); assert ok
    buf.tofile(str(A.OUT / (nom_planche if nom_planche else "Synth_15_banque_individuelles.png")))


def main():
    out = []
    for nom in sys.argv[1:] or ["rosette", "m8_sho", "m57", "horsehead"]:
        b = construire(nom)
        if b is not None:
            sauver(b)
            out.append(b)
    if out:
        planche(out)


if __name__ == "__main__":
    main()
