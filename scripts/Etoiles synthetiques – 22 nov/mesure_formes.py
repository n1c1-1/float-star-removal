# ======================================================================
# mesure_formes.py
# Role     : Shapes of the medium stars (G 12-13.5, isolated): real versus v2, to calibrate the « shapes » step.
# Step     : 4. Realistic synthetic stars (22 Nov) (see scripts/LISEZMOI.md)
# Inputs   : star-free backgrounds (pair targets), star shape banks
# Outputs  : pairs (input, target), sheets, blind test
# Example  : Usage: python mesure_formes.py [ellip_sigma irreg_p irreg_q_max irreg_d_max]
# ======================================================================
"""Shapes of the medium stars (G 12-13.5, isolated): real versus v2, to calibrate the « shapes » step.
Per star: ellipticity 1 - b/a (2nd-order moments, r <= 2 FWHM) and asymmetry = sum |s - s rotated by 180° around the
centroid| / sum s (r <= 2 FWHM). Output: mesure_formes.json; prints the 10/50/90 percentiles.
Usage: python mesure_formes.py [ellip_sigma irreg_p irreg_q_max irreg_d_max]"""
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
import etoiles2 as E                                                 # noqa: E402
from images_mesurees import IMP                                      # noqa: E402


def forme(s, fw):
    h = s.shape[-1] // 2
    L = np.nansum(s, 0)
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1].astype(np.float32)
    r = np.hypot(xx, yy)
    bg = np.median(L[r > h - 2])
    v = np.clip(L - bg, 0, None) * (r <= 2 * fw)
    sw = v.sum()
    mx, my = (v * xx).sum() / sw, (v * yy).sum() / sw
    sxx = (v * (xx - mx) ** 2).sum() / sw; syy = (v * (yy - my) ** 2).sum() / sw; sxy = (v * (xx - mx) * (yy - my)).sum() / sw
    tr, det = sxx + syy, sxx * syy - sxy ** 2
    l1 = tr / 2 + math.sqrt(max(tr * tr / 4 - det, 0)); l2 = tr / 2 - math.sqrt(max(tr * tr / 4 - det, 0))
    e = 1 - math.sqrt(max(l2, 1e-9) / max(l1, 1e-9))
    M = cv2.getRotationMatrix2D((h + mx, h + my), 180, 1.0)
    w = (L - bg).astype(np.float32)
    rot = cv2.warpAffine(w, M, w.shape[::-1], flags=cv2.INTER_CUBIC)
    rr = np.hypot(xx - mx, yy - my) <= 2 * fw
    asym = float(np.abs(w - rot)[rr].sum() / max(np.abs(w)[rr].sum(), 1e-9))
    return e, asym


def main():
    over = [float(v) for v in sys.argv[1:]]
    out = {}
    for name in A.IMAGES:
        cfg = A.IMAGES[name]
        d = A.SR / cfg["dir"] / "Piste1" / "sorties" / name
        a = A.IM.load(name)
        aud = json.loads((HERE / "audit_mesures.json").read_text(encoding="utf-8"))[name]
        fw, zp, kap = aud["fwhm"], aud["zp"], aud["stats_image"]["k_ouverture_sur_catalogue"]
        sig, imax = np.array(aud["bruit"]), np.array(aud["max_image"])
        cat = np.load(d / "astrometrie.npz", allow_pickle=True)["cat"].item()
        x, y, g = (np.asarray(cat[k], float) for k in ("x", "y", "g"))
        tree = cKDTree(np.c_[x, y])
        h = int(math.ceil(2 * fw)) + 4
        real, Gs = [], []
        for i in np.flatnonzero((g >= 12) & (g <= 13.5)):
            if not (h < x[i] < a.shape[2] - h and h < y[i] < a.shape[1] - h):
                continue
            if any(j != i and g[j] < g[i] + 4 for j in tree.query_ball_point([x[i], y[i]], 4 * fw)):
                continue
            s = A.stamp(a, x[i], y[i], h)
            if np.isnan(s).any() or np.any(s == 0):
                continue
            real.append(forme(s, fw)); Gs.append(g[i])
            if len(real) >= 150:
                break
        imp = dict(IMP[name], sat=imax)
        if over:
            imp.update(ellip_sigma=over[0], irreg_p=over[1], irreg_q=over[2], irreg_d=over[3])
            if len(over) > 4:
                imp["grumeaux"] = over[4]
        P = E.tirer_image(np.random.default_rng([5, len(name)]), 201, imp, E.ETAPES)
        P["couleur_fn"] = lambda b: np.full((3, len(np.atleast_1d(b))), 1 / 3, np.float32)
        rng = np.random.default_rng(0)
        syn = []
        for k, G in enumerate(Gs[:100]):
            fond = (sig[:, None, None] * rng.standard_normal((3, 2 * h + 1, 2 * h + 1))).astype(np.float32)
            F = kap * 10 ** (-0.4 * (G - zp))
            champ = dict(x=[h + rng.uniform(-0.5, 0.5)], y=[h + rng.uniform(-0.5, 0.5)], flux=[F], bp_rp=[1.0])
            o = E.ajouter_etoiles(fond, [3, k], champ=champ, P=P)
            syn.append(forme(o["entree"], fw))
        R, S = np.array(real), np.array(syn)
        pc = lambda v: np.percentile(v, [10, 50, 90]).round(3).tolist()
        out[name] = dict(n=len(R), reel_ellip=pc(R[:, 0]), v2_ellip=pc(S[:, 0]), reel_asym=pc(R[:, 1]), v2_asym=pc(S[:, 1]))
        print(name, out[name], flush=True)
    (HERE / ("mesure_formes.json" if not over else "mesure_formes_essai.json")).write_text(
        json.dumps(dict(reglage=over, **out), ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
