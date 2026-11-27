# ======================================================================
# mesure_halos.py
# Role     : Texture of the bright stars' halos (G < 9.5, no brighter star within 60 px): real versus v2 (round 2).
# Step     : 4. Realistic synthetic stars (22 Nov) (see scripts/LISEZMOI.md)
# Inputs   : star-free backgrounds (pair targets), star shape banks
# Outputs  : pairs (input, target), sheets, blind test
# Example  : Usage: python mesure_halos.py [grain_halo]
# ======================================================================
"""Texture of the bright stars' halos (G < 9.5, no brighter star within 60 px): real versus v2 (round 2).
Per star, luminance with background removed, 3 rings between max(4 FWHM, 2 x plateau radius) and 60 px; median per 3°
sector (spikes at +-6° and catalogue neighbours within 3 FWHM discarded); a + harmonics 1 and 2 are removed; texture =
standard deviation of the remainder / ring level, minus the noise share. Output: mesure_halos.json.
Usage: python mesure_halos.py [grain_halo]"""
from __future__ import annotations

import os as _os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    _os.environ[_k] = "2"                                  # au plus 2 fils (consigne du 22 nov après la saturation)

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_etoiles as A                                            # noqa: E402
import etoiles2 as E                                                 # noqa: E402
from images_mesurees import IMP                                      # noqa: E402

H = 80


def texture(s, fw, sig, angles, others_xy):
    h = s.shape[-1] // 2
    r, th, *_ = A.rings(h)
    L = np.nansum(s, 0)
    bg = np.nanmedian(L[(r > h - 6)])
    L = L - bg
    pk, npl = A.plateau(s, 15, 60)
    r0 = max(4 * fw, 2 * math.sqrt(max(npl[1], 1) / math.pi))
    thd = np.degrees(th) % 360
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1]
    bad = np.zeros_like(r, bool)
    for (ox, oy) in others_xy:
        bad |= np.hypot(xx - ox, yy - oy) < 3 * fw
    sigL = math.sqrt((sig ** 2).sum())
    vals = []
    for a, b in zip(np.geomspace(r0, 60, 4)[:-1], np.geomspace(r0, 60, 4)[1:]):
        m = (r >= a) & (r < b) & ~bad & np.isfinite(L)
        bins = (thd[m] // 3).astype(int)
        v = L[m]
        med = np.array([np.median(v[bins == k]) if np.sum(bins == k) > 2 else np.nan for k in range(120)])
        cnt = np.array([np.sum(bins == k) for k in range(120)])
        cen = np.arange(120) * 3 + 1.5
        keep = np.isfinite(med)
        for ag in angles:
            keep &= np.abs((cen - ag + 180) % 360 - 180) > 6
        if keep.sum() < 40:
            continue
        t = np.radians(cen[keep])
        X = np.c_[np.ones(t.size), np.cos(t), np.sin(t), np.cos(2 * t), np.sin(2 * t)]
        co, *_ = np.linalg.lstsq(X, med[keep], rcond=None)
        if co[0] < 10 * sigL:
            continue
        res = med[keep] - __import__('operator').matmul(X, co)
        noise = 1.25 * sigL / np.sqrt(np.maximum(cnt[keep], 1))
        tex = math.sqrt(max(float(np.mean(res ** 2) - np.mean(noise ** 2)), 0.0)) / co[0]
        vals.append(tex)
    return float(np.median(vals)) if vals else None


def main():
    over = [float(v) for v in sys.argv[1:]]
    M = json.loads((HERE / "mesure_aigrettes.json").read_text(encoding="utf-8"))
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
        imp = dict(IMP[name], sat=imax)
        if over:
            imp["grain_halo"] = over[0]
        P = E.tirer_image(np.random.default_rng([8, len(name)]), 2 * H + 1, imp, E.ETAPES)
        rng = np.random.default_rng(1)
        R, S = [], []
        for e in M[name]["etoiles"][:15]:
            if e["G"] < 4.5:
                continue
            xs, ys = e["xy"]
            s = A.stamp(a, xs, ys, H)
            idx = [j for j in tree.query_ball_point([xs, ys], H + 5) if abs(x[j] - xs) + abs(y[j] - ys) > 1]
            oth = [(x[j] - round(xs), y[j] - round(ys)) for j in idx]
            tr = texture(s, fw, sig, cfg["angles"], oth)
            pr = A.ring_median(s)
            sh = np.clip(np.nanmean(pr[:, int(3 * fw):int(8 * fw)] - np.nanmedian(pr[:, 70:80], 1)[:, None], 1), 1e-6, None)
            parts = (sh / sh.sum()).astype(np.float32)
            P["couleur_fn"] = lambda b, p=parts: np.repeat(p[:, None], len(np.atleast_1d(b)), 1)
            fond = (sig[:, None, None] * rng.standard_normal((3, 2 * H + 1, 2 * H + 1))).astype(np.float32)
            F = kap * 10 ** (-0.4 * (e["G"] - zp))
            o = E.ajouter_etoiles(fond, [4, len(S)], champ=dict(x=[H + 0.2], y=[H - 0.3], flux=[F], bp_rp=[1.0]), P=P)
            ts = texture(o["entree"], fw, sig, cfg["angles"], [])
            if tr is not None and ts is not None:
                R.append(tr); S.append(ts)
        pc = lambda v: np.percentile(v, [10, 50, 90]).round(3).tolist() if len(v) else None
        out[name] = dict(n=len(R), reel=pc(R), v2=pc(S))
        print(name, out[name], flush=True)
    (HERE / ("mesure_halos.json" if not over else "mesure_halos_essai.json")).write_text(
        json.dumps(dict(reglage=over, **out), ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
