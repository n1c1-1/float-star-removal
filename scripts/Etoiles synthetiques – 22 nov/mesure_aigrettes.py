# ======================================================================
# mesure_aigrettes.py
# Rôle     : Étape 2.0 : les aigrettes réelles mesurées sur nos 4 images, étoiles croisées écartées (pour les plages du générateur v2).
# Étape    : 4. Étoiles synthétiques réalistes (22 nov) (voir scripts/LISEZMOI.md)
# Entrées  : fonds sans étoiles (cibles de paires), banques de formes d'étoiles
# Sorties  : paires (entrée, cible), planches, test à l'aveugle
# Exemple  : python mesure_aigrettes.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Étape 2.0 : les aigrettes réelles mesurées sur nos 4 images, étoiles croisées écartées (pour les plages du générateur v2).

Pour les étoiles G < 9,5 de chaque image (assez loin du bord, sans plus brillante à 60 px), le long de chaque angle connu :
excès = moyenne sur 3 px en travers moins la moyenne de deux parallèles à +-d px (d = max(3 FWHM, 8)) ; les points à
moins de 3 FWHM d'une autre étoile du catalogue (de la ligne ou des parallèles) sont écartés. Par étoile et par couleur :
ajustement linéaire de log10(excès / flux) = a + b log10(1 + (r / fw)^2) + c r^2 sur les points > 2 bruits :
  s0 = 10^a (niveau près du cœur), pente p = -2b, longueur de coupure L = 1 / sqrt(-c ln 10) (si c < 0) ;
r_fin = dernier r où l'excès dépasse 3 bruits ; largeur en travers (sigma gaussien) à r 15-40 px ; pointillés = MAD de
(excès / ajustement - 1) et période dominante.
Sortie : mesure_aigrettes.json.
"""
from __future__ import annotations

import os as _os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    _os.environ[_k] = "2"                                  # au plus 2 fils (consigne du 22 nov après la saturation)

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_etoiles as A                                            # noqa: E402

R_MAX = 330


def sample(s, h, t, r, o):
    xs = h + r * math.cos(t) - o * math.sin(t)
    ys = h + r * math.sin(t) + o * math.cos(t)
    return map_coordinates(np.nan_to_num(s), [ys, xs], order=1), xs, ys


def one_image(name):
    cfg = A.IMAGES[name]
    d = A.SR / cfg["dir"] / "Piste1" / "sorties" / name
    a = A.IM.load(name)
    mod = json.loads((d / "modele_psf_image.json").read_text(encoding="utf-8"))
    fw, zp = float(mod["fwhm"]), float(mod["zp"])
    cat = np.load(d / "astrometrie.npz", allow_pickle=True)["cat"].item()
    x, y, g = (np.asarray(cat[k], float) for k in ("x", "y", "g"))
    for (ex, ey, eg, en) in cfg.get("extra", []):
        x, y, g = np.r_[x, ex], np.r_[y, ey], np.r_[g, eg]
    Hh, W = a.shape[1:]
    sig = A.sky_noise(a)
    tree = cKDTree(np.c_[x, y])
    dpar = max(3 * fw, 8.0)
    rows = []
    for i in np.argsort(g):
        if g[i] > 9.5 or len(rows) >= 25:
            break
        m = 340
        if not (m < x[i] < W - m and m < y[i] < Hh - m):
            continue
        if any(j != i and g[j] < g[i] for j in tree.query_ball_point([x[i], y[i]], 60)):
            continue
        h = R_MAX + 20
        s = A.stamp(a, x[i], y[i], h)
        flux = 10 ** (-0.4 * (g[i] - zp))
        others = [j for j in tree.query_ball_point([x[i], y[i]], h + 10) if j != i and g[j] < g[i] + 9]
        ox, oy = x[others] - x[i] + h, y[others] - y[i] + h
        r = np.arange(R_MAX, dtype=float)
        per_spike = []
        widths = []
        for ang in cfg["angles"]:
            t = math.radians(ang)
            ok = np.ones(R_MAX, bool)
            for o in (-dpar, 0.0, dpar):
                xs = h + r * math.cos(t) - o * math.sin(t)
                ys = h + r * math.sin(t) + o * math.cos(t)
                if len(others):
                    dist = np.min(np.hypot(xs[:, None] - ox[None], ys[:, None] - oy[None]), axis=1)
                    ok &= dist > 3 * fw
            ex = np.zeros((a.shape[0], R_MAX))
            for c in range(a.shape[0]):
                main = sum(sample(s[c], h, t, r, o)[0] for o in (-1.0, 0.0, 1.0)) / 3
                par = 0.5 * (sample(s[c], h, t, r, -dpar)[0] + sample(s[c], h, t, r, dpar)[0])
                ex[c] = main - par
            ex[:, ~ok] = np.nan
            per_spike.append(ex)
            # largeur en travers à r 15-40 px (vert)
            rr = np.arange(15, 41, dtype=float)
            offs = np.arange(-10, 11, dtype=float)
            prof = np.array([np.mean(sample(s[1], h, t, rr, o)[0]) for o in offs])
            prof -= 0.5 * (prof[:3].mean() + prof[-3:].mean())
            if prof[10] > 5 * sig[1]:
                w = np.clip(prof, 0, None)
                widths.append(float(math.sqrt((w * offs ** 2).sum() / max(w.sum(), 1e-9))))
        ex = np.nanmedian(np.array(per_spike), axis=0)                   # médiane des aigrettes de l'étoile
        noise = sig * math.sqrt(1 / 3 + 1 / 2)
        res = dict(G=round(float(g[i]), 2), xy=[int(x[i]), int(y[i])], par_couleur=[])
        for c in range(a.shape[0]):
            v = ex[c]
            sel = (r >= 2 * fw) & np.isfinite(v) & (v > 2 * noise[c])
            if sel.sum() < 15:
                res["par_couleur"].append(None)
                continue
            rr = r[sel]
            yv = np.log10(v[sel] / flux)
            X = np.c_[np.ones(rr.size), np.log10(1 + (rr / fw) ** 2), rr ** 2]
            co, *_ = np.linalg.lstsq(X, yv, rcond=None)
            fit = __import__('operator').matmul(X, co)
            q = 10 ** (yv - fit) - 1
            above = np.flatnonzero(np.isfinite(v) & (v > 3 * noise[c]) & (r >= 2 * fw))
            r_end = int(r[above[-1]]) if above.size else 0
            # période : sur la partie continue la plus longue
            per = None
            if q.size >= 24:
                f = np.abs(np.fft.rfft(q - q.mean())); fr = np.fft.rfftfreq(q.size)
                okf = (fr > 1 / 60) & (fr < 1 / 4)
                if okf.any():
                    per = float(1 / fr[okf][np.argmax(f[okf])])
            res["par_couleur"].append(dict(
                s0=float(f"{10 ** co[0]:.3g}"), pente=round(float(-2 * co[1]), 2),
                L=None if co[2] >= 0 else round(float(1 / math.sqrt(-co[2] * math.log(10))), 1),
                r_fin=r_end, I_sur_F_r30=float(f"{np.nanmedian(v[28:33]) / flux:.3g}"),
                I_sur_F_r100=float(f"{np.nanmedian(v[95:106]) / flux:.3g}"),
                pointilles=round(float(1.4826 * np.median(np.abs(q))), 3),
                periode_px=None if per is None else round(per, 1), points=int(sel.sum())))
        res["largeur_sigma_px"] = round(float(np.median(widths)), 2) if widths else None
        rows.append(res)
        print(name, res["G"], [(p["pente"], p["L"], p["r_fin"], p["pointilles"]) if p else None
                               for p in res["par_couleur"]], res["largeur_sigma_px"], flush=True)
    return dict(fwhm=fw, bruit=sig.round(4).tolist(), etoiles=rows)


def main():
    out = {}
    for n in sys.argv[1:] or list(A.IMAGES):
        out[n] = one_image(n)
    (HERE / "mesure_aigrettes.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
