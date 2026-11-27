# ======================================================================
# diag_paires.py
# Role     : Questions 3 and 4 of the discrepancy diagnostic (25 Nov): what the N2 training pairs teach.
# Step     : diagnostic (25 Nov) (see scripts/LISEZMOI.md)
# Inputs   : pairs and validation outputs
# Outputs  : diagnostic tables
# Example  : python diag_paires.py [arguments]  (see the docstring; paths via the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Questions 3 and 4 of the discrepancy diagnostic (25 Nov): what the N2 training pairs teach.

Pairs read one by one (none modified): <DONNEES>/Paires étoiles v33 – 23 nov/*_v33d.npz (denoised and raw, held-out ones
included: they are flagged) and <DONNEES>/Paires étoiles APOD – 24 nov/*_v33d.npz (web).
(3) Gaia stars G 14-17: SAME criterion as valider_reseau.py (2 FWHM aperture minus the median of the 3-5 FWHM ring, on the
    mean of the 3 colours; star kept if input flux > 10 noise units x sqrt(area); noise = MAD of L - 5x5 median on every
    other pixel); TARGET flux / INPUT flux > 0.1; weight at the centre and on the aperture.
(4) sky far from stars (same radius as the validation, factor 4 FWHM, stars G <= 19) with weight > 0: target - input
    (offset per colour, in tile noise units), grain ratio (high frequency of the target / of the input), share of pixels
    changed, blotches (4 px smoothed difference > 0.5 noise).
Output: <DONNEES>/Diagnostic écarts – 25 nov/paires.jsonl and paires_resume.txt.
"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "4")
import json, re, sys
from pathlib import Path
import numpy as np
import cv2
from scipy import ndimage as ndi

cv2.setNumThreads(4)
sys.stdout.reconfigure(encoding="utf-8")
SR = (Path(__import__("os").environ.get("RACINE_SCRIPTS") or Path(__file__).resolve().parents[1]))
DEEP = SR / "Methode v33b – 23 nov" / "Catalogue Gaia profond – 20 nov"
DEEP2 = SR / "Methode v33 production – 23 nov" / "Catalogue Gaia profond – 20 nov"
SORT = [SR / "Methode v33b – 23 nov" / "Piste1" / "sorties", SR / "Methode v33 production – 23 nov" / "Piste1" / "sorties"]
P_V33 = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Paires étoiles v33 – 23 nov")
P_APOD = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Paires étoiles APOD – 24 nov")
OUT = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Diagnostic écarts – 25 nov")
TENUES = {"m13_g1408_1408", "m57_g1664_2688", "ngc6888_g896_1408", "ngc7000_tiers1_g2432_1920", "rosette_40da_g1920_1920",
          "wizard_a7s_g2944_1664"}
MOT = re.compile(r"^(.+)_(g\d+)_(\d+)_(debruitee|brute)_v33d$")


def catalogue(image):
    for d in (DEEP, DEEP2):
        f = d / f"gaia_dr3_profond_{image}.npz"
        if f.exists():
            z = np.load(f)
            ok = np.isfinite(z["Gmag"]) & np.isfinite(z["x_pix"]) & np.isfinite(z["y_pix"])
            return z["x_pix"][ok].astype(float), z["y_pix"][ok].astype(float), z["Gmag"][ok].astype(float)
    return None


def fwhm_image(image):
    for s in SORT:
        f = s / image / "modele_psf_image.json"
        if f.exists():
            return float(json.loads(f.read_text(encoding="utf-8"))["fwhm"])
    return None


def bruit_val(L):
    """Estimateur de valider_reseau.mesures_entree."""
    sub = L[::2, ::2]
    res = sub - cv2.medianBlur(np.clip(sub, -50, 1e4).astype(np.float32), 5)
    return float(1.4826 * np.median(np.abs(res[sub != 0]))) if (sub != 0).any() else 1.0


def ouverture(L, x, y, fw):
    H, W = L.shape
    r_out = int(np.ceil(5 * fw)) + 1
    yy, xx = np.mgrid[-r_out:r_out + 1, -r_out:r_out + 1]
    r2 = (yy ** 2 + xx ** 2).astype(np.float32)
    ap = r2 <= (2 * fw) ** 2
    an = (r2 >= (3 * fw) ** 2) & (r2 <= (5 * fw) ** 2)
    res = np.full(len(x), np.nan)
    for i, (a, b) in enumerate(zip(x, y)):
        xi, yi = int(round(a)), int(round(b))
        if xi - r_out < 0 or yi - r_out < 0 or xi + r_out >= W or yi + r_out >= H:
            continue
        s = L[yi - r_out:yi + r_out + 1, xi - r_out:xi + r_out + 1]
        res[i] = float((s[ap] - np.median(s[an])).sum())
    return res, ap, r_out


def poids_ouverture(Wt, x, y, ap, r_out):
    H, W = Wt.shape
    c, m = [], []
    for a, b in zip(x, y):
        xi, yi = int(round(a)), int(round(b))
        c.append(float(Wt[yi, xi]))
        m.append(Wt[yi - r_out:yi + r_out + 1, xi - r_out:xi + r_out + 1][ap])
    return np.array(c), m


def masque_loin(shape, x, y, g, fw):
    H, W = shape
    for fac in (4.0, 3.0, 2.0):
        r = np.maximum(fac * fw, fw * (fac + 2.5 * np.maximum(0.0, 14.0 - g)))
        m = np.ones((H, W), bool)
        # par rayon arrondi (peu de valeurs distinctes) : distance à l'étoile la plus proche du groupe
        for rv in np.unique(np.round(r)):
            k = np.round(r) == rv
            seeds = np.ones((H, W), bool)
            xi = np.clip(np.round(x[k]).astype(int), 0, W - 1); yi = np.clip(np.round(y[k]).astype(int), 0, H - 1)
            ins = (x[k] >= -rv) & (x[k] < W + rv) & (y[k] >= -rv) & (y[k] < H + rv)
            seeds[yi[ins], xi[ins]] = False
            if (~seeds).any():
                m &= ndi.distance_transform_edt(seeds) > rv
        if m.mean() > 0.03 or fac == 2.0:
            return m, fac


def hp(a, s=2.0):
    return a - cv2.GaussianBlur(a, (0, 0), s)


def mad(v):
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if v.size else float("nan")


def une_paire(f, genre_src):
    m = MOT.match(f.stem)
    if not m:
        return None
    image, gx_, gy_, genre = m.group(1), m.group(2), m.group(3), m.group(4)
    tx, ty = int(gx_[1:]), int(gy_)
    nom = f"{image}_{gx_}_{gy_}"
    z = np.load(f)
    I, T, Wt = z["input"].astype(np.float32), z["target"].astype(np.float32), z["weight"].astype(np.float32)
    cl = z["classes"] if "classes" in z.files else None
    if Wt.ndim == 3:
        Wt = Wt[0]
    T = np.nan_to_num(T)
    C, H, W = I.shape
    L, TL = I.mean(0), T.mean(0)
    b = bruit_val(L)
    bh = mad(hp(L)[::2, ::2].ravel())                     # bruit haute fréquence de l'entrée (pour les écarts du fond)
    fw = fwhm_image(image)
    cat = catalogue(image)
    r = {"paire": nom, "image": image, "sorte": "web" if genre_src == "apod" else genre, "tenue": nom in TENUES,
         "bruit_val": b, "bruit_hf": bh, "fwhm": fw,
         "poids": {k: float(v) for k, v in zip(("0", "0.5", "1"), ((Wt == 0).mean(), ((Wt > 0) & (Wt < 1)).mean(), (Wt >= 1).mean()))}}
    if cat is None or fw is None:
        r["manque"] = "catalogue" if cat is None else "fwhm"
        return r
    X, Y, G = cat
    s = (X >= tx - 50) & (X < tx + W + 50) & (Y >= ty - 50) & (Y < ty + H + 50)
    x, y, g = X[s] - tx, Y[s] - ty, G[s]
    # (3) G 14-17
    k = (g >= 14) & (g < 17)
    nap = np.pi * (2 * fw) ** 2
    fi, ap, r_out = ouverture(L, x[k], y[k], fw)
    ft, _, _ = ouverture(TL, x[k], y[k], fw)
    ok = np.isfinite(fi) & np.isfinite(ft) & (fi > 10 * b * np.sqrt(nap))
    rat = ft[ok] / fi[ok]
    wc, wm = poids_ouverture(Wt, x[k][ok], y[k][ok], ap, r_out)
    wmoy = np.array([v.mean() for v in wm]) if len(wm) else np.zeros(0)
    pix = np.concatenate(wm) if len(wm) else np.zeros(0)
    r["g14_17"] = {"n": int(ok.sum()), "ratio": [round(float(v), 4) for v in rat], "w_centre": [float(v) for v in wc],
                   "w_moy": [round(float(v), 3) for v in wmoy],
                   "pix_w": [int((pix == 0).sum()), int(((pix > 0) & (pix < 1)).sum()), int((pix >= 1).sum())]}
    # (4) ciel loin des étoiles au poids > 0
    sel = g <= 19.0
    far, fac = masque_loin((H, W), x[sel], y[sel], g[sel], fw)
    bord = np.zeros((H, W), bool); bord[8:-8, 8:-8] = True
    m4 = far & (Wt > 0) & bord
    r["fond"] = {"facteur": fac, "part_loin_w": float(m4.mean()), "pixels": int(m4.sum())}
    if m4.sum() >= 2000:
        D = T - I
        dl = D.mean(0)
        hI = np.stack([hp(I[c]) for c in range(C)]); hT = np.stack([hp(T[c]) for c in range(C)])
        lis = cv2.GaussianBlur(dl, (0, 0), 4.0)
        r["fond"].update({
            "decalage_bruits": [float(np.median(D[c][m4]) / bh) for c in range(C)],
            "abs_med_brut": float(np.median(np.abs(D).mean(0)[m4])),
            "abs_med_bruits": float(np.median(np.abs(D).mean(0)[m4]) / bh),
            "part_changes": float((np.abs(D).max(0)[m4] > 1e-4 * max(bh, 1e-6)).mean()),
            "grain_cible_sur_entree": float(np.mean([mad(hT[c][m4]) / max(mad(hI[c][m4]), 1e-12) for c in range(C)])),
            "taches_part": float((np.abs(lis[m4]) > 0.5 * bh).mean()),
            "classes_non0": float((cl[m4] != 0).mean()) if cl is not None and cl.shape == (H, W) else None,
        })
    return r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fichiers = [(f, "v33") for f in sorted(P_V33.glob("*_v33d.npz"))] + [(f, "apod") for f in sorted(P_APOD.glob("*_v33d.npz"))]
    print(len(fichiers), "paires", flush=True)
    with open(OUT / "paires.jsonl", "w", encoding="utf-8") as fo:
        for i, (f, src) in enumerate(fichiers):
            try:
                r = une_paire(f, src)
            except Exception as e:      # noqa: BLE001 - une paire illisible est notée, pas silencieuse
                r = {"paire": f.stem, "erreur": repr(e)}
            if r is None:
                print("nom ignoré", f.name, flush=True); continue
            fo.write(json.dumps(r, ensure_ascii=False) + "\n"); fo.flush()
            if i % 10 == 0:
                print(i, f.name, flush=True)
    print("fini", flush=True)


if __name__ == "__main__":
    main()
