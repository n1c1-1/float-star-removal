# ======================================================================
# preparer_g_h.py
# Role     : v33g targets (pedestal restored) and v33h weights (faint stars repaired), pairs one by one, CPU.
# Stage    : 7-8. N5 (v33g targets) and N6 (v33h weights) (25 Nov) (see scripts/LISEZMOI.md)
# Inputs   : v33d -> v33g -> v33h pairs, start E3 step 20000
# Outputs  : N5/N6 snapshots, tableau_n56.md
# Example  : preparer_g_h.py g   : *_v33d.npz -> *_v33g.npz (target + L everywhere; weights and classes unchanged) + checks + contact sheet
# ======================================================================
r"""v33g targets (pedestal restored) and v33h weights (faint stars repaired), pairs one by one, CPU.

    preparer_g_h.py g   : *_v33d.npz -> *_v33g.npz (target + L everywhere; weights and classes unchanged) + checks + contact sheet
    preparer_g_h.py h   : *_v33g.npz -> *_v33h.npz (G 14-17 weights) + surface shares at weight > 0

v33g: D = input - target; M = class 0 (untouched) minus the disks of Gaia stars G < 8 (radius = halo measured in the input,
V2 variant of deux_echelles_v1v2.py: first radius where the median radial profile - local background < 2 noises, background = median of the ring
400-500 px (otherwise 300-400), cap 300 px; with no measurable ring: 300 px); L = G*(D.M) / G*M, Gaussian sigma 40 px, per channel, with iterative pruning of the M pixels that deviate from the smoothing by more than 2.5 robust deviations (see lissage_elague);
target_g = target + L. Where G*M is too low (< 1e-3), L comes from a wider Gaussian (120, then 400 px), otherwise 0.
v33h: for each G 14-17 star kept by the validation criterion (input > 10 noises x square root of the area), 2 FWHM disk:
weight 1 if the flux remaining in the target is between -5 % and +5 % of the input flux; weight 0 if it exceeds +10 %; otherwise unchanged (the 1s first,
then the 0s: the 0 wins).
No existing file is modified; the *_v33g / _v33h are written next to the pairs.
"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "4"
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(4)
sys.stdout.reconfigure(encoding="utf-8")
SR = (Path(__import__("os").environ.get("RACINE_SCRIPTS") or Path(__file__).resolve().parents[1]))
OUT = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles N5 N6 – 25 nov")
_s = importlib.util.spec_from_file_location("dp", SR / "Diagnostic écarts – 25 nov" / "diag_paires.py")
DP = importlib.util.module_from_spec(_s)
_s.loader.exec_module(DP)


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


def halo_mesure(L, cx, cy, bruit, valid):
    """Variante V2 de deux_echelles_v1v2.py, sur la tuile."""
    H, W = L.shape
    R = 520
    x0, x1, y0, y1 = max(0, int(cx) - R), min(W, int(cx) + R + 1), max(0, int(cy) - R), min(H, int(cy) + R + 1)
    if x1 <= x0 or y1 <= y0:
        return 300.0, False
    p = L[y0:y1, x0:x1]
    ok = valid[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    c = ok & (d >= 400) & (d < 500)
    if c.sum() < 500:
        c = ok & (d >= 300) & (d < 400)
    if c.sum() < 500:
        return 300.0, False
    fond = float(np.median(p[c]))
    for r in range(8, 300, 4):
        m = ok & (d >= r) & (d < r + 4)
        if m.sum() < 30:
            continue
        if np.median(p[m]) - fond < 2 * bruit:
            return float(max(r, 30)), True
    return 300.0, True


def contexte(f, src):
    m = DP.MOT.match(f.stem[:-4] + "v33d")
    image, gx_, gy_ = m.group(1), m.group(2), m.group(3)
    tx, ty = int(gx_[1:]), int(gy_)
    cat = DP.catalogue(image)
    fw = DP.fwhm_image(image)
    return image, tx, ty, cat, fw


def gauss_norm(DM, M, s):
    k = cv2.GaussianBlur
    num = np.stack([k(DM[c], (0, 0), s, borderType=cv2.BORDER_CONSTANT) for c in range(DM.shape[0])])
    den = k(M.astype(np.float32), (0, 0), s, borderType=cv2.BORDER_CONSTANT)
    return num, den


def lissage(D, M):
    C, H, W = D.shape
    DM = D * M[None]
    Lc = np.zeros_like(D)
    fait = np.zeros((H, W), bool)
    for s in (40.0, 120.0, 400.0):
        num, den = gauss_norm(DM, M, s)
        ok = (den > 1e-3) & ~fait
        Lc[:, ok] = num[:, ok] / den[ok][None]
        fait |= ok
    return Lc, float(fait.mean())


def lissage_elague(D, M, bh, n=4, k=2.5):
    """Même lissage, mais les pixels de M dont l'écart au lissage dépasse k écarts robustes (plancher : bruit hf) sont retirés, n tours :
    la moyenne gaussienne simple est tirée vers le haut par les ailes d'étoiles G >= 8 restées dans la classe 0 (essai du 25 nov : médiane
    de (entrée - cible_g) sur M de -0,4 bruit hf en moyenne, contre ~0 élagué)."""
    Mk = M.copy()
    for _ in range(n):
        L, _c = lissage(D, Mk.astype(np.float32))
        res = D - L
        nouv = M.copy()
        for c in range(D.shape[0]):
            r = res[c][M]
            med = np.median(r)
            sg = max(1.4826 * np.median(np.abs(r - med)), bh)
            nouv &= np.abs(res[c] - med) < k * sg
        Mk = nouv
    Lc, couvert = lissage(D, Mk.astype(np.float32))
    return Lc, couvert, float(Mk.sum() / max(M.sum(), 1))


def etape_g(f, src):
    z = np.load(f)
    I, T, Wt, cl = z["input"].astype(np.float32), np.nan_to_num(z["target"].astype(np.float32)), z["weight"], z["classes"]
    C, H, W = I.shape
    image, tx, ty, cat, fw = contexte(f, src)
    Li = I.mean(0)
    b = DP.bruit_val(Li)
    valid = Li != 0
    M = (cl == 0) & valid
    info = {"paire": f.stem[:-5], "part_classe0": float((cl == 0).mean()), "etoiles_g8": 0, "rayons": [], "catalogue": cat is not None}
    if cat is not None:
        X, Y, G = cat
        s = (G < 8) & (X > tx - 300) & (X < tx + W + 300) & (Y > ty - 300) & (Y < ty + H + 300)
        for x, y, g in zip(X[s] - tx, Y[s] - ty, G[s]):
            r, mes = halo_mesure(Li, x, y, b, valid)
            y0, y1, x0, x1 = max(0, int(y - r)), min(H, int(y + r) + 1), max(0, int(x - r)), min(W, int(x + r) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            yy, xx = np.ogrid[y0:y1, x0:x1]
            M[y0:y1, x0:x1] &= ~(np.sqrt((yy - y) ** 2 + (xx - x) ** 2) <= r)
            info["rayons"].append([round(float(g), 2), int(r), bool(mes)])
        info["etoiles_g8"] = len(info["rayons"])
    info["part_M"] = float(M.mean())
    D = I - T
    if M.sum() < 4000:
        Lc = np.zeros_like(D)
        info["sans_correction"] = True
        info["couvert"] = 0.0
    else:
        Lc, info["couvert"], info["part_gardee"] = lissage_elague(D, M, DP.mad(DP.hp(Li)[::2, ::2].ravel()))
        info["sans_correction"] = False
    Tg = (T + Lc).astype(np.float32)
    # contrôles
    bh = DP.mad(DP.hp(Li)[::2, ::2].ravel())
    info["bruit_hf"] = bh
    info["mediane_avant"] = [float(np.median(D[c][M])) for c in range(C)] if M.sum() else None
    info["mediane_apres"] = [float(np.median((I - Tg)[c][M])) for c in range(C)] if M.sum() else None
    info["L_moyen"] = float(Lc.mean())
    gy, gx = np.gradient(Lc.mean(0))
    info["L_grad_max"] = float(np.hypot(gx, gy).max())              # unités par pixel : marge de marche
    # étoiles G 14-17 : part > 10 % restante, avant / après (critère de la validation, poids > 0 seulement)
    info["g1417"] = None
    if cat is not None and fw is not None:
        X, Y, G = cat
        s = (X >= tx - 50) & (X < tx + W + 50) & (Y >= ty - 50) & (Y < ty + H + 50)
        x, y, g = X[s] - tx, Y[s] - ty, G[s]
        k = (g >= 14) & (g < 17)
        fi, ap, r_out = DP.ouverture(Li, x[k], y[k], fw)
        f0, _, _ = DP.ouverture(T.mean(0), x[k], y[k], fw)
        f1, _, _ = DP.ouverture(Tg.mean(0), x[k], y[k], fw)
        nap = np.pi * (2 * fw) ** 2
        ok = np.isfinite(fi) & np.isfinite(f0) & np.isfinite(f1) & (fi > 10 * b * np.sqrt(nap))
        wc, _ = DP.poids_ouverture(Wt, x[k][ok], y[k][ok], ap, r_out)
        r0, r1 = f0[ok] / fi[ok], f1[ok] / fi[ok]
        info["g1417"] = {"n": int(ok.sum()), "n_wpos": int((wc > 0).sum()),
                         "gt10_avant": int((r0 > 0.1).sum()), "gt10_apres": int((r1 > 0.1).sum()),
                         "gt10_wpos_avant": int(((r0 > 0.1) & (wc > 0)).sum()), "gt10_wpos_apres": int(((r1 > 0.1) & (wc > 0)).sum())}
    z2 = {k: z[k] for k in z.files}
    z2["target"] = Tg
    return z2, info, I, T, Tg


def planche(choix, dest):
    rows = []
    for nom, I, T, Tg in choix:
        top = float(np.arcsinh(np.percentile(I.mean(0), 99.5) / 60)) + 1e-6
        tl = []
        for lab, a in (("entree", I), ("cible v33d", T), ("cible v33g", Tg)):
            v = np.clip(np.arcsinh(np.clip(a, 0, None) / 60) / top, 0, 1)
            im = (v.transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8).copy()
            cv2.putText(im, lab, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            tl.append(im)
        cv2.putText(tl[0], nom[:34], (6, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        rows.append(np.hstack(tl))
    _, buf = cv2.imencode(".jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    buf.tofile(str(dest))


def lister(suffixe):
    return ([(f, "v33") for f in sorted(DP.P_V33.glob(f"*_{suffixe}.npz"))] +
            [(f, "apod") for f in sorted(DP.P_APOD.glob(f"*_{suffixe}.npz"))])


def main_g():
    OUT.mkdir(parents=True, exist_ok=True)
    infos, decal = [], {}
    fich = lister("v33d")
    log(f"{len(fich)} paires")
    for i, (f, src) in enumerate(fich):
        z2, info, I, T, Tg = etape_g(f, src)
        dest = f.with_name(f.stem[:-5] + "_v33g.npz")
        np.savez_compressed(dest, **z2)
        info["fichier"] = dest.name
        info["decalage_lisse_median"] = float(np.median((Tg - T).mean(0)))
        infos.append(info)
        decal[f.stem] = (info["decalage_lisse_median"] / max(info["bruit_hf"], 1e-9))
        if i % 10 == 0:
            log(f"{i}/{len(fich)} {f.stem} M={info['part_M']:.2f} g8={info['etoiles_g8']} L={info['L_moyen']:.4f}")
        # on garde en mémoire seulement les candidats de la planche : décalage et noms
        info["_cle"] = f.stem
    (OUT / "controles_g.json").write_text(json.dumps(infos, indent=1, ensure_ascii=False), encoding="utf-8")
    # résumé
    ok = [d for d in infos if d["mediane_avant"] is not None]
    av = np.array([np.mean(d["mediane_avant"]) / max(d["bruit_hf"], 1e-9) for d in ok])
    ap = np.array([np.mean(d["mediane_apres"]) / max(d["bruit_hf"], 1e-9) for d in ok])
    g = [d["g1417"] for d in infos if d["g1417"]]
    tot = lambda k: sum(d[k] for d in g)
    lines = [f"{len(infos)} paires, {sum(d['sans_correction'] for d in infos)} sans correction (M < 4000 px)",
             f"médiane (entrée - cible) sur M, en bruits hf : avant méd {np.median(av):.3f} [p10 {np.percentile(av,10):.3f} ; p90 {np.percentile(av,90):.3f}] ; "
             f"après méd {np.median(ap):.4f} [p10 {np.percentile(ap,10):.4f} ; p90 {np.percentile(ap,90):.4f}]",
             f"G 14-17 retenues {tot('n')} : > 10 % restant : {tot('gt10_avant')/tot('n'):.4f} -> {tot('gt10_apres')/tot('n'):.4f} ; "
             f"au poids > 0 : {tot('gt10_wpos_avant')/max(tot('n_wpos'),1):.4f} -> {tot('gt10_wpos_apres')/max(tot('n_wpos'),1):.4f}",
             f"L_grad_max (unités/px) : méd {np.median([d['L_grad_max'] for d in infos]):.5f}, max {max(d['L_grad_max'] for d in infos):.5f} ; "
             f"part M méd {np.median([d['part_M'] for d in infos]):.2f}"]
    top = sorted(infos, key=lambda d: -abs(d["decalage_lisse_median"] / max(d["bruit_hf"], 1e-9)))[:8]
    lines.append("plus gros décalages (bruits hf) : " + ", ".join(f"{d['paire']} {d['decalage_lisse_median']/max(d['bruit_hf'],1e-9):.2f}" for d in top))
    (OUT / "resume_g.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    # planche : la paire web au plus gros décalage, la plus grosse en général, une m101 A7S (galaxie), et d'autres
    web = [d for d in infos if d["paire"].startswith("apod_")]
    cand = [max(web, key=lambda d: abs(d["decalage_lisse_median"] / max(d["bruit_hf"], 1e-9)))["paire"]]
    cand += [d["paire"] for d in top if d["paire"] not in cand][:3]
    cand += [d["paire"] for d in infos if d["paire"].startswith(("m101_a7s", "sorciere_a7s", "m82"))][:2]
    cand = list(dict.fromkeys(cand))[:6]
    choix = []
    for n in cand:
        d0 = [f for f, _ in fich if f.stem == n + "_v33d" or f.stem.startswith(n)][0]
        z = np.load(d0)
        g_ = np.load(d0.with_name(d0.stem[:-5] + "_v33g.npz"))
        choix.append((n, z["input"], np.nan_to_num(z["target"]), g_["target"]))
    planche(choix, OUT / "planche_v33g.jpg")
    log("planche : " + ", ".join(cand))


def etape_h(f, src):
    z = np.load(f)
    I, T, Wt = z["input"].astype(np.float32), z["target"].astype(np.float32), z["weight"].astype(np.float32).copy()
    if Wt.ndim == 3:
        Wt = Wt[0]
    C, H, W = I.shape
    image, tx, ty, cat, fw = contexte(f, src)
    w0 = Wt.copy()
    nb = {"a1": 0, "a0": 0, "n": 0}
    if cat is not None and fw is not None:
        Li = I.mean(0)
        b = DP.bruit_val(Li)
        X, Y, G = cat
        s = (X >= tx - 50) & (X < tx + W + 50) & (Y >= ty - 50) & (Y < ty + H + 50)
        x, y, g = X[s] - tx, Y[s] - ty, G[s]
        k = (g >= 14) & (g < 17)
        xk, yk = x[k], y[k]
        fi, ap, r_out = DP.ouverture(Li, xk, yk, fw)
        ft, _, _ = DP.ouverture(np.nan_to_num(T).mean(0), xk, yk, fw)
        nap = np.pi * (2 * fw) ** 2
        ok = np.isfinite(fi) & np.isfinite(ft) & (fi > 10 * b * np.sqrt(nap))
        rat = np.where(ok, ft / np.where(fi != 0, fi, 1), np.nan)
        un, zero = np.zeros((H, W), bool), np.zeros((H, W), bool)
        for i in np.flatnonzero(ok):
            xi, yi = int(round(xk[i])), int(round(yk[i]))
            sl = (slice(yi - r_out, yi + r_out + 1), slice(xi - r_out, xi + r_out + 1))
            if -0.05 <= rat[i] <= 0.05:
                un[sl] |= ap
                nb["a1"] += 1
            elif rat[i] > 0.10:
                zero[sl] |= ap
                nb["a0"] += 1
        nb["n"] = int(ok.sum())
        Wt[un] = 1.0
        Wt[zero] = 0.0
    z2 = {k_: z[k_] for k_ in z.files}
    z2["weight"] = Wt.reshape(z["weight"].shape).astype(z["weight"].dtype)
    return z2, {"paire": f.stem[:-5], "etoiles": nb, "surface_avant": float((w0 > 0).mean()), "surface_apres": float((Wt > 0).mean()),
                "p1_avant": float((w0 >= 1).mean()), "p1_apres": float((Wt >= 1).mean())}


def main_h():
    infos = []
    fich = lister("v33g")
    log(f"{len(fich)} paires v33g")
    for i, (f, src) in enumerate(fich):
        z2, info = etape_h(f, src)
        np.savez_compressed(f.with_name(f.stem[:-5] + "_v33h.npz"), **z2)
        infos.append(info)
        if i % 20 == 0:
            log(f"{i}/{len(fich)} {f.stem} {info['surface_avant']:.3f} -> {info['surface_apres']:.3f}")
    (OUT / "controles_h.json").write_text(json.dumps(infos, indent=1, ensure_ascii=False), encoding="utf-8")
    av = np.array([d["surface_avant"] for d in infos]); ap = np.array([d["surface_apres"] for d in infos])
    dl = (ap - av) * 100
    grands = [(d["paire"], round(float((d["surface_apres"] - d["surface_avant"]) * 100), 1)) for d in infos
              if abs(d["surface_apres"] - d["surface_avant"]) > 0.03]
    lines = [f"{len(infos)} paires ; surface au poids > 0 : moyenne par paire {av.mean()*100:.2f} % -> {ap.mean()*100:.2f} % ; "
             f"changement méd {np.median(dl):.2f} pts [p10 {np.percentile(dl,10):.2f} ; p90 {np.percentile(dl,90):.2f}] ; min {dl.min():.2f} max {dl.max():.2f}",
             f"étoiles : retenues {sum(d['etoiles']['n'] for d in infos)}, poids -> 1 : {sum(d['etoiles']['a1'] for d in infos)}, -> 0 : {sum(d['etoiles']['a0'] for d in infos)}",
             f"paires qui changent de plus de 3 points : {len(grands)} {grands}"]
    (OUT / "resume_h.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    {"g": main_g, "h": main_h}[sys.argv[1]]()
