# Comparison with a model that is not part of this method; path to be supplied (placeholder "<other-model>.pt" below).
# ======================================================================
# diag_validation.py
# Role     : Questions 1 and 2 of the discrepancy diagnostic (25 Nov): what the “median background” and “median structure” of N2 are made of.
# Step     : diagnostic (25 Nov) (see scripts/LISEZMOI.md)
# Inputs   : pairs and validation outputs
# Outputs  : diagnostic tables
# Example  : python diag_validation.py [arguments]  (see the docstring; paths via the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Questions 1 and 2 of the discrepancy diagnostic (25 Nov): what the “median background” and “median structure” of N2 are made of.

Validation measure (valider_reseau.py, unmodified, imported): D = |output - input| averaged over the 3 colours, in units
of the input (not divided by the noise, despite its docstring); “background” = median of D over the pixels far from any detectable
Gaia star; “structure” = same over those AND smoothed(input, 6 px) > its 90th percentile; the table takes the median over the 13 images.

Here, on 2 crops per image (1280 px, analysis on the central 1024; crop 1 = the richest in “far” pixels, crop 2 = the
richest in “far AND structure”), outputs of N2 step 26000 and of the FLOAT model recomputed on CPU (4 threads, same call
star_removal.apply tile 1024 / overlap 256):
    --inferer N2|livre    writes the crop outputs (<DONNEES>/Diagnostic écarts – 25 nov/coupes/<model>/)
    --analyser            measurements (offset, colour, grain, blotches, pedestal share) -> validation.json, validation_resume.txt,
                          planche.jpg
"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""               # processeur seulement : la carte est à un autre agent
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import cv2

cv2.setNumThreads(4)
SR = (Path(__import__("os").environ.get("RACINE_SCRIPTS") or Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(SR / "Validation – 22 nov"))
import valider_reseau as V  # noqa: E402  (ses chemins et réglages ; rien n'y est modifié)
os.environ["CUDA_VISIBLE_DEVICES"] = ""               # valider_reseau remet la 4060 Ti : on la recache avant torch
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "4"

OUT = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Diagnostic écarts – 25 nov")
REF = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Comparaison référence – 23 nov" / "sorties")
MODELES = {"N2": (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles nuit – 24 nov" / "N2" / "snap" / "pas_26000.pt"),
           "livre": Path(r"<FLOAT>\astro_gpu\resources\stars\<other-model>.pt")}
S, M = 1024, 128


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


def noms():
    return [n for n in V.LISTE if (V.DATA / n / "gaia_dr3_profond.npz").exists()]


def preparer(nom):
    """Entrée, masques de la validation (loin, structure), coupes choisies."""
    img = np.array(np.load(V.entree_de(nom), mmap_mode="r"), np.float32)
    C, H, W = img.shape
    L = img.mean(0)
    valid = L != 0
    gx, gy, gg = V.charger_gaia(nom, (H, W))
    me = V.mesures_entree(nom, L, gx, gy, gg)
    far, _ = V.masque_loin((H, W), gx, gy, gg, me["fwhm_px"], min(me["g_detectable"], 19.0))
    far &= valid
    lis = cv2.GaussianBlur(L, (0, 0), 6)
    struct = lis > float(np.percentile(lis[valid], 90))
    T = S + 2 * M
    ii_f = cv2.integral(far.astype(np.uint8)).astype(np.int64)
    ii_s = cv2.integral((far & struct).astype(np.uint8)).astype(np.int64)
    ys = list(range(0, max(H - T, 0) + 1, 128)); xs = list(range(0, max(W - T, 0) + 1, 128))

    def somme(ii, y, x):
        y0, x0, y1, x1 = y + M, x + M, y + M + S, x + M + S
        return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
    cand = [(y, x) for y in ys for x in xs]
    c1 = max(cand, key=lambda p: somme(ii_f, *p))
    c2 = max(cand, key=lambda p: (somme(ii_s, *p), -abs(p[0] - c1[0]) - abs(p[1] - c1[1])))
    coupes = [c1] if c2 == c1 else [c1, c2]
    return img, far, struct, lis, me, coupes


def inferer(modele):
    import torch
    torch.set_num_threads(4)
    dev = torch.device("cpu")
    net, info = V.charger_reseau(MODELES[modele], dev)
    d = OUT / "coupes" / modele
    d.mkdir(parents=True, exist_ok=True)
    log(f"{modele} {info}")
    for nom in noms():
        img, far, struct, lis, me, coupes = preparer(nom)
        for k, (y, x) in enumerate(coupes):
            f = d / f"{nom}_{k}.npy"
            if f.exists():
                continue
            t0 = time.time()
            c = np.ascontiguousarray(img[:, y:y + S + 2 * M, x:x + S + 2 * M])
            with torch.no_grad():
                o = V.inference(net, c, dev)
            np.save(f, o.astype(np.float32))
            log(f"{nom} coupe {k} ({y},{x}) {c.shape} : {time.time() - t0:.0f} s")
        del img


def mad(v):
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if v.size > 10 else float("nan")


def hp(a, s=2.0):
    return a - cv2.GaussianBlur(a, (0, 0), s)


def mesures(I, O, m, bruit, lis):
    """I, O : (3, h, w) ; m : masque ; bruit : bruit de l'entrée (validation)."""
    if m.sum() < 1000:
        return None
    D = O - I
    Dm = D.mean(0)
    A = np.abs(D).mean(0)
    Dl = np.stack([cv2.GaussianBlur(D[c], (0, 0), 4.0) for c in range(3)])
    Dh = D - Dl
    off = [float(np.median(D[c][m])) for c in range(3)]
    offL = float(np.median(Dm[m]))
    a = A[m]
    tri = np.sort(a)[::-1]
    return {
        "pixels": int(m.sum()),
        "D_med": float(np.median(a)),                                      # la mesure de la validation
        "D_med_bruits": float(np.median(a) / bruit),
        "decalage": offL, "decalage_bruits": offL / bruit,
        "derive_couleur_bruits": float(max(abs(o - offL) for o in off) / bruit),
        "lisse_med": float(np.median(np.abs(Dl).mean(0)[m])),             # part lente (> 4 px) de l'écart
        "fin_med": float(np.median(np.abs(Dh).mean(0)[m])),               # part fine (grain)
        "grain_sortie_sur_entree": float(np.mean([mad(hp(O[c])[m]) / max(mad(hp(I[c])[m]), 1e-12) for c in range(3)])),
        "taches_0.5b": float((np.abs(Dl.mean(0))[m] > 0.5 * bruit).mean()),
        "taches_2b": float((np.abs(Dl.mean(0))[m] > 2 * bruit).mean()),
        "niveau_perdu": float(np.median(Dl.mean(0)[m] / np.maximum(np.abs(lis[m]), 1e-6))),   # < 0 : piédestal effacé
        "niveau_entree_med": float(np.median(lis[m])),
        "part_top5": float(tri[:max(1, len(tri) // 20)].sum() / max(tri.sum(), 1e-12)),
    }


def analyser():
    res = {}
    vues = []
    for nom in noms():
        img, far, struct, lis, me, coupes = preparer(nom)
        ref = np.load(REF / f"{nom}.npy", mmap_mode="r") if (REF / f"{nom}.npy").exists() else None
        b = me["bruit_pixel"]
        res[nom] = {"bruit": b, "coupes": []}
        for k, (y, x) in enumerate(coupes):
            sl = (slice(None), slice(y + M, y + M + S), slice(x + M, x + M + S))
            I = img[sl]
            fm, sm, lm = far[sl[1:]], (far & struct)[sl[1:]], lis[sl[1:]]
            outs = {}
            for mo in MODELES:
                f = OUT / "coupes" / mo / f"{nom}_{k}.npy"
                if f.exists():
                    outs[mo] = np.load(f)[:, M:M + S, M:M + S]
            if ref is not None:
                outs["reference"] = np.array(ref[sl], np.float32)
            row = {"y": y, "x": x, "part_loin": float(fm.mean()), "part_structure": float(sm.mean())}
            for mo, O in outs.items():
                row[mo] = {"fond": mesures(I, O, fm, b, lm), "structure": mesures(I, O, sm, b, lm)}
            res[nom]["coupes"].append(row)
            if "N2" in outs and "livre" in outs and k == 0:
                vues.append((nom, I, outs, b))
        log(f"{nom} : " + " | ".join(
            f"c{k} " + " ".join(f"{mo} {r[mo]['fond']['D_med']:.4f}" for mo in ("N2", "livre", "reference")
                                 if mo in r and r[mo]["fond"]) for k, r in enumerate(res[nom]["coupes"])))
        del img
    (OUT / "validation.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    resume(res)
    planche(vues)


def resume(res):
    cles = ["D_med", "D_med_bruits", "decalage_bruits", "derive_couleur_bruits", "lisse_med", "fin_med",
            "grain_sortie_sur_entree", "taches_0.5b", "taches_2b", "niveau_perdu", "part_top5"]
    lignes = []
    for zone in ("fond", "structure"):
        lignes.append(f"== {zone} (coupe 0 = plus riche en fond, 1 = plus riche en structure) ==")
        lignes.append("image coupe modele | " + " ".join(cles))
        for nom, r in res.items():
            for k, row in enumerate(r["coupes"]):
                for mo in ("N2", "livre", "reference"):
                    v = row.get(mo, {}).get(zone)
                    if v:
                        lignes.append(f"{nom} {k} {mo} | " + " ".join(f"{v[c]:.4g}" for c in cles))
    (OUT / "validation_resume.txt").write_text("\n".join(lignes), encoding="utf-8")
    print("\n".join(lignes), flush=True)


def planche(vues):
    """Une rangée par image (les 3 pires écarts de fond de N2 / FLOAT) : entrée | N2 | FLOAT | N2-entrée | FLOAT-entrée,
    agrandissement 384 px x2, écarts à ±3 bruits (gris = 0, clair = sortie plus claire)."""
    if not vues:
        return
    def score(v):
        nom, I, outs, b = v
        return np.median(np.abs(outs["N2"] - I).mean(0)) / max(np.median(np.abs(outs["livre"] - I).mean(0)), 1e-9)
    vues = sorted(vues, key=score, reverse=True)[:3]
    rows = []
    for nom, I, outs, b in vues:
        z = slice(320, 704)
        Ic = I[:, z, z]
        top = float(np.percentile(np.arcsinh(np.clip(Ic.mean(0), 0, None) / (3 * b)), 99.5)) + 1e-6

        def rgb(a):
            v = np.clip(np.arcsinh(np.clip(a, 0, None) / (3 * b)) / top, 0, 1)
            return (v.transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)

        def ecart(a):
            d = np.clip((a - Ic) / (3 * b), -1, 1)
            return ((d.transpose(1, 2, 0)[:, :, ::-1] * 0.5 + 0.5) * 255).astype(np.uint8)
        N, F = outs["N2"][:, z, z], outs["livre"][:, z, z]
        r = np.ascontiguousarray(np.hstack([rgb(Ic), rgb(N), rgb(F), ecart(N), ecart(F)]))
        cv2.putText(r, f"{nom} : entree | N2 | FLOAT | N2-entree | FLOAT-entree (+-3 bruits)", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        rows.append(r)
    sheet = np.vstack(rows)
    cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(str(OUT / "planche.jpg"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--inferer", choices=list(MODELES))
    ap.add_argument("--analyser", action="store_true")
    a = ap.parse_args()
    if a.inferer:
        inferer(a.inferer)
    if a.analyser:
        analyser()
