# ======================================================================
# restaurer.py
# Role     : Slow restoration at inference (no training), CPU only.
# Stage    : 10. Discarded trial: restoring the slow part (25 Nov) (see scripts/LISEZMOI.md)
# Inputs   : N5 outputs
# Outputs  : tableau_rl.md
# Example  : restaurer.py [--images nom ...] [--sigmas 40 80]
# ======================================================================
r"""Slow restoration at inference (no training), CPU only.

    restaurer.py [--images nom ...] [--sigmas 40 80]

Per image: N5 (best step 23000) in two-scale V2 inference (copy of Tours N5 N6\inferer_v2.py, no GPU) -> R = input - output
-> L = normalised Gaussian smoothing (sigma) of R outside the halo disks of Gaia G < 8, iterative clipping (4 rounds, 2.5 robust deviations,
floor = hf noise: lissage_elague of preparer_g_h.py) -> output + L * (1 - V2 weight).
Writes <DONNEES>\Restauration lente – 25 nov\starless_<N5|S40|S80>\<image>.npy (one file per image; already present = skipped).
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ASTRO_GPU_OFFLINE"] = "1"
import sys, json, time
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np, cv2
cv2.setNumThreads(4)
SR = (Path(__import__("os").environ.get("RACINE_SCRIPTS") or Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(SR)); sys.path.insert(0, str(SR / "Validation – 22 nov")); sys.path.insert(0, r"<FLOAT>")
import torch
torch.set_num_threads(4)
from models import chain_nafnet
from astro_gpu.stages import star_removal as SRm
from liste import LISTE

OUT = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Restauration lente – 25 nov")
N5 = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles N5 N6 – 25 nov")
DATA = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Validation étoiles – 22 nov")
DS = 4


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


def halo_mesure(L, cx, cy, bruit, valid):
    H, W = L.shape; R = 520
    x0, x1, y0, y1 = max(0, int(cx) - R), min(W, int(cx) + R + 1), max(0, int(cy) - R), min(H, int(cy) + R + 1)
    if x1 <= x0 or y1 <= y0:
        return 300.0, False
    p = L[y0:y1, x0:x1]; ok = valid[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]; d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    c = ok & (d >= 400) & (d < 500)
    if c.sum() < 500: c = ok & (d >= 300) & (d < 400)
    if c.sum() < 500: return 300.0, False
    fond = float(np.median(p[c]))
    for r in range(8, 300, 4):
        m = ok & (d >= r) & (d < r + 4)
        if m.sum() < 30: continue
        if np.median(p[m]) - fond < 2 * bruit:
            return float(max(r, 30)), True
    return 300.0, True


def poids(shape, gx, gy, rr):
    H, W = shape; w = np.zeros((H, W), np.float32)
    for x, y, r in zip(gx, gy, rr):
        x0, x1 = max(0, int(x - r)), min(W, int(x + r) + 1); y0, y1 = max(0, int(y - r)), min(H, int(y + r) + 1)
        if x1 <= x0 or y1 <= y0: continue
        yy, xx = np.ogrid[y0:y1, x0:x1]; d = np.sqrt((yy - y) ** 2 + (xx - x) ** 2)
        t = np.clip((d - 0.7 * r) / (0.3 * r), 0, 1)
        w[y0:y1, x0:x1] = np.maximum(w[y0:y1, x0:x1], 0.5 * (1 + np.cos(np.pi * t)))
    return w


def disques(shape, gx, gy, rr):
    """Vrai dans les disques (rayon plein) : pixels exclus du lissage."""
    H, W = shape; m = np.zeros((H, W), bool)
    for x, y, r in zip(gx, gy, rr):
        x0, x1 = max(0, int(x - r)), min(W, int(x + r) + 1); y0, y1 = max(0, int(y - r)), min(H, int(y + r) + 1)
        if x1 <= x0 or y1 <= y0: continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        m[y0:y1, x0:x1] |= np.sqrt((yy - y) ** 2 + (xx - x) ** 2) <= r
    return m


def reduire(a):
    """Moyenne par aire sur DS x DS (a : (h, w), bords tronqués au multiple de DS puis complétés)."""
    h, w = a.shape
    return cv2.resize(a, (w // DS, h // DS), interpolation=cv2.INTER_AREA)


def lissage(D, Mk, sigma):
    """L = G*(D.M) / G*M par canal, à l'échelle réduite DS ; secours sigma 3x puis 10x là où G*M < 1e-3 ; ré-agrandi à pleine taille."""
    C, H, W = D.shape
    h, w = H // DS, W // DS
    Mf = Mk[:h * DS, :w * DS].astype(np.float32)
    den0 = reduire(Mf)
    L = np.zeros((C, H, W), np.float32)
    for c in range(C):
        num0 = reduire(D[c, :h * DS, :w * DS] * Mf)
        res = np.zeros((h, w), np.float32); fait = np.zeros((h, w), bool)
        for s in (sigma, 3 * sigma, 10 * sigma):
            num = cv2.GaussianBlur(num0, (0, 0), s / DS, borderType=cv2.BORDER_CONSTANT)
            den = cv2.GaussianBlur(den0, (0, 0), s / DS, borderType=cv2.BORDER_CONSTANT)
            ok = (den > 1e-3) & ~fait
            res[ok] = num[ok] / den[ok]; fait |= ok
        up = cv2.resize(res, (w * DS, h * DS), interpolation=cv2.INTER_LINEAR)
        L[c, :h * DS, :w * DS] = up
        if h * DS < H: L[c, h * DS:, :w * DS] = up[-1:]
        if w * DS < W: L[c, :, w * DS:] = L[c, :, w * DS - 1:w * DS]
    return L


def lissage_elague(D, M, bh, sigma, n=4, k=2.5):
    Mk = M.copy()
    for _ in range(n):
        L = lissage(D, Mk, sigma)
        nouv = M.copy()
        for c in range(D.shape[0]):
            res = D[c] - L[c]
            r = res[::2, ::2][M[::2, ::2]]
            med = np.median(r)
            sg = max(1.4826 * np.median(np.abs(r - med)), bh)
            nouv &= np.abs(res - med) < k * sg
            del res
        Mk = nouv
        del L
    return lissage(D, Mk, sigma), float(Mk.sum() / max(M.sum(), 1))


def main():
    sigmas = [40, 80]
    if "--sigmas" in sys.argv:
        sigmas = [int(x) for x in sys.argv[sys.argv.index("--sigmas") + 1:] if x.isdigit()]
    noms = [n for n in LISTE if (DATA / n / "gaia_dr3_profond.npz").exists()]
    if "--images" in sys.argv:
        noms = [x for x in sys.argv[sys.argv.index("--images") + 1:] if x in LISTE]
    tags = ["N5"] + [f"S{s}" for s in sigmas]
    for t in tags:
        (OUT / f"starless_{t}").mkdir(parents=True, exist_ok=True)
    dev = torch.device("cpu")
    pas = int((N5 / "N5_meilleur_pas.txt").read_text().strip())
    ck = torch.load(N5 / "N5" / "snap" / f"pas_{pas}.pt", map_location=dev, weights_only=False)
    net = chain_nafnet(); net.load_state_dict({k: v.float() for k, v in ck["net"].items()}); net.eval()
    log(f"N5 pas {pas}, processeur, {torch.get_num_threads()} fils")
    info_all = {}
    for nom in noms:
        if all((OUT / f"starless_{t}" / f"{nom}.npy").exists() for t in tags):
            continue
        t0 = time.time()
        img = np.array(np.load(Path(LISTE[nom][0]) / "input.npy", mmap_mode="r"), np.float32)
        C, H, W = img.shape
        z = np.load(DATA / nom / "gaia_dr3_profond.npz"); g = z["Gmag"].astype(float); gx = z["x_pix"].astype(float); gy = z["y_pix"].astype(float)
        bruit = json.loads((DATA / nom / "mesures_entree.json").read_text(encoding="utf-8"))["bruit_pixel"]
        Li = img.mean(0); valid = Li != 0
        # N5 : inférence deux échelles V2 (copie de inferer_v2.py)
        o2f = OUT / "tmp" / f"{nom}_o2.npy"; w2f = OUT / "tmp" / f"{nom}_w2.npy"
        (OUT / "tmp").mkdir(exist_ok=True)
        inb = np.isfinite(g) & (gx > -300) & (gx < W + 300) & (gy > -300) & (gy < H + 300)
        s2 = np.flatnonzero(inb & (g < 6.5))
        r2 = np.array([halo_mesure(Li, gx[i], gy[i], bruit, valid)[0] for i in s2])
        w2 = poids((H, W), gx[s2], gy[s2], r2)
        if o2f.exists():
            o2 = np.load(o2f)
        else:
            full = SRm.apply(net, img, dev, None, tile=1024, overlap=256)
            o2 = full
            if len(s2):
                k = 2; hh, ww = H // k * k, W // k * k
                pe = img[:, :hh, :ww].reshape(3, hh // k, k, ww // k, k).mean((2, 4)).astype(np.float32)
                po = SRm.apply(net, pe, dev, None, tile=1024, overlap=256)
                up = full.copy()
                for c in range(3): up[c, :hh, :ww] = cv2.resize(po[c], (ww, hh), interpolation=cv2.INTER_CUBIC)
                o2 = full * (1 - w2) + up * w2
                del po, pe, up
            o2 = o2.astype(np.float32)
            np.save(o2f, o2)
            del full
        log(f"{nom} {img.shape}: N5 fait en {time.time() - t0:.0f} s, {len(s2)} etoiles G<6,5")
        np.save(OUT / "starless_N5" / f"{nom}.npy", o2)
        # correction lente
        s8 = np.flatnonzero(inb & (g < 8))
        r8 = np.array([halo_mesure(Li, gx[i], gy[i], bruit, valid)[0] for i in s8])
        M = valid & ~disques((H, W), gx[s8], gy[s8], r8)
        bh = float(1.4826 * np.median(np.abs((lambda a: a - np.median(a))((Li - cv2.GaussianBlur(Li, (0, 0), 2.0))[::2, ::2].ravel()))))
        D = (img - o2).astype(np.float32)
        info = {"etoiles_g8": int(len(s8)), "part_M": float(M.mean()), "bruit_hf": bh}
        for s in sigmas:
            t1 = time.time()
            L, gardee = lissage_elague(D, M, bh, float(s))
            L *= (1 - w2)[None]
            L[:, ~valid] = 0
            sortie = (o2 + L).astype(np.float32)
            np.save(OUT / f"starless_S{s}" / f"{nom}.npy", sortie)
            info[f"S{s}"] = {"part_gardee": round(gardee, 4), "L_moyen": round(float(L.mean()), 4), "L_max": round(float(L.max()), 3),
                             "L_min": round(float(L.min()), 3), "secondes": round(time.time() - t1)}
            log(f"  sigma {s}: {info[f'S{s}']}")
            del L, sortie
        info_all[nom] = info
        (OUT / "tmp" / f"{nom}_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        del img, o2, D, M, w2, Li
        log(f"{nom} fini en {time.time() - t0:.0f} s")
    log("fini")


if __name__ == "__main__":
    main()
