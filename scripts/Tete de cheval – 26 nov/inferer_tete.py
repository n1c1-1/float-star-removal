# ======================================================================
# inferer_tete.py
# Role     : Inference on the prepared Horsehead Nebula (preparer.py).
# Stage    : 13. Trial on the Horsehead Nebula (26 Nov) (see scripts/LISEZMOI.md)
# Inputs   : linear stack of the Horsehead Nebula (not published)
# Outputs  : starless images (not published)
# Example  : python inferer_tete.py [arguments]  (see the docstring; paths via variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
r"""Inference on the prepared Horsehead Nebula (preparer.py). Usage: inferer_tete.py reseau <checkpoint.pt> <tag> | float <tag>
reseau : two-scale V2 (G<6.5, radius = measured halo, cap 300 px, full at 70 %, cosine after), as in inferer_v2_mem.py.
float  : the FLOAT model, a single simple pass (as in the Stars stage).
Alnitak (G ~ 1.7) is absent from the Gaia extract (too bright for Gaia): added by hand (SIMBAD, 05h40m45.5s -01d56m33s)."""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"; os.environ["ASTRO_GPU_OFFLINE"] = "1"
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np, cv2
SR = (Path(__import__("os").environ.get("RACINE_SCRIPTS") or Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(SR)); sys.path.insert(0, r"<FLOAT>")
import torch
torch.set_num_threads(4)
from astro_gpu.stages import star_removal as SRm
import importlib.util
OUT = next((Path(__import__("os").environ.get("DONNEES", "donnees"))).glob("Tete de cheval*")) / "travail"
mode, tag = sys.argv[1], sys.argv[-1]
def log(s): print(time.strftime("%H:%M:%S"), s, flush=True)
dev = torch.device("cuda:0"); assert "4060" in torch.cuda.get_device_name(0)
if mode == "reseau":
    from models import chain_nafnet
    ck = torch.load(sys.argv[2], map_location=dev, weights_only=False)
    net = chain_nafnet().to(dev); net.load_state_dict({k: v.float() for k, v in ck["net"].items()}); net.eval()
else:
    net = SRm.network(dev)
z = np.load(OUT / "entree.npz"); lin = z["lin"]; bk = z["bk"]; unit = float(z["unit"])
img = ((lin - bk) / unit).astype(np.float32); C, H, W = img.shape
g, gx, gy = z["G"].astype(float), z["gx"].astype(float), z["gy"].astype(float)
g = np.append(g, 1.7); gx = np.append(gx, 1752.97); gy = np.append(gy, 859.70)
t0 = time.time()
full = SRm.apply(net, img, dev, None, tile=1024, overlap=256)
log(f"passe pleine {time.time()-t0:.0f} s")
o2 = full
if mode == "reseau":
    def halo_mesure(L, cx, cy, bruit, valid):
        R = 520
        x0, x1, y0, y1 = max(0, int(cx) - R), min(W, int(cx) + R + 1), max(0, int(cy) - R), min(H, int(cy) + R + 1)
        p = L[y0:y1, x0:x1]; ok = valid[y0:y1, x0:x1]
        yy, xx = np.ogrid[y0:y1, x0:x1]; d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        c = ok & (d >= 400) & (d < 500)
        if c.sum() < 500: c = ok & (d >= 300) & (d < 400)
        if c.sum() < 500: return 300.0
        fond = float(np.median(p[c]))
        for r in range(8, 300, 4):
            m = ok & (d >= r) & (d < r + 4)
            if m.sum() < 30: continue
            if np.median(p[m]) - fond < 2 * bruit: return float(max(r, 30))
        return 300.0
    def poids(gx, gy, rr):
        w = np.zeros((H, W), np.float32)
        for x, y, r in zip(gx, gy, rr):
            x0, x1 = max(0, int(x - r)), min(W, int(x + r) + 1); y0, y1 = max(0, int(y - r)), min(H, int(y + r) + 1)
            if x1 <= x0 or y1 <= y0: continue
            yy, xx = np.ogrid[y0:y1, x0:x1]; d = np.sqrt((yy - y) ** 2 + (xx - x) ** 2)
            t = np.clip((d - 0.7 * r) / (0.3 * r), 0, 1)
            w[y0:y1, x0:x1] = np.maximum(w[y0:y1, x0:x1], 0.5 * (1 + np.cos(np.pi * t)))
        return w
    BRUIT = 0.02   # image non debruitee : on prend le bruit de pixel type des mesures de validation (0,007-0,04)
    L = img.mean(0); valid = np.ones((H, W), bool)
    s2 = np.flatnonzero((g < 6.5) & (gx > -300) & (gx < W + 300) & (gy > -300) & (gy < H + 300))
    r2 = np.array([halo_mesure(L, gx[i], gy[i], BRUIT, valid) for i in s2])
    log("rayons de halo (G, px): " + str([(round(float(g[i]), 2), int(r)) for i, r in zip(s2, r2)]))
    w2 = poids(gx[s2], gy[s2], r2)
    k = 2; hh, ww = H // k * k, W // k * k
    pe = img[:, :hh, :ww].reshape(3, hh // k, k, ww // k, k).mean((2, 4)).astype(np.float32)
    po = SRm.apply(net, pe, dev, None, tile=1024, overlap=256)
    up = full.copy()
    for c in range(3): up[c, :hh, :ww] = cv2.resize(po[c], (ww, hh), interpolation=cv2.INTER_CUBIC)
    o2 = full * (1 - w2) + up * w2
    log(f"part de l'image en V2 : {float((w2 > 0).mean()):.3f}")
sl = o2 * unit + bk
np.save(OUT / f"starless_{tag}.npy", sl.astype(np.float32))
log("fini")
