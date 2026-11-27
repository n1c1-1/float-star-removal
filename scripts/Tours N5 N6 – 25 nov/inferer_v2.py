# ======================================================================
# inferer_v2.py
# Role     : Two-scale V2 inference (copy of Portée M27 – 25 nov\deux_echelles_v1v2.py, V2 variant only; the original is unchanged).
# Stage    : 7-8. N5 (v33g targets) and N6 (v33h weights) (25 Nov) (see scripts/LISEZMOI.md)
# Inputs   : v33d -> v33g -> v33h pairs, start E3 step 20000
# Outputs  : N5/N6 snapshots, tableau_n56.md
# Example  : inferer_v2.py <checkpoint.pt> <tag> [--sans-starless]
# ======================================================================
r"""Two-scale V2 inference (copy of Portée M27 – 25 nov\deux_echelles_v1v2.py, V2 variant only; the original is unchanged).

    inferer_v2.py <checkpoint.pt> <tag> [--sans-starless]

V2: G < 6.5, radius = halo measured in the input (cap 300 px), full up to 70 % of the radius, cosine up to 100 %, network output on
the x2 reduced image enlarged back inside the zone. Writes <DONNEES>\Entrainement étoiles N5 N6 – 25 nov\starless_<tag>\<image>.npy (erased after
measurement) and, for 4 images (M31 A7S, Leo core, M42 Dobson, M27), an 800 px crop of the input, of the result and of RefExt:
crops\<tag>_<image>.npz (centre: maximum of the smoothed image for M31 and Leo, brightest Gaia star for M42 Dobson and M27).
"""
import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"; os.environ["ASTRO_GPU_OFFLINE"] = "1"
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np, cv2
SR = (Path(__import__("os").environ.get("RACINE_SCRIPTS") or Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(SR)); sys.path.insert(0, str(SR / "Validation – 22 nov")); sys.path.insert(0, r"<FLOAT>")
import torch
torch.set_num_threads(4)
from models import chain_nafnet
from astro_gpu.stages import star_removal as SRm
from liste import LISTE

OUT = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles N5 N6 – 25 nov")
DATA = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Validation étoiles – 22 nov")
CK, TAG = sys.argv[1], sys.argv[2]
SANS = "--sans-starless" in sys.argv
PLANCHES = {"m31_a7s_newton": "max", "leotriplet_canon6d_newton": "max", "m42_dobson_grande_fwhm": "etoile", "m27_mono_nb": "etoile"}
(OUT / "crops").mkdir(parents=True, exist_ok=True)
if not SANS:
    (OUT / f"starless_{TAG}").mkdir(exist_ok=True)


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


dev = torch.device("cuda:0")
assert "4060" in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
ck = torch.load(CK, map_location=dev, weights_only=False)
net = chain_nafnet().to(dev); net.load_state_dict({k: v.float() for k, v in ck["net"].items()}); net.eval()


def halo_mesure(L, cx, cy, bruit, valid):
    H, W = L.shape; R = 520
    x0, x1, y0, y1 = max(0, int(cx) - R), min(W, int(cx) + R + 1), max(0, int(cy) - R), min(H, int(cy) + R + 1)
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


for nom in LISTE:
    if not (DATA / nom / "gaia_dr3_profond.npz").exists(): continue
    t0 = time.time()
    d0 = Path(LISTE[nom][0])
    img = np.array(np.load(d0 / "input.npy", mmap_mode="r"), np.float32)
    C, H, W = img.shape
    z = np.load(DATA / nom / "gaia_dr3_profond.npz"); g = z["Gmag"].astype(float); gx = z["x_pix"].astype(float); gy = z["y_pix"].astype(float)
    bruit = json.loads((DATA / nom / "mesures_entree.json").read_text(encoding="utf-8"))["bruit_pixel"]
    L = img.mean(0); valid = L != 0
    full = SRm.apply(net, img, dev, None, tile=1024, overlap=256)
    inb = np.isfinite(g) & (gx > -300) & (gx < W + 300) & (gy > -300) & (gy < H + 300)
    s2 = np.flatnonzero(inb & (g < 6.5))
    r2 = np.array([halo_mesure(L, gx[i], gy[i], bruit, valid)[0] for i in s2])
    w2 = poids((H, W), gx[s2], gy[s2], r2)
    o2 = full
    if len(s2):
        k = 2; hh, ww = H // k * k, W // k * k
        pe = img[:, :hh, :ww].reshape(3, hh // k, k, ww // k, k).mean((2, 4)).astype(np.float32)
        po = SRm.apply(net, pe, dev, None, tile=1024, overlap=256)
        up = full.copy()
        for c in range(3): up[c, :hh, :ww] = cv2.resize(po[c], (ww, hh), interpolation=cv2.INTER_CUBIC)
        o2 = full * (1 - w2) + up * w2
        del po, pe, up
    if not SANS:
        np.save(OUT / f"starless_{TAG}" / f"{nom}.npy", o2.astype(np.float32))
    if nom in PLANCHES:
        h = 400
        if PLANCHES[nom] == "max":
            lis = cv2.GaussianBlur(np.minimum(L, np.percentile(L[valid], 99.9)), (0, 0), 60)
            cy, cx = np.unravel_index(int(np.argmax(lis)), lis.shape)
        else:
            j = np.flatnonzero(np.isfinite(g) & (gx > 0) & (gx < W) & (gy > 0) & (gy < H)); j = j[np.argmin(g[j])]
            cx, cy = gx[j], gy[j]
        Y0 = int(np.clip(cy - h, 0, H - 2 * h)); X0 = int(np.clip(cx - h, 0, W - 2 * h))
        sn = np.load(d0 / "starless.npy", mmap_mode="r")
        np.savez_compressed(OUT / "crops" / f"{TAG}_{nom}.npz", entree=img[:, Y0:Y0 + 2 * h, X0:X0 + 2 * h].astype(np.float32),
                            sortie=o2[:, Y0:Y0 + 2 * h, X0:X0 + 2 * h].astype(np.float32),
                            refext=np.asarray(sn[:, Y0:Y0 + 2 * h, X0:X0 + 2 * h]).astype(np.float32), origine=np.array([X0, Y0]))
        del sn
    log(f"{nom}: {len(s2)} etoiles G<6,5, part {float((w2 > 0).mean()):.4f} ; {time.time() - t0:.0f} s")
    del img, full, o2, L
log("fini")
