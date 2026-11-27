# ======================================================================
# deux_echelles_v1v2.py
# Role     : Two variants of N2 two scales.
# Stage    : 12. Two-scale V2 inference (25 Nov) (see scripts/LISEZMOI.md)
# Inputs   : .pt network + validation image + Gaia extract
# Outputs  : starless images, tables
# Example  : python deux_echelles_v1v2.py [arguments]  (see the docstring; paths via variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
r"""Two variants of N2 two scales. V1: G < 5.5, radius = 0.5 x formula. V2: G < 6.5, radius = halo measured in the input
(median radial profile: first radius where profile - local background < 2 noises; background = median of the 400-500 px ring; cap 300 px).
Formula: 300 px x 10^(-0.2 (G - 5.6)). Blend: full up to 70 % of the radius, cosine up to 100 %.
Writes starless_v1/, starless_v2/ (erased after measurement), M27 and M42 contact sheets, parts_v1v2.json."""
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
OUT = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Portée M27 – 25 nov")
for v in ("v1", "v2"): (OUT / f"starless_{v}").mkdir(exist_ok=True)
CK = r"<DONNEES>\Entrainement étoiles nuit – 24 nov\N2\snap\pas_26000.pt"
DATA = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Validation étoiles – 22 nov")
def log(s): print(time.strftime("%H:%M:%S"), s, flush=True)
dev = torch.device("cuda:0"); assert "4060" in torch.cuda.get_device_name(0)
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

def planche(nom, cols, cx, cy, dest, H, W):
    h = 400; Y0 = int(np.clip(cy - h, 0, H - 2 * h)); X0 = int(np.clip(cx - h, 0, W - 2 * h))
    cr = [(lab, a[:, Y0:Y0 + 2 * h, X0:X0 + 2 * h]) for lab, a in cols]
    top = float(np.arcsinh(np.percentile(cr[0][1].mean(0), 99.5) / 60)) + 1e-6
    tl = []
    for lab, a in cr:
        v = np.clip(np.arcsinh(np.clip(np.asarray(a), 0, None) / 60) / top, 0, 1)
        im = cv2.resize((v.transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8), (500, 500), interpolation=cv2.INTER_AREA).copy()
        cv2.putText(im, lab, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2); tl.append(im)
    _, buf = cv2.imencode(".jpg", np.hstack(tl), [cv2.IMWRITE_JPEG_QUALITY, 90]); buf.tofile(str(dest))

info = {}
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
    Rf = 300 * 10 ** (-0.2 * (g - 5.6))
    def inb(R): return np.isfinite(g) & (gx > -R) & (gx < W + R) & (gy > -R) & (gy < H + R)
    s1 = np.flatnonzero(inb(0.5 * Rf) & (g < 5.5)); s2 = np.flatnonzero(inb(300) & (g < 6.5))
    r1 = 0.5 * Rf[s1]
    r2 = np.array([halo_mesure(L, gx[i], gy[i], bruit, valid)[0] for i in s2])
    w1 = poids((H, W), gx[s1], gy[s1], r1); w2 = poids((H, W), gx[s2], gy[s2], r2)
    up = None
    if len(s1) or len(s2):
        k = 2; hh, ww = H // k * k, W // k * k
        pe = img[:, :hh, :ww].reshape(3, hh // k, k, ww // k, k).mean((2, 4)).astype(np.float32)
        po = SRm.apply(net, pe, dev, None, tile=1024, overlap=256)
        up = full.copy()
        for c in range(3): up[c, :hh, :ww] = cv2.resize(po[c], (ww, hh), interpolation=cv2.INTER_CUBIC)
        del po, pe
    o1 = full * (1 - w1) + up * w1 if len(s1) else full
    o2 = full * (1 - w2) + up * w2 if len(s2) else full
    np.save(OUT / "starless_v1" / f"{nom}.npy", o1.astype(np.float32)); np.save(OUT / "starless_v2" / f"{nom}.npy", o2.astype(np.float32))
    info[nom] = {"V1": {"etoiles": int(len(s1)), "part_image": round(float((w1 > 0).mean()), 4)},
                 "V2": {"etoiles": int(len(s2)), "part_image": round(float((w2 > 0).mean()), 4), "rayons_px": [[round(float(g[i]), 2), round(float(r))] for i, r in zip(s2, r2)]}}
    log(f"{nom}: V1 {info[nom]['V1']} ; V2 {info[nom]['V2']['etoiles']} etoiles, part {info[nom]['V2']['part_image']} ; {time.time()-t0:.0f} s")
    if nom in ("m27_mono_nb", "m42_canon40da_lunette"):
        j = np.flatnonzero(np.isfinite(g) & (gx > 0) & (gx < W) & (gy > 0) & (gy < H))
        j = j[np.argmin(g[j])]
        sn = np.load(d0 / "starless.npy", mmap_mode="r")
        planche(nom, [("entree", img), ("N2", full), ("V1", o1), ("V2", o2), ("RefExt", sn)], gx[j], gy[j], OUT / f"planche_v1v2_{nom}.jpg", H, W)
        del sn
    del img, full, o1, o2, up, L
(OUT / "parts_v1v2.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
log("fini")
