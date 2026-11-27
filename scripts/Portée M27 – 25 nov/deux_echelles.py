# ======================================================================
# deux_echelles.py
# Role     : N2 two scales: full resolution everywhere, except around Gaia stars G < 6.5 (N2 output on the x2 reduced image).
# Stage    : 12. Two-scale V2 inference (25 Nov) (see scripts/LISEZMOI.md)
# Inputs   : .pt network + validation image + Gaia extract
# Outputs  : starless images, tables
# Example  : python deux_echelles.py [arguments]  (see the docstring; paths via variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
r"""N2 two scales: full resolution everywhere, except around Gaia stars G < 6.5 (N2 output on the x2 reduced image).
Radius = 300 px * 10^(-0.2 (G - 5.6)) (halo measured on M27, square root of the flux), full up to 70 % of the radius, cosine blend up to 100 %.
Writes the starless images (<nom>.npy) in <DONNEES>\Portée M27 – 25 nov\starless_deux_echelles and the M27 crops."""
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
OUT = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Portée M27 – 25 nov"); SL = OUT / "starless_deux_echelles"; SL.mkdir(exist_ok=True)
CK = r"<DONNEES>\Entrainement étoiles nuit – 24 nov\N2\snap\pas_26000.pt"
DATA = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Validation étoiles – 22 nov")
def log(s): print(time.strftime("%H:%M:%S"), s, flush=True)
dev = torch.device("cuda:0"); assert "4060" in torch.cuda.get_device_name(0)
ck = torch.load(CK, map_location=dev, weights_only=False)
net = chain_nafnet().to(dev); net.load_state_dict({k: v.float() for k, v in ck["net"].items()}); net.eval()
info = {}
for nom in LISTE:
    if not (DATA / nom / "gaia_dr3_profond.npz").exists(): continue
    t0 = time.time()
    img = np.array(np.load(Path(LISTE[nom][0]) / "input.npy", mmap_mode="r"), np.float32)
    C, H, W = img.shape
    z = np.load(DATA / nom / "gaia_dr3_profond.npz"); g = z["Gmag"].astype(float); gx = z["x_pix"].astype(float); gy = z["y_pix"].astype(float)
    full = SRm.apply(net, img, dev, None, tile=1024, overlap=256)
    R = 300 * 10 ** (-0.2 * (g - 5.6))
    sel = np.flatnonzero(np.isfinite(g) & (g < 6.5) & (gx > -R) & (gx < W + R) & (gy > -R) & (gy < H + R))
    w = np.zeros((H, W), np.float32)
    for i in sel:
        r = R[i]; x0, x1 = max(0, int(gx[i] - r)), min(W, int(gx[i] + r) + 1); y0, y1 = max(0, int(gy[i] - r)), min(H, int(gy[i] + r) + 1)
        if x1 <= x0 or y1 <= y0: continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        d = np.sqrt((yy - gy[i]) ** 2 + (xx - gx[i]) ** 2)
        t = np.clip((d - 0.7 * r) / (0.3 * r), 0, 1)
        w[y0:y1, x0:x1] = np.maximum(w[y0:y1, x0:x1], 0.5 * (1 + np.cos(np.pi * t)))
    out = full
    if len(sel):
        k = 2; hh, ww = H // k * k, W // k * k
        pe = img[:, :hh, :ww].reshape(3, hh // k, k, ww // k, k).mean((2, 4)).astype(np.float32)
        po = SRm.apply(net, pe, dev, None, tile=1024, overlap=256)
        up = full.copy()
        for c in range(3): up[c, :hh, :ww] = cv2.resize(po[c], (ww, hh), interpolation=cv2.INTER_CUBIC)
        out = full * (1 - w) + up * w
        del po, pe, up
    np.save(SL / f"{nom}.npy", out.astype(np.float32))
    info[nom] = {"etoiles_zone": [[round(float(g[i]), 2), round(float(R[i]))] for i in sel], "part_image_remplacee": round(float((w > 0).mean()), 4)}
    log(f"{nom} {img.shape}: {len(sel)} etoiles, zone {info[nom]['part_image_remplacee']:.3f} de l'image, {time.time()-t0:.0f} s")
    if nom == "m27_mono_nb":
        z0 = json.load(open(OUT / "mesures.json", encoding="utf-8"))["etoile"]["decoupe_origine_xy"]; X0, Y0 = z0
        c = np.load(OUT / "decoupes.npz")
        cr = {"entree": c["entree"], "n2": c["a_comme_mesure"], "deux": out[:, Y0:Y0 + 800, X0:X0 + 800].copy(), "refext": c["refext"]}
        top = float(np.arcsinh(np.percentile(cr["entree"].mean(0), 99.5) / 60)) + 1e-6
        tl = []
        for k2, lab in (("entree", "entree"), ("n2", "N2"), ("deux", "N2 deux echelles"), ("refext", "RefExt")):
            v = np.clip(np.arcsinh(np.clip(cr[k2], 0, None) / 60) / top, 0, 1)
            im = cv2.resize((v.transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8), (700, 700), interpolation=cv2.INTER_AREA).copy()
            cv2.putText(im, lab, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2); tl.append(im)
        _, buf = cv2.imencode(".jpg", np.hstack(tl), [cv2.IMWRITE_JPEG_QUALITY, 90]); buf.tofile(str(OUT / "planche_deux_echelles_M27.jpg"))
    del img, full, out
(OUT / "deux_echelles.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
log("fini")
