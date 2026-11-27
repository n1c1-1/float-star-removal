# ======================================================================
# portee_m27.py
# Role     : N2 step 26000 on M27: (a) as in the measurement, (b) large tiles, (c) reduced x2, (d) reduced x3.
# Stage    : 12. Two-scale V2 inference (25 Nov) (see scripts/LISEZMOI.md)
# Inputs   : .pt network + validation image + Gaia extract
# Outputs  : starless images, tables
# Example  : python portee_m27.py [arguments]  (see the docstring; paths via variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
r"""N2 step 26000 on M27: (a) as in the measurement, (b) large tiles, (c) reduced x2, (d) reduced x3. Outputs in <DONNEES>\Portée M27 – 25 nov."""
import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"; os.environ["ASTRO_GPU_OFFLINE"] = "1"
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np, cv2
SR = (Path(__import__("os").environ.get("RACINE_SCRIPTS") or Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(SR)); sys.path.insert(0, r"<FLOAT>")
import torch
torch.set_num_threads(4)
from models import chain_nafnet
from astro_gpu.stages import star_removal as SRm
from astro_gpu.core import devices
OUT = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Portée M27 – 25 nov")
CK = r"<DONNEES>\Entrainement étoiles nuit – 24 nov\N2\snap\pas_26000.pt"
D = (Path(__import__("os").environ.get("ESSAIS", "essais")) / "stars_dataset" / "_revue_nb" / "56_comb")
GAIA = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Validation étoiles – 22 nov" / "m27_mono_nb" / "gaia_dr3_profond.npz")
def log(s): print(time.strftime("%H:%M:%S"), s, flush=True)
dev = torch.device("cuda:0"); assert "4060" in torch.cuda.get_device_name(0)
ck = torch.load(CK, map_location=dev, weights_only=False)
net = chain_nafnet().to(dev); net.load_state_dict({k: v.float() for k, v in ck["net"].items()}); net.eval()
img = np.array(np.load(D / "input.npy", mmap_mode="r"), np.float32)
ref = np.load(D / "starless.npy", mmap_mode="r")
C, H, W = img.shape
z = np.load(GAIA); g = z["Gmag"].astype(float); gx = z["x_pix"].astype(float); gy = z["y_pix"].astype(float)
ok = np.isfinite(g) & (gx > 0) & (gx < W) & (gy > 0) & (gy < H)
j = np.flatnonzero(ok)[np.argmin(g[ok])]
# recentrage sur le pic local de l'entree (le centre Gaia peut etre decale de quelques pixels)
x0, y0 = int(gx[j]), int(gy[j]); r = 15
s = img.mean(0)[max(0, y0 - r):y0 + r + 1, max(0, x0 - r):x0 + r + 1]
dy, dx = np.unravel_index(np.argmax(s), s.shape); cy, cx = max(0, y0 - r) + dy, max(0, x0 - r) + dx
log(f"image {img.shape}, plus grosse etoile Gaia G={g[j]:.2f} en ({gx[j]:.0f},{gy[j]:.0f}) ; pic de l'entree en ({cx},{cy})")
yy, xx = np.ogrid[:H, :W]
d2 = (yy - cy) ** 2 + (xx - cx) ** 2
anneau = (d2 >= 50 ** 2) & (d2 <= 300 ** 2)
ext = (d2 >= 450 ** 2) & (d2 <= 650 ** 2)
disque = d2 <= 300 ** 2
def mesure(o):
    res = (img - o).mean(0)
    return {"residu_moyen_anneau_50_300": float(res[anneau].mean()), "residu_median_anneau": float(np.median(res[anneau])),
            "sortie_moyenne_anneau": float(o.mean(0)[anneau].mean()), "sortie_moyenne_couronne_450_650": float(o.mean(0)[ext].mean()),
            "sortie_min_moyenne_disque300": float(o.mean(0)[disque].min())}
h = 400; Y0 = int(np.clip(cy - h, 0, H - 2 * h)); X0 = int(np.clip(cx - h, 0, W - 2 * h))
crops, stats = {"entree": img[:, Y0:Y0 + 2 * h, X0:X0 + 2 * h].copy()}, {}
crops["refext"] = np.array(ref[:, Y0:Y0 + 2 * h, X0:X0 + 2 * h]); stats["refext"] = mesure(np.asarray(ref))
stats["entree"] = {"residu_moyen_anneau_50_300": 0.0, "sortie_moyenne_anneau": float(img.mean(0)[anneau].mean()), "sortie_moyenne_couronne_450_650": float(img.mean(0)[ext].mean())}
def noter(nom, o, t0, extra=None):
    crops[nom] = o[:, Y0:Y0 + 2 * h, X0:X0 + 2 * h].copy(); stats[nom] = mesure(o); stats[nom]["secondes"] = round(time.time() - t0)
    if extra: stats[nom].update(extra)
    log(f"{nom}: {stats[nom]}")
    (OUT / "mesures.json").write_text(json.dumps({"etoile": {"G": float(g[j]), "gaia_xy": [float(gx[j]), float(gy[j])], "pic_xy": [int(cx), int(cy)], "decoupe_origine_xy": [X0, Y0]}, "mesures": stats}, indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(OUT / "decoupes.npz", **crops)
t0 = time.time(); o = SRm.apply(net, img, dev, None, tile=1024, overlap=256); noter("a_comme_mesure", o, t0, {"tuile": 1024, "recouvrement": 256}); del o
# (b) la plus grande tuile que la memoire permet, recouvrement de la moitie
want = min(H, W) // 64 * 64
tile_gb = SRm.TILE_GB * (want / SRm.NET_TILE) ** 2
fits = devices.tile_that_fits(want, dev, tile_gb)
log(f"(b) tuile demandee {want}, que la memoire permet {fits}")
t0 = time.time(); o = SRm.apply(net, img, dev, None, tile=want, overlap=want // 2, batch=1); noter("b_grandes_tuiles", o, t0, {"tuile_demandee": want, "tuile_permise": int(fits), "recouvrement": want // 2}); del o
for k, nom in ((2, "c_reduit_x2"), (3, "d_reduit_x3")):
    t0 = time.time()
    hh, ww = H // k * k, W // k * k
    pe = img[:, :hh, :ww].reshape(3, hh // k, k, ww // k, k).mean((2, 4)).astype(np.float32)
    po = SRm.apply(net, pe, dev, None, tile=1024, overlap=256)
    o = img.copy()
    for c in range(3):
        o[c, :hh, :ww] = cv2.resize(po[c], (ww, hh), interpolation=cv2.INTER_CUBIC)
    noter(nom, o, t0, {"facteur": k}); del o, po, pe
# planche
cols = [("entree", "entree"), ("a_comme_mesure", "(a) comme la mesure"), ("b_grandes_tuiles", "(b) grandes tuiles"), ("c_reduit_x2", "(c) reduit x2"), ("d_reduit_x3", "(d) reduit x3"), ("refext", "RefExt")]
top = float(np.arcsinh(np.percentile(crops["entree"].mean(0), 99.5) / 60)) + 1e-6
tiles = []
for k, lab in cols:
    v = np.clip(np.arcsinh(np.clip(crops[k], 0, None) / 60) / top, 0, 1)
    im = (v.transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)
    im = cv2.resize(im, (560, 560), interpolation=cv2.INTER_AREA).copy()
    cv2.putText(im, lab, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    tiles.append(im)
ok_, buf = cv2.imencode(".jpg", np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90]); buf.tofile(str(OUT / "planche_portee_M27.jpg"))
log("fini")
