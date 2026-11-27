# ======================================================================
# coeur_m31.py
# Role     : Core of M31 (m31_a7s_newton): N2 (step 26000) and N5 (step 23000), single scale and V2, on a crop; CPU only.
# Step     : diagnostic (25 Nov) (see scripts/LISEZMOI.md)
# Inputs   : M31 validation image
# Outputs  : core flux per network
# Example  : python coeur_m31.py [arguments]  (see the docstring; paths via the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
r"""Core of M31 (m31_a7s_newton): N2 (step 26000) and N5 (step 23000), single scale and V2, on a crop; CPU only.
Reuses halo_mesure / poids from deux_echelles_v1v2.py (copy, nothing modified). Outputs: <DONNEES>\Cœur M31 – 25 nov."""
import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["ASTRO_GPU_OFFLINE"] = "1"
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np, cv2
SR = (Path(__import__("os").environ.get("RACINE_SCRIPTS") or Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(SR)); sys.path.insert(0, r"<FLOAT>")
import torch
torch.set_num_threads(4)
from models import chain_nafnet
from astro_gpu.stages import star_removal as SRm
OUT = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Cœur M31 – 25 nov"); OUT.mkdir(exist_ok=True)
DATA = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Validation étoiles – 22 nov" / "m31_a7s_newton"); D0 = (Path(__import__("os").environ.get("ESSAIS", "essais")) / "stars_dataset" / "user_M31")
CKS = {"N2": r"<DONNEES>\Entrainement étoiles nuit – 24 nov\N2\snap\pas_26000.pt", "N5": r"<DONNEES>\Entrainement étoiles N5 N6 – 25 nov\N5\snap\pas_23000.pt"}
def log(s): print(time.strftime("%H:%M:%S"), s, flush=True)
dev = torch.device("cpu")
src = open(SR / "Portée M27 – 25 nov" / "deux_echelles_v1v2.py", encoding="utf-8").read()
exec("import numpy as np\ndef halo_mesure" + src.split("def halo_mesure", 1)[1].split("def planche", 1)[0])
img = np.array(np.load(D0 / "input.npy", mmap_mode="r"), np.float32); C, H, W = img.shape
sn = np.load(D0 / "starless.npy", mmap_mode="r")
z = np.load(DATA / "gaia_dr3_profond.npz"); g = z["Gmag"].astype(float); gx = z["x_pix"].astype(float); gy = z["y_pix"].astype(float)
bruit = json.loads((DATA / "mesures_entree.json").read_text(encoding="utf-8"))["bruit_pixel"]
L = img.mean(0); Ls = cv2.GaussianBlur(cv2.resize(L, (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 25)
my, mx = np.unravel_index(np.argmax(Ls), Ls.shape); cx, cy = mx * 4 + 2, my * 4 + 2
log(f"image {W}x{H} bruit {bruit} ; coeur (pic lissé) x={cx} y={cy}")
# (a) sources G<8 dans la zone 800 px
R = 400; m = np.isfinite(g) & (g < 8) & (abs(gx - cx) < R) & (abs(gy - cy) < R)
lst = [(int(z["Source"][i]), round(float(g[i]), 2), round(float(gx[i]), 1), round(float(gy[i]), 1), round(float(np.hypot(gx[i] - cx, gy[i] - cy)), 1)) for i in np.flatnonzero(m)]
mm = np.isfinite(g) & (abs(gx - cx) < R) & (abs(gy - cy) < R)
plus_brillante = min(((float(g[i]), round(float(gx[i]), 1), round(float(gy[i]), 1)) for i in np.flatnonzero(mm)), default=None)
log(f"sources G<8 dans la zone 800 px: {lst} ; plus brillante: {plus_brillante}")
# decoupe 1500 px (marge pour la mesure de halo et les tuiles), affichage 800 central
hw = 750; X0 = int(np.clip(cx - hw, 0, W - 2 * hw)); Y0 = int(np.clip(cy - hw, 0, H - 2 * hw)); hh = ww = 2 * hw
crop = np.ascontiguousarray(img[:, Y0:Y0 + hh, X0:X0 + ww]); Lc = crop.mean(0); valid = Lc != 0
sx, sy = gx - X0, gy - Y0
s2 = np.flatnonzero(np.isfinite(g) & (g < 6.5) & (sx > -300) & (sx < ww + 300) & (sy > -300) & (sy < hh + 300))
rays = [halo_mesure(Lc, sx[i], sy[i], bruit, valid)[0] for i in s2]
log(f"V2 : {len(s2)} etoiles G<6.5 dans la decoupe, rayons {[(round(float(g[i]),2), round(r), round(float(sx[i])), round(float(sy[i]))) for i, r in zip(s2, rays)]}")
def poids(shape, gx_, gy_, rr):
    Hh, Ww = shape; w = np.zeros((Hh, Ww), np.float32)
    for x, y, r in zip(gx_, gy_, rr):
        x0, x1 = max(0, int(x - r)), min(Ww, int(x + r) + 1); y0, y1 = max(0, int(y - r)), min(Hh, int(y + r) + 1)
        if x1 <= x0 or y1 <= y0: continue
        yy, xx = np.ogrid[y0:y1, x0:x1]; d = np.sqrt((yy - y) ** 2 + (xx - x) ** 2)
        t = np.clip((d - 0.7 * r) / (0.3 * r), 0, 1)
        w[y0:y1, x0:x1] = np.maximum(w[y0:y1, x0:x1], 0.5 * (1 + np.cos(np.pi * t)))
    return w
w2 = poids((hh, ww), sx[s2], sy[s2], rays)
cxl, cyl = int(cx - X0), int(cy - Y0)
log(f"poids V2 au coeur: {float(w2[cyl, cxl]):.2f} ; part image {float((w2>0).mean()):.3f}")
res = {}
pe = crop.reshape(3, hh // 2, 2, ww // 2, 2).mean((2, 4)).astype(np.float32)
for nom, ck in CKS.items():
    t0 = time.time()
    c = torch.load(ck, map_location=dev, weights_only=False)
    net = chain_nafnet().to(dev); net.load_state_dict({k: v.float() for k, v in c["net"].items()}); net.eval()
    full = SRm.apply(net, crop, dev, None, tile=1024, overlap=256)
    po = SRm.apply(net, pe, dev, None, tile=1024, overlap=256)
    up = full.copy()
    for k in range(3): up[k] = cv2.resize(po[k], (ww, hh), interpolation=cv2.INTER_CUBIC)
    v2 = full * (1 - w2) + up * w2
    res[nom] = (full, v2); log(f"{nom} fait en {time.time()-t0:.0f} s")
    np.save(OUT / f"{nom}_simple.npy", full.astype(np.float16)); np.save(OUT / f"{nom}_v2.npy", v2.astype(np.float16))
    del net
# planche 800 central
a = hw - 400; b = hw + 400
cols = [("entree", crop), ("N2 simple", res["N2"][0]), ("N2 V2", res["N2"][1]), ("N5 simple", res["N5"][0]), ("N5 V2", res["N5"][1]), ("RefExt", np.asarray(sn[:, Y0:Y0 + hh, X0:X0 + ww]))]
top = float(np.arcsinh(np.percentile(cols[0][1][:, a:b, a:b].mean(0), 99.9) / 60)) + 1e-6
tl = []
for lab, arr in cols:
    v = np.clip(np.arcsinh(np.clip(arr[:, a:b, a:b], 0, None) / 60) / top, 0, 1)
    im = cv2.resize((v.transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8), (420, 420), interpolation=cv2.INTER_AREA).copy()
    cv2.putText(im, lab, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2); tl.append(im)
_, buf = cv2.imencode(".jpg", np.hstack(tl), [cv2.IMWRITE_JPEG_QUALITY, 90]); buf.tofile(str(OUT / "planche_coeur_m31.jpg"))
# chiffres au coeur : flux moyen disque 15 px, rapport a l'entree et a RefExt
yy, xx = np.ogrid[:hh, :ww]; dm = np.hypot(yy - cyl, xx - cxl) < 15
mes = {lab: [round(float(arr[k][dm].mean()), 1) for k in range(3)] for lab, arr in cols}
log(f"flux moyen (disque 15 px au coeur) RVB: {mes}")
(OUT / "coeur_resultats.json").write_text(json.dumps({"coeur_xy": [int(cx), int(cy)], "sources_G_lt_8": lst, "plus_brillante_zone": plus_brillante,
  "V2_etoiles": [[round(float(g[i]),2), round(float(sx[i]+X0)), round(float(sy[i]+Y0)), round(r)] for i, r in zip(s2, rays)], "poids_V2_coeur": float(w2[cyl, cxl]), "flux_coeur": mes}, indent=1, ensure_ascii=False), encoding="utf-8")
log("fini")
