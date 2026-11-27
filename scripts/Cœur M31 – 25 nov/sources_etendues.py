# ======================================================================
# sources_etendues.py
# Role     : Gaia sources G<6.5 of the validation images, with the V2 halo radius and the excess of the 40-80 px ring (noise units): a bright extended object gives a strong excess and a radius capped at 300.
# Step     : diagnostic (25 Nov) (see scripts/LISEZMOI.md)
# Inputs   : M31 validation image
# Outputs  : core flux per network
# Example  : python sources_etendues.py [arguments]  (see the docstring; paths via the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
r"""Gaia sources G<6.5 of the validation images, with the V2 halo radius and the excess of the 40-80 px ring (noise units): a bright extended object gives a strong excess and a radius capped at 300."""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np
SR = (Path(__import__("os").environ.get("RACINE_SCRIPTS") or Path(__file__).resolve().parents[1])); sys.path.insert(0, str(SR / "Validation – 22 nov"))
from liste import LISTE
src = open(SR / "Portée M27 – 25 nov" / "deux_echelles_v1v2.py", encoding="utf-8").read()
exec("import numpy as np\ndef halo_mesure" + src.split("def halo_mesure", 1)[1].split("def poids", 1)[0])
DATA = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Validation étoiles – 22 nov"); out = []
for nom in LISTE:
    if not (DATA / nom / "gaia_dr3_profond.npz").exists(): continue
    img = np.load(Path(LISTE[nom][0]) / "input.npy", mmap_mode="r"); C, H, W = img.shape
    z = np.load(DATA / nom / "gaia_dr3_profond.npz"); g = z["Gmag"].astype(float); gx = z["x_pix"].astype(float); gy = z["y_pix"].astype(float)
    bruit = json.loads((DATA / nom / "mesures_entree.json").read_text(encoding="utf-8"))["bruit_pixel"]
    for i in np.flatnonzero(np.isfinite(g) & (g < 6.5) & (gx > 0) & (gx < W) & (gy > 0) & (gy < H)):
        x, y = gx[i], gy[i]; R = 520
        x0, x1, y0, y1 = max(0, int(x) - R), min(W, int(x) + R + 1), max(0, int(y) - R), min(H, int(y) + R + 1)
        p = np.asarray(img[:, y0:y1, x0:x1], np.float32).mean(0); v = p != 0
        yy, xx = np.ogrid[y0:y1, x0:x1]; d = np.hypot(yy - y, xx - x)
        a = v & (d >= 40) & (d < 80); b = v & (d >= 400) & (d < 500)
        if a.sum() < 100 or b.sum() < 500: continue
        ex = (np.median(p[a]) - np.median(p[b])) / bruit
        r = halo_mesure(p, x - x0, y - y0, bruit, v)[0]
        out.append((nom, round(float(g[i]), 2), int(x), int(y), round(float(ex)), round(r)))
        print(out[-1], flush=True)
((Path(__import__("os").environ.get("DONNEES", "donnees")) / "Cœur M31 – 25 nov") / "sources_g65.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
print("fini")
