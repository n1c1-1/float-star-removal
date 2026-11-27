# ======================================================================
# b_tests.py
# Role     : Horsehead diagnostic: tests on CROPS (1024 px around each star, measurement and sheet on the centre), one change at a time.
# Step     : 13. Check of saturated low-signal stars (Horsehead, 26 Nov) (see scripts/LISEZMOI.md)
# Inputs   : crops of the real image and of the validation images, .pt networks
# Outputs  : table of remaining flux per star
# Example  : python b_tests.py [arguments]  (see the docstring; paths via the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
r"""Horsehead diagnostic: tests on CROPS (1024 px around each star, measurement and sheet on the centre), one change at a time.
Tests: x1 (single pass) for E3, N5, N7, N7/N5 0.75 blend, FLOAT; blend + V2 x2 (as in the chain); blend + x4; blend with gain x0.5, x4, x8;
blend on an input denoised by FLOAT's denoiser (x1 and V2). Nothing is written outside <DONNEES>\Diagnostic... ; no existing script modified."""
import os, sys, time, json
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
OUT = next((Path(__import__("os").environ.get("DONNEES", "donnees"))).glob("Diagnostic Tete de cheval*")); TR = OUT / "travail"; TR.mkdir(exist_ok=True)
T = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Tete de cheval – 26 nov" / "travail")
def log(s): print(time.strftime("%H:%M:%S"), s, flush=True)
dev = torch.device("cuda:0"); assert "4060" in torch.cuda.get_device_name(0)

def charger(p):
    ck = torch.load(p, map_location=dev, weights_only=False)
    net = chain_nafnet().to(dev); net.load_state_dict({k: v.float() for k, v in ck["net"].items()}); net.eval(); return net
def pas(f): return int(Path(f).read_text().split()[0])
N56 = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles N5 N6 – 25 nov"); N7D = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles N7 – 25 nov")
NETS = {"E3": r"<ESSAIS>\starRemoval\train\E3\snap\step_20000.pt",
        "N5": str(N56 / "N5" / "snap" / f"pas_{pas(N56 / 'N5_meilleur_pas.txt')}.pt"),
        "N7": str(N7D / "N7" / "snap" / f"pas_{pas(N7D / 'N7_meilleur_pas.txt')}.pt"),
        "melange": r"<DONNEES>\Melanges N5 N6 N7 – 26 nov\poids\N7N5_075.pt",
        "melange50": r"<DONNEES>\Melanges N5 N6 N7 – 26 nov\poids\N7N5_050.pt"}
nets = {k: charger(v) for k, v in NETS.items()}; nets["FLOAT"] = SRm.network(dev)
log("reseaux charges")

# ---- entree Tete de cheval et version debruitee (debruiteur de FLOAT, pretrained, memes appels que Denoise.run sur le plein empilement)
z = np.load(T / "entree.npz"); lin = z["lin"]; bk = z["bk"]; unit = float(z["unit"])
img_h = ((lin - bk) / unit).astype(np.float32); C, H, W = img_h.shape
pd = TR / "lin_debruitee.npy"
if not pd.exists():
    from astro_gpu.stages import denoise as DN
    dn = DN.pretrained(dev); dn.eval()
    bk_n, sg_n = [], []
    for c in range(3):
        x = lin[c][::4, ::4]; m = np.median(x[x != 0]); bk_n.append(m); sg_n.append(1.4826 * np.median(np.abs(x - m)[x != 0]))
    bk_n, sg_n = np.array(bk_n, np.float32), np.array(sg_n, np.float32)
    nm = np.full((H, W), 1 / np.sqrt(2), np.float32)       # plein empilement : bruit relatif 1 / racine 2 (Denoise.run), poids uniformes
    t0 = time.time(); d = DN.apply_net(dn, lin, nm, bk_n, sg_n, dev); log(f"debruitage plein {time.time()-t0:.0f} s, bruit ciel avant {sg_n.mean():.5f}")
    d = DN.keep_level(d, lin, np.ones((H, W), np.float32))
    np.save(pd, d.astype(np.float32)); del d, dn
lin_dn = np.load(pd); img_dn = ((lin_dn - bk) / unit).astype(np.float32)

# ---- etoiles
SD = (Path(__import__("os").environ.get("ESSAIS", "essais")) / "stars_dataset")
STARS = [("G3,79 (haut droite)", "h", 3257, 495), ("G6,18", "h", 3288, 1466), ("moyenne A (3212,1969)", "h", 3212, 1969),
         ("moyenne B (998,173)", "h", 998, 173), ("moyenne C (1961,2099)", "h", 1961, 2099), ("moyenne D (2675,1050)", "h", 2675, 1050),
         ("valid IC405 G4,1 (A7S)", "v", "user_FlamingStar", 5763, 3786), ("valid IC5146 G6,4", "v", "set_08_dslr_refractor_canon60da_cocoon", 475, 2735)]
R = 1024
def decoupe(s):
    if s[1] == "h":
        x, y = s[2], s[3]; src_a, src_b = img_h, img_dn
    else:
        x, y = s[3], s[4]; src_a = np.load(SD / s[2] / "input.npy", mmap_mode="r"); src_b = None
    Hh, Ww = src_a.shape[1:]
    x0 = int(np.clip(x - R // 2, 0, Ww - R)); y0 = int(np.clip(y - R // 2, 0, Hh - R))
    a = np.array(src_a[:, y0:y0 + R, x0:x0 + R], np.float32)
    b = np.array(src_b[:, y0:y0 + R, x0:x0 + R], np.float32) if src_b is not None else None
    return a, b, x - x0, y - y0, (x0, y0)

def zone(cx, cy, r=300.0):
    yy, xx = np.ogrid[:R, :R]; d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return (0.5 * (1 + np.cos(np.pi * np.clip((d - 0.7 * r) / (0.3 * r), 0, 1)))).astype(np.float32)
def passe(net, a): return SRm.apply(net, a, dev, None, tile=1024, overlap=256)
def reduit_passe(net, a, k):
    hh = R // k * k; pe = a[:, :hh, :hh].reshape(3, hh // k, k, hh // k, k).mean((2, 4)).astype(np.float32)
    po = passe(net, pe); return np.stack([cv2.resize(po[c], (hh, hh), interpolation=cv2.INTER_CUBIC) for c in range(3)])
def distance(cx, cy):
    yy, xx = np.ogrid[:R, :R]; return np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
def reste(inp, out, cx, cy, r, ring):
    d = distance(cx, cy); L_i, L_o = inp.mean(0), out.mean(0); bg = np.median(L_o[(d >= ring[0]) & (d < ring[1])]); m = d <= r
    fi = np.clip(L_i - bg, 0, None)[m].sum(); fo = np.clip(L_o - bg, 0, None)[m].sum(); return float(fo / max(fi, 1e-9))
def fwhm_de(a, cx, cy):
    d = distance(cx, cy); L = a.mean(0); bgl = np.median(L[(d >= 40) & (d < 60)]); pk = L[d <= 4].max() - bgl
    for r in range(1, 40):
        if np.median(L[(d >= r) & (d < r + 1)]) - bgl < pk / 2: return 2.0 * r
    return 80.0
def sigma_fond(a):
    d = a[1] - np.roll(a[1], 1, 1); s = []
    for y in range(0, R - 64, 64):
        for x in range(0, R - 64, 64):
            q = d[y:y+64, x:x+64]; s.append(1.4826 * np.median(np.abs(q - np.median(q))) / np.sqrt(2))
    return float(np.percentile(s, 25))

rows, fiches = [], []
for s in STARS:
    a, b, cx0, cy0, org = decoupe(s)
    L = a.max(0); y1, x1 = max(0, cy0 - 12), max(0, cx0 - 12); sub = L[y1:cy0 + 13, x1:cx0 + 13]
    dy, dx = np.unravel_index(int(np.argmax(sub)), sub.shape); cx, cy = x1 + dx, y1 + dy
    fw = fwhm_de(a, cx, cy); d = distance(cx, cy)
    pic = float(a[:, cy - 1:cy + 2, cx - 1:cx + 2].max()); nplat = int((L[d <= 3 * fw] >= 0.95 * pic).sum())
    rv = a.mean(0)[(d >= 4 * fw) & (d < 10 * fw)]; aig = float(np.percentile(rv, 99.5) / max(abs(np.percentile(rv, 50)), 0.5))
    nb = sigma_fond(a)
    pic_lin = float(lin[:, org[1] + cy - 1:org[1] + cy + 2, org[0] + cx - 1:org[0] + cx + 2].max()) if s[1] == "h" else None
    fiches.append(dict(nom=s[0], pic_bruits=round(pic, 1), pic_lineaire=pic_lin, px_ge_095pic_3fwhm=nplat, fwhm=round(fw, 1),
                       aigrettes_p995_sur_med=round(aig, 1), bruit_fond=round(nb, 3), origine=list(org), c=[int(cx), int(cy)]))
    r2 = max(2 * fw, 6.0); rw = 25.0
    V = {}; t0 = time.time(); w = zone(cx, cy)
    for k in ("E3", "N5", "N7", "melange", "FLOAT"): V["x1 " + k] = passe(nets[k], a)
    m1 = V["x1 melange"]
    V["x1 melange50"] = passe(nets["melange50"], a)
    V["melange50 V2 x2"] = V["x1 melange50"] * (1 - w) + reduit_passe(nets["melange50"], a, 2) * w
    V["N5 V2 x2"] = V["x1 N5"] * (1 - w) + reduit_passe(nets["N5"], a, 2) * w
    V["melange V2 x2"] = m1 * (1 - w) + reduit_passe(nets["melange"], a, 2) * w
    V["melange V2 x4"] = m1 * (1 - w) + reduit_passe(nets["melange"], a, 4) * w
    for g in (0.5, 4.0, 8.0): V[f"melange gain x{g:g}"] = passe(nets["melange"], a * g) / g
    if b is not None:
        V["melange debruite x1"] = passe(nets["melange"], b)
        V["melange debruite V2"] = V["melange debruite x1"] * (1 - w) + reduit_passe(nets["melange"], b, 2) * w
    ligne = {k: (reste(a, o, cx, cy, r2, (2.5 * fw + 8, 3.5 * fw + 12)), reste(a, o, cx, cy, rw, (35, 45))) for k, o in V.items()}
    rows.append((s[0], ligne))
    log(f"{s[0]} fwhm {fw:.1f} pic {pic:.0f} : {time.time()-t0:.0f} s ; " + " ".join(f"{k}={v[0]:.2f}/{v[1]:.2f}" for k, v in ligne.items()))
    np.savez_compressed(TR / f"crop_{len(rows)}.npz", entree=a, c=np.array([cx, cy]), **{k.replace(" ", "_"): o for k, o in V.items()})
    del V
json.dump(dict(fiches=fiches, tableau={n: l for n, l in rows}), open(OUT / "resultats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log("fini")
