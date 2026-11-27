# ======================================================================
# diag_coeurs.py
# Role     : Core diagnostic: radial profiles (input, N7, N8L 24000, RefExt) of the 6 cores + comparison with synthetic cores.
# Step     : diagnostic (26 Nov) (see scripts/LISEZMOI.md)
# Inputs   : validation outputs
# Outputs  : radial profiles, sheet
# Example  : python diag_coeurs.py [arguments]  (see the docstring; paths via the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Core diagnostic: radial profiles (input, N7, N8L 24000, RefExt) of the 6 cores + comparison with synthetic cores.
Read-only on the data; CPU only."""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["ASTRO_GPU_OFFLINE"] = "1"
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np, cv2
SR = (Path(__import__("os").environ.get("RACINE_SCRIPTS") or Path(__file__).resolve().parents[1])); OUT = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Diagnostic cœurs – 26 nov")
sys.path[:0] = [str(SR), str(SR / "Validation – 22 nov"), r"<FLOAT>"]
from liste import LISTE
import torch; torch.set_num_threads(4)
from models import chain_nafnet
from astro_gpu.stages import star_removal as SRm
COEURS = json.loads((SR / "Tour N8L – 26 nov" / "coeurs_liste.json").read_text(encoding="utf-8-sig"))
NOMS = ["M31 A7S coeur", "M32 A7S", "M31 60D coeur", "M32 60D", "M65", "M66"]
CK = {"N8L": r"<DONNEES>\Entrainement étoiles N8L – 26 nov\N8L\last.pt", "N7": r"<DONNEES>\Entrainement étoiles N7 – 25 nov\N7\last.pt"}
HW = 512; dev = torch.device("cpu")
nets = {}
for k, p in CK.items():
    c = torch.load(p, map_location=dev, weights_only=False); n = chain_nafnet(); n.load_state_dict({a: b.float() for a, b in c["net"].items()}); n.eval(); nets[k] = n
    print(k, "pas", c.get("step", c.get("pas", "?")), flush=True)

def prof(a, cx, cy, R=40):
    yy, xx = np.indices(a.shape); r = np.hypot(yy - cy, xx - cx)
    return np.array([a[(r >= i - .5) & (r < i + .5)].mean() for i in range(R + 1)])
def fenetre(nom, x, y):
    d0 = Path(LISTE[nom][0]); img = np.load(d0 / "input.npy", mmap_mode="r"); sn = np.load(d0 / "starless.npy", mmap_mode="r")
    C, H, W = img.shape
    X0 = int(np.clip(x - HW, 0, max(W - 2 * HW, 0))); Y0 = int(np.clip(y - HW, 0, max(H - 2 * HW, 0)))
    sl = (slice(None), slice(Y0, Y0 + 2 * HW), slice(X0, X0 + 2 * HW))
    return np.ascontiguousarray(img[sl], np.float32), np.ascontiguousarray(sn[sl], np.float32), X0, Y0
res = []; vign = {}
for (nom, x, y), lab in zip(COEURS, NOMS):
    ent, sn, X0, Y0 = fenetre(nom, x, y)
    bruit = json.loads(((Path(__import__("os").environ.get("DONNEES", "donnees")) / "Validation étoiles – 22 nov") / nom / "mesures_entree.json").read_text(encoding="utf-8"))["bruit_pixel"]
    fwhm_et = json.loads(((Path(__import__("os").environ.get("DONNEES", "donnees")) / "Validation étoiles – 22 nov") / nom / "mesures_entree.json").read_text(encoding="utf-8"))["fwhm_px"]
    outs = {"entree": ent, "RefExt": sn}
    for k, n in nets.items():
        outs[k] = SRm.apply(n, ent, dev, None, tile=1024, overlap=256)
    cx, cy = x - X0, y - Y0
    L = cv2.GaussianBlur(ent.mean(0), (0, 0), 2); sub = L[cy - 12:cy + 13, cx - 12:cx + 13]
    j, i = np.unravel_index(sub.argmax(), sub.shape); cy2, cx2 = cy - 12 + j, cx - 12 + i
    P = {k: prof(v.mean(0), cx2, cy2) for k, v in outs.items()}
    e = P["entree"]; pic = float(e[0]); fond = float(np.median(ent.mean(0)))
    half = np.argmax(e < pic / 2) if (e < pic / 2).any() else 99
    yy, xx = np.indices(ent.shape[1:]); r = np.hypot(yy - cy2, xx - cx2)
    m15 = r < 15
    flux = {k: float(v.mean(0)[m15].sum()) for k, v in outs.items()}
    c3 = float(ent.mean(0)[r < 3].sum() / ent.mean(0)[m15].sum())
    mx = float(ent.max()); sat = int((ent.mean(0) >= 0.999 * ent.mean(0).max()).sum())
    # 3 anneaux : perte par anneau pour chaque sortie
    anneaux = {}
    for k in ("N7", "N8L", "RefExt"):
        anneaux[k] = [round(float(outs[k].mean(0)[(r >= a) & (r < b)].sum() / max(ent.mean(0)[(r >= a) & (r < b)].sum(), 1e-9)), 3) for a, b in ((0, 3), (3, 8), (8, 15))]
    res.append(dict(nom=lab, image=nom, x=x, y=y, bruit=bruit, fwhm_etoiles=fwhm_et, pic_entree=round(pic, 1), pic_bruits=round(pic / bruit, 1),
                    demi_hauteur_px=int(half), frac_flux_3px=round(c3, 3), flux15=round(flux["entree"], 0),
                    rapports={k: round(flux[k] / flux["entree"], 3) for k in ("N7", "N8L", "RefExt")}, anneaux_0_3_3_8_8_15=anneaux,
                    profils={k: [round(float(t), 2) for t in v] for k, v in P.items()}, centre_fen=[int(cx2), int(cy2)]))
    vign[lab] = {k: v[:, max(cy2 - 40, 0):cy2 + 41, max(cx2 - 40, 0):cx2 + 41] for k, v in outs.items()}
    print(lab, res[-1]["pic_bruits"], res[-1]["demi_hauteur_px"], res[-1]["rapports"], anneaux, flush=True)
    del outs
(OUT / "profils_coeurs.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
np.savez_compressed(OUT / "vignettes.npz", **{f"{l}|{k}": v.astype(np.float16) for l, d in vign.items() for k, v in d.items()})

# --- cœurs synthétiques
sys.path.insert(0, str(SR / "Tour N8L – 26 nov"))
import paires_synthetiques_n7 as PS
rng = np.random.default_rng(1); S = []
for _ in range(300):
    fw = float(rng.uniform(3, 8))
    c, inf = PS.coeur_synthetique((256, 256), rng, fw, 1.0, 0.0)
    m = c.mean(0); cy_, cx_ = np.unravel_index(m.argmax(), m.shape)
    yy, xx = np.indices(m.shape); r = np.hypot(yy - cy_, xx - cx_)
    pe = np.array([m[(r >= i - .5) & (r < i + .5)].mean() for i in range(41)])
    hh = int(np.argmax(pe < pe[0] / 2)) if (pe < pe[0] / 2).any() else 99
    S.append((inf["pic"], hh, float((m[r < 3].sum()) / max(m[r < 15].sum(), 1e-9)), inf["sature"], inf["fwhm"], fw, pe[0]))
S = np.array(S, float)
syn = dict(n=len(S), pic_bruits_q=[float(np.percentile(S[:, 0], q)) for q in (0, 10, 50, 90, 100)],
           demi_hauteur_px_q=[float(np.percentile(S[:, 1], q)) for q in (0, 10, 50, 90, 100)],
           frac_flux_3px_q=[round(float(np.percentile(S[:, 2], q)), 3) for q in (0, 10, 50, 90, 100)],
           fwhm_q=[round(float(np.percentile(S[:, 4], q)), 1) for q in (0, 10, 50, 90, 100)],
           part_saturee=float(S[:, 3].mean()), facteur_largeur=PS.FACTEUR_LARGEUR,
           part_fwhm_sous_9=float((S[:, 4] < 9).mean()), part_pic_sous_100=float((S[:, 0] < 100).mean()))
(OUT / "synthetiques.json").write_text(json.dumps(syn, indent=1), encoding="utf-8"); print(syn)
