# ======================================================================
# mesurer_controle.py
# Role     : Measures a network on the compact-galaxy control set (used only to JUDGE, never to tune or to train).
# Step     : 13. Control set of small galaxies (26 Nov) (see scripts/LISEZMOI.md)
# Inputs   : web images and Gaia extracts (unpublished)
# Outputs  : controle.json, per-network measurements
# Example  : mesurer_controle.py <checkpoint.pt | folder of starless outputs> [--tag name] [--device cpu]
# ======================================================================
r"""Measures a network on the compact-galaxy control set (used only to JUDGE, never to tune or to train).

    mesurer_controle.py <checkpoint.pt | folder of starless outputs> [--tag name] [--device cpu]

For each galaxy in controle.json: takes a 384x384 crop of the input around the galaxy, applies the network (same loading as the N7/N8L chain:
chain_nafnet + state_dict["net"], astro_gpu.stages.star_removal.apply, overlapping 256 px tiles; CPU accepted) and writes the
output flux / input flux ratio within 3 px and within 8 px (background = median of the 24-40 px ring of each image, mean of the 3 colours).
Ideal for a galaxy: 1 (it must stay). Safeguard: same ratios on the Gaia star of the same brightness for each galaxy (ideal: 0).
Output folder: <image>.npy or <image>/starless.npy, (3, H, W), same units as the input (APOD: entrees\<image>\input.npy).
Output: <DONNEES>\Controle galaxies compactes – 26 nov\Resultats\<tag>.json
"""
import os, sys, json, time, glob
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # la carte est a l'entrainement : processeur seulement
os.environ["ASTRO_GPU_OFFLINE"] = "1"
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np

SR = (Path(__import__("os").environ.get("RACINE_SCRIPTS") or Path(__file__).resolve().parents[1]))
OUT = Path(glob.glob(r"<DONNEES>\Controle galaxies compactes*")[0])
APOD = Path(glob.glob(r"<DONNEES>\Paires ?toiles APOD*")[0])
sys.path.insert(0, str(SR)); sys.path.insert(0, r"<FLOAT>")
HW = 192


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


def entree(nom):
    return np.load(APOD / "entrees" / nom / "input.npy", mmap_mode="r")


def fenetre(a, x, y):
    """(3, 384, 384) autour de (x, y) ; renvoie (fenetre, x0, y0) -- le centre est toujours a 192 sauf si l'image est trop petite."""
    C, H, W = a.shape
    x0 = int(np.clip(round(x) - HW, 0, W - 2 * HW)); y0 = int(np.clip(round(y) - HW, 0, H - 2 * HW))
    return np.ascontiguousarray(a[:, y0:y0 + 2 * HW, x0:x0 + 2 * HW], np.float32), x0, y0


def flux(L, cx, cy):
    """flux moyen des 3 couleurs dans 3 px et dans 8 px, fond = mediane de la couronne 24-40 px."""
    yy, xx = np.mgrid[:L.shape[0], :L.shape[1]]
    r = np.hypot(yy - cy, xx - cx)
    bg = float(np.median(L[(r >= 24) & (r <= 40)]))
    return float((L[r <= 3] - bg).sum()), float((L[r <= 8] - bg).sum())


def charger_sorties(src):
    src = Path(src)
    def get(nom):
        for f in (src / f"{nom}.npy", src / nom / "starless.npy"):
            if f.exists():
                return np.load(f, mmap_mode="r")
        raise FileNotFoundError(f"{nom} : ni {src / (nom + '.npy')} ni {src / nom / 'starless.npy'}")
    return get


def main():
    a = sys.argv
    src = a[1]
    tag = a[a.index("--tag") + 1] if "--tag" in a else Path(src).stem
    ctl = json.loads((OUT / "controle.json").read_text(encoding="utf-8"))
    net = dev = SRm = None
    info = {"source": str(src)}
    if Path(src).is_dir():
        get = charger_sorties(src)
        info["type"] = "dossier de sorties sans etoiles"
    else:
        import torch
        torch.set_num_threads(4)
        from models import chain_nafnet
        from astro_gpu.stages import star_removal as SRm
        dev = torch.device("cpu")
        ck = torch.load(src, map_location=dev, weights_only=False)
        net = chain_nafnet().to(dev)
        net.load_state_dict({k: v.float() for k, v in ck["net"].items()})
        net.eval()
        info.update(type="checkpoint", step=ck.get("step"))
        get = None
    log(f"mesure {tag} : {len(ctl['galaxies'])} galaxies, {info}")
    lignes = []
    cache = {}
    for g in ctl["galaxies"]:
        t0 = time.time()
        nom = g["image"]
        ent = entree(nom)
        row = {"id": g["id"], "image": nom, "x": g["x"], "y": g["y"]}
        for cle, (x, y) in (("galaxie", (g["x"], g["y"])), ("etoile", (g["etoile_comparaison"]["x"], g["etoile_comparaison"]["y"]))):
            win, x0, y0 = fenetre(ent, x, y)
            if net is not None:
                out = SRm.apply(net, win, dev, None, tile=1024, overlap=256)
            else:
                out = np.ascontiguousarray(get(nom)[:, y0:y0 + 2 * HW, x0:x0 + 2 * HW], np.float32)
            cx, cy = x - x0, y - y0
            fe3, fe8 = flux(win.mean(0), cx, cy)
            fs3, fs8 = flux(out.mean(0), cx, cy)
            row[cle] = {"flux_entree_3px": round(fe3, 1), "flux_sortie_3px": round(fs3, 1), "rapport_3px": round(fs3 / fe3, 4) if fe3 > 0 else None,
                        "flux_entree_8px": round(fe8, 1), "flux_sortie_8px": round(fs8, 1), "rapport_8px": round(fs8 / fe8, 4) if fe8 > 0 else None}
        lignes.append(row)
        log(f"  {g['id']:>2} {nom} ({g['x']:.0f},{g['y']:.0f}) galaxie 3px {row['galaxie']['rapport_3px']} 8px {row['galaxie']['rapport_8px']} ; "
            f"etoile 3px {row['etoile']['rapport_3px']} 8px {row['etoile']['rapport_8px']} ({time.time() - t0:.0f} s)")

    def med(cle, k):
        v = [r[cle][k] for r in lignes if r[cle][k] is not None]
        return round(float(np.median(v)), 4) if v else None
    resume = {"galaxie": {"mediane_3px": med("galaxie", "rapport_3px"), "mediane_8px": med("galaxie", "rapport_8px")},
              "etoile_meme_brillance": {"mediane_3px": med("etoile", "rapport_3px"), "mediane_8px": med("etoile", "rapport_8px")}}
    (OUT / "Resultats").mkdir(exist_ok=True)
    (OUT / "Resultats" / f"{tag}.json").write_text(json.dumps({"tag": tag, "info": info, "resume": resume, "galaxies": lignes}, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"RESUME {tag} : {resume}")


if __name__ == "__main__":
    main()
