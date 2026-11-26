# ======================================================================
# mesurer_controle.py
# Rôle     : Mesure d'un reseau sur le jeu de controle de galaxies compactes (ne sert qu'a JUGER, jamais a regler ni a entrainer).
# Étape    : 13. Jeu de contrôle des petites galaxies (26 nov) (voir scripts/LISEZMOI.md)
# Entrées  : images web et extraits Gaia (non publiés)
# Sorties  : controle.json, mesures par réseau
# Exemple  : mesurer_controle.py <checkpoint.pt | dossier de sorties sans etoiles> [--tag nom] [--device cpu]
# ======================================================================
r"""Mesure d'un reseau sur le jeu de controle de galaxies compactes (ne sert qu'a JUGER, jamais a regler ni a entrainer).

    mesurer_controle.py <checkpoint.pt | dossier de sorties sans etoiles> [--tag nom] [--device cpu]

Pour chaque galaxie de controle.json : decoupe 384x384 de l'entree autour de la galaxie, applique le reseau (meme chargement que la chaine N7/N8L :
chain_nafnet + state_dict["net"], astro_gpu.stages.star_removal.apply, tuiles de 256 px recouvertes ; processeur accepte) et ecrit le rapport
flux sortie / flux entree dans 3 px et dans 8 px (fond = mediane de la couronne 24-40 px de chaque image, moyenne des 3 couleurs).
Idéal pour une galaxie : 1 (elle doit rester). Garde-fou : memes rapports sur l'etoile Gaia de meme brillance de chaque galaxie (ideal : 0).
Dossier de sorties : <image>.npy ou <image>/starless.npy, (3, H, W), memes unites que l'entree (APOD : entrees\<image>\input.npy).
Sortie : <DONNEES>\Controle galaxies compactes – 26 nov\Resultats\<tag>.json
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
