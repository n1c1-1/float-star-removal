# ======================================================================
# coeurs_mesure.py
# Rôle     : Colonne « cœurs » : rapport moyen flux sortie / flux entrée dans 15 px autour de chaque cœur de galaxie des images de validation (idéal 1 ; RefExt ~ 1).
# Étape    : 11. Mélanges de poids (26 nov) (voir scripts/LISEZMOI.md)
# Entrées  : instantanés N5, N6, N7
# Sorties  : poids mélangés .pt, tableau_melanges.md
# Exemple  : coeurs_mesure.py detecter                                  -> candidats (DoG sur le starless de référence), à regarder
# ======================================================================
r"""Colonne « cœurs » : rapport moyen flux sortie / flux entrée dans 15 px autour de chaque cœur de galaxie des images de validation
(idéal 1 ; RefExt ~ 1). Même méthode que Cœur M31 – 25 nov\coeur_m31.py (fenêtre autour du cœur, SRm.apply tuile 1024 recouvrement 256).

    coeurs_mesure.py detecter                                  -> candidats (DoG sur le starless de référence), à regarder
    coeurs_mesure.py mesurer <tag> <checkpoint.pt|refext> [--device cpu|cuda:0]
        -> <DONNEES>\Melanges N5 N6 N7 – 26 nov\coeurs\coeurs_<tag>.json (par cœur et moyenne) + fen_<tag>.npy (800 px, M31 A7S)
    coeurs_mesure.py planche                                   -> planche_coeur_m31.jpg (entrée | N5 | N6 | N7 | RefExt, si présents)

Liste des cœurs : COEURS ci-dessous (M31 A7S : x=2802 y=2314 de Cœur M31 ; les autres repérés par « detecter » puis regardés).
Le calcul de rapport suppose la sortie V2 = sortie simple au cœur (aucune étoile G < 6,5 à proximité : vérifié pour M31 A7S)."""
import os, sys, json, time
os.environ.setdefault("ASTRO_GPU_OFFLINE", "1")
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np, cv2

SR = (Path(__import__("os").environ.get("RACINE_SCRIPTS") or Path(__file__).resolve().parents[1]))
OUT = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Melanges N5 N6 N7 – 26 nov")
sys.path.insert(0, str(SR)); sys.path.insert(0, str(SR / "Validation – 22 nov")); sys.path.insert(0, r"<FLOAT>")
from liste import LISTE

#: (image, x, y) ; rempli après « detecter » et contrôle visuel (voir JOURNAL.md)
COEURS = json.loads((Path(__file__).resolve().parent / "coeurs_liste.json").read_text(encoding="utf-8-sig")) \
    if (Path(__file__).resolve().parent / "coeurs_liste.json").exists() else [["m31_a7s_newton", 2802, 2314]]
GAL = ["m31_a7s_newton", "m31_canon60d_objectif_filee", "leotriplet_canon6d_newton", "m33_osc_lunette"]
HW = 512


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


def detecter():
    DATA = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Validation étoiles – 22 nov")
    res = {}
    for nom in GAL:
        d0 = Path(LISTE[nom][0])
        sn = np.load(d0 / "starless.npy", mmap_mode="r")
        C, H, W = sn.shape
        bruit = json.loads((DATA / nom / "mesures_entree.json").read_text(encoding="utf-8"))["bruit_pixel"]
        L = np.zeros((H // 2, W // 2), np.float32)
        for c in range(C):
            L += cv2.resize(np.asarray(sn[c], np.float32), (W // 2, H // 2), interpolation=cv2.INTER_AREA) / C
        dog = cv2.GaussianBlur(L, (0, 0), 3) - cv2.GaussianBlur(L, (0, 0), 30)
        mx = cv2.dilate(dog, np.ones((61, 61), np.uint8))
        pk = np.argwhere((dog >= mx) & (dog > 0))
        pk = sorted(pk, key=lambda p: -dog[p[0], p[1]])[:6]
        res[nom] = [(int(p[1] * 2), int(p[0] * 2), round(float(dog[p[0], p[1]]), 1), round(float(dog[p[0], p[1]] / bruit))) for p in pk]
        log(f"{nom} {W}x{H} bruit {bruit} : candidats (x, y, DoG, DoG/bruit) {res[nom]}")
    (OUT / "coeurs_candidats.json").write_text(json.dumps(res), encoding="utf-8")


def fenetre(nom, x, y):
    d0 = Path(LISTE[nom][0])
    img = np.load(d0 / "input.npy", mmap_mode="r"); sn = np.load(d0 / "starless.npy", mmap_mode="r")
    C, H, W = img.shape
    X0 = int(np.clip(x - HW, 0, max(W - 2 * HW, 0))); Y0 = int(np.clip(y - HW, 0, max(H - 2 * HW, 0)))
    sl = (slice(None), slice(Y0, Y0 + 2 * HW), slice(X0, X0 + 2 * HW))
    return np.ascontiguousarray(img[sl], np.float32), np.ascontiguousarray(sn[sl], np.float32), X0, Y0


def mesurer(tag, ck, device):
    import torch
    torch.set_num_threads(4)
    dev = torch.device(device)
    if dev.type == "cuda":
        assert "4060" in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
    net = SRm = None
    if tag != "RefExt":
        from models import chain_nafnet
        from astro_gpu.stages import star_removal as SRm
        c = torch.load(ck, map_location=dev, weights_only=False)
        net = chain_nafnet().to(dev); net.load_state_dict({k: v.float() for k, v in c["net"].items()}); net.eval()
    rows = []
    (OUT / "coeurs").mkdir(exist_ok=True)
    for nom, x, y in COEURS:
        ent, sn, X0, Y0 = fenetre(nom, x, y)
        sortie = sn if net is None else SRm.apply(net, ent, dev, None, tile=1024, overlap=256)
        yy, xx = np.ogrid[:ent.shape[1], :ent.shape[2]]
        dm = np.hypot(yy - (y - Y0), xx - (x - X0)) < 15
        fe, fs = float(ent.mean(0)[dm].mean()), float(sortie.mean(0)[dm].mean())
        rows.append({"image": nom, "x": x, "y": y, "flux_entree": round(fe, 1), "flux_sortie": round(fs, 1), "rapport": round(fs / fe, 4)})
        log(f"{tag} {nom} ({x},{y}) : entrée {fe:.1f} sortie {fs:.1f} rapport {fs / fe:.3f}")
        if nom == "m31_a7s_newton" and (x, y) == (2802, 2314):
            a, b = int(x - Y0 * 0 - X0) - 400, int(y - Y0) - 400
            np.save(OUT / "coeurs" / f"fen_{tag}.npy", sortie[:, max(b, 0):max(b, 0) + 800, max(a, 0):max(a, 0) + 800].astype(np.float16))
            if tag == "RefExt":
                np.save(OUT / "coeurs" / "fen_entree.npy", ent[:, max(b, 0):max(b, 0) + 800, max(a, 0):max(a, 0) + 800].astype(np.float16))
        del ent, sn, sortie
    moy = float(np.mean([r["rapport"] for r in rows]))
    (OUT / "coeurs" / f"coeurs_{tag}.json").write_text(json.dumps({"tag": tag, "checkpoint": str(ck), "coeurs": rows, "moyenne": round(moy, 4)},
                                                                 indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"{tag} : rapport moyen {moy:.4f} sur {len(rows)} cœurs")


def planche():
    cols = []
    for lab in ("entree", "N5", "N6", "N7", "RefExt"):
        f = OUT / "coeurs" / f"fen_{lab}.npy"
        if f.exists():
            cols.append((lab, np.load(f).astype(np.float32)))
    if len(cols) < 2:
        return
    top = float(np.arcsinh(np.percentile(cols[0][1].mean(0), 99.9) / 60)) + 1e-6
    tl = []
    for lab, a in cols:
        v = np.clip(np.arcsinh(np.clip(a, 0, None) / 60) / top, 0, 1)
        im = cv2.resize((v.transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8), (420, 420), interpolation=cv2.INTER_AREA).copy()
        cv2.putText(im, lab, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        tl.append(im)
    _, buf = cv2.imencode(".jpg", np.hstack(tl), [cv2.IMWRITE_JPEG_QUALITY, 90])
    buf.tofile(str(OUT / "planche_coeur_m31.jpg"))


if __name__ == "__main__":
    a = sys.argv[1]
    if a == "detecter":
        detecter()
    elif a == "mesurer":
        dev = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cpu"
        if sys.argv[2] == "RefExt" or sys.argv[3] == "refext":
            mesurer("RefExt", None, "cpu")
        else:
            mesurer(sys.argv[2], sys.argv[3], dev)
        planche()
    else:
        planche()
