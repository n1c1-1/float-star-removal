# ======================================================================
# valider_tuiles.py
# Rôle     : Valide choix_orchestrateur.csv : convertit en coins (x,y) dans l'entree, mesure chaque tuile, ecrit mesures_tuiles.csv + vignettes (npz) dans le dossier de sortie.
# Étape    : 2. Choix des tuiles (23 nov) (voir scripts/LISEZMOI.md)
# Entrées  : images, mosaïques
# Sorties  : liste de tuiles
# Exemple  : python valider_tuiles.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Valide choix_orchestrateur.csv : convertit en coins (x,y) dans l'entree, mesure chaque tuile, ecrit mesures_tuiles.csv + vignettes (npz) dans le dossier de sortie.
Une image a la fois, 2 fils, sans GPU. Le verdict final (a l'oeil) est dans verdict.py."""
import os, sys, csv, re, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np, cv2
cv2.setNumThreads(2)
ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parent / "Paires jeu 2 \u2013 23 nov"))
import planches_choix as pc
V2 = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Choix des tuiles v2 " / "u2013 23 nov")
V1 = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Choix des tuiles " / "u2013 23 nov")
CACHE = ICI / "validation_v2"
cands = {}
for r in csv.DictReader(open(V2 / "candidats.csv", encoding="utf-8-sig"), delimiter=";"):
    cands.setdefault(r["image"], {})[int(r["num"])] = (int(r["x"]), int(r["y"]))
j1 = {}
for f in V1.glob("tuiles_*.csv"):
    for r in csv.DictReader(open(f, encoding="utf-8-sig"), delimiter=";"):
        if r["jeu1"] == "1":
            j1.setdefault(r["image"], []).append((int(r["x"]), int(r["y"])))
choix = {}
for l in open(V2 / "choix_orchestrateur.csv", encoding="utf-8-sig"):
    l = l.strip()
    if not l or l.startswith("#") or l.startswith("image;"):
        continue
    nom, *items = l.split(";")
    choix[nom] = items


def etire(sub, bk, top):
    o = [np.arcsinh(60 * np.clip((sub[c] - bk[c]) / max(top[c] - bk[c], 1e-9), 0, None)) / np.arcsinh(60) for c in range(3)]
    return np.clip(np.stack(o, -1), 0, 1)


def mesurer(t, e):
    """t : tuile (3,512,512) brute ; e : etiree (512,512,3) 0-1."""
    g = e.mean(-1)
    zer = float(np.mean(np.all(t == 0, axis=0)))
    lum = t.mean(0)
    sig = 1.4826 * float(np.median(np.abs(lum - np.median(lum))))
    bl = cv2.blur(lum, (8, 8))[::8, ::8]
    diff = float(np.mean(bl > np.median(lum) + 5 * sig / 8 ** 0 * 0.35))   # lumiere diffuse approx.
    # fraction du ciel presque vide : etire < 0.12 en moyenne
    vide = float(np.mean(cv2.blur(g, (16, 16)) < 0.10))
    # gradient : blocs 64 px, mediane ; ecart p95-p5 en unites etirees
    b = np.median(g.reshape(8, 64, 8, 64).transpose(0, 2, 1, 3).reshape(64, -1), axis=1)
    grad = float(np.percentile(b, 95) - np.percentile(b, 5))
    # etoile geante : plus grande composante g>0.6 (apres flou), boite englobante / surface
    m = (cv2.blur(g, (5, 5)) > 0.7).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    geante = 0.0
    if n > 1:
        i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        geante = float(st[i, cv2.CC_STAT_WIDTH] * st[i, cv2.CC_STAT_HEIGHT] / 512 ** 2)
        aire = float(st[i, cv2.CC_STAT_AREA] / 512 ** 2)
    else:
        aire = 0.0
    # bandes : lignes/colonnes constantes
    bande = float(max(np.mean(np.all(g == g[:, :1], axis=1)), np.mean(np.all(g == g[:1, :], axis=0))))
    return dict(zero=round(zer, 4), vide=round(vide, 3), grad=round(grad, 3), geante=round(geante, 3), aire_geante=round(aire, 3), bande=round(bande, 3), moy=round(float(g.mean()), 3))


rows = []
only = sys.argv[1:]
for nom in sorted(choix):
    if only and nom not in only:
        continue
    items = choix[nom]
    if items and items[0].startswith("AUCUNE"):
        rows.append(dict(image=nom, n="-", x="", y="", origine="aucune", note=items[0]))
        print(nom, "AUCUNE", flush=True)
        continue
    a = pc.charger(nom)
    a = a[None] if a.ndim == 2 else a
    C, H, W = a.shape
    ch = [np.nan_to_num(np.asarray(a[min(c, C - 1)], np.float32)) for c in range(3)]
    bk = [float(np.median(c[::4, ::4])) for c in ch]
    top = [float(np.percentile(c[::4, ::4], 99.8)) for c in ch]
    for k, it in enumerate(items):
        if it == "J1":
            # tuile du jeu 1 contenant l'objet : choisie a l'oeil, voir verdict.py (J1 = indice dans j1[nom] trie)
            cs = sorted(set(j1.get(nom, [])))
            for (x, y) in cs:
                x = min(max(x, 0), W - 512); y = min(max(y, 0), H - 512)
                t = np.stack([c[y:y + 512, x:x + 512] for c in ch]); e = etire(t, bk, top)
                rows.append(dict(image=nom, n="J1?", x=x, y=y, origine="J1-candidate", **mesurer(t, e)))
                np.save(CACHE / f"{nom}__J1_{x}_{y}.npy", (e * 255 + .5).astype(np.uint8))
            continue
        m = re.match(r"\[([\d.]+),([\d.]+)\]", it)
        if m:
            fx, fy = float(m.group(1)), float(m.group(2))
            x = int(round(min(max(fx * W - 256, 0), W - 512))); y = int(round(min(max(fy * H - 256, 0), H - 512)))
            orig = f"[{fx},{fy}]"
        else:
            n_ = int(it); x, y = cands[nom][n_]; orig = f"cand {n_}"
        t = np.stack([c[y:y + 512, x:x + 512] for c in ch]); e = etire(t, bk, top)
        rows.append(dict(image=nom, n=str(k + 1), x=x, y=y, origine=orig, W=W, H=H, **mesurer(t, e)))
        np.save(CACHE / f"{nom}__{k + 1}.npy", (e * 255 + .5).astype(np.uint8))
    print(nom, W, H, flush=True)
keys = ["image", "n", "x", "y", "origine", "W", "H", "zero", "vide", "grad", "geante", "aire_geante", "bande", "moy", "note"]
suffix = "_" + "_".join(only) if only else ""
with open(CACHE / f"mesures_tuiles{suffix}.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, keys, delimiter=";", extrasaction="ignore", restval=""); w.writeheader(); w.writerows(rows)
print("fini", len(rows))
