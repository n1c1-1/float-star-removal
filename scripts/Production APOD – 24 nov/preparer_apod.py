# ======================================================================
# preparer_apod.py
# Rôle     : Prepare les entrees APOD pour la chaine v33b (aucun reseau, aucun telechargement) : - entree en unites de bruit, fond 0, comme les autres images (norm_params de la chaine : fond et bruit par couleur, unite = bruit moyen des 3 coul
# Étape    : 3. Production des paires web APOD (24 nov) (voir scripts/LISEZMOI.md)
# Entrées  : images APOD délinéarisées
# Sorties  : paires _debruitee .npz
# Exemple  : usage : preparer_apod.py
# ======================================================================
"""Prepare les entrees APOD pour la chaine v33b (aucun reseau, aucun telechargement) :
 - entree en unites de bruit, fond 0, comme les autres images (norm_params de la chaine : fond et bruit par couleur, unite = bruit moyen des 3 couleurs,
   pas de balance : images deja equilibrees) -> <DONNEES>/Paires étoiles APOD – 24 nov/entrees/<image>/input.npy + meta.json ;
 - astrometrie NNN_astrometrie.npz -> Methode v33b/Piste1/sorties/<image>/astrometrie.npz ; Gaia profond -> Methode v33b/Catalogue Gaia profond – 20 nov ;
 - images.py de v33b : ajout des images apod_* (une fonction, rien d'autre) ;
 - tuiles.csv : choix de l'orchestrateur resolu en x, y (coin haut gauche), borne a [128, W-640] x [128, H-640] ; tout decalage est note.
usage : preparer_apod.py"""
import csv, json, re, shutil, sys
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
LIN = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Delinearisation APOD – 24 nov" / "lineaire"); GAIA = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Catalogue Gaia profond APOD – 24 nov")
CH = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Choix des tuiles APOD – 24 nov")
OUT = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Paires étoiles APOD – 24 nov"); ENT = OUT / "entrees"; ENT.mkdir(parents=True, exist_ok=True)
V = (Path(__import__("os").environ.get("RACINE_SCRIPTS") or Path(__file__).resolve().parents[1]) / "Methode v33b – 23 nov")
ICI = Path(__file__).resolve().parent
cand = {(r["image"], int(r["num"])): (int(r["x"]), int(r["y"])) for r in csv.DictReader(open(CH / "candidats.csv", encoding="utf-8-sig"), delimiter=";")}
lignes = [l.strip() for l in open(CH / "choix_orchestrateur.csv", encoding="utf-8-sig") if l.strip() and not l.startswith("#") and not l.startswith("image;")]
tuiles, notes = [], []
for l in lignes:
    im, *ch = l.split(";")
    n = im.split("_")[1]
    a = np.load(LIN / f"{n}.npy", mmap_mode="r"); H, W = a.shape[1:]
    for c in ch:
        if c.startswith("["):
            fx, fy = map(float, c.strip("[]").split(",")); x, y = int(round(fx * W - 256)), int(round(fy * H - 256)); src = c
        else:
            x, y = cand[(im, int(c))]; src = c
        x2, y2 = min(max(x, 128), W - 640), min(max(y, 128), H - 640)
        if (x2, y2) != (x, y):
            notes.append(f"{im} {src} : ({x},{y}) -> ({x2},{y2}) (bord de l'image {W}x{H})")
        tuiles.append((im, x2, y2, src))
    # entree
    d = ENT / im
    if not (d / "input.npy").exists():
        full = np.array(a, np.float32)
        x4 = full[:, ::4, ::4]
        bk = np.array([np.median(x4[c][x4[c] != 0]) for c in range(3)], np.float32)
        sg = np.array([1.4826 * np.median(np.abs(x4[c] - bk[c])[x4[c] != 0]) for c in range(3)], np.float32)
        unit = float(sg.mean())
        img = (full - bk[:, None, None]) / unit
        d.mkdir(exist_ok=True)
        np.save(d / "input.npy", img.astype(np.float32))
        (d / "meta.json").write_text(json.dumps(dict(name=im, source=str(LIN / f"{n}.npy"), shape=list(img.shape), background=bk.tolist(), noise=sg.tolist(), unit=unit,
                                                      units="noise sigmas (mean of the 3 channels' noise), background 0; published image, not denoised again"), indent=1), encoding="utf-8")
    s = V / "Piste1" / "sorties" / im; s.mkdir(parents=True, exist_ok=True)
    if not (s / "astrometrie.npz").exists():
        shutil.copy2(LIN / f"{n}_astrometrie.npz", s / "astrometrie.npz")
    g = GAIA / f"gaia_dr3_profond_apod_{n}.npz"
    shutil.copy2(g, V / "Catalogue Gaia profond – 20 nov" / f"gaia_dr3_profond_{im}.npz")
    print(im, (H, W), len(ch), "tuiles", flush=True)
with open(ICI / "tuiles.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter=";"); w.writerow(["image", "x", "y", "choix"]); w.writerows(tuiles)
(ICI / "decalages.txt").write_text("\n".join(notes) or "aucun decalage", encoding="utf-8")
p = V / "Piste1" / "images.py"; t = p.read_text(encoding="utf-8")
if "_apod" not in t:
    t += r'''

# 24 nov (images web APOD) : entrees faites par Production APOD – 24 nov\preparer_apod.py (unites de bruit, fond 0), une par dossier.
def _apod():
    for d in sorted((Path(__import__("os").environ.get("DONNEES", "donnees")) / "Paires étoiles APOD – 24 nov" / "entrees").glob("apod_*")):
        IMAGES[d.name] = (d / "input.npy", "APOD", "couleur")


_apod()
'''
    p.write_text(t, encoding="utf-8")
print(len(tuiles), "tuiles ;", len(notes), "decalees")
