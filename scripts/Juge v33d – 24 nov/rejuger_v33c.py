# ======================================================================
# rejuger_v33c.py
# Rôle     : Rejuge v33c : recalcule les poids des paires v33 existantes (rien d'autre).
# Étape    : 2. Juge v33c/v33d : halos et étoiles faibles restées (24 nov) (voir scripts/LISEZMOI.md)
# Entrées  : paires v33
# Sorties  : poids mis à 0 (fichiers _v33d)
# Exemple  : python rejuger_v33c.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Rejuge v33c : recalcule les poids des paires v33 existantes (rien d'autre). usage : rejuger_v33c.py [--ecrire] [--seuil 3] [noms...]
Sans --ecrire : seulement le tableau. Les paires « lachee » de la revue ne sont pas traitees."""
import csv, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import regles_v33c as R
P = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Paires étoiles v33 – 23 nov")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
lach = []
for l in open(P / "revue_orchestrateur.csv", encoding="utf-8"):
    if l.startswith("#") or l.startswith("image;"):
        continue
    a = l.strip().split(";")
    if len(a) >= 3 and a[2] == "lâchée":
        lach.append((a[0], a[1]))
def lachee(im, t):
    return any(i == im and (g == "*" or g == t) for i, g in lach)
ecrire = "--ecrire" in sys.argv
args = [a for a in sys.argv[1:] if a != "--ecrire"]
seuil = R.SEUIL_PLAT
if "--seuil" in args:
    k = args.index("--seuil"); seuil = float(args[k + 1]); del args[k:k + 2]
noms = args or [f.name[:-len("_debruitee.npz")] for f in sorted(P.glob("*_debruitee.npz"))]
lignes, tot = [], []
lignes_pl = {}
for nom in noms:
    im, t = nom.rsplit("_g", 1); t = "g" + t
    if lachee(im, t):
        continue
    z = dict(np.load(P / f"{nom}_debruitee.npz"))
    tx, ty = map(int, t[1:].split("_"))
    W0 = z["weight"]; W = W0.copy()
    r = R.appliquer_v33c(z["input"], z["target"], W, z["classes"], im, tx, ty, seuil=seuil)
    a0, a1 = float((W0 == 0).mean()) * 100, float((W == 0).mean()) * 100
    lignes.append((nom, a0, a1, r["plateaux"], r["ilots"], r["ilots_px"], r["pris"], r["zones05"]))
    if ecrire and not np.array_equal(W, W0):
        z["weight"] = W
        np.savez_compressed(P / f"{nom}_v33c_debruitee.npz".replace("_v33c_debruitee", "_debruitee_v33c"), **z)
        import vue_v33c
        lignes_pl.setdefault(im, []).append(vue_v33c.rangee(z["input"], z["target"], W0, W, r["pris"]))
        fb = P / f"{nom}_brute.npz"
        if fb.exists():
            zb = dict(np.load(fb)); zb["weight"] = W
            np.savez_compressed(P / f"{nom}_brute_v33c.npz", **zb)
out = ["paire ; poids0 avant % ; apres % ; gain points ; plateaux ; ilots ; px ilots ; zones 0,5 prises"]
for nom, a0, a1, n2, ni, px, pris, n3 in lignes:
    out.append(f"{nom} ; {a0:.2f} ; {a1:.2f} ; {a1 - a0:+.2f} ; {n2} ; {ni} ; {px} ; {n3}")
g = [l[2] - l[1] for l in lignes]
out.append(f"MOYENNE gain {np.mean(g):+.3f} ; max {max(g):+.2f} ({lignes[int(np.argmax(g))][0]}) ; paires > +3 : {sum(x > 3 for x in g)} ; paires changees {sum(x > 0 for x in g)}/{len(g)}")
print("\n".join(out))
if ecrire:
    (P / "controle_v33c").mkdir(exist_ok=True)
    (P / "controle_v33c" / "poids0_v33c.txt").write_text("\n".join(out), encoding="utf-8")
    (P / "controle_v33c" / "plateaux_v33c.json").write_text(json.dumps({l[0]: l[6] for l in lignes if l[6]}), encoding="utf-8")

    import cv2
    for im, rows in lignes_pl.items():
        cv2.imencode(".jpg", np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(str(P / "controle_v33c" / f"{im}.jpg"))
