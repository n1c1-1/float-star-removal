# ======================================================================
# choisir.py
# Rôle     : 8 tuiles candidates par image, d'apres mesures/<image>.csv -> candidats.csv (image;num;x;y;mesures), choix_vierge.csv, resume.csv
# Étape    : 2. Choix des tuiles (23 nov) (voir scripts/LISEZMOI.md)
# Entrées  : images, mosaïques
# Sorties  : liste de tuiles
# Exemple  : python choisir.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""8 tuiles candidates par image, d'apres mesures/<image>.csv -> candidats.csv (image;num;x;y;mesures), choix_vierge.csv, resume.csv"""
import csv, glob, sys, math
from pathlib import Path
sys.dont_write_bytecode = True
ICI = Path(__file__).resolve().parent
OUT = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Choix des tuiles v2 " / "u2013 23 nov")
OUT.mkdir(parents=True, exist_ok=True)


def ov(a, b):
    return max(0, 512 - abs(a[0] - b[0])) * max(0, 512 - abs(a[1] - b[1])) / 512 ** 2


def bump(d):
    return max(0.0, 1 - abs(d - 0.5) / 0.5)


cand, vierge, resume = [], [], []
for f in sorted(glob.glob(str(ICI / "mesures" / "*.csv"))):
    R = list(csv.DictReader(open(f, encoding="utf-8"), delimiter=";"))
    nom = R[0]["image"]
    pris = [(int(r["x"]), int(r["y"])) for r in R if r["jeu1"] == "1"]
    T = []
    for r in R:
        v = {k: float(r[k]) for k in ("diffus", "nb_etoiles", "petites", "moyennes", "brillantes", "amp_max", "gros", "invalide", "fond")}
        v["x"], v["y"] = int(r["x"]), int(r["y"]); v["num"] = r["numero"]
        if r["jeu1"] == "1" or any(ov((v["x"], v["y"]), p) > 0.25 for p in pris):
            continue
        if v["invalide"] > 0.005 or v["gros"] > 0.02 or v["fond"] > 80:
            continue
        v["tout"] = float(r["tout"]); v["relache"] = 0
        if v["diffus"] < 0.12 or v["nb_etoiles"] < 40:
            if v["diffus"] >= 0.03 and v["tout"] >= 0.12 and v["nb_etoiles"] >= 100:
                v["relache"] = 1          # lumiere diffuse faible : seulement pour completer
            else:
                continue
        cl = (v["petites"] >= 10) + (v["moyennes"] >= 5) + (v["brillantes"] >= 1)
        v["score"] = 0.45 * bump(v["diffus"]) + 0.30 * cl / 3 + 0.15 * min(1, math.log(v["nb_etoiles"]) / math.log(300)) + 0.10 * (1 <= v["brillantes"] <= 8)
        v["score"] *= 0.8 if v["relache"] else 1.0
        v["feat"] = (v["diffus"], math.log10(v["nb_etoiles"]) / 2, math.log10(v["amp_max"] + 10) / 4)
        T.append(v)
    ch = []
    while T and len(ch) < 8:
        def val(t):
            d = min((math.dist(t["feat"], c["feat"]) for c in ch), default=1.0)
            return t["score"] + 0.5 * min(d, 0.6)
        strict = [t for t in T if not t["relache"] and all(ov((t["x"], t["y"]), (c["x"], c["y"])) <= 0.25 for c in ch)]
        ok = strict or [t for t in T if all(ov((t["x"], t["y"]), (c["x"], c["y"])) <= 0.25 for c in ch)]
        if not ok:
            break
        b = max(ok, key=val); ch.append(b); T.remove(b)
    # numeroter 1-8 de gauche a droite, haut en bas
    ch.sort(key=lambda t: (t["y"], t["x"]))
    for i, t in enumerate(ch, 1):
        m = f"diffus={t['diffus']:.2f}, etoiles={int(t['nb_etoiles'])} (petites {int(t['petites'])}, moyennes {int(t['moyennes'])}, brillantes {int(t['brillantes'])}), amp_max={t['amp_max']:.0f}, fond={t['fond']:.1f}, score={t['score']:.2f}{', diffus faible' if t['relache'] else ''}, grille={t['num']}"
        cand.append([nom, i, t["x"], t["y"], m])
    vierge.append([nom, ""])
    resume.append([nom, len(ch), min((t["score"] for t in ch), default=0)])
with open(OUT / "candidats.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f, delimiter=";"); w.writerow(["image", "num", "x", "y", "mesures"]); w.writerows(cand)
with open(OUT / "choix_vierge.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f, delimiter=";"); w.writerow(["image", "nums choisis"]); w.writerows(vierge)
with open(ICI / "resume.csv", "w", newline="") as f:
    w = csv.writer(f, delimiter=";"); w.writerow(["image", "n", "score_min"]); w.writerows([[a, b, round(c, 2)] for a, b, c in resume])
for a, b, c in resume:
    print(a, b, round(c, 2))
