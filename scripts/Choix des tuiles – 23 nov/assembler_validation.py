# ======================================================================
# assembler_validation.py
# Rôle     : Assemble tuiles_retenues.csv et validation.jpg (<DONNEES>/Choix des tuiles v2 - 23 nov/) a partir des mesures, du verdict a l'oeil (AR) et des propositions verifiees.
# Étape    : 2. Choix des tuiles (23 nov) (voir scripts/LISEZMOI.md)
# Entrées  : images, mosaïques
# Sorties  : liste de tuiles
# Exemple  : python assembler_validation.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Assemble tuiles_retenues.csv et validation.jpg (<DONNEES>/Choix des tuiles v2 - 23 nov/) a partir des mesures, du verdict a l'oeil (AR) et des propositions verifiees.
Une seule passe, sans image entiere : lit les vignettes deja ecrites dans validation_v2/."""
import sys, csv
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np, cv2
ICI = Path(__file__).resolve().parent
D = ICI / "validation_v2"
OUT = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Choix des tuiles v2 " / "u2013 23 nov")
rd = lambda f: list(csv.DictReader(open(f, encoding="utf-8-sig"), delimiter=";"))
rows = [r for r in rd(D / "mesures_tuiles.csv") if r["image"] not in ("m57", "ngc7662")]
rows += rd(D / "mesures_tuiles_m57_ngc7662.csv")
prop = {(p["image"], p["n"]): p for p in rd(D / "mesures_propositions.csv")}
J1 = {"m57": (3424, 2208), "ngc7662": (1152, 640)}      # tuile du jeu 1 contenant l'anneau / la nebuleuse planetaire (vu a l'oeil)
AR = {  # (image, n) -> motif ; la proposition vient de propositions.csv (mesuree)
    ("bubble_a7s", "3"): "ciel presque vide, etoiles deformees, pas de nebuleuse (0,89 vide)",
    ("ic1318_stellina", "1"): "ciel noir presque vide, taches de bruit de couleur (0,97 vide)",
    ("ic1318_stellina", "2"): "ciel noir presque vide, taches vertes et rouges de bruit de couleur (0,95 vide)",
    ("ic1318_stellina", "3"): "ciel noir presque vide, taches bleues de bruit de couleur (0,96 vide)",
    ("ic5070", "4"): "surtout du ciel noir, nebuleuse seulement dans le coin (0,86 vide)",
    ("m101_a7s", "1"): "degrade fort : disque gris de la pile brute dans la tuile (galaxie bien placee)",
    ("m101_a7s", "2"): "ciel noir presque vide (0,96 vide), quelques etoiles",
    ("m45_40da", "4"): "ciel presque vide (0,95 vide), une etoile rouge, pas de nebuleuse",
    ("m45_asi1600", "1"): "etoile geante avec halo et 6 branches sur toute la tuile (boite 23 %)",
    ("m45_asi1600", "2"): "etoile geante avec halo et 6 branches sur toute la tuile (boite 25 %)",
    ("m45_asi1600", "3"): "ciel noir presque vide (0,94 vide), une etoile a branches",
    ("m45_asi1600", "4"): "ciel noir presque vide (0,97 vide)",
    ("m51_canon6d", "2"): "ciel noir presque vide (0,98 vide), pas de galaxie",
    ("m51_canon6d", "3"): "ciel noir presque vide (0,98 vide), pas de galaxie",
    ("m8_sho", "4"): "moitie droite en ciel noir (0,85 vide), nebuleuse en bord gauche seulement",
    ("ngc7000_gfx", "4"): "ciel presque vide (0,98 vide), nebuleuse invisible",
    ("pleiades_a7s", "3"): "ciel noir presque vide (0,98 vide), pas de nebuleuse",
    ("pleiades_g2014", "2"): "ciel presque vide (0,97 vide), etoiles filees",
    ("pleiades_g2014", "3"): "etoile geante avec halo sur toute la tuile (boite 37 %)",
}
NOTES = {  # remarques (f) ajoutees au motif des tuiles OK de l'image
    "m101_a7s": "entree = pile brute avec degrade (disque gris) ; ",
    "m51_canon6d": "entree = pile brute ; degrade de bord a droite et en bas, hors tuile ; ",
    "ic1318_stellina": "entree = pile ; bandes de bord haut/bas hors tuile ; ",
    "m51_origin": "entree = pile ; bords degrades hors tuile, bruit de couleur ; ",
    "m51_seestar_s30": "entree = pile ; bords degrades hors tuile, bruit de couleur ; ",
}
EXTRA_OK = {("m51_origin", "3"): "galaxie coupee en haut (companion et moitie de M51) ; ", ("m97", "1"): "disque de la nebuleuse > 1/4 mais c'est l'objet, image tres bruitee ; ",
            ("m82", "2"): "fines lignes de pile (colonne verte) faibles ; ", ("m82", "4"): "fines lignes bleues horizontales faibles ; ", ("m82", "3"): "fines lignes de pile faibles ; ",
            ("ngc7000_tiers1", "1"): "etoiles filees ; ", ("pleiades_g2014", "1"): "etoiles peu nombreuses (0,83 vide) ; "}
res = []
for r in rows:
    if r["n"] == "-":
        res.append(dict(image=r["image"], x="", y="", origine="aucune", verdict="OK", motif="AUCUNE tuile (decision de l'orchestrateur : entree bruit de couleur + degrade)", proposition="", n="-", thumb=None)); continue
    nom = r["image"]
    if r["n"] == "J1?":
        if (int(r["x"]), int(r["y"])) != J1[nom]:
            continue
        n = "J1"; orig = "J1"
        th = f"{nom}__J1_{r['x']}_{r['y']}.npy"
    else:
        n = r["n"]; orig = r["origine"]
        th = f"{nom}__{n}.npy"
    d = dict(image=nom, x=int(r["x"]), y=int(r["y"]), origine=f"n{n} {orig}", n=n, thumb=D / th)
    if (nom, n) in AR:
        p = prop[(nom, n)]
        d["verdict"] = "A_REVOIR"; d["motif"] = AR[(nom, n)]
        d["proposition"] = f"remplacer par ({p['x']},{p['y']}) (vide {p['vide']}, geante {p['geante']}, recouvrement {p['recouvrement']})"
        if nom == "m101_a7s" and n == "1":
            d["proposition"] = f"retirer le degrade de l'entree (pile brute) ; a defaut decaler a ({p['x']},{p['y']}), disque un peu reduit"
        dx, dy = int(p["x"]) - int(r["x"]), int(p["y"]) - int(r["y"])
        d["proposition"] += f" ; deplacement ({dx:+d},{dy:+d})"
    else:
        d["verdict"] = "OK"
        mot = NOTES.get(nom, "") + EXTRA_OK.get((nom, n), "")
        d["motif"] = (mot + "contient objet/etoiles, pas de bord ni zero, pas d'etoile geante").strip()
        d["proposition"] = ""
    res.append(d)


def pos(d):
    if d["verdict"] == "A_REVOIR" and (d["image"], d["n"]) in prop:
        p = prop[(d["image"], d["n"])]; return int(p["x"]), int(p["y"])
    return d["x"], d["y"]


# (e) recouvrement entre tuiles d'une meme image (positions retenues : OK telles quelles, A_REVOIR a leur position proposee)
for i, a in enumerate(res):
    if a["x"] == "":
        continue
    for b in res[i + 1:]:
        if b["image"] != a["image"] or b["x"] == "":
            continue
        (ax, ay), (bx, by) = pos(a), pos(b)
        rec = max(0, 512 - abs(ax - bx)) * max(0, 512 - abs(ay - by)) / 512 ** 2
        if rec > 0.2501:
            print("RECOUVREMENT proposition", a["image"], a["n"], b["n"], round(rec, 3))
        rec0 = max(0, 512 - abs(a["x"] - b["x"])) * max(0, 512 - abs(a["y"] - b["y"])) / 512 ** 2
        if rec0 > 0.2501:
            print("RECOUVREMENT origine", a["image"], a["n"], b["n"], round(rec0, 3))
            for t, o in ((a, b), (b, a)):
                t["motif"] += f" ; recouvrement {rec0:.0%} avec n{o['n']}"
                t["verdict"] = "A_REVOIR"
                if not t["proposition"]:
                    t["proposition"] = "decaler d'au moins 150 px"
with open(OUT / "tuiles_retenues.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f, delimiter=";"); w.writerow(["image", "x", "y", "origine", "verdict", "motif", "proposition"])
    for d in res:
        w.writerow([d["image"], d["x"], d["y"], d["origine"], d["verdict"], d["motif"], d["proposition"]])
nok = sum(d["verdict"] == "OK" and d["x"] != "" for d in res); nar = sum(d["verdict"] == "A_REVOIR" for d in res)
print("OK", nok, "A_REVOIR", nar, "total", len(res))
T = 200; COLS = 12
cells = []
for d in res:
    if d["thumb"] is None:
        continue
    t = np.load(d["thumb"])[:, :, ::-1]
    t = cv2.resize(t, (T, T), interpolation=cv2.INTER_AREA).copy()
    s = f"{d['image']} {d['n']}"
    cv2.putText(t, s, (3, 14), 0, 0.4, (0, 0, 0), 3, cv2.LINE_AA); cv2.putText(t, s, (3, 14), 0, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    if d["verdict"] == "A_REVOIR":
        cv2.rectangle(t, (0, 0), (T - 1, T - 1), (0, 0, 255), 5)
    cells.append(t)
while len(cells) % COLS:
    cells.append(np.zeros((T, T, 3), np.uint8))
im = np.vstack([np.hstack(cells[i:i + COLS]) for i in range(0, len(cells), COLS)])
cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(str(OUT / "validation.jpg"))
print(im.shape)
