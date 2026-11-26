# ======================================================================
# choisir_tuiles.py
# Rôle     : Candidates de tuiles de 512 px pour les images web APOD gardées (comme Choix des tuiles – 23 nov : candidats.csv, mosaïques numérotées, choix_vierge.csv), 2 à 4 par image sur du contenu (nébuleuse, galaxie, champ d'étoiles), jamai
# Étape    : 3. Délinéarisation et astrométrie des images web APOD (24 nov) (voir scripts/LISEZMOI.md)
# Entrées  : images web APOD téléchargées
# Sorties  : images linéaires, astrométrie, tuiles
# Exemple  : python choisir_tuiles.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
r"""Candidates de tuiles de 512 px pour les images web APOD gardées (comme Choix des tuiles – 23 nov : candidats.csv, mosaïques
numérotées, choix_vierge.csv), 2 à 4 par image sur du contenu (nébuleuse, galaxie, champ d'étoiles), jamais du ciel vide.
Une image à la fois, sans GPU. Sortie : <DONNEES>\Choix des tuiles APOD – 24 nov\. Usage : choisir_tuiles.py [numéros...]"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

from outils_apod import CTRL, LIN, lire  # noqa: E402

OUT = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Choix des tuiles APOD – 24 nov")
OUT.mkdir(parents=True, exist_ok=True)
T, PAS = 512, 128
JAUNE = (255, 215, 0)


def ov(a, b):
    return max(0, T - abs(a[0] - b[0])) * max(0, T - abs(a[1] - b[1])) / T ** 2


def bump(d):
    return max(0.0, 1 - abs(d - 0.5) / 0.5)


def mesurer(D, y):
    """Mesures par tuile (grille de pas 128) : lumière diffuse, étoiles, zones écrêtées ou noires de l'original."""
    lum = D.mean(0)
    H, W = lum.shape
    b = 4
    h, w = H // b, W // b
    blocs = np.median(lum[: h * b, : w * b].reshape(h, b, w, b).swapaxes(1, 2).reshape(h, w, -1), axis=2)   # médiane : sans étoiles
    ciel = float(np.percentile(blocs, 20))
    bas = blocs[blocs <= np.percentile(blocs, 40)]
    bruit = max(1.4826 * float(np.median(np.abs(bas - np.median(bas)))), 1e-9)
    diff = ndi.uniform_filter(blocs, 5) > ciel + 5 * bruit
    fond = np.kron(ndi.median_filter(blocs, 9), np.ones((b, b), np.float32))
    fond = np.pad(fond, ((0, H - fond.shape[0]), (0, W - fond.shape[1])), mode="edge")
    s = ndi.gaussian_filter(lum, 1.2) - fond
    sub = s[::7, ::7]
    seuil = 10 * max(1.4826 * float(np.median(np.abs(sub - np.median(sub)))), 1e-12)
    pics = (s == ndi.maximum_filter(s, size=7)) & (s > seuil)
    py, px = np.nonzero(pics)
    amp = s[py, px]
    q50, q95 = (np.percentile(amp, 50), np.percentile(amp, 95)) if len(amp) else (0, 0)
    ecr = (y.max(0) >= 253.5 / 255) | (y.max(0) <= 0.5 / 255)
    R = []
    for y0 in range(0, H - T + 1, PAS):
        for x0 in range(0, W - T + 1, PAS):
            m = (py >= y0) & (py < y0 + T) & (px >= x0) & (px < x0 + T)
            a = amp[m]
            R.append(dict(x=x0, y=y0, diffus=float(diff[y0 // b:(y0 + T) // b, x0 // b:(x0 + T) // b].mean()), nb=int(m.sum()),
                          petites=int((a < q50).sum()), moyennes=int(((a >= q50) & (a < q95)).sum()), brillantes=int((a >= q95).sum()),
                          invalide=float(ecr[y0:y0 + T, x0:x0 + T].mean()),
                          fond=float(np.median(blocs[y0 // b:(y0 + T) // b, x0 // b:(x0 + T) // b]))))
    return R


def attrib(v):
    cl = (v["petites"] >= 10) + (v["moyennes"] >= 5) + (v["brillantes"] >= 1)
    v["score"] = 0.45 * bump(v["diffus"]) + 0.30 * cl / 3 + 0.15 * min(1, math.log(max(v["nb"], 1)) / math.log(300)) + 0.10 * (1 <= v["brillantes"] <= 8)
    v["feat"] = (v["diffus"], math.log10(max(v["nb"], 1)) / 2)


def choisir(R, nmax=4):
    T_ = []
    for v in R:
        if v["invalide"] > 0.005:
            continue
        if v["diffus"] < 0.12 and v["nb"] < 150:
            continue                                  # ciel presque vide
        if v["nb"] < 25 and v["diffus"] < 0.3:
            continue
        attrib(v)
        T_.append(v)
    n_ok = len(T_)
    cible = 4 if n_ok >= 12 else 3 if n_ok >= 4 else 2
    ch = []
    while T_ and len(ch) < min(nmax, cible):
        def val(t):
            d = min((math.dist(t["feat"], c["feat"]) for c in ch), default=1.0)
            return t["score"] + 0.5 * min(d, 0.6)
        ok = [t for t in T_ if all(ov((t["x"], t["y"]), (c["x"], c["y"])) <= 0.1 for c in ch)]
        if not ok:
            break
        b = max(ok, key=val); ch.append(b); T_.remove(b)
    ch.sort(key=lambda t: (t["y"], t["x"]))
    for t in ch:
        t.pop("feat", None)
    return ch, n_ok


def etire(sub, bk, top):
    o = [np.arcsinh(60 * np.clip((sub[c] - bk[c]) / max(top[c] - bk[c], 1e-9), 0, None)) / np.arcsinh(60) for c in range(3)]
    return Image.fromarray((np.clip(np.stack(o, -1), 0, 1) * 255 + .5).astype(np.uint8))


def ligne(nom, D, ch):
    H, W = D.shape[1:]
    bk = [float(np.median(c[::4, ::4])) for c in D]
    top = [float(np.percentile(c[::4, ::4], 99.8)) for c in D]
    s = min(1.0, 450 / max(H, W))
    w, h = max(1, round(W * s)), max(1, round(H * s))
    small = np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(c)).resize((w, h), Image.BOX)) for c in D])
    ovi = etire(small, bk, top)
    dr = ImageDraw.Draw(ovi)
    font = ImageFont.load_default()
    th = []
    for i, t in enumerate(ch, 1):
        x, y = t["x"], t["y"]
        dr.rectangle([round(x * s), round(y * s), round((x + T) * s), round((y + T) * s)], outline=JAUNE, width=2)
        dr.text((round(x * s) + 4, round(y * s) + 4), str(i), fill=JAUNE, font=font)
        tile = etire(D[:, y:y + T, x:x + T], bk, top).resize((160, 160), Image.BOX)
        ImageDraw.Draw(tile).text((4, 4), str(i), fill=JAUNE, font=font)
        th.append(tile)
    g = 4
    row = Image.new("RGB", (450 + 10 + 4 * 160 + 3 * g, max(h, 2 * 160 + g + 24)), (30, 30, 30))
    row.paste(ovi, (0, 0))
    ImageDraw.Draw(row).text((462, 5), nom, fill=(255, 255, 255), font=font)
    for i, t in enumerate(th):
        r_, c_ = divmod(i, 4)
        row.paste(t, (460 + c_ * (160 + g), 24 + r_ * (160 + g)))
    return row


NM = [0]


def mosaique(rows):
    NM[0] += 1
    Wm = max(r.width for r in rows)
    m = Image.new("RGB", (Wm, sum(r.height + 6 for r in rows)), (0, 0, 0))
    y = 0
    for r in rows:
        m.paste(r, (0, y)); y += r.height + 6
    m.save(OUT / f"mosaique_{NM[0]:02d}.jpg", quality=88)
    print("mosaique", NM[0], flush=True)


def main():
    dec = json.loads((CTRL / "decisions.json").read_text(encoding="utf-8"))
    ecartees_astro = set()
    if (CTRL / "astrometrie.jsonl").exists():
        ok = {}
        for l in open(CTRL / "astrometrie.jsonl", encoding="utf-8"):
            r = json.loads(l); ok[r["numero"]] = r["resolu"]
        ecartees_astro = {n for n, v in ok.items() if not v}
    gardees = sorted(int(n) for n, d in dec.items() if d["gardee"] and int(n) not in ecartees_astro)
    fich = {}
    for l in open(CTRL / "mesures.jsonl", encoding="utf-8"):
        r = json.loads(l); fich[r["numero"]] = r["fichier"]
    part = OUT / "cand_part.jsonl"
    done = {}
    if part.exists():
        for l in open(part, encoding="utf-8"):
            r = json.loads(l); done[r["numero"]] = r
    cand, resume, rows = [], [], []
    for n in gardees:
        nom = f"apod_{n:03d}"
        D = np.load(LIN / f"{n:03d}.npy")
        if n in done:
            ch, n_ok = done[n]["ch"], done[n]["n_ok"]
        else:
            ch, n_ok = choisir(mesurer(D, lire(fich[n])))
            open(part, "a", encoding="utf-8").write(json.dumps({"numero": n, "ch": ch, "n_ok": n_ok}) + "\n")
        print(nom, len(ch), "candidates ;", n_ok, "tuiles admissibles", flush=True)
        for i, t in enumerate(ch, 1):
            cand.append([nom, i, t["x"], t["y"], f"diffus={t['diffus']:.2f}, etoiles={t['nb']} (petites {t['petites']}, moyennes {t['moyennes']}, brillantes {t['brillantes']}), fond={t['fond']:.4f}, score={t['score']:.2f}"])
        resume.append([nom, len(ch), n_ok])
        rows.append(ligne(nom, D, ch))
        if len(rows) == 4:
            mosaique(rows); rows = []
    if rows:
        mosaique(rows)
    with open(OUT / "candidats.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";"); w.writerow(["image", "num", "x", "y", "mesures"]); w.writerows(cand)
    with open(OUT / "choix_vierge.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";"); w.writerow(["image", "nums choisis"]); w.writerows([[r[0], ""] for r in resume])
    with open(OUT / "resume.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";"); w.writerow(["image", "candidates", "tuiles admissibles"]); w.writerows(resume)


if __name__ == "__main__":
    main()
