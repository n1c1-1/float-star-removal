# ======================================================================
# planche.py
# Rôle     : Planche JPEG : pour 3 piles, original linéaire | étirée (telle quelle) | délinéarisée FLOAT | délinéarisée améliorée.
# Étape    : 3. Délinéarisation des images web (23 nov) (voir scripts/LISEZMOI.md)
# Entrées  : image étirée
# Sorties  : image rendue linéaire
# Exemple  : Usage : planche.py <recette> <méthode améliorée> <pile> <pile> <pile>
# ======================================================================
"""Planche JPEG : pour 3 piles, original linéaire | étirée (telle quelle) | délinéarisée FLOAT | délinéarisée améliorée.
Les trois linéaires sont ramenées à l'échelle de l'original (même ajustement que les mesures) puis montrées avec le même
étirement d'affichage arcsinh 60 ; gros plans x4 sur une étoile brillante et sur la zone de nébuleuse la plus forte.
Usage : planche.py <recette> <méthode améliorée> <pile> <pile> <pile>"""
from __future__ import annotations

import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, __import__("os").path.dirname(__file__))
from commun import BANC, RES, SORTIE, lire_etiree  # noqa: E402
from mesurer import Reference, echelle  # noqa: E402

V, Z, ZC = 470, 235, 59          # côté d'une vue, d'un gros plan, zone recadrée pour le gros plan (x4)


def affiche(x, noir, blanc):
    u = np.clip((x - noir) / (blanc - noir), 0, 1)
    return np.arcsinh(60 * u) / np.arcsinh(60)


def rgb8(a):
    return Image.fromarray(np.round(np.moveaxis(np.clip(a, 0, 1), 0, -1) * 255).astype(np.uint8))


def main():
    rec, meth, piles = sys.argv[1], sys.argv[2], sys.argv[3:6]
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    titres = ["original linéaire", f"étirée ({rec})", "délinéarisée FLOAT", f"délinéarisée améliorée ({meth})"]
    W = 4 * V + 5 * 10
    H = 40 + len(piles) * (30 + V + 8 + Z + 30)
    page = Image.new("RGB", (W, H), (24, 24, 24))
    dr = ImageDraw.Draw(page)
    dr.text((10, 10), f"Délinéarisation – banc du 23 nov : recette {rec} ; affichage commun arcsinh 60 "
                      f"(étirée montrée telle quelle) ; gros plans x4 : étoile brillante | nébuleuse", fill=(230, 230, 230), font=font)
    y0 = 40
    for p in piles:
        L = np.load(BANC / f"{p}_lin.npy").astype(np.float64)
        ref = Reference(L)
        S = lire_etiree(next(BANC.glob(f"{p}__{rec}.*[fg]")))
        vues = [L]
        for m in ("float", meth):
            f = RES / f"{p}__{rec}__{m}.npy"
            if not f.exists():                      # FLOAT a refusé l'image (« the stretch cannot be estimated »)
                vues.append(None)
                continue
            D = np.load(f).astype(np.float64)
            a, b = echelle(L, D)
            vues.append((D - b[:, None, None]) / a)
        lum = L.mean(0)
        sky = float(np.median(lum))
        noir, blanc = sky - 3 * ref.sigL, float(np.percentile(lum, 99.95))
        gris = np.full_like(S, 0.25)
        imgs = [affiche(v, noir, blanc) if v is not None else gris for v in vues]
        imgs.insert(1, S)
        # étoile brillante non saturée dans l'original ; zone de nébuleuse la plus forte sans étoile
        k = int(np.argsort(ref.FLl)[-max(1, len(ref.FLl) // 50)])
        ey, ex = int(ref.ys[k]), int(ref.xs[k])
        iy, ix = np.unravel_index(np.argmax(np.where(ref.libre, ref.lvl, -np.inf)), ref.lvl.shape)
        ny, nx = iy * ref.B + ref.B // 2, ix * ref.B + ref.B // 2
        dr.text((10, y0 + 5), p, fill=(255, 210, 120), font=font)
        y0 += 30
        for j, im in enumerate(imgs):
            x0 = 10 + j * (V + 10)
            page.paste(rgb8(im).resize((V, V), Image.LANCZOS), (x0, y0))
            dr.text((x0 + 6, y0 + 4), titres[j], fill=(255, 255, 255), font=font)
            if j == 2 and vues[1] is None:
                dr.text((x0 + 40, y0 + V // 2), "FLOAT refuse cette image", fill=(255, 120, 120), font=font)
        y1 = y0 + V + 8
        for zi, nomz in enumerate(("étoile brillante x4", "nébuleuse x4")):
            dr.text((10 + zi * 4 * (Z + 5) + 4, y1 + Z + 4),
                    f"{nomz} : original | étirée | FLOAT | améliorée", fill=(200, 200, 200), font=font)
        for zi, (cy, cx) in enumerate(((ey, ex), (ny, nx))):
            h = ZC // 2
            cy = int(np.clip(cy, h, L.shape[1] - h - 1)); cx = int(np.clip(cx, h, L.shape[2] - h - 1))
            for j, im in enumerate(imgs):
                c = im[:, cy - h:cy - h + ZC, cx - h:cx - h + ZC]
                x0 = 10 + (zi * 4 + j) * (Z + 5)
                page.paste(rgb8(c).resize((Z, Z), Image.NEAREST), (x0, y1))
        y0 = y1 + Z + 30
    out = SORTIE / f"planche_{rec}_{meth}.jpg"
    page.save(out, quality=92)
    print("écrit", out, page.size)


if __name__ == "__main__":
    main()
