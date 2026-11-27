# ======================================================================
# planches.py
# Role     : Control sheet per image: original | linear re-stretched (plain arcsinh) | x4 close-ups (bright and faint star, before then after).
# Step     : 3. De-linearisation and astrometry of the APOD web images (24 Nov) (see scripts/LISEZMOI.md)
# Inputs   : downloaded APOD web images
# Outputs  : linear images, astrometry, tiles
# Example  : python planches.py [arguments]  (see the docstring; paths via the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Control sheet per image: original | linear re-stretched (plain arcsinh) | x4 close-ups (bright and faint
star, before then after). Title: etoiles_auto measurements. Usage: planches.py [numbers...]"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from outils_apod import CTRL, LIN, lire  # noqa: E402

LARG, C, Z = 640, 48, 4


def rgb8(a):
    return Image.fromarray(np.round(np.clip(np.moveaxis(a, 0, -1), 0, 1) * 255).astype(np.uint8))


def arcsinh_simple(D):
    lum = D.mean(0)
    noir, blanc = np.percentile(lum, 0.5), np.percentile(lum, 99.9)
    u = np.clip((D - noir) / max(blanc - noir, 1e-12), 0, 1)
    return np.arcsinh(500 * u) / np.arcsinh(500)


def crop(x, cy, cx):
    a, b = int(round(cy)), int(round(cx))
    h = C // 2
    a = min(max(a, h), x.shape[1] - h); b = min(max(b, h), x.shape[2] - h)
    return x[:, a - h:a + h, b - h:b + h]


def local(c):
    """Gros plan linéaire : noir = fond du cadre, blanc = son maximum (étirement linéaire local)."""
    noir = np.median(np.r_[c[:, 0].ravel(), c[:, -1].ravel()])
    return np.clip((c - noir) / max(float(c.max() - noir), 1e-12), 0, 1)


def main():
    sel = {int(a) for a in sys.argv[1:]}
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    for l in open(CTRL / "mesures.jsonl", encoding="utf-8"):
        r = json.loads(l)
        n = r["numero"]
        if (sel and n not in sel) or "echec" in r or not (LIN / f"{n:03d}.npy").exists():
            continue
        y, D = lire(r["fichier"]), np.load(LIN / f"{n:03d}.npy")
        H, W = y.shape[1:]
        h = int(LARG * H / W)
        A = Image.fromarray(np.round(np.moveaxis(y, 0, -1) * 255).astype(np.uint8)).resize((LARG, h), Image.LANCZOS)
        B = rgb8(arcsinh_simple(D)).resize((LARG, h), Image.LANCZOS)
        page = Image.new("RGB", (2 * LARG + 30, 70 + h + 20 + C * Z + 30), (24, 24, 24))
        page.paste(A, (10, 70)); page.paste(B, (20 + LARG, 70))
        dr = ImageDraw.Draw(page)
        i, eo, el = r.get("info", {}), r["etoiles_orig"], r["etoiles_lin"]
        dr.text((10, 5), f"{n} {r['fichier']} ({r['verdict_orch']}) {W}x{H}", fill=(255, 255, 255), font=font)
        dr.text((10, 25), f"etoiles_auto : couleur {i.get('couleur')}, courbe cout {i.get('cout_depart')} -> {i.get('cout_fin')}, "
                          f"{i.get('etoiles_utilisees')} etoiles de courbe, {el['n_etoiles']} mesurees, FWHM {el['fwhm_px']} px", fill=(230, 230, 230), font=font)
        dr.text((10, 45), f"taille r50 brillantes/faibles : etiree {eo['r50_rapport']:.2f} -> lineaire {el['r50_rapport']:.2f} ; "
                          f"pente taille {el['pente_taille']:.2f} ; flux sur {el['decades_flux']:.2f} decades ; fond noir {100*r['noir_2']:.1f} % ; ecrete larges {100*r['ecrete_larges']:.2f} %",
                fill=(230, 230, 230), font=font)
        dr.text((10, 70 + h + 2), "original | lineaire re-etiree arcsinh (a=500)", fill=(200, 200, 120), font=font)
        z = r.get("zoom_etoiles")
        if z:
            x0 = 10
            for nom in ("brillante", "faible"):
                cy, cx, F, r50 = z[nom]
                av = crop(y, cy, cx); ap = local(crop(D, cy, cx))
                for lab, c in (("avant", av), ("apres", ap)):
                    im = rgb8(c).resize((C * Z, C * Z), Image.NEAREST)
                    page.paste(im, (x0, 70 + h + 20)); dr.text((x0, 70 + h + 20 + C * Z + 2), f"{nom} {lab} x4", fill=(200, 200, 120), font=font)
                    x0 += C * Z + 8
        page.convert("RGB").save(CTRL / f"planche_{n:03d}.jpg", quality=88)
        print("planche", n, flush=True)


if __name__ == "__main__":
    main()
