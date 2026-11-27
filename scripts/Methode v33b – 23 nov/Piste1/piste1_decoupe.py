# ======================================================================
# piste1_decoupe.py
# Role     : Track 1: fast loop on a small crop of the Rosette around the 6 stars circled by the owner.
# Step     : 2. Classic Gaia-guided subtraction chain (v33b) (see scripts/LISEZMOI.md)
# Inputs   : stacked image (noise units), astrometry, offline deep Gaia extract
# Outputs  : star-free image, class maps
# Example  : python piste1_decoupe.py <tag> [JSON settings]
# ======================================================================
"""Track 1: fast loop on a small crop of the Rosette around the 6 stars circled by the owner.

    python piste1_decoupe.py <tag> [JSON settings]
Working crop 768 px (x 1530-2298, y 790-1558 in the whole image), judged zone 512 px at the centre
(x 1658-2170, y 918-1430), which contains the 6 stars. Outputs: sorties/rosette/<tag>_decoupe6/; views and measurements:
sorties/essai_tuiles/<tag>_decoupe6/: input | subtraction only | star-free | layer (review and Post stretches),
the 6 stars at 100 %, point sources left, hollows with control, lumps.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
import images  # noqa: E402
import retrait_modele as RM  # noqa: E402

tag = sys.argv[1]
over = json.loads(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else {}
CROP = (1530, 790, 2298, 1558)
ZONE = (128, 128, 640, 640)             # dans la découpe
SIX = [(1706, 1009), (2043, 968), (2038, 1036), (1762, 1159), (2078, 1220), (1999, 1327)]
case = "decoupe6"
over = dict(over, crop=list(CROP), region_grid=1)
if "--rendu" not in sys.argv:
    RM.run("rosette", f"{tag}_{case}", over)
d = HERE / "sorties" / "rosette" / f"{tag}_{case}"
out = HERE / "sorties" / "essai_tuiles" / f"{tag}_{case}"
out.mkdir(parents=True, exist_ok=True)
full = images.load("rosette")
I = full[:, CROP[1]:CROP[3], CROP[0]:CROP[2]]
A = np.load(d / "soustraction.npy"); S = np.load(d / "starless.npy")
top = float(full.max()); bk = float(np.median(full[1, ::7, ::7])); t998 = float(np.percentile(full[1, ::5, ::5], 99.8))
rev = lambda a: (np.transpose(np.clip(np.arcsinh(np.clip(a + 3, 0, None) / 3) / np.arcsinh((top + 3) / 3), 0, 1), (1, 2, 0)) * 255).astype(np.uint8)
post = lambda a: (np.transpose(np.clip(np.arcsinh(40 * np.clip((a - bk) / (t998 - bk), 0, None)) / np.arcsinh(40), 0, 1), (1, 2, 0)) * 255).astype(np.uint8)


def save(p, im):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 93])
    p.write_bytes(buf.tobytes())


x0, y0, x1, y1 = ZONE
z_ = (slice(None), slice(y0, y1), slice(x0, x1))
for nm, f in (("1_revue", rev), ("2_post", post)):
    sep = np.full((y1 - y0, 6, 3), 255, np.uint8)
    save(out / f"{case}_{nm}_vue_entree_soustraction_sansetoiles_couche.jpg",
         np.concatenate([f(I[z_]), sep, f(A[z_]), sep, f(S[z_]), sep, f(I[z_] - S[z_])], 1))
# les 6 étoiles
rows = []
sig = 0.0931
six = {}
for X, Y in SIX:
    x, y = X - CROP[0], Y - CROP[1]
    sl = (slice(None), slice(y - 32, y + 32), slice(x - 32, x + 32))
    sp = np.full((64, 4, 3), 255, np.uint8)
    rows.append(np.concatenate([post(I[sl]), sp, post(S[sl]), sp, post(I[sl] - S[sl])], 1))
    rows.append(np.full((4, rows[-1].shape[1], 3), 255, np.uint8))
    c = I.mean(0)[y - 6:y + 7, x - 6:x + 7]
    py, px = np.unravel_index(c.argmax(), c.shape)
    xp, yp = x - 6 + px, y - 6 + py
    Ls = S.mean(0)
    six[f"{X},{Y}"] = round(float(Ls[yp - 2:yp + 3, xp - 2:xp + 3].max() - np.median(Ls[yp - 12:yp + 13, xp - 12:xp + 13])) / sig, 1)
save(out / f"{case}_six_etoiles_post_entree_sansetoiles_couche.jpg", np.concatenate(rows, 0))
# mesures (module de mesure de la piste 1)
import piste1_mesures as PM  # noqa: E402
sys.argv = ["", "rosette", tag, case, f"{x0},{y0},{x1},{y1}", f"{CROP[0]},{CROP[1]}"]
PM.main("rosette", tag, case, [x0, y0, x1, y1], [CROP[0], CROP[1]])
m = json.loads((HERE / "sorties" / "essai_tuiles" / tag / f"piste1_mesures_{case}.json").read_text(encoding="utf-8"))
# grumeaux : écart-type de la luminance filtrée à l'échelle de la PSF
from scipy import ndimage as ndi  # noqa: E402
fw = json.loads((d / "resume.json").read_text(encoding="utf-8"))["fwhm"]
mf = lambda a: cv2.GaussianBlur((a - ndi.median_filter(a, 15)).astype(np.float32), (0, 0), fw / 2.355)
gr = round(float(mf(S.mean(0)[y0:y1, x0:x1]).std() / sig), 3)
res = dict(six_reste_bruits=six, ponctuelles_5s=(m["ponctuelles_5s_sans_etoiles"], m["ponctuelles_5s_entree"]),
           ponctuelles_3s=(m["ponctuelles_3s_sans_etoiles"], m["ponctuelles_3s_entree"]),
           creux=m["creux_sans_etoiles"], temoin=m["creux_temoin"], grumeaux_bruits=gr,
           grande_echelle_s16=m["grande_echelle_sans_etoiles_moins_entree_s16"],
           hors_brillantes=dict(part_zone_pct=m["part_zone_hors_brillantes_pct"],
                                ponctuelles_5s=(m["hors_brillantes_ponctuelles_5s_sans_etoiles"], m["hors_brillantes_ponctuelles_5s_entree"]),
                                ponctuelles_3s=(m["hors_brillantes_ponctuelles_3s_sans_etoiles"], m["hors_brillantes_ponctuelles_3s_entree"]),
                                creux=m["hors_brillantes_creux"], temoin=m["hors_brillantes_temoin"]))
(out / f"{case}_resume.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res))
