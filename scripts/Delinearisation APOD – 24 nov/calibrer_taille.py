# ======================================================================
# calibrer_taille.py
# Role     : Reference for the star size measurement: on the 6 linear stacks of the 23 Nov bench, the r50 ratio of bright / faint stars and the slope of the size, for the true linear stack, the stretched image, and the etoiles_auto version.
# Step     : 3. De-linearisation and astrometry of the APOD web images (24 Nov) (see scripts/LISEZMOI.md)
# Inputs   : downloaded APOD web images
# Outputs  : linear images, astrometry, tiles
# Example  : python calibrer_taille.py [arguments]  (see the docstring; paths via the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Reference for the star size measurement: on the 6 linear stacks of the 23 Nov bench, the r50 ratio of bright /
faint stars and the slope of the size, for the true linear stack, the stretched image, and the etoiles_auto version."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from outils_apod import mesures_etoiles  # noqa: E402
from commun import BANC, RES, bruit_laplacien, lire_etiree, mad_sigma  # noqa: E402
from etirer import REC  # noqa: E402

for p in ("rosette", "m51", "ngc869", "pleiades"):
    L = np.load(BANC / f"{p}_lin.npy").astype(np.float64)
    for r in REC:
        f = next(BANC.glob(f"{p}__{r}.*[fg]"))
        y = lire_etiree(f)
        lum = y.mean(0)
        sig = max(mad_sigma(bruit_laplacien(lum)[::2, ::2]), 1 / 255 / np.sqrt(12))
        e, _ = mesures_etoiles(lum, lum, sig)
        a = RES / f"{p}__{r}__etoiles_auto.npy"
        l, _ = mesures_etoiles(lum, L.mean(0), sig)       # mêmes étoiles, sur la VRAIE linéaire
        d = mesures_etoiles(lum, np.load(a).mean(0), sig)[0] if a.exists() else {}
        print(p, r, "etiree %.2f/%.2f | vraie lin %.2f/%.2f | auto %.2f/%.2f" % (
            e["r50_rapport"], e["pente_taille"], l["r50_rapport"], l["pente_taille"],
            d.get("r50_rapport", np.nan), d.get("pente_taille", np.nan)), flush=True)
