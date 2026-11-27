# ======================================================================
# resoudre.py
# Role     : Astrometry of the kept web images, with the unmodified production code (Methode v33b – 23 nov\Piste1\astrometrie.py: FLOAT's local solver + TAN-SIP refinement on the offline Gaia extract).
# Step     : 3. De-linearisation and astrometry of the APOD web images (24 Nov) (see scripts/LISEZMOI.md)
# Inputs   : downloaded APOD web images
# Outputs  : linear images, astrometry, tiles
# Example  : python resoudre.py [arguments]  (see the docstring; paths via the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
r"""Astrometry of the kept web images, with the unmodified production code (Methode v33b – 23 nov\Piste1\astrometrie.py:
FLOAT's local solver + TAN-SIP refinement on the offline Gaia extract). The approximate centre comes from a table (object in the APOD
title); without a centre, blind solving. Output: lineaire\NNN_astrometrie.npz (cat, scale, wres, wcs: v33b format)
and controle\astrometrie.jsonl. Usage: resoudre.py [numbers...]"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))
sys.path.insert(0, str(ICI.parent / "Methode v33b – 23 nov" / "Piste1"))
import numpy as np  # noqa: E402

import astrometrie  # noqa: E402  (production v33b)
from outils_apod import CTRL, LIN  # noqa: E402

# centre approché (RA, Dec en degrés) lu dans le titre APOD ; absent = résolution aveugle d'abord
CENTRES = {22: (270.5, -24.0), 27: (258.43, -37.10), 33: (100.0, 8.73), 37: (287.44, -63.86), 39: (184.74, 47.30), 41: (328.37, 47.27),
           49: (40.44, 0.44), 51: (53.40, -36.14), 57: (54.72, -26.34), 58: (234.9, 59.3), 73: (119.5, -47.0)}

# essai 2 : quand la recherche libre échoue (champ étroit, peu d'étoiles Tycho-2), même appel avec une échelle supposée tirée d'une grille
from astro_gpu.core import asterion  # noqa: E402
_solve0 = asterion.solve
ECHELLES = (0.2, 0.3, 0.45, 0.7, 1.0, 1.5, 2.2, 3.3)


def _solve(*a, scale=None, **k):
    r = _solve0(*a, scale=scale, **k)
    if r is None and scale is None and k.get("ra") is not None:
        for e in ECHELLES:
            r = _solve0(*a, scale=e, **k)
            if r is not None:
                break
    return r


asterion.solve = _solve

dec = json.loads((CTRL / "decisions.json").read_text(encoding="utf-8"))
sel = [int(a) for a in sys.argv[1:]] or [int(n) for n, d in dec.items() if d["gardee"]]
mj = CTRL / "astrometrie.jsonl"
for n in sel:
    t0 = time.time()
    img = np.load(LIN / f"{n:03d}.npy")
    H, W = img.shape[1:]
    log = lambda s: print(n, s, flush=True)  # noqa: E731
    row = {"numero": n, "taille": [W, H]}
    c = CENTRES.get(n)
    astrometrie.find_offline = (lambda t, c=c: c if c else (None, None))
    try:
        wcs, scale, wres = astrometrie.solve(img, str(n), LIN / f"tmp_{n}", log)
        cat = astrometrie.gaia_stars(wcs, (H, W), scale, log=log)
        np.savez(LIN / f"{n:03d}_astrometrie.npz", cat=np.array(cat, dtype=object), scale=scale, wres=wres,
                 wcs=wcs.to_header_string(relax=True))
        ce = wcs.pixel_to_world(W / 2, H / 2)
        row.update(resolu=True, centre_donne=c, echelle=round(scale, 4), residu_px=round(wres, 3), centre=[round(float(ce.ra.deg), 4), round(float(ce.dec.deg), 4)],
                   champ_deg=[round(W * scale / 3600, 3), round(H * scale / 3600, 3)], etoiles_gaia=int(len(cat["x"])))
    except Exception as e:  # noqa: BLE001
        row.update(resolu=False, centre_donne=c, erreur=repr(e)[:300])
    import shutil
    shutil.rmtree(LIN / f"tmp_{n}", ignore_errors=True)
    row["s"] = round(time.time() - t0, 1)
    open(mj, "a", encoding="utf-8").write(json.dumps(row, ensure_ascii=False) + "\n")
    print(n, "->", {k: row.get(k) for k in ("resolu", "echelle", "residu_px", "centre", "etoiles_gaia", "s")}, flush=True)
