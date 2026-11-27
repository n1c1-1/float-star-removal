# ======================================================================
# verifier_compacts.py
# Role     : N9: 200 synthetic cores drawn as in training (40 % compact); share of the flux within 3 px (3 px disk / 15 px disk, around the maximum, core alone).
# Stage    : 10. Discarded trial N9 (compact cores, 26 Nov) (see scripts/LISEZMOI.md)
# Inputs   : v33g pairs
# Outputs  : snapshots, tableau_n9.md
# Example  : verifier_compacts.py [--part 0.4]  -> compacts.json
# ======================================================================
r"""N9: 200 synthetic cores drawn as in training (40 % compact); share of the flux within 3 px (3 px disk / 15 px disk, around the maximum, core alone).
    verifier_compacts.py [--part 0.4]  -> compacts.json"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, r"<SCRIPTS>")
import paires_synthetiques_n7 as PS
S = PS._charger()
rng = np.random.default_rng(12345)
res = {"compact": [], "etendu": []}
for i in range(200):
    fw = float(S["meta"][int(rng.integers(len(S["meta"])))]["fwhm"])
    compact = bool(np.random.default_rng([i, 777]).random() < PS.PART_COMPACTS)
    c, inf = PS.coeur_synthetique((256, 256), np.random.default_rng([i, 5]), fw, 1.0, 0.0, compact=compact)
    L = c.mean(0); yy, xx = np.indices(L.shape)
    j, k = np.unravel_index(L.argmax(), L.shape)
    r = np.hypot(yy - j, xx - k)
    f3 = float(L[r < 3].sum() / L[r < 15].sum())
    res["compact" if compact else "etendu"].append(dict(f3=round(f3, 3), fw_etoile=round(fw, 2), fwhm=round(inf["fwhm"], 2), re=round(inf["re"], 2), n=round(inf["sersic_n"], 2), sature=inf["sature"], sigma_lissage=round(inf["sigma_lissage"], 2)))
for k, v in res.items():
    f = np.array([x["f3"] for x in v])
    print(k, len(v), "f3 centiles 0/10/25/50/75/90/100 :", [round(float(np.percentile(f, q)), 3) for q in (0, 10, 25, 50, 75, 90, 100)],
          "dans 15-30 % :", f"{np.mean((f >= .15) & (f <= .30)):.0%}", "fwhm/fw étoile médiane", round(float(np.median([x['fwhm'] / x['fw_etoile'] for x in v])), 2))
Path(__file__).with_name("compacts.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
