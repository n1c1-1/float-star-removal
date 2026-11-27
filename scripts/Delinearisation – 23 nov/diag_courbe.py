# ======================================================================
# diag_courbe.py
# Role     : Diagnostic: the true inverse curve (median of the linear luminance per level of the stretched luminance) against the estimated curves.
# Step     : 3. De-linearisation of web images (23 Nov) (see scripts/LISEZMOI.md)
# Inputs   : stretched image
# Outputs  : image made linear
# Example  : python diag_courbe.py [arguments]  (see the docstring; paths via the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Diagnostic: the true inverse curve (median of the linear luminance per level of the stretched luminance) against the
estimated curves. Usage: diag_courbe.py <method>[,...] <stack>:<recipe> ...  -> resultats/courbe_<stack>__<recipe>.png"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__file__))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from commun import BANC, RES, lire_etiree  # noqa: E402
from mesurer import echelle  # noqa: E402


def main():
    meths = sys.argv[1].split(",")
    for cas in sys.argv[2:]:
        un(*cas.split(":"), meths)


def un(p, r, meths):
    L = np.load(BANC / f"{p}_lin.npy").astype(np.float64)
    y = lire_etiree(next(BANC.glob(f"{p}__{r}.*[fg]")))
    Y, LL = y.mean(0).ravel(), L.mean(0).ravel()
    edges = np.linspace(0, 1, 201)
    idx = np.digitize(Y, edges) - 1
    yc, lc = [], []
    for k in range(200):
        m = idx == k
        if m.sum() > 30:
            yc.append(Y[m].mean()); lc.append(np.median(LL[m]))
    yc, lc = np.array(yc), np.array(lc)
    sky = np.median(LL)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for a_ in ax:
        a_.plot(yc, lc - sky, "k.", label="vraie (médiane de L par niveau)")
    for m in meths:
        D = np.load(RES / f"{p}__{r}__{m}.npy").astype(np.float64)
        a, b = echelle(L, D)
        DD = ((D - b[:, None, None]) / a).mean(0).ravel()
        dc = np.array([np.median(DD[idx == k]) for k in range(200) if (idx == k).sum() > 30])
        for a_ in ax:
            a_.plot(yc, dc - sky, "-", label=m)
    ax[1].set_yscale("symlog", linthresh=max(1e-6, float(np.std(LL[np.abs(LL - sky) < 5 * np.std(LL)]))))
    ax[0].set_xlabel("niveau étiré"); ax[0].set_ylabel("linéaire - ciel"); ax[0].legend()
    ax[1].set_xlabel("niveau étiré"); ax[1].set_title("échelle symlog")
    fig.suptitle(f"{p} {r}")
    out = RES / f"courbe_{p}__{r}.png"
    fig.savefig(out, dpi=80)
    print("écrit", out)


if __name__ == "__main__":
    main()
