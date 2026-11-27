# ======================================================================
# a0_choix.py
# Role     : Step 0: spot the stars that remain (input vs N7/N5 0.75 blend vs FLOAT) on the Horsehead.
# Step     : 13. Check of saturated low-signal stars (Horsehead, 26 Nov) (see scripts/LISEZMOI.md)
# Inputs   : crops of the real image and of the validation images, .pt networks
# Outputs  : table of remaining flux per star
# Example  : python a0_choix.py [arguments]  (see the docstring; paths via the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
r"""Step 0: spot the stars that remain (input vs N7/N5 0.75 blend vs FLOAT) on the Horsehead."""
import sys; sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np, scipy.ndimage as ndi
T = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Tete de cheval – 26 nov" / "travail")
z = np.load(T/"entree.npz"); lin = z["lin"]; bk = z["bk"]; unit = float(z["unit"])
print("unit", unit, "bk", bk.ravel(), "shape", lin.shape)
mix = np.load(T/"starless_N7N5_075.npy"); fl = np.load(T/"starless_FLOAT.npy")
pk = lin.max(0)
lab, n = ndi.label(ndi.binary_dilation(pk > 0.5, iterations=2)); idx = np.arange(1, n+1)
mx = ndi.maximum(pk, lab, idx); cen = ndi.center_of_mass(pk*(pk>0.5), lab, idx); area = ndi.sum(np.ones_like(pk), lab, idx)
rows = []
for i,(m,(cy,cx),a) in enumerate(zip(mx,cen,area)):
    if a>6000: continue
    y,x = int(cy),int(cx); r=12
    sl = (slice(None), slice(max(0,y-r),y+r+1), slice(max(0,x-r),x+r+1))
    e = (lin[sl]-bk).sum((1,2)).mean(); rm=(mix[sl]-bk).sum((1,2)).mean(); rf=(fl[sl]-bk).sum((1,2)).mean()
    rows.append((e,x,y,m,rm,rf,a))
rows.sort(reverse=True)
print("n", len(rows))
for e,x,y,m,rm,rf,a in rows[:60]:
    print(f"x{x:5d} y{y:5d} crete{m:.2f} flux_in{e:8.2f} reste_mix{rm/e:6.2f} reste_FLOAT{rf/e:6.2f} aire{a:.0f}")
