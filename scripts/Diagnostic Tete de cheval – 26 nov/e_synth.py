# ======================================================================
# e_synth.py
# Role     : Test e: does the synthetic generator (paires_synthetiques_n7.paire_synthetique, the training one) produce clipped cores, at what level (in noise units) and in what proportion? Read-only use of the generator.
# Step     : 13. Check of saturated low-signal stars (Horsehead, 26 Nov) (see scripts/LISEZMOI.md)
# Inputs   : crops of the real image and of the validation images, .pt networks
# Outputs  : table of remaining flux per star
# Example  : python e_synth.py [arguments]  (see the docstring; paths via the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
r"""Test e: does the synthetic generator (paires_synthetiques_n7.paire_synthetique, the training one) produce clipped cores,
at what level (in noise units) and in what proportion? Read-only use of the generator."""
import sys, json; sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"<SCRIPTS>\Tour N7 – 25 nov"); sys.path.insert(0, r"<SCRIPTS>\Etoiles synthetiques – 22 nov"); sys.path.insert(0, r"<SCRIPTS>")
import numpy as np, scipy.ndimage as ndi
import paires_synthetiques_n7 as P
S = P._charger(); meta = S["meta"]
sat = np.array([m["sat"] for m in meta]); pl = np.array([m["plateau"] for m in meta])
print("styles", len(meta), "sat vert (bruits) quantiles 0/10/25/50/75/100 :", np.percentile(sat[:,1], [0,10,25,50,75,100]).round(0))
for s in (150, 300, 1000): print(f"  part des styles avec sat vert < {s} : {float((sat[:,1] < s).mean()):.3f}")
print("plateau (lo,hi) : ", pl.min(0), pl.max(0), "moyenne", pl.mean(0).round(2))
print("part de styles avec banque de formes (ecretees):", float(np.mean([len(m["banque_pic"])>0 for m in meta])))
rng = np.random.default_rng(1)
comps = []   # (pic, nb pixels >= 0,97 pic, tuile)
nt = 80; sat_tuile = []
for t in range(nt):
    fond = rng.standard_normal((3, 512, 512)).astype(np.float32)
    e, c = P.paire_synthetique(fond, 1000 + t)
    s = (e - c).max(0)
    lab, n = ndi.label(s > 20)
    if n == 0: continue
    idx = np.arange(1, n + 1)
    pk = ndi.maximum(s, lab, idx)
    for i, p in zip(idx, pk):
        m = lab == i
        comps.append((float(p), int((s[m] >= 0.97 * p).sum()), t))
comps = np.array(comps)
print("composantes > 20 bruits :", len(comps), "sur", nt, "tuiles")
for lo, hi in ((20,100),(100,300),(300,1000),(1000,3000),(3000,1e9)):
    m = (comps[:,0] >= lo) & (comps[:,0] < hi)
    plats = (comps[m,1] >= 6).mean() if m.any() else float('nan')
    print(f"  pic {lo}-{hi:g} : {int(m.sum())} composantes, dont plates (>=6 px a 97 % du pic) : {plats:.2f}")
m = comps[:,0] >= 60; print("pic >= 60 : n", int(m.sum()), "part plates", float((comps[m,1] >= 6).mean()))
print("pics max par tuile : mediane", np.median([comps[comps[:,2]==t,0].max() for t in range(nt) if (comps[:,2]==t).any()]))
