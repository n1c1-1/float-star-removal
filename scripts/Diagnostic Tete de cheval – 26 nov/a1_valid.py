# ======================================================================
# a1_valid.py
# Role     : Chain module (see its docstring below and scripts/LISEZMOI.md).
# Step     : 13. Check of saturated low-signal stars (Horsehead, 26 Nov) (see scripts/LISEZMOI.md)
# Inputs   : crops of the real image and of the validation images, .pt networks
# Outputs  : table of remaining flux per star
# Example  : python a1_valid.py [arguments]  (see the docstring; paths via the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
import sys, json; sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"<SCRIPTS>\Validation – 22 nov")
from pathlib import Path
import numpy as np
from liste import LISTE, DATA
for nom in LISTE:
    d0 = Path(LISTE[nom][0]); 
    if not (DATA/nom/"gaia_dr3_profond.npz").exists(): continue
    img = np.load(d0/"input.npy", mmap_mode="r"); C,H,W = img.shape
    z = np.load(DATA/nom/"gaia_dr3_profond.npz"); g=z["Gmag"].astype(float); gx=z["x_pix"]; gy=z["y_pix"]
    m = json.loads((DATA/nom/"mesures_entree.json").read_text(encoding="utf-8"))
    ok = np.flatnonzero((g<8)&(gx>20)&(gx<W-20)&(gy>20)&(gy<H-20))
    out=[]
    for i in ok[np.argsort(g[ok])][:6]:
        x,y=int(gx[i]),int(gy[i]); p=np.asarray(img[:,y-8:y+9,x-8:x+9]).max(); out.append((round(g[i],1),x,y,round(float(p))))
    print(nom, (C,H,W), "bruit", m.get("bruit_pixel"), "max", float(np.max(img[:, ::4, ::4])), out)
