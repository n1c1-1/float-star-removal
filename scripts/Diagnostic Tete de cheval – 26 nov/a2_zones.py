# ======================================================================
# a2_zones.py
# Rôle     : Module de la chaîne (voir sa docstring ci-dessous et scripts/LISEZMOI.md).
# Étape    : 13. Contrôle des étoiles saturées à faible signal (Tête de Cheval, 26 nov) (voir scripts/LISEZMOI.md)
# Entrées  : découpes de l'image réelle et des images de validation, réseaux .pt
# Sorties  : tableau de flux restant par étoile
# Exemple  : python a2_zones.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
import sys; sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np
T = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Tete de cheval – 26 nov" / "travail"); Z = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Tete de cheval zones – 26 nov" / "travail")
z = np.load(T/"entree.npz"); lin = z["lin"]; bk = z["bk"]
mix = np.load(T/"starless_N7N5_075.npy", mmap_mode="r"); zon = np.load(Z/"starless_zones.npy", mmap_mode="r"); w = np.load(Z/"poids_zones.npy", mmap_mode="r")
zd = np.load(Z/"zones_detectees.npz")
P = [("G3,79",3257,495),("G6,18",3288,1466),("A",3212,1969),("B",998,173),("C",1961,2099),("D",2675,1050)]
for n,x,y in P:
    sl=(slice(None),slice(y-16,y+17),slice(x-16,x+17))
    i=(lin[sl]-bk).sum((1,2)).mean(); a=(mix[sl]-bk).sum((1,2)).mean(); b=(zon[sl]-bk).sum((1,2)).mean()
    d=np.hypot(zd["x"]-x,zd["y"]-y); j=int(np.argmin(d))
    print(n, "poids", float(w[y,x]), "reste mix", round(a/i,2), "reste zones", round(b/i,2), "zone detectee la plus proche: dist", round(float(d[j])), "rayon", float(zd["r"][j]))
print("rayons detectes: quantiles", np.percentile(zd["r"],[0,25,50,75,100]), "gaia", zd["gr"])
