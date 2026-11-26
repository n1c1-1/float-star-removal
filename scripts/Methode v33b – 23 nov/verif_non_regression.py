# ======================================================================
# verif_non_regression.py
# Rôle     : Module de la chaîne (voir sa docstring ci-dessous et scripts/LISEZMOI.md).
# Étape    : 2. Chaîne classique de soustraction guidée par Gaia (v33b) (voir scripts/LISEZMOI.md)
# Entrées  : image empilée (unités de bruit), astrométrie, extrait Gaia profond hors ligne
# Sorties  : image sans étoiles, cartes de classes
# Exemple  : python verif_non_regression.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
import sys, os, shutil, numpy as np
P1 = r"<SCRIPTS>\Methode v33b – 23 nov\Piste1"
REF = r"<SCRIPTS>\Methode v33 production – 23 nov\Piste1\sorties\m1"
img = sys.argv[1]
REF = REF.replace("m1", img) if img != "m1" else REF
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, P1)
d = os.path.join(P1, "sorties", img)
bak = os.path.join(r"<DONNEES>\v33b_essais", "cache_" + img); os.makedirs(bak, exist_ok=True)
for n in ("modele_psf_image.npz", "modele_psf_image.json", "etalonnage_image.npz", "etalonnage_image.json"):
    if os.path.exists(os.path.join(d, n)): shutil.move(os.path.join(d, n), os.path.join(bak, n))
import etalonnage_image as EI
EI.obtenir(img, print)
for n in ("modele_psf_image.npz", "etalonnage_image.npz"):
    a = np.load(os.path.join(d, n), allow_pickle=True); b = np.load(os.path.join(REF, n), allow_pickle=True)
    for k in a.files:
        x, y = a[k], b[k]
        if x.dtype.kind in "fc":
            print(n, k, x.shape, "maxabs diff", float(np.nanmax(np.abs(x - y))) if x.size else 0, "max", float(np.nanmax(np.abs(y))) if y.size else 0)
        else:
            print(n, k, "egal", bool(np.array_equal(x, y)))
