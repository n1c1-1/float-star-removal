# ======================================================================
# planche_diag.py
# Rôle     : Module de la chaîne (voir sa docstring ci-dessous et scripts/LISEZMOI.md).
# Étape    : diagnostic (26 nov) (voir scripts/LISEZMOI.md)
# Entrées  : sorties de validation
# Sorties  : profils radiaux, planche
# Exemple  : python planche_diag.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
import sys, json, numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
OUT = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Diagnostic cœurs – 26 nov")
R = {r["nom"]: r for r in json.loads((OUT / "profils_coeurs.json").read_text(encoding="utf-8"))}
V = np.load(OUT / "vignettes.npz")
noms = ["M32 60D", "M32 A7S", "M31 A7S coeur"]; cols = ["entree", "N7", "N8L", "RefExt"]
fig, ax = plt.subplots(2, 4, figsize=(18, 8))
for i, n in enumerate(noms):
    a = ax[0, i]
    for k, c in zip(cols, ["k", "tab:orange", "tab:red", "tab:blue"]):
        a.plot(R[n]["profils"][k], c, label=k)
    a.set_yscale("log"); a.set_title(f"{n} : profil radial 0-40 px (log)"); a.set_xlabel("rayon (px)"); a.legend()
ax[0, 3].axis("off")
ax[0, 3].text(0, .5, "Perte = pic central retire\n(noyau pris pour une etoile),\npas un creux annulaire:\nau-dela de ~8 px la sortie\nrejoint l'entree.", fontsize=12)
n = "M32 60D"
for j, k in enumerate(cols):
    v = V[f"{n}|{k}"].astype(np.float32).mean(0); top = np.arcsinh(np.percentile(V[f"{n}|entree"].astype(np.float32).mean(0), 99.9) / 5)
    ax[1, j].imshow(np.clip(np.arcsinh(np.clip(v, 0, None) / 5) / top, 0, 1), cmap="gray"); ax[1, j].set_title(f"M32 Canon 60D : {k}"); ax[1, j].axis("off")
plt.tight_layout(); plt.savefig(OUT / "planche_pire_coeur.jpg", dpi=80, pil_kwargs={"quality": 88})
