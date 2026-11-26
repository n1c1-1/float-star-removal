# ======================================================================
# mesurer_v33c.py
# Rôle     : Mesure de fin d'essai : choisit le meilleur pas (plus petite erreur « sous les étoiles » des tenues, history.json), le passe à valider_reseau.py (mêmes 13 images + IC 1848 que pour A, GPU 4060 Ti, 4 fils), puis écrit le tableau A 
# Étape    : 6. Entraînement d'essai 2 h (24 nov) (voir scripts/LISEZMOI.md)
# Entrées  : paires v33c
# Sorties  : instantanés, tableau de mesure
# Exemple  : valider_reseau.py (mêmes 13 images + IC 1848 que pour A, GPU 4060 Ti, 4 fils), puis écrit le tableau
# ======================================================================
"""Mesure de fin d'essai : choisit le meilleur pas (plus petite erreur « sous les étoiles » des tenues, history.json), le passe à
valider_reseau.py (mêmes 13 images + IC 1848 que pour A, GPU 4060 Ti, 4 fils), puis écrit le tableau
A pas 8000 / meilleur pas v33c / modèle FLOAT (livre) / outil de référence, mêmes colonnes que <DONNEES>\\agreger.py.

    lancer_borne.ps1 ... -Args "-I <ce script>"           (un seul calcul GPU à la fois ; jamais pendant l'entraînement)
    option --pas N pour imposer un pas ; --tableau-seul pour refaire le tableau sans remesurer.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

SORTIE = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles v33c – 24 nov" / "V33C")
VAL = Path(__file__).resolve().parent.parent / "Validation – 22 nov"
R = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Validation étoiles – 22 nov" / "Resultats")
PY = r"python"
BANDES = ["<8", "8-11", "11-14", "14-17", ">17"]


def ligne(tag):
    D = [json.load(open(f, encoding="utf-8")) for f in glob.glob(str(R / tag / "*.json")) if not f.endswith("resume.json")]
    D = [d for d in D if "etoiles_restantes" in d]

    def bande(b, k="part_gt_0.1"):
        v = [d["etoiles_restantes"][b][k] for d in D if d["etoiles_restantes"].get(b, {}).get("n", 0) >= 5 and k in d["etoiles_restantes"][b]]
        return float(np.mean(v)) if v else float("nan")
    sb = [d["fond"]["structure_brillante"]["mediane"] for d in D if d["fond"].get("structure_brillante") and "mediane" in d["fond"]["structure_brillante"]]
    sy = [d["synthetique"]["toutes"]["l1"] for d in D if d.get("synthetique") and "toutes" in d["synthetique"] and d["synthetique"]["toutes"]]
    return (tag, len(D), [bande(b) for b in BANDES], float(np.median([d["fond"]["fond"]["mediane"] for d in D])),
            float(np.median([d["fond"]["fond"]["p99"] for d in D])), float(np.median(sb)) if sb else float("nan"),
            float(np.mean(sy)) if sy else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pas", type=int, default=0)
    ap.add_argument("--tableau-seul", action="store_true")
    a = ap.parse_args()
    hist = json.load(open(SORTIE / "history.json", encoding="utf-8"))
    pas = a.pas or min(hist, key=lambda h: h["val_etoiles"])["pas"]
    tag = f"V33C_pas_{pas}"
    print(f"meilleur pas : {pas} (sous les étoiles {[h['val_etoiles'] for h in hist if h['pas'] == pas]})", flush=True)
    if not a.tableau_seul:
        env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", NUMBA_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                   CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0", PYTHONIOENCODING="utf-8")
        subprocess.run([PY, "-I", str(VAL / "valider_reseau.py"), str(SORTIE / "snap" / f"pas_{pas}.pt"), "--tag", tag,
                        "--device", "cuda:0", "--attendre"], cwd=str(VAL), env=env, check=True)
    noms = {"A_E3_pas_8000": "A pas 8000", tag: f"v33c pas {pas}", "livre": "modèle FLOAT", "reference": "outil de référence"}
    out = ["| essai | images | " + " | ".join(f"étoiles >10 % restantes G {b}" for b in BANDES) + " | fond méd. | fond p99 | structure méd. | L1 synthétique |",
           "|---|---|" + "---|" * 9]
    for t, nom in noms.items():
        r = ligne(t)
        out.append(f"| {nom} | {r[1]} | " + " | ".join(f"{x:.3f}" for x in r[2]) + f" | {r[3]:.4f} | {r[4]:.2f} | {r[5]:.3f} | {r[6]:.4f} |")
    txt = "\n".join(out)
    (SORTIE / "tableau_validation.md").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    sys.exit(main())
