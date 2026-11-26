# ======================================================================
# mesurer_n3.py
# Rôle     : COPIE du 25 nov de mesurer_nuit.py : essais N3a, N3b (meme mesure, meme tableau, lignes N1/N2/N3 ajoutees).
# Étape    : 10. Essai écarté N3 (grands halos synthétiques, 25 nov) (voir scripts/LISEZMOI.md)
# Entrées  : paires v33d/v33e
# Sorties  : instantanés, tableau
# Exemple  : mesurer_nuit.py N1            meilleur pas (plus petite erreur « sous les étoiles » des tenues) -> valider_reseau.py
# ======================================================================
"""COPIE du 25 nov de mesurer_nuit.py : essais N3a, N3b (meme mesure, meme tableau, lignes N1/N2/N3 ajoutees).
Mesure de fin d'essai (N1 ou N2) : même mesure de validation que V33C (mesurer_v33c.py), même tableau.

    mesurer_nuit.py N1            meilleur pas (plus petite erreur « sous les étoiles » des tenues) -> valider_reseau.py
                                  (13 images + IC 1848, GPU 4060 Ti) -> tableau + planche de M27
    options : --pas N, --tableau-seul
Tableau : A pas 8000 / V33C pas 13000 / N1 / N2 / modèle FLOAT / outil de référence (lignes absentes ignorées).
Sorties : <DONNEES>\\Entrainement étoiles nuit – 24 nov\\tableau_nuit.md et planches\\M27_<essai>_pas_<n>.jpg (+ M27 des autres).
"""
import argparse
import glob
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ICI = Path(__file__).resolve().parent
SORTIE = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles N3 – 25 nov")
VAL = ICI.parent / "Validation – 22 nov"
R = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Validation étoiles – 22 nov" / "Resultats")
PY = r"python"
_s = importlib.util.spec_from_file_location("mesurer_v33c", ICI.parent / "Entrainement v33c – 24 nov" / "mesurer_v33c.py")
M = importlib.util.module_from_spec(_s)
_s.loader.exec_module(M)


def tableau():
    tags = [("A_E3_pas_8000", "A pas 8000"), ("V33C_pas_13000", "V33C pas 13000")]
    for e in ("N1", "N2", "N3a", "N3b"):
        for d in sorted(glob.glob(str(R / f"{e}_pas_*"))):
            tags.append((Path(d).name, Path(d).name.replace("_pas_", " pas ")))
    tags += [("livre", "modèle FLOAT"), ("reference", "outil de référence")]
    out = ["| essai | images | " + " | ".join(f"étoiles >10 % restantes G {b}" for b in M.BANDES)
           + " | fond méd. | fond p99 | structure méd. | L1 synthétique |", "|---|---|" + "---|" * 9]
    for t, nom in tags:
        if not glob.glob(str(R / t / "*.json")):
            continue
        r = M.ligne(t)
        out.append(f"| {nom} | {r[1]} | " + " | ".join(f"{x:.3f}" for x in r[2])
                   + f" | {r[3]:.4f} | {r[4]:.2f} | {r[5]:.3f} | {r[6]:.4f} |")
    txt = "\n".join(out)
    (SORTIE / "tableau_nuit.md").write_text(txt + "\n", encoding="utf-8")
    print(txt, flush=True)
    (SORTIE / "planches").mkdir(exist_ok=True)
    for t, _ in tags:
        p = R / t / "m27_mono_nb_planche.jpg"
        if p.exists():
            shutil.copy(p, SORTIE / "planches" / f"M27_{t}.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("essai", choices=["N1", "N2", "N3a", "N3b"])
    ap.add_argument("--pas", type=int, default=0)
    ap.add_argument("--tableau-seul", action="store_true")
    a = ap.parse_args()
    hist = json.load(open(SORTIE / a.essai / "history.json", encoding="utf-8"))
    pas = a.pas or min(hist, key=lambda h: h["val_etoiles"])["pas"]
    tag = f"{a.essai}_pas_{pas}"
    print(f"{a.essai} : meilleur pas {pas} ({[h['val_etoiles'] for h in hist if h['pas'] == pas]})", flush=True)
    if not a.tableau_seul and not (R / tag / "resume.json").exists():
        env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", NUMBA_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                   CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0", PYTHONIOENCODING="utf-8")
        subprocess.run([PY, "-I", str(VAL / "valider_reseau.py"), str(SORTIE / a.essai / "snap" / f"pas_{pas}.pt"), "--tag", tag,
                        "--device", "cuda:0", "--attendre"], cwd=str(VAL), env=env, check=True)
    tableau()


if __name__ == "__main__":
    sys.exit(main())
