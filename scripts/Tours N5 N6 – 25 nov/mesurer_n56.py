# ======================================================================
# mesurer_n56.py
# Rôle     : Mesure d'un essai N5 / N6 : meilleur pas (plus petite erreur « sous les étoiles » des tenues) -> inférence deux échelles V2 (inferer_v2.py) -> validation habituelle (valider_n56.py = copie de valider_copie.py, mêmes images, mêmes 
# Étape    : 7-8. N5 (cibles v33g) et N6 (poids v33h) (25 nov) (voir scripts/LISEZMOI.md)
# Entrées  : paires v33d -> v33g -> v33h, départ E3 pas 20000
# Sorties  : instantanés N5/N6, tableau_n56.md
# Exemple  : mesurer_n56.py N5 | N6 | N2crops | tableau
# ======================================================================
r"""Mesure d'un essai N5 / N6 : meilleur pas (plus petite erreur « sous les étoiles » des tenues) -> inférence deux échelles V2 (inferer_v2.py)
-> validation habituelle (valider_n56.py = copie de valider_copie.py, mêmes images, mêmes colonnes) -> tableau.

    mesurer_n56.py N5 | N6 | N2crops | tableau
"""
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ICI = Path(__file__).resolve().parent
SORTIE = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles N5 N6 – 25 nov")
RES = SORTIE / "Resultats"
PORTEE = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Portée M27 – 25 nov" / "Resultats")
VALR = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Validation étoiles – 22 nov" / "Resultats")
N2CK = r"<DONNEES>\Entrainement étoiles nuit – 24 nov\N2\snap\pas_26000.pt"
PY = r"python"
import importlib.util
_s = importlib.util.spec_from_file_location("mesurer_v33c", ICI.parent / "Entrainement v33c – 24 nov" / "mesurer_v33c.py")
M = importlib.util.module_from_spec(_s)
_s.loader.exec_module(M)
ENV = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", NUMBA_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
           CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0", PYTHONIOENCODING="utf-8")
IMAGES = [("m31_a7s_newton", "M31 A7S"), ("leotriplet_canon6d_newton", "cœur de Leo"), ("m42_dobson_grande_fwhm", "M42 Dobson"), ("m27_mono_nb", "M27")]


def lancer(args):
    r = subprocess.run([PY, "-I"] + args, env=ENV)
    if r.returncode:
        raise SystemExit(f"échec {args[0]} code {r.returncode}")


def tableau():
    out = ["| essai | images | " + " | ".join(f"restantes G {b}" for b in M.BANDES) + " | fond méd. | fond p99 | structure méd. | L1 synthétique |",
           "|---|---|" + "---|" * 9]
    for root, t, nom in ((PORTEE, "N2_v2", "N2 (V2)"), (RES, "N5", "N5"), (RES, "N6", "N6"), (VALR, "livre", "modèle FLOAT"), (VALR, "reference", "RefExt")):
        if not glob.glob(str(root / t / "*.json")):
            continue
        M.R = root
        r = M.ligne(t)
        out.append(f"| {nom} | {r[1]} | " + " | ".join(f"{x:.3f}" for x in r[2]) + f" | {r[3]:.4f} | {r[4]:.2f} | {r[5]:.3f} | {r[6]:.4f} |")
    txt = "\n".join(out)
    (SORTIE / "tableau_n56.md").write_text(txt + "\n", encoding="utf-8")
    print(txt, flush=True)
    planches()


def planches():
    """Une planche par image : entrée | N2 V2 | N5 | N6 | RefExt (arcsinh 60, découpe 800 px) ; FLOAT : pas de starless conservé."""
    cr = SORTIE / "crops"
    for nom, lab in IMAGES:
        cols = []
        for tag in ("N2", "N5", "N6"):
            f = cr / f"{tag}_{nom}.npz"
            if f.exists():
                z = np.load(f)
                if not cols:
                    cols.append(("entree", z["entree"]))
                cols.append((tag + (" V2" if tag == "N2" else ""), z["sortie"]))
                sn = z["refext"]
        if not cols:
            continue
        cols.append(("RefExt", sn))
        top = float(np.arcsinh(np.percentile(cols[0][1].mean(0), 99.5) / 60)) + 1e-6
        tl = []
        for l, a in cols:
            v = np.clip(np.arcsinh(np.clip(a.astype(np.float32), 0, None) / 60) / top, 0, 1)
            im = cv2.resize((v.transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8), (460, 460), interpolation=cv2.INTER_AREA).copy()
            cv2.putText(im, l, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            tl.append(im)
        (SORTIE / "planches").mkdir(exist_ok=True)
        _, buf = cv2.imencode(".jpg", np.hstack(tl), [cv2.IMWRITE_JPEG_QUALITY, 90])
        buf.tofile(str(SORTIE / "planches" / f"{nom}.jpg"))


def mesurer(essai):
    hist = json.load(open(SORTIE / essai / "history.json", encoding="utf-8"))
    pas = min(hist, key=lambda h: h["val_etoiles"])["pas"]
    print(f"{essai} : meilleur pas {pas}", flush=True)
    (SORTIE / f"{essai}_meilleur_pas.txt").write_text(str(pas) + "\n", encoding="utf-8")
    lancer([str(ICI / "inferer_v2.py"), str(SORTIE / essai / "snap" / f"pas_{pas}.pt"), essai])
    lancer([str(ICI / "valider_n56.py"), str(SORTIE / f"starless_{essai}"), "--tag", essai])
    shutil.rmtree(SORTIE / f"starless_{essai}", ignore_errors=True)


if __name__ == "__main__":
    a = sys.argv[1]
    if a in ("N5", "N6"):
        mesurer(a)
        tableau()
    elif a == "N2crops":
        lancer([str(ICI / "inferer_v2.py"), N2CK, "N2", "--sans-starless"])
        tableau()
    else:
        tableau()
