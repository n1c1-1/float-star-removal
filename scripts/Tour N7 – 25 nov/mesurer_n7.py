# ======================================================================
# mesurer_n7.py
# Rôle     : Mesure de N7 (copie de mesurer_n56.py) : meilleur pas -> inférence deux échelles V2 (inferer_v2.py) -> colonne cœurs (coeurs_mesure.py) -> validation habituelle (valider_n7.py = copie de valider_n56.py) -> tableau N5 / N6 / N7 / m
# Étape    : 9. N7 : cœurs de galaxie synthétiques (25 nov) (voir scripts/LISEZMOI.md)
# Entrées  : paires v33g + cœurs synthétiques
# Sorties  : instantanés N7, tableau_n7.md
# Exemple  : mesurer_n7.py N7 | coeurs_ref | tableau
# ======================================================================
r"""Mesure de N7 (copie de mesurer_n56.py) : meilleur pas -> inférence deux échelles V2 (inferer_v2.py) -> colonne cœurs (coeurs_mesure.py)
-> validation habituelle (valider_n7.py = copie de valider_n56.py) -> tableau N5 / N6 / N7 / modèle FLOAT / RefExt avec la colonne cœurs.

    mesurer_n7.py N7 | coeurs_ref | tableau
      coeurs_ref : colonne cœurs de N5, N6 (meilleur pas de leur history.json) et de RefExt (GPU, quelques secondes)
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
SORTIE = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles N7 – 25 nov")
N56 = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles N5 N6 – 25 nov")
RES = SORTIE / "Resultats"
VALR = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Validation étoiles – 22 nov" / "Resultats")
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


def pas_de(dossier, essai):
    hist = json.load(open(dossier / essai / "history.json", encoding="utf-8"))
    return min(hist, key=lambda h: h["val_etoiles"])["pas"]


def coeurs_de(tag):
    f = SORTIE / "coeurs" / f"coeurs_{tag}.json"
    return json.loads(f.read_text(encoding="utf-8"))["moyenne"] if f.exists() else float("nan")


def coeurs_ref():
    for essai in ("N5", "N6"):
        if (N56 / essai / "history.json").exists() and not (SORTIE / "coeurs" / f"coeurs_{essai}.json").exists():
            lancer([str(ICI / "coeurs_mesure.py"), "mesurer", essai, str(N56 / essai / "snap" / f"pas_{pas_de(N56, essai)}.pt"), "--device", "cuda:0"])
    if not (SORTIE / "coeurs" / "coeurs_RefExt.json").exists():
        lancer([str(ICI / "coeurs_mesure.py"), "mesurer", "RefExt", "refext"])


def tableau():
    out = ["| essai | images | " + " | ".join(f"restantes G {b}" for b in M.BANDES) + " | fond méd. | fond p99 | structure méd. | L1 synthétique | cœurs (sortie/entrée, idéal 1) |",
           "|---|---|" + "---|" * 10]
    for root, t, nom, ctag in ((N56 / "Resultats", "N5", "N5", "N5"), (N56 / "Resultats", "N6", "N6", "N6"), (RES, "N7", "N7", "N7"),
                               (VALR, "livre", "modèle FLOAT", "FLOAT"), (VALR, "reference", "RefExt", "RefExt")):
        if not glob.glob(str(root / t / "*.json")):
            continue
        M.R = root
        r = M.ligne(t)
        c = coeurs_de(ctag)
        out.append(f"| {nom} | {r[1]} | " + " | ".join(f"{x:.3f}" for x in r[2]) + f" | {r[3]:.4f} | {r[4]:.2f} | {r[5]:.3f} | {r[6]:.4f} | "
                   + ("" if c != c else f"{c:.3f}") + " |")
    txt = "\n".join(out)
    (SORTIE / "tableau_n7.md").write_text(txt + "\n", encoding="utf-8")
    print(txt, flush=True)
    planches()


def planches():
    """Une planche par image : entrée | N5 | N6 | N7 | RefExt (arcsinh 60, découpe 800 px, mêmes découpes) ; les crops de N5, N6 viennent du dossier N5 N6."""
    for nom, lab in IMAGES:
        cols, sn = [], None
        for tag, cr in (("N5", N56 / "crops"), ("N6", N56 / "crops"), ("N7", SORTIE / "crops")):
            f = cr / f"{tag}_{nom}.npz"
            if f.exists():
                z = np.load(f)
                if not cols:
                    cols.append(("entree", z["entree"]))
                cols.append((tag, z["sortie"]))
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
    pas = pas_de(SORTIE, essai)
    print(f"{essai} : meilleur pas {pas}", flush=True)
    (SORTIE / f"{essai}_meilleur_pas.txt").write_text(str(pas) + "\n", encoding="utf-8")
    ck = str(SORTIE / essai / "snap" / f"pas_{pas}.pt")
    lancer([str(ICI / "inferer_v2.py"), ck, essai])
    lancer([str(ICI / "coeurs_mesure.py"), "mesurer", essai, ck, "--device", "cuda:0"])
    lancer([str(ICI / "valider_n7.py"), str(SORTIE / f"starless_{essai}"), "--tag", essai])
    shutil.rmtree(SORTIE / f"starless_{essai}", ignore_errors=True)


if __name__ == "__main__":
    a = sys.argv[1]
    if a == "N7":
        mesurer(a)
        coeurs_ref()
        tableau()
    elif a == "coeurs_ref":
        coeurs_ref()
        tableau()
    else:
        tableau()
