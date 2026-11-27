# ======================================================================
# banque_bibliotheque.py
# Role     : Round 5: the individual-star bank extended to the LIBRARY (« Bibliotheque images – 22 nov », inventaire.csv), READ-ONLY on the images.
# Step     : 4. Realistic synthetic stars (22 Nov) (see scripts/LISEZMOI.md)
# Inputs   : star-free backgrounds (pair targets), star shape banks
# Outputs  : pairs (input, target), sheets, blind test
# Example  : python banque_bibliotheque.py [id ...]   -> banque\banque2_<id>.npz ; banque\bibliotheque_notes.csv ;
# ======================================================================
"""Round 5: the individual-star bank extended to the LIBRARY (« Bibliotheque images – 22 nov »,
inventaire.csv), READ-ONLY on the images. One ready linear stack per instrument configuration: first the
"to process first" ranking of the BILAN (section 5), then variety; no Euclid; not the stacks of the 4 test images
(Rosette E001/T005, M57 E050/S010, M8 E058, Horsehead). Bright stars found by detection
(banque_detection.py), no astrometry or catalogue.

    python banque_bibliotheque.py [id ...]   -> banque\\banque2_<id>.npz ; banque\\bibliotheque_notes.csv ;
                                                   sheet Synth_17_banque_bibliotheque.png
"""
from __future__ import annotations

import os as _os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    _os.environ[_k] = "2"                                  # au plus 2 fils (consigne du 22 nov après la saturation)

import csv
import gc
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import banque_detection as BD                                        # noqa: E402
import banque_individuelle as BI                                     # noqa: E402
from ajouter_image_banque import lire_image                          # noqa: E402

INV = HERE.parent / "Bibliotheque images – 22 nov" / "inventaire.csv"
#: une pile par configuration : BILAN section 5 d'abord (rangs 1, 2, 4-9), puis la variété (section 2)
CHOIX = ["E065", "E067", "E010", "W010", "W004", "W008", "W002", "E051", "W006",
         "E006", "E011", "E018", "E019", "E020", "E021", "E022", "E023", "E027", "E031", "E035", "E036", "E039",
         "E042", "E052", "E053", "E062", "E068", "E069", "S004", "E060", "E032"]


def main():
    ids = sys.argv[1:] or CHOIX
    rows = {r["id"]: r for r in csv.DictReader(open(INV, encoding="utf-8-sig"), delimiter=";")}
    notes_p = BI.OUTB / "bibliotheque_notes.csv"
    BI.OUTB.mkdir(exist_ok=True)
    done = {}
    if notes_p.exists():
        for r in csv.DictReader(open(notes_p, encoding="utf-8"), delimiter=";"):
            done[r["id"]] = r
    for i in ids:
        r = rows.get(i)
        if r is None:
            print(i, "absent de l'inventaire", flush=True)
            continue
        if i in done and done[i].get("etat") == "ajoutée":
            continue
        t0 = time.time()
        note = dict(id=i, cible=r["cible_canon"], instrument=f"{r['capteur']} | {r['instrument_focale']}",
                    chemin=r["chemin"], fwhm="", aigrettes="", angles="", etoiles="", etat="", secondes="")
        try:
            a = lire_image(Path(r["chemin"].split(" (")[0].strip()))       # le chemin peut porter une note « (+ …) »
            if a.shape[0] not in (1, 3):
                raise ValueError(f"forme {a.shape}")
            b = BD.construire(i, a)
            del a
            gc.collect()
            if b is None:
                note["etat"] = "rien (trop peu d'étoiles utilisables)"
            else:
                BI.sauver(b)
                note.update(fwhm=f"{b['fwhm']:.2f}", aigrettes=str(len(b["angles"])),
                            angles=" ".join(str(round(v)) for v in b["angles"]), etoiles=str(len(b["G"])),
                            etat="ajoutée")
        except Exception as e:                                       # image illisible : notée, on passe
            note["etat"] = f"échec : {repr(e)[:120]}"
        note["secondes"] = f"{time.time() - t0:.0f}"
        done[i] = note
        print(i, note["cible"], note["etat"], note["fwhm"], note["aigrettes"], note["etoiles"], note["secondes"], "s",
              flush=True)
        with open(notes_p, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(note), delimiter=";")
            w.writeheader()
            for v in done.values():
                w.writerow({k: v.get(k, "") for k in note})
    bs = [BI.charger(i) for i, v in done.items() if v.get("etat") == "ajoutée"]
    bs = [b for b in bs if b is not None]
    if bs:
        BI.planche(bs, "Synth_17_banque_bibliotheque.png")


if __name__ == "__main__":
    main()
