# ======================================================================
# day_e2.py
# Rôle     : E2 : paires régénérées avec les couleurs de saturation et les noyaux de galaxie, puis entraînement.
# Étape    : 1. Entraînement sur données synthétiques seules (E2 / E3, 7-9 nov) (voir scripts/LISEZMOI.md)
# Entrées  : générateur : graines + catalogue.json (fiches d'instruments publiques) ; aucune image
# Sorties  : paires synthétiques (input/target/info.json + manifest.json), réseau E3
# Exemple  : python day_e2.py [--hours 6.5]
# ======================================================================
"""The night of 7 to 8 November, second part: the pairs regenerated with the saturation's colours and the galaxies'
nuclei corrected (pairs_v3), then run E taken on from its step 16000 on them (E2), then its evaluation against the
chain's network: evaluate.py, diagnose.py gap, apply_real.py's pictures.

    python day_e2.py [--hours 6.5]
"""
from __future__ import annotations

import argparse
import sys
import time

from day import CLONG, DEV, ROOT, log, run

PAIRS = ROOT / "pairs_v3"
VAL = ROOT / "diagnose" / "valset_pairs_v3.npz"
START = ROOT / "train" / "E" / "snap" / "step_16000.pt"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hours", type=float, default=6.5)
    a = ap.parse_args(argv)
    log("night E2 starts: waiting for pairs_v3")
    while not (PAIRS / "manifest.json").exists():
        time.sleep(60)
    run(["make_dataset.py", "--verify", "2", "--workers", "4", "--out", str(PAIRS)], "verify_v3")
    run(["diagnose.py", "units", "--pairs", str(PAIRS)], "units_v3")
    run(["diagnose.py", "valset", "--pairs", str(PAIRS), "--models", f"C_long={CLONG}"], "valset_v3")
    run(["train_e.py", "--tag", "E2", "--hours", str(a.hours), "--device", DEV, "--pairs", str(PAIRS), "--val",
         str(VAL), "--init", str(START), "--lr", "3.7e-4"], "train_E2")
    best = ROOT / "train" / "E2" / "best.pt"
    if not best.exists():
        log("no E2 checkpoint: stop")
        return 1
    run(["evaluate.py", f"E2={best}", f"E={START}", "current", "--device", DEV, "--held", "6", "--pairs",
         str(PAIRS), "--out", str(ROOT / "eval" / "E2")], "eval_E2")
    run(["diagnose.py", "gap", "--models", f"E2={best}", f"E={START}", "current"], "gap_E2")
    run(["apply_real.py", "--model", str(best), "--device", DEV, "--out", str(ROOT / "final_E2")], "pictures_E2")
    log("night E2 done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
