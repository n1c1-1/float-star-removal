# ======================================================================
# day_e3.py
# Rôle     : E3 : paires pairs_v4 (étoiles géantes gardées dans la cible), puis reprise de E2 depuis ses meilleurs poids.
# Étape    : 1. Entraînement sur données synthétiques seules (E2 / E3, 7-9 nov) (voir scripts/LISEZMOI.md)
# Entrées  : générateur : graines + catalogue.json (fiches d'instruments publiques) ; aucune image
# Sorties  : paires synthétiques (input/target/info.json + manifest.json), réseau E3
# Exemple  : python day_e3.py [--hours 6.5] [--init <checkpoint>] [--tag E3]
# ======================================================================
"""The giants left whole: the pairs regenerated (pairs_v4: stars.py keeps the stars of Alnitak's class in the target; the
saturation, the bright end and the widest stars set on our images' measures), then E2 taken on from its best weights on
them (E3), on the RTX 4060 Ti.

    python day_e3.py [--hours 6.5] [--init <checkpoint>] [--tag E3]
"""
from __future__ import annotations

import argparse
import sys

from day import DEV, ROOT, CLONG, log, run

PAIRS = ROOT / "pairs_v4"
VAL = ROOT / "diagnose" / "valset_pairs_v4.npz"
START = ROOT / "train" / "E2" / "best.pt"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hours", type=float, default=6.5)
    ap.add_argument("--init", default=str(START))
    ap.add_argument("--lr", default="3.7e-4")
    ap.add_argument("--tag", default="E3")
    ap.add_argument("--workers", default="6")
    a = ap.parse_args(argv)
    if not (PAIRS / "manifest.json").exists():
        log("E3 starts: the pairs")
        # the pairs already written stay; fewer workers if the memory ran out
        for w in (a.workers, "4", "3", "3", "2"):
            if run(["make_dataset.py", "--n", "500", "--seed", "1", "--sets", "clean,defects", "--workers", w,
                    "--device", DEV, "--out", str(PAIRS), "--resume"], f"pairs_v4_{w}"):
                break
        else:
            log("the pairs failed: stop")
            return 1
        run(["make_dataset.py", "--verify", "2", "--workers", "4", "--out", str(PAIRS)], "verify_v4")
        run(["diagnose.py", "valset", "--pairs", str(PAIRS), "--models", f"C_long={CLONG}"], "valset_v4")
    train = ["train_e.py", "--tag", a.tag, "--hours", str(a.hours), "--device", DEV, "--pairs", str(PAIRS), "--val",
             str(VAL), "--lr", a.lr]
    if a.init:
        train += ["--init", a.init]
    run(train, f"train_{a.tag}")
    log(f"{a.tag} training over")
    return 0


if __name__ == "__main__":
    sys.exit(main())
