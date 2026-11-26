# ======================================================================
# day.py
# Rôle     : Orchestration d'une journée d'entraînement sans surveillance (carte graphique seule) : fabrique les paires puis entraîne.
# Étape    : 1. Entraînement sur données synthétiques seules (E2 / E3, 7-9 nov) (voir scripts/LISEZMOI.md)
# Entrées  : générateur : graines + catalogue.json (fiches d'instruments publiques) ; aucune image
# Sorties  : paires synthétiques (input/target/info.json + manifest.json), réseau E3
# Exemple  : python day.py [--hours 8.5]
# ======================================================================
"""Step 2 of the gap study, unattended, on the RTX 4060 Ti only:

1. the end of the calibrated pairs' generation (pairs_v2/manifest.json), their verification (2 pairs per set), and
   their stars measured as the real images' (diagnose.py units);
2. the fixed validation set cut from them (diagnose.py valset);
3. the training D: the bench's winner (LaMa) from C_long's weights, on the calibrated pairs, stars added on the fly
   to half the tiles (synth.py), the denoised version 65 % of the tiles, 8.5 hours (the current network's recipe
   time), a cosine rate; the best checkpoint by the fixed validation set;
4. D against C_long and the current network: evaluate.py (held-out calibrated pairs, our test images with stars
   added), the four ways of diagnose.py gap, and the final pictures of our real images (apply_real.py).

    python day.py [--hours 8.5]

Log: <ESSAIS>/starRemoval/day/log.txt.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
ROOT = (Path(__import__("os").environ.get("ESSAIS", "essais")) / "starRemoval")
PAIRS = ROOT / "pairs_v2"
LOGD = ROOT / "day"
DEV = "cuda:0"
CLONG = ROOT / "train" / "C_long" / "best.pt"
VAL = ROOT / "diagnose" / "valset_pairs_v2.npz"


def log(msg: str):
    LOGD.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now():%H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOGD / "log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(args: list[str], name: str) -> bool:
    log(f"start {name}: {' '.join(args)}")
    t = time.time()
    with open(LOGD / f"{name}.log", "w", encoding="utf-8") as f:
        r = subprocess.run([PY, *args], cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    log(f"end {name}: exit {r.returncode} after {(time.time() - t) / 60:.0f} min")
    return r.returncode == 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hours", type=float, default=8.5)
    a = ap.parse_args(argv)
    log("day starts")
    while not (PAIRS / "manifest.json").exists():
        time.sleep(60)
    log("generation done")
    run(["make_dataset.py", "--verify", "2", "--workers", "4", "--out", str(PAIRS)], "verify")
    run(["diagnose.py", "units", "--pairs", str(PAIRS)], "units")
    run(["diagnose.py", "valset", "--pairs", str(PAIRS), "--models", f"C_long={CLONG}"], "valset")
    run(["train.py", "--model", "lama", "--tag", "D", "--hours", str(a.hours), "--device", DEV, "--patch", "384",
         "--batch", "6", "--workers", "5", "--pairs", str(PAIRS), "--init", str(CLONG), "--lr", "1.5e-4",
         "--synthetic", "0.5", "--denoised-share", "0.65", "--val-file", str(VAL), "--val-every", "15"], "train_D")
    best = ROOT / "train" / "D" / "best.pt"
    if not best.exists():
        log("no D checkpoint: stop")
        return 1
    run(["evaluate.py", f"D={best}", f"C_long={CLONG}", "current", "--device", DEV, "--held", "6", "--pairs",
         str(PAIRS), "--out", str(ROOT / "eval" / "D")], "eval_D")
    run(["diagnose.py", "gap", "--models", f"D={best}", f"C_long={CLONG}", "current"], "gap_D")
    run(["apply_real.py", "--model", str(best), "--device", DEV, "--out", str(ROOT / "final_D")], "pictures_D")
    log("day done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
