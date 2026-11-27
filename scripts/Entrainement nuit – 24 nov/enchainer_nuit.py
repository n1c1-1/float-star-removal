# ======================================================================
# enchainer_nuit.py
# Role     : Chain of the night of 24 Nov: N1 (5 h), N1 measurement, N2 (5 h), N2 measurement.
# Step     : 6. Night training N1/N2 (24 Nov) (see scripts/LISEZMOI.md)
# Inputs   : v33d pairs + de-linearised web pairs, start E3 step 20000
# Outputs  : N1/N2 snapshots, table
# Example  : lancer_borne.ps1 -Fils 4 -Log ...\chaine.out -Exe python -Args "-I enchainer_nuit.py [--sans-apod]"
# ======================================================================
"""Chain of the night of 24 Nov: N1 (5 h), N1 measurement, N2 (5 h), N2 measurement. A single command, resumable.

Each finished stage leaves a marker <DONNEES>\\Entrainement étoiles nuit – 24 nov\\etapes\\<stage>.fini; rerunning the same command skips
what is finished. A stage that crashes is redone up to 3 times after a 2 min pause (training restarts from last.pt with
--reprendre: network, optimiser, step). A stage that fails 3 times is noted in chaine.log and the chain moves on to the next one
(N2 does not depend on the N1 measurement). The computations are child processes of the lancer_borne launch (4 threads, low priority,
4060 Ti only).

    lancer_borne.ps1 -Fils 4 -Log ...\\chaine.out -Exe python -Args "-I enchainer_nuit.py [--sans-apod]"
"""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ICI = Path(__file__).resolve().parent
SORTIE = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles nuit – 24 nov")
PY = r"python"
APOD = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Paires étoiles APOD – 24 nov")
ETAPES = SORTIE / "etapes"
extra = [a for a in sys.argv[1:] if a == "--sans-apod"]


def log(m):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {m}"
    print(line, flush=True)
    with open(SORTIE / "chaine.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def etape(nom, cmd, reprise=None):
    rep = ETAPES / f"{nom}.fini"
    if rep.exists():
        log(f"{nom} : déjà fait, sauté")
        return True
    for essai in range(1, 4):
        c = cmd + (reprise if (reprise and essai > 1) or (reprise and (SORTIE / nom.split('_')[0] / 'last.pt').exists()) else [])
        log(f"{nom} : essai {essai}/3 : {' '.join(c[2:])}")
        t0 = time.time()
        r = subprocess.run([PY, "-I"] + c)
        if r.returncode == 0:
            rep.write_text(f"{datetime.now():%Y-%m-%d %H:%M:%S}\n", encoding="utf-8")
            log(f"{nom} : fini en {(time.time() - t0) / 60:.0f} min")
            return True
        log(f"{nom} : code {r.returncode} après {(time.time() - t0) / 60:.0f} min ; pause 120 s")
        time.sleep(120)
    log(f"{nom} : ÉCHEC après 3 essais, on passe à la suite")
    return False


def main():
    ETAPES.mkdir(parents=True, exist_ok=True)
    if not extra and not list(APOD.glob("*_v33d.npz")):
        log("ARRÊT : aucune paire web *_v33d.npz dans " + str(APOD) + " (relancer après le rejugement, ou --sans-apod)")
        return 2
    log("début de la chaîne de nuit")
    tr, me = str(ICI / "train_nuit.py"), str(ICI / "mesurer_nuit.py")
    etape("N1_train", [tr, "--tag", "N1"] + extra, ["--reprendre"])
    etape("N1_mesure", [me, "N1"])
    etape("N2_train", [tr, "--tag", "N2", "--n2"] + extra, ["--reprendre"])
    etape("N2_mesure", [me, "N2"])
    log("chaîne terminée")
    return 0


if __name__ == "__main__":
    sys.exit(main())
