# ======================================================================
# enchainer_n3.py
# Role     : Chain of 25 Nov: N3a (3 h) then its measurement, N3b (3 h) then its measurement.
# Step     : 10. Discarded trial N3 (large synthetic halos, 25 Nov) (see scripts/LISEZMOI.md)
# Inputs   : v33d/v33e pairs
# Outputs  : snapshots, table
# Example  : lancer_borne.ps1 -Fils 4 -Log ...\chaine.out -Exe python -Args "-I enchainer_n3.py"
# ======================================================================
"""Chain of 25 Nov: N3a (3 h) then its measurement, N3b (3 h) then its measurement. A single command, resumable.
(The duration control N3L, 8 h from E3, was removed from the chain on 25 Nov by Nicolas's decision.)

N3a: lever A alone (large synthetic halos; v33d pairs), start from N2 snapshot step 26000.
N3b: levers A + B (large halos; v33e pairs from the v33e judge), start from N2 step 26000.
Both keep the N2 recipe (weights 0.5 -> 1, very bright stars on nebula) and add 15 % of batches of 2 large-halo 512 tiles
(= 25 % of the synthetic tiles by area: 60 % x 25 % = 15 % of the pixels). lr 1e-4 (the N2 cosine was at 7.5e-5 at step 26000), 500-step warm-up,
cosine over time (3 h): the end of the run reaches zero rate whatever the speed.

Each finished stage leaves a marker <DONNEES>\\Entrainement etoiles N3 - 25 nov\\etapes\\<stage>.fini; rerunning skips what is finished.
A stage that crashes is redone up to 3 times (training: restarts from last.pt with --reprendre). Only one GPU process at a time.

    lancer_borne.ps1 -Fils 4 -Log ...\\chaine.out -Exe python -Args "-I enchainer_n3.py"
"""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ICI = Path(__file__).resolve().parent
SORTIE = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles N3 – 25 nov")
PY = r"python"
ETAPES = SORTIE / "etapes"
N2 = r"<DONNEES>\Entrainement étoiles nuit – 24 nov\N2\snap\pas_26000.pt"
COMMUN = ["--n2", "--depart", N2, "--part-grands", "0.15", "--pas", "24000", "--hours", "3.0", "--lr", "1e-4", "--warm", "500"]


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
        c = cmd + (reprise if reprise and (SORTIE / nom.split('_')[0] / 'last.pt').exists() else [])
        log(f"{nom} : essai {essai}/3 : {' '.join(c[1:])}")
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
    log("début de la chaîne N3 (N3a, N3b)")
    tr, me = str(ICI / "train_n3.py"), str(ICI / "mesurer_n3.py")
    etape("N3a_train", [tr, "--tag", "N3a", "--suffixe", "v33d"] + COMMUN, ["--reprendre"])
    etape("N3a_mesure", [me, "N3a"])
    etape("N3b_train", [tr, "--tag", "N3b", "--suffixe", "v33e"] + COMMUN, ["--reprendre"])
    etape("N3b_mesure", [me, "N3b"])
    log("chaîne terminée")
    return 0


if __name__ == "__main__":
    sys.exit(main())
