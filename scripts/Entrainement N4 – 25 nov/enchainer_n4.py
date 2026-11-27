# ======================================================================
# enchainer_n4.py
# Role     : Chain of 25 Nov (N4): N4 (3 h) + measurement, then AUTOMATICALLY N4bis (3 h, start = best step of N4, lower lr) + measurement.
# Step     : 10. Discarded trial N4 (judge by Gaia magnitude, 25 Nov) (see scripts/LISEZMOI.md)
# Inputs   : v33f pairs
# Outputs  : snapshots, table
# Example  : python enchainer_n4.py [arguments]  (see the docstring; paths via the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Chain of 25 Nov (N4): N4 (3 h) + measurement, then AUTOMATICALLY N4bis (3 h, start = best step of N4, lower lr) + measurement.
A single command, resumable (markers etapes\\*.fini). Only one GPU process at a time.

N4: v33f pairs (v33f judge: halo radius from the Gaia magnitude) + large synthetic halos on 50 % of the synthetic tiles
     (--part-grands 0.30: 30 % of the batches in 2 tiles of 512 px = 50 % of the 60 % synthetic pixels), start from N3a best step (12000), lr 1e-4.
N4bis: same recipe, start from the best step of N4 (smallest “under the stars” error), lr 5e-5 (half), 3 h more.
"""
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ICI = Path(__file__).resolve().parent
SORTIE = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles N4 – 25 nov")
PY = r"python"
ETAPES = SORTIE / "etapes"
N3A = r"<DONNEES>\Entrainement étoiles N3 – 25 nov\N3a\snap\pas_12000.pt"
COMMUN = ["--n2", "--suffixe", "v33f", "--part-grands", "0.30", "--pas", "24000", "--hours", "3.0", "--warm", "500"]


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


def meilleur_pas(tag):
    h = json.loads((SORTIE / tag / "history.json").read_text(encoding="utf-8"))
    return min(h, key=lambda x: x["val_etoiles"])["pas"]


def main():
    ETAPES.mkdir(parents=True, exist_ok=True)
    log("début de la chaîne N4 (N4, N4bis)")
    tr, me = str(ICI / "train_n4.py"), str(ICI / "mesurer_n4.py")
    etape("N4_train", [tr, "--tag", "N4", "--depart", N3A, "--lr", "1e-4"] + COMMUN, ["--reprendre"])
    etape("N4_mesure", [me, "N4"])
    try:
        dep = str(SORTIE / "N4" / "snap" / f"pas_{meilleur_pas('N4')}.pt")
    except Exception as e:
        log(f"N4bis : pas de meilleur pas de N4 ({e}) ; départ N3a")
        dep = N3A
    log(f"N4bis : départ {dep}")
    etape("N4bis_train", [tr, "--tag", "N4bis", "--depart", dep, "--lr", "5e-5"] + COMMUN, ["--reprendre"])
    etape("N4bis_mesure", [me, "N4bis"])
    log("chaîne terminée")
    return 0


if __name__ == "__main__":
    sys.exit(main())
