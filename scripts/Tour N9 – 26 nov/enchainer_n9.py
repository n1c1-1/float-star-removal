# ======================================================================
# enchainer_n9.py
# Role     : N9 chain (short round, 10,000 steps): training train_n9.py --reprendre up to step 10000, then full measurement (mesurer_n9.py) at the best validation step and at step 10000.
# Stage    : 10. Discarded trial N9 (compact cores, 26 Nov) (see scripts/LISEZMOI.md)
# Inputs   : v33g pairs
# Outputs  : snapshots, tableau_n9.md
# Example  : lancer_borne.ps1 -Fils 4 -Log ...\chaine_b.out -Exe python -Args "-I enchainer_n9.py"
# ======================================================================
r"""N9 chain (short round, 10,000 steps): training train_n9.py --reprendre up to step 10000, then full measurement (mesurer_n9.py) at the best validation step and at step 10000.
If the round stops on "cores too low" (arret_coeurs.txt), the chain measures nothing.
After a "memory stop" (code 4) or a refusal of the measurement (code 2, less than 8 GB free), the chain waits for 8 GB free (checked every 60 s) and restarts.
    lancer_borne.ps1 -Fils 4 -Log ...\chaine_b.out -Exe python -Args "-I enchainer_n9.py"
"""
import subprocess, sys, time
from datetime import datetime
from pathlib import Path
import psutil

ICI = Path(__file__).resolve().parent
SORTIE = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles N9 – 26 nov")
PY = r"python"
ETAPES = SORTIE / "etapes"


def log(m):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {m}"
    print(line, flush=True)
    with open(SORTIE / "chaine.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def libre():
    return psutil.virtual_memory().available / 2**30


def etape(nom, cmd, essais=6):
    rep = ETAPES / f"{nom}.fini"
    if rep.exists():
        log(f"{nom} : déjà fait, sauté"); return True
    for n in range(1, essais + 1):
        while libre() <= 8:
            time.sleep(60)
        log(f"{nom} : essai {n}/{essais} (libre {libre():.1f} Go) : {' '.join(cmd[1:])}")
        t0 = time.time()
        r = subprocess.run([PY, "-I"] + cmd)
        if r.returncode == 0:
            rep.write_text(f"{datetime.now():%Y-%m-%d %H:%M:%S}\n", encoding="utf-8")
            log(f"{nom} : fini en {(time.time() - t0) / 60:.0f} min"); return True
        log(f"{nom} : code {r.returncode} après {(time.time() - t0) / 60:.0f} min ; pause 120 s")
        time.sleep(120)
    log(f"{nom} : ÉCHEC"); return False


def main():
    ETAPES.mkdir(parents=True, exist_ok=True)
    log("chaîne N9 lancée (10 000 pas, 25 % de cœurs dont 40 % compacts, départ E3)")
    ok = etape("N9_train", [str(ICI / "train_n9.py"), "--tag", "N9", "--n2", "--suffixe", "v33g", "--coeurs", "0.25", "--pas", "10000", "--reprendre"])
    if ok and (SORTIE / "N9" / "arret_coeurs.txt").exists():
        log("arrêt : cœurs trop bas ; pas de mesure complète")
    elif ok:
        etape("N9_mesure", [str(ICI / "mesurer_n9.py")])
    log("chaîne N9 terminée")
    return 0


if __name__ == "__main__":
    sys.exit(main())
