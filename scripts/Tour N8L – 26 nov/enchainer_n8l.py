# ======================================================================
# enchainer_n8l.py
# Role     : N8L chain (long round, 24,000 steps): training train_n8l.py --reprendre up to step 24000, then full measurement (mesurer_n8l.py) at the best validation step and at step 24000.
# Stage    : 10. Discarded trial N8L (cores 10 %, 24,000 steps, 26 Nov) (see scripts/LISEZMOI.md)
# Inputs   : v33g pairs
# Outputs  : snapshots, table
# Example  : lancer_borne.ps1 -Fils 4 -Log ...\chaine_b.out -Exe python -Args "-I enchainer_n8l.py"
# ======================================================================
r"""N8L chain (long round, 24,000 steps): training train_n8l.py --reprendre up to step 24000, then full measurement (mesurer_n8l.py) at the best validation step and at step 24000.
If the round stops on "cores too low" (arret_coeurs.txt), the chain measures nothing.
After a "memory stop" (code 4) or a refusal of the measurement (code 2, less than 8 GB free), the chain waits for 8 GB free (checked every 60 s) and restarts.
    lancer_borne.ps1 -Fils 4 -Log ...\chaine_b.out -Exe python -Args "-I enchainer_n8l.py"
"""
import subprocess, sys, time
from datetime import datetime
from pathlib import Path
import psutil

ICI = Path(__file__).resolve().parent
SORTIE = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles N8L – 26 nov")
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
    log("chaîne N8L lancée (24 000 pas, départ E3)")
    ok = etape("N8L_train", [str(ICI / "train_n8l.py"), "--tag", "N8L", "--n2", "--suffixe", "v33g", "--coeurs", "0.10", "--pas", "24000", "--reprendre"])
    if ok and (SORTIE / "N8L" / "arret_coeurs.txt").exists():
        log("arrêt : cœurs trop bas ; pas de mesure complète")
    elif ok:
        etape("N8L_mesure", [str(ICI / "mesurer_n8l.py")])
    log("chaîne N8L terminée")
    return 0


if __name__ == "__main__":
    sys.exit(main())
