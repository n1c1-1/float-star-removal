# ======================================================================
# enchainer_n56.py
# Rôle     : Chaîne N5 -> mesure -> N6 -> mesure -> crops N2 -> tableau, un seul processus, reprenable (copie de enchainer_nuit.py du 24 nov).
# Étape    : 7-8. N5 (cibles v33g) et N6 (poids v33h) (25 nov) (voir scripts/LISEZMOI.md)
# Entrées  : paires v33d -> v33g -> v33h, départ E3 pas 20000
# Sorties  : instantanés N5/N6, tableau_n56.md
# Exemple  : lancer_borne.ps1 -Fils 4 -Log ...\chaine.out -Exe python -Args "-I enchainer_n56.py"
# ======================================================================
r"""Chaîne N5 -> mesure -> N6 -> mesure -> crops N2 -> tableau, un seul processus, reprenable (copie de enchainer_nuit.py du 24 nov).

N5 = recette exacte de N2 (train_nuit.py --n2 : même départ E3 pas 20000, 34500 pas, mêmes proportions) avec les paires v33g (cibles
piédestal rendu) ; N6 = N5 avec les paires v33h (poids des étoiles G 14-17 réparés). Repère <DONNEES>\Entrainement étoiles N5 N6 – 25 nov\etapes\<étape>.fini ;
relancer la même commande saute ce qui est fini ; un entraînement interrompu repart de last.pt (--reprendre), 3 essais par étape.

    lancer_borne.ps1 -Fils 4 -Log ...\chaine.out -Exe python -Args "-I enchainer_n56.py"
"""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ICI = Path(__file__).resolve().parent
SORTIE = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles N5 N6 – 25 nov")
PY = r"python"
ETAPES = SORTIE / "etapes"


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
        c = cmd + (reprise if (reprise and (SORTIE / nom.split("_")[0] / "last.pt").exists()) else [])
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
    tr, me = str(ICI / "train_n56.py"), str(ICI / "mesurer_n56.py")
    log("début de la chaîne N5 N6")
    etape("N5_train", [tr, "--tag", "N5", "--n2", "--suffixe", "v33g"], ["--reprendre"])
    etape("N5_mesure", [me, "N5"])
    etape("N6_train", [tr, "--tag", "N6", "--n2", "--suffixe", "v33h"], ["--reprendre"])
    etape("N6_mesure", [me, "N6"])
    etape("N2_crops", [me, "N2crops"])
    log("chaîne terminée")
    return 0


if __name__ == "__main__":
    sys.exit(main())
