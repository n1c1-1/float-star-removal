# ======================================================================
# enchainer_n3.py
# Rôle     : Chaine du 25 nov : N3a (3 h) puis sa mesure, N3b (3 h) puis sa mesure.
# Étape    : 10. Essai écarté N3 (grands halos synthétiques, 25 nov) (voir scripts/LISEZMOI.md)
# Entrées  : paires v33d/v33e
# Sorties  : instantanés, tableau
# Exemple  : lancer_borne.ps1 -Fils 4 -Log ...\chaine.out -Exe python -Args "-I enchainer_n3.py"
# ======================================================================
"""Chaine du 25 nov : N3a (3 h) puis sa mesure, N3b (3 h) puis sa mesure. Une seule commande, reprenable.
(Le controle de duree N3L, 8 h depuis E3, a ete retire de la chaine le 25 nov sur decision de Nicolas.)

N3a : levier A seul (grands halos synthetiques ; paires v33d), depart snapshot N2 pas 26000.
N3b : leviers A + B (grands halos ; paires v33e du juge v33e), depart N2 pas 26000.
Les deux gardent la recette N2 (poids 0,5 -> 1, etoiles ttres brillantes sur nebuleuse) et ajoutent 15 % de lots de 2 tuiles 512 a grands halos
(= 25 % des tuiles synthetiques par surface : 60 % x 25 % = 15 % des pixels). lr 1e-4 (le cosinus de N2 etait a 7,5e-5 au pas 26000), montee 500 pas,
cosinus sur le temps (3 h) : la fin de run arrive a taux nul quelle que soit la vitesse.

Chaque etape finie laisse un repere <DONNEES>\\Entrainement etoiles N3 - 25 nov\\etapes\\<etape>.fini ; relancer saute ce qui est fini.
Une etape qui plante est refaite jusqu'a 3 fois (entrainement : repart de last.pt avec --reprendre). Un seul processus GPU a la fois.

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
