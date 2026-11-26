# ======================================================================
# enchainer_n4.py
# Rôle     : Chaine du 25 nov (N4) : N4 (3 h) + mesure, puis AUTOMATIQUEMENT N4bis (3 h, depart = meilleur pas de N4, lr plus bas) + mesure.
# Étape    : 10. Essai écarté N4 (juge par magnitude Gaia, 25 nov) (voir scripts/LISEZMOI.md)
# Entrées  : paires v33f
# Sorties  : instantanés, tableau
# Exemple  : python enchainer_n4.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Chaine du 25 nov (N4) : N4 (3 h) + mesure, puis AUTOMATIQUEMENT N4bis (3 h, depart = meilleur pas de N4, lr plus bas) + mesure.
Une seule commande, reprenable (reperes etapes\\*.fini). Un seul processus GPU a la fois.

N4 : paires v33f (juge v33f : rayon du halo par la magnitude Gaia) + grands halos synthetiques sur 50 % des tuiles synthetiques
     (--part-grands 0,30 : 30 % des lots en 2 tuiles de 512 px = 50 % des 60 % de pixels synthetiques), depart N3a meilleur pas (12000), lr 1e-4.
N4bis : meme recette, depart du meilleur pas de N4 (plus petite erreur « sous les etoiles »), lr 5e-5 (moitie), 3 h de plus.
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
