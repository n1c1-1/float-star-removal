# ======================================================================
# enchainer_n7.py
# Role     : N7 chain (copy of enchainer_n56.py), a single process, resumable.
# Stage    : 9. N7: synthetic galaxy cores (25 Nov) (see scripts/LISEZMOI.md)
# Inputs   : v33g pairs + synthetic cores
# Outputs  : N7 snapshots, tableau_n7.md
# Example  : lancer_borne.ps1 -Fils 4 -Log ...\chaine.out -Exe python -Args "-I enchainer_n7.py"
# ======================================================================
r"""N7 chain (copy of enchainer_n56.py), a single process, resumable. It first WAITS for the end of the N5 N6 chain (PID 321448: one check
per minute), then:
  1. chooses the v33h pairs if N6 has G 14-17 <= that of N5 AND median structure <= that of N5 + 0.001 (tableau_n56.md of the N5 N6 folder), otherwise v33g;
  2. cores column of N5, N6 and RefExt (mesurer_n7.py coeurs_ref);
  3. N7 = exact N2/N5 recipe (start E3 step 20000, 34500 steps, same proportions), only change: synthetic cores on 25 % of the synthetic tiles;
  4. measurement of N7 (two-scale V2, cores column, validation) and table N5 / N6 / N7 / FLOAT model / RefExt, contact sheets.
Marker <DONNEES>\Entrainement étoiles N7 – 25 nov\etapes\<étape>.fini; rerunning the same command skips what is finished.

    lancer_borne.ps1 -Fils 4 -Log ...\chaine.out -Exe python -Args "-I enchainer_n7.py"
"""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ICI = Path(__file__).resolve().parent
SORTIE = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles N7 – 25 nov")
N56 = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles N5 N6 – 25 nov")
PY = r"python"
ETAPES = SORTIE / "etapes"
PID_ATTENDU = 321448


def log(m):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {m}"
    print(line, flush=True)
    with open(SORTIE / "chaine.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def existe(pid):
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"], capture_output=True, text=True)
    return f'"{pid}"' in r.stdout


def choisir_suffixe():
    """v33h si N6 : G 14-17 <= N5 ET structure médiane <= N5 + 0,001 ; sinon v33g. Retourne (suffixe, explication)."""
    f = N56 / "tableau_n56.md"
    if not f.exists():
        return "v33g", "tableau_n56.md absent : v33g par défaut"
    lig = {}
    for l in f.read_text(encoding="utf-8").splitlines():
        c = [x.strip() for x in l.strip().strip("|").split("|")]
        if c and c[0] in ("N5", "N6"):
            lig[c[0]] = (float(c[5]), float(c[9]))          # G 14-17, structure médiane
    if "N5" not in lig or "N6" not in lig:
        return "v33g", f"ligne N5 ou N6 absente du tableau ({sorted(lig)}) : v33g par défaut"
    g5, s5 = lig["N5"]; g6, s6 = lig["N6"]
    ok = g6 <= g5 and s6 <= s5 + 0.001
    return ("v33h" if ok else "v33g"), (f"N5 : G 14-17 {g5:.3f}, structure {s5:.3f} ; N6 : G 14-17 {g6:.3f}, structure {s6:.3f} ; "
                                        f"N6 G 14-17 <= N5 : {g6 <= g5}, structure N6 <= N5 + 0,001 : {s6 <= s5 + 0.001} -> {'v33h' if ok else 'v33g'}")


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
    log(f"chaîne N7 lancée ; attente de la fin du processus {PID_ATTENDU} (une vérification par minute)")
    while existe(PID_ATTENDU):
        time.sleep(60)
    log(f"processus {PID_ATTENDU} terminé")
    choix = ETAPES / "choix_paires.txt"
    if choix.exists():
        suf = choix.read_text(encoding="utf-8").split()[0]
        log(f"choix des paires déjà fait : {suf}")
    else:
        suf, expl = choisir_suffixe()
        choix.write_text(f"{suf} {expl}\n", encoding="utf-8")
        log(f"CHOIX DES PAIRES : {suf} ({expl})")
        with open(ICI / "JOURNAL.md", "a", encoding="utf-8") as f:
            f.write(f"\n## {datetime.now():%d %b %H:%M} – choix des paires de N7 (chaîne)\n- {suf} : {expl}\n")
    tr, me = str(ICI / "train_n7.py"), str(ICI / "mesurer_n7.py")
    etape("coeurs_ref", [me, "coeurs_ref"])
    etape("N7_train", [tr, "--tag", "N7", "--n2", "--suffixe", suf, "--coeurs", "0.25"], ["--reprendre"])
    etape("N7_mesure", [me, "N7"])
    log("chaîne terminée")
    return 0


if __name__ == "__main__":
    sys.exit(main())
