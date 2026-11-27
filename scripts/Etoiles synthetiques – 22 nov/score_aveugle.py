# ======================================================================
# score_aveugle.py
# Role     : Blind test score: the judge's answers (JSON {planche: {numero: {"verdict": "vraie"|"synthetique", "raison": ...}}}) against the key (cle_aveugle.json).
# Step     : 4. Realistic synthetic stars (22 Nov) (see scripts/LISEZMOI.md)
# Inputs   : star-free backgrounds (pair targets), star shape banks
# Outputs  : pairs (input, target), sheets, blind test
# Example  : Usage: python score_aveugle.py <cle_aveugle.json> <reponses.json>  -> score_aveugle.json (here)
# ======================================================================
"""Blind test score: the judge's answers (JSON {planche: {numero: {"verdict": "vraie"|"synthetique", "raison": ...}}})
against the key (cle_aveugle.json). Error rate per image and per class (target ~50 %), fakes taken for real,
real taken for fake; reasons given for the fakes spotted.
Usage: python score_aveugle.py <cle_aveugle.json> <reponses.json>  -> score_aveugle.json (here)"""
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    cle = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    raw = Path(sys.argv[2]).read_bytes()
    try:
        rep = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:                          # un juge qui écrit en cp1252
        rep = json.loads(raw.decode("cp1252"))
    tab = defaultdict(lambda: dict(n=0, erreurs=0, fausses=0, fausses_prises_vraies=0, vraies=0, vraies_prises_fausses=0))
    raisons = defaultdict(list)
    manquants = 0
    for img, v in cle.items():
        for it in v["cle"]:
            r = rep.get(it["planche"], {}).get(str(it["numero"]))
            if r is None:
                manquants += 1
                continue
            dit_vraie = r["verdict"].lower().startswith("vra")
            for k in (f"{img}/{it['classe']}", img, it["classe"], "total"):
                t = tab[k]
                t["n"] += 1
                t["erreurs"] += int(dit_vraie != it["vraie"])
                if it["vraie"]:
                    t["vraies"] += 1; t["vraies_prises_fausses"] += int(not dit_vraie)
                else:
                    t["fausses"] += 1; t["fausses_prises_vraies"] += int(dit_vraie)
            if not it["vraie"] and not dit_vraie:
                raisons[f"{img}/{it['classe']}"].append(r.get("raison", ""))
    out = {k: dict(v, taux_erreur=round(v["erreurs"] / max(v["n"], 1), 3)) for k, v in sorted(tab.items())}
    res = dict(manquants=manquants, scores=out, raisons_fausses_reperees=raisons)
    (HERE / "score_aveugle.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in out.items():
        print(f"{k:32} n {v['n']:3}  erreur {100 * v['taux_erreur']:5.1f} %  fausses prises vraies "
              f"{v['fausses_prises_vraies']}/{v['fausses']}  vraies prises fausses {v['vraies_prises_fausses']}/{v['vraies']}")
    print("manquants", manquants)


if __name__ == "__main__":
    main()
