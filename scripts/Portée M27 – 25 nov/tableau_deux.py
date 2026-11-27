# ======================================================================
# tableau_deux.py
# Role     : Chain module (see its docstring below and scripts/LISEZMOI.md).
# Stage    : 12. Two-scale V2 inference (25 Nov) (see scripts/LISEZMOI.md)
# Inputs   : .pt network + validation image + Gaia extract
# Outputs  : starless images, tables
# Example  : python tableau_deux.py [arguments]  (see the docstring; paths via variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
import sys, importlib.util
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
s = importlib.util.spec_from_file_location("m", r"<SCRIPTS>\Entrainement v33c – 24 nov\mesurer_v33c.py"); M = importlib.util.module_from_spec(s); s.loader.exec_module(M)
V = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Validation étoiles – 22 nov" / "Resultats"); Mine = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Portée M27 – 25 nov" / "Resultats")
out = ["| essai | images | " + " | ".join(f"restantes G {b}" for b in M.BANDES) + " | fond méd. | fond p99 | structure méd. | L1 synthétique |", "|---|---|" + "---|" * 9]
for root, t, nom in ((V, "N2_pas_26000", "N2"), (Mine, "N2_deux_echelles", "N2 deux échelles"), (Mine, "N2_v1", "V1"), (Mine, "N2_v2", "V2"), (V, "livre", "modèle FLOAT"), (V, "reference", "RefExt")):
    M.R = root; r = M.ligne(t)
    out.append(f"| {nom} | {r[1]} | " + " | ".join(f"{x:.3f}" for x in r[2]) + f" | {r[3]:.4f} | {r[4]:.2f} | {r[5]:.3f} | {r[6]:.4f} |")
txt = "\n".join(out); print(txt); (Mine / "tableau_v1v2.md").write_text(txt + "\n", encoding="utf-8")
