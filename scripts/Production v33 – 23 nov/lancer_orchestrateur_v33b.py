# ======================================================================
# lancer_orchestrateur_v33b.py
# Role     : Chain module (see its docstring below and scripts/LISEZMOI.md).
# Stage    : 2. Production of the v33 pairs (23 Nov) (see scripts/LISEZMOI.md)
# Inputs   : chosen tiles, v33/v33b chain
# Outputs  : _debruitee/_brute .npz pairs
# Example  : python lancer_orchestrateur_v33b.py [arguments]  (see the docstring; paths via variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
import subprocess
d = r"<SCRIPTS>\Production v33 – 23 nov"
py = r"python"
r = subprocess.Popen(["powershell", "-NoProfile", "-File", r"<SCRIPTS>\lancer_borne.ps1", "-Fils", "1", "-Log",
                      d + "\journaux_v33b\orchestrateur_sortie.log", "-Exe", py, "-Args", f'"{d}\produire_v33b.py"', "-Dossier", d],
                     stdout=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
print(r.stdout.readline())
