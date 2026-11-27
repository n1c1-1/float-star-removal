# ======================================================================
# lancer_orchestrateur_apod.py
# Role     : Chain module (see its docstring below and scripts/LISEZMOI.md).
# Stage    : 3. Production of the APOD web pairs (24 Nov) (see scripts/LISEZMOI.md)
# Inputs   : de-linearised APOD images
# Outputs  : _debruitee .npz pairs
# Example  : python lancer_orchestrateur_apod.py [arguments]  (see the docstring; paths via variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
import subprocess
d = r"<SCRIPTS>\Production APOD – 24 nov"
py = r"python"
r = subprocess.Popen(["powershell", "-NoProfile", "-File", r"<SCRIPTS>\lancer_borne.ps1", "-Fils", "1", "-Log",
                      d + "\journaux\orchestrateur_sortie.log", "-Exe", py, "-Args", f'"{d}\produire_apod.py"', "-Dossier", d],
                     stdout=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
print(r.stdout.readline())
