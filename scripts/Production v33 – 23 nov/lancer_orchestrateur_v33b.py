# ======================================================================
# lancer_orchestrateur_v33b.py
# Rôle     : Module de la chaîne (voir sa docstring ci-dessous et scripts/LISEZMOI.md).
# Étape    : 2. Production des paires v33 (23 nov) (voir scripts/LISEZMOI.md)
# Entrées  : tuiles choisies, chaîne v33/v33b
# Sorties  : paires _debruitee/_brute .npz
# Exemple  : python lancer_orchestrateur_v33b.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
import subprocess
d = r"<SCRIPTS>\Production v33 – 23 nov"
py = r"python"
r = subprocess.Popen(["powershell", "-NoProfile", "-File", r"<SCRIPTS>\lancer_borne.ps1", "-Fils", "1", "-Log",
                      d + "\journaux_v33b\orchestrateur_sortie.log", "-Exe", py, "-Args", f'"{d}\produire_v33b.py"', "-Dossier", d],
                     stdout=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
print(r.stdout.readline())
