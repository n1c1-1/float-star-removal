# ======================================================================
# lancer_orchestrateur_apod.py
# Rôle     : Module de la chaîne (voir sa docstring ci-dessous et scripts/LISEZMOI.md).
# Étape    : 3. Production des paires web APOD (24 nov) (voir scripts/LISEZMOI.md)
# Entrées  : images APOD délinéarisées
# Sorties  : paires _debruitee .npz
# Exemple  : python lancer_orchestrateur_apod.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
import subprocess
d = r"<SCRIPTS>\Production APOD – 24 nov"
py = r"python"
r = subprocess.Popen(["powershell", "-NoProfile", "-File", r"<SCRIPTS>\lancer_borne.ps1", "-Fils", "1", "-Log",
                      d + "\journaux\orchestrateur_sortie.log", "-Exe", py, "-Args", f'"{d}\produire_apod.py"', "-Dossier", d],
                     stdout=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
print(r.stdout.readline())
