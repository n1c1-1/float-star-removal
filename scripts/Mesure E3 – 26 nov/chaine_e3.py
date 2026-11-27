# ======================================================================
# chaine_e3.py
# Role     : Runs the E3 (step 20000) inference then its validation on CPU, one after the other.
# Stage    : 13. Measurement and controls: the E3 rows of the README tables (26 Nov) (see scripts/README.md)
# Inputs   : F:\astro_essais\starRemoval\train\E3\snap\step_20000.pt
# Outputs  : starless_E3_pas20000 + validation results E3_V2
# Example  : python chaine_e3.py
# ======================================================================
import subprocess, sys, time
PY = r"C:\Users\Nicolas\sandbox\astro\.venv\Scripts\python.exe"
H = r"C:\Users\Nicolas\sandbox\starRemoval\Article – 26 nov"
CK = r"F:\astro_essais\starRemoval\train\E3\snap\step_20000.pt"
for args in ([PY, "-I", H + r"\inferer_v2_cpu.py", CK, "E3_pas20000", "--sans-starless-NON"],):
    pass
r = subprocess.run([PY, "-I", H + r"\inferer_v2_cpu.py", CK, "E3_pas20000"], cwd=H)
print("inference rc", r.returncode, flush=True)
if r.returncode == 0:
    r = subprocess.run([PY, "-I", H + r"\valider_cpu.py", r"F:\float_ia\Article – 26 nov\starless_E3_pas20000", "--tag", "E3_V2", "--device", "cpu"], cwd=H)
    print("valider rc", r.returncode, flush=True)
