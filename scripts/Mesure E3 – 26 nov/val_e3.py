# ======================================================================
# val_e3.py
# Role     : Runs only the CPU validation of the E3 (step 20000) starless images (second half of chaine_e3.py).
# Stage    : 13. Measurement and controls: the E3 rows of the README tables (26 Nov) (see scripts/README.md)
# Inputs   : F:\float_ia\Article – 26 nov\starless_E3_pas20000
# Outputs  : validation results E3_V2
# Example  : python val_e3.py
# ======================================================================
import subprocess
PY = r"C:\Users\Nicolas\sandbox\astro\.venv\Scripts\python.exe"
H = r"C:\Users\Nicolas\sandbox\starRemoval\Article – 26 nov"
r = subprocess.run([PY, H + r"\valider_cpu.py", r"F:\float_ia\Article – 26 nov\starless_E3_pas20000", "--tag", "E3_V2", "--device", "cpu"], cwd=H)
print("valider rc", r.returncode, flush=True)
