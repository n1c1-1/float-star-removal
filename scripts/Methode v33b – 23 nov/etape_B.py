# ======================================================================
# etape_B.py
# Role     : Step B of the validated method: the three final passes (star remnants, dark points / 3-8 px remnants, judge's defects) on the output of step A.
# Step     : 2. Classic Gaia-guided subtraction chain (v33b) (see scripts/LISEZMOI.md)
# Inputs   : stacked image (noise units), astrometry, offline deep Gaia extract
# Outputs  : star-free image, class maps
# Example  : Usage: python etape_B.py <nom>      (reads sorties/<nom>/A/cas.json ; writes sorties/<nom>/B/{t1,t2,t3}/)
# ======================================================================
"""Step B of the validated method: the three final passes (star remnants, dark points / 3-8 px remnants, judge's defects)
on the output of step A. Scripts of Passes/: passe_finale_p.py, passe_finale_t2_p.py, passe_finale_t3_p.py (parametrised copies,
made by Passes/fabriquer_copies_p.py from the exact copies of the originals).
Usage: python etape_B.py <nom>      (reads sorties/<nom>/A/cas.json ; writes sorties/<nom>/B/{t1,t2,t3}/)
Thresholds of pass 1 ONLY: PF_HI=7, PF_LO=4, PF_ZK=3 (test « 7 4 3 » of the validated result); t2 and t3 keep the
module's default values (5 / 3 / 4), as in the session: do not generalise these variables to the three passes."""
import os, sys, json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
nom = sys.argv[1]
A = ROOT / "sorties" / nom / "A"
cas = json.loads((A / "cas.json").read_text(encoding="utf-8"))
case, crop, tag = cas["case"], cas["crop"], cas["tag_piste1"]
a_out = A / "sorties" / "hybride_c9" / case
mes = json.loads((a_out / "mesures.json").read_text(encoding="utf-8"))
decoupe6 = nom == "decoupe6"
cfg = dict(
    case=case, crop=crop, tile=[128, 128, 640, 640],
    a_out=str(a_out),
    base_csv=str(next((A / "sorties" / "base_piste1" / case).glob(f"{tag}_m*")) / "etoiles.csv"),   # g1 : marge 200 ou 520 (mode geantes)
    b_root=str(ROOT / "sorties" / nom / "B"),
    p1_dir=str(ROOT / "Piste1"),
    six=[[1706, 1009], [2043, 968], [2038, 1036], [1762, 1159], [2078, 1220], [1999, 1327]] if decoupe6 else [],
    angles=mes["angles"],
    fwhm=json.loads((ROOT / "Piste1" / "sorties" / cas.get("image", "rosette") / cas["tag_piste1"] / "resume.json").read_text(encoding="utf-8"))["fwhm"],
    named_fixes=decoupe6,
)
if cas.get("image", "rosette") != "rosette":     # SHO 21 nov : image et catalogue profond de cette image (absent -> aucun)
    cfg["image"] = cas["image"]
    cfg["deep"] = str(ROOT / "Catalogue Gaia profond – 20 nov" / f"gaia_dr3_profond_{cas['image']}.npz")
(ROOT / "sorties" / nom / "B").mkdir(parents=True, exist_ok=True)
cfg_path = ROOT / "sorties" / nom / "B" / "cfg.json"
cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
env = dict(os.environ, MV_CFG=str(cfg_path), PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1",
           ASTRO_GPU_GAIA_EXTRACTS="<CACHE_FLOAT>/cache/gaia_extracts", PYTHONIOENCODING="utf-8")
for script in ("passe_finale_p.py", "passe_finale_t2_p.py", "passe_finale_t3_p.py"):
    e = dict(env, PF_HI="7", PF_LO="4", PF_ZK="3") if script == "passe_finale_p.py" else env
    r = subprocess.run([PY, script], cwd=ROOT / "Passes", env=e, capture_output=True)
    (ROOT / "sorties" / nom / "B" / (script[:-3] + ".log")).write_bytes(r.stdout + r.stderr)
    if r.returncode != 0:
        sys.exit(f"{script} a echoue (voir le journal)")
    print(script, "ok", flush=True)
