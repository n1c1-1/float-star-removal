# ======================================================================
# methode_validee.py
# Role     : VALIDATED METHOD v1: classical star removal, on a crop of the denoised Rosette image.
# Stage    : 2. Classical Gaia-guided subtraction chain (v33b) (see scripts/LISEZMOI.md)
# Inputs   : stacked image (noise units), astrometry, offline deep Gaia extract
# Outputs  : starless image, class maps
# Example  : python methode_validee.py <decoupe> [--sans-vues] [--reutiliser-piste1]
# ======================================================================
"""VALIDATED METHOD v1: classical star removal, on a crop of the denoised Rosette image.
    python methode_validee.py <decoupe> [--sans-vues] [--reutiliser-piste1]
<decoupe> : decoupe6 (reference case, crop 1530,790,2298,1558), filaments, lisse, or x0,y0,x1,y1 with --nom <nom>.

Chain (each step reads the frozen code of its own folder, never the current code of the working folders):
  1. etape_piste1.py  Piste1/retrait_modele.py (c9 state) : deep Gaia + PSF subtraction, no filling;
  2. etape_A.py       Piste3/ (c9 state)  : "hybride_c9" = track 3, round 4, patch-based filler;
  3. etape_B.py       Passes/               : three final passes (parameterised);
  4. vues.py                                : 4 quarters input | result x4 (Post), strong stretch, data.
Outputs: sorties/<decoupe>/ (sans_etoiles_tuile.npy = final result of the tile, views, mesures.json) ;
           sorties/<decoupe>/A/ and B/{t1,t2,t3}/ = intermediate steps. CPU only."""
import sys, os, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
CROPS = {
    "decoupe6": (1530, 790, 2298, 1558),
    "filaments": (1466, 1591, 2234, 2359),     # centre (1850, 1975)
    "lisse": (2016, 1666, 2784, 2434),         # centre (2400, 2050)
    "coeur": (1236, 1316, 2004, 2084),
    "coeur_trompe": (1396, 1516, 2164, 2284),
    "coeur_est": (2066, 1266, 2834, 2034),
}
# SHO 21 nov (plomberie) : decoupes de M8 SHO (image "m8_sho" de Piste1/images.py), crop 768 = tuile 512 + 128 de chaque cote
CROPS_IMAGE = {"m8_sho": {}, "m57": {}, "horsehead": {}}
import json as _json
# v22 (M57, 22 nov) : un fichier decoupes_<image>.json par image (avant : m8_sho seul, nom ecrit en dur)
for _img in CROPS_IMAGE:
    _f = ROOT / f"decoupes_{_img}.json"       # ecrit par carte_m8_sho.py / carte_m57.py (choix des tuiles)
    if _f.exists():
        CROPS_IMAGE[_img] = {k: tuple(v) for k, v in _json.loads(_f.read_text(encoding="utf-8")).items()}
image = sys.argv[sys.argv.index("--image") + 1] if "--image" in sys.argv else "rosette"
arg = sys.argv[1]
if image != "rosette" and arg in CROPS_IMAGE.get(image, {}):
    nom, crop = arg, CROPS_IMAGE[image][arg]
elif image == "rosette" and arg in CROPS:
    nom, crop = arg, CROPS[arg]
else:
    crop = tuple(int(v) for v in arg.split(","))
    nom = sys.argv[sys.argv.index("--nom") + 1]
tag_p1 = "c9_decoupe6" if nom == "decoupe6" else "mv_" + nom
c = ",".join(map(str, crop))
t0 = time.time()


def step(label, args):
    t = time.time()
    r = subprocess.run([PY, *args], cwd=ROOT)
    print(f"[{nom}] {label} : rc={r.returncode} en {time.time() - t:.0f} s", flush=True)
    if r.returncode != 0:
        sys.exit(f"{label} a echoue")


if "--reutiliser-piste1" in sys.argv and (ROOT / "Piste1" / "sorties" / image / tag_p1 / "resume.json").exists():
    print(f"[{nom}] piste 1 reutilisee : {tag_p1}")
else:
    # v32 : --piste1-gpu = meme piste 1 sur la 4060 Ti (etape_piste1_gpu.py de v30, meme resultat a 0,0005 bruit)
    step("1 piste 1", (["etape_piste1_gpu.py", tag_p1, c, image] if "--piste1-gpu" in sys.argv
                       else ["etape_piste1.py", tag_p1, c, image]))
step("2 etape A", ["etape_A.py", nom, tag_p1, c, image])
step("3 etape B", ["etape_B.py", nom])
if "--sans-vues" not in sys.argv:
    step("4 vues", ["vues.py", nom])
print(f"[{nom}] termine en {time.time() - t0:.0f} s -> {ROOT / 'sorties' / nom}")
