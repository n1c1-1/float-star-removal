# ======================================================================
# astrometrie_image.py
# Role     : Astrometry of an image of the validation list: local solver (Piste1/astrometrie.solve, offline Gaia), then astrometrie.npz (cat, scale, wres, wcs) in <DONNEES>/Validation étoiles – 22 nov/<nom>/.
# Stage    : 13. Validation measurement (see scripts/LISEZMOI.md)
# Inputs   : .pt network (or folder of outputs) + list of validation images
# Outputs  : json per image, contact sheets, resume.md
# Example  : usage : astrometrie_image.py <nom> [<nom> ...]
# ======================================================================
"""Astrometry of an image of the validation list: local solver (Piste1/astrometrie.solve, offline Gaia), then
astrometrie.npz (cat, scale, wres, wcs) in <DONNEES>/Validation étoiles – 22 nov/<nom>/. Fallback: adapted astrometrie_cone.py.
usage : astrometrie_image.py <nom> [<nom> ...]"""
import os, sys, json, time
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["ASTRO_GPU_OFFLINE"] = "1"
os.environ["ASTRO_GPU_GAIA_EXTRACTS"] = "<CACHE_FLOAT>/cache/gaia_extracts"
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np
V = (Path(__import__("os").environ.get("RACINE_SCRIPTS") or Path(__file__).resolve().parents[1]) / "Methode v32 production – 22 nov")
sys.path.insert(0, str(V / "Piste1")); sys.path.insert(0, str(Path(__file__).parent))
import astrometrie
from liste import LISTE, DATA

for nom in sys.argv[1:]:
    src, cible, *_ = LISTE[nom]
    d = DATA / nom; d.mkdir(parents=True, exist_ok=True)
    if (d / "astrometrie.npz").exists():
        print(nom, "deja fait"); continue
    t0 = time.time()
    img = np.load(src / "input.npy", mmap_mode="r"); img = np.array(img, np.float32)
    C, H, W = img.shape
    print(f"=== {nom} {img.shape} cible {cible}", flush=True)
    try:
        wcs, scale, wres = astrometrie.solve(img, cible, d / "tmp", print)
    except Exception as e:
        print("ECHEC", nom, str(e)[:300], flush=True); continue
    cat = astrometrie.gaia_stars(wcs, (H, W), scale)
    np.savez(d / "astrometrie.npz", cat=np.array(cat, dtype=object), scale=scale, wres=wres, wcs=wcs.to_header_string(relax=True))
    import shutil; shutil.rmtree(d / "tmp", ignore_errors=True)
    (d / "astrometrie_resume.json").write_text(json.dumps(dict(image=nom, methode="solveur local + affinage TAN-SIP", echelle_arcsec_px=round(scale, 4),
        residu_px=round(wres, 3), secondes=round(time.time() - t0)), ensure_ascii=False, indent=1), encoding="utf-8")
    print("fait", nom, f"{scale:.4f}\"/px residu {wres:.3f} px {time.time()-t0:.0f} s", flush=True)
