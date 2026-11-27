# ======================================================================
# lancer_chaine_tuiles.py
# Role     : CPU orchestrator: onboarding of each image, then the classical chain v33b on its tile, results copied out.
# Stage    : 15. Classical baseline versus networks on 512 px tiles (26-27 Nov) (see scripts/README.md)
# Inputs   : tuiles.json, the chain copy chaine_v33b\ (see README.md), validation images
# Outputs  : classique_<image>.npy, journal.json, logs in F:\float_ia\Article – 26 nov\baseline\
# Example  : lancer_borne.ps1 -Fils 4 ... lancer_chaine_tuiles.py --budget-h 3   (launch line in _lancer.ps1)
# ======================================================================
r"""CPU orchestrator of the baseline: for each image of LISTE (list order), in order, ONE tile at a time:
  1. onboarding: batir_etalonnage_image.py then batir_modele_image.py (image sheet: calibration + star model; <= 30 min per step);
  2. the chain v33b (copy in chaine_v33b\) on the tile crop +/-128: methode_validee.py x0-128,y0-128,x0+640,y0+640 --nom <image>_b<x0>_<y0> --image <image> --sans-vues;
  3. copy of the final result (sorties/<nom>/B/t3/sans_etoiles_tuile.npy, 3x512x512) and of the logs to F:\float_ia\Article – 26 nov\baseline\, cleaning of the working outputs.
Resumable (an image done or failed is skipped, journal.json). Stops when the total exceeds the budget (--budget-h, default 3).
To be launched by lancer_borne.ps1 (4 threads). CUDA_VISIBLE_DEVICES="": no GPU. usage: lancer_chaine_tuiles.py [--budget-h 3] [--seulement name ...]"""
import os, sys, json, time, shutil, subprocess
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"       # (vide = variable supprimee sous Windows : "-1")
os.environ["ASTRO_GPU_OFFLINE"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
import psutil
ICI = Path(__file__).resolve().parent
CH = ICI / "chaine_v33b"
OUT = Path(r"F:\float_ia\Article – 26 nov\baseline")
PY = r"C:\Users\Nicolas\sandbox\astro\.venv\Scripts\python.exe"
RAM_MAX = 8 * 2 ** 30
sys.path.insert(0, r"C:\Users\Nicolas\sandbox\starRemoval\Validation – 22 nov")
from liste import LISTE
budget = float(sys.argv[sys.argv.index("--budget-h") + 1]) * 3600 if "--budget-h" in sys.argv else 3 * 3600
seul = sys.argv[sys.argv.index("--seulement") + 1:] if "--seulement" in sys.argv else None
(OUT / "journaux").mkdir(parents=True, exist_ok=True)
JF = OUT / "journal.json"
J = json.loads(JF.read_text(encoding="utf-8")) if JF.exists() else {}
tuiles = json.loads((ICI / "tuiles.json").read_text(encoding="utf-8"))


def sauve():
    JF.write_text(json.dumps(J, indent=1, ensure_ascii=False), encoding="utf-8")


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


def tuer(p):
    subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)


def lancer(args, timeout, logf):
    """Lance un script (python -I n'est pas utilise : le script est dans le dossier de la copie) ; surveille temps et RAM (arbre du processus)."""
    t0 = time.time(); pic = 0
    with open(logf, "ab") as f:
        p = subprocess.Popen([PY] + args, cwd=CH, stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ))
        why = None
        while p.poll() is None:
            time.sleep(5)
            try:
                kids = [p] + psutil.Process(p.pid).children(recursive=True)
                rss = sum(k.memory_info().rss for k in kids if k.is_running())
            except Exception:
                rss = 0
            pic = max(pic, rss)
            if rss > RAM_MAX:
                why = "RAM > 8 Go"; tuer(p); break
            if time.time() - t0 > timeout:
                why = f"delai {timeout / 60:.0f} min depasse"; tuer(p); break
    return (p.returncode if why is None else -1), why, time.time() - t0, pic / 2 ** 30


t_total = sum(v.get("secondes", 0) for v in J.values())
for nom in LISTE:
    if nom not in tuiles or tuiles[nom].get("x0") is None:
        continue
    if seul and nom not in seul:
        continue
    if nom in J:
        continue
    if t_total > budget:
        log(f"budget {budget / 3600:.1f} h atteint : arret avant {nom}"); break
    x0, y0 = tuiles[nom]["x0"], tuiles[nom]["y0"]
    tag = f"{nom}_b{x0}_{y0}"
    r = {"image": nom, "x0": x0, "y0": y0}
    t0 = time.time()
    lf = OUT / "journaux" / f"{nom}.log"
    log(f"=== {nom} tuile ({x0},{y0})")
    ok = True
    for step, scr in (("etalonnage", "batir_etalonnage_image.py"), ("modele", "batir_modele_image.py")):
        rc, why, dt, ram = lancer([scr, nom], 30 * 60, lf)
        r[f"onboarding_{step}"] = {"rc": rc, "secondes": round(dt), "ram_go": round(ram, 2), "arret": why}
        log(f"  {step}: rc={rc} {dt:.0f} s RAM {ram:.1f} Go {why or ''}")
        if rc != 0:
            ok = False; r["echec"] = f"onboarding {step} : {why or 'code ' + str(rc)} (voir journaux/{nom}.log)"; break
    if ok:
        c = f"{x0 - 128},{y0 - 128},{x0 + 640},{y0 + 640}"
        t1 = time.time()
        rc, why, dt, ram = lancer(["methode_validee.py", c, "--nom", tag, "--image", nom, "--sans-vues"], 90 * 60, lf)
        r["chaine"] = {"rc": rc, "secondes": round(dt), "ram_go": round(ram, 2), "arret": why, "crop": c}
        log(f"  chaine: rc={rc} {dt:.0f} s RAM {ram:.1f} Go {why or ''}")
        fin = CH / "sorties" / tag / "B" / "t3" / "sans_etoiles_tuile.npy"
        if rc == 0 and fin.exists():
            shutil.copy2(fin, OUT / f"classique_{nom}.npy")
            for sub in ("A", "B"):
                for g in (CH / "sorties" / tag / sub).rglob("*.json"):
                    pass
            m = CH / "sorties" / tag / "B" / "t3" / "mesures.json"
            if m.exists():
                shutil.copy2(m, OUT / "journaux" / f"{tag}_mesures_t3.json")
            cl = CH / "sorties" / tag / "B" / "t3" / "carte_remplissage_tuile.npy"
            if cl.exists():
                shutil.copy2(cl, OUT / f"classique_{nom}_carte.npy")
        else:
            r["echec"] = f"chaine : {why or 'code ' + str(rc)} (voir journaux/{nom}.log)"
        jp = CH / "Piste1" / "sorties" / nom / f"mv_{tag}" / "journal.txt"
        if jp.exists():
            shutil.copy2(jp, OUT / "journaux" / f"{tag}_piste1_journal.txt")
        shutil.rmtree(CH / "sorties" / tag, ignore_errors=True)
        shutil.rmtree(CH / "Piste1" / "sorties" / nom / f"mv_{tag}", ignore_errors=True)
    r["secondes"] = round(time.time() - t0)
    r["fin"] = time.strftime("%Y-%m-%d %H:%M:%S")
    J[nom] = r; sauve(); t_total += r["secondes"]
    log(f"  {nom} : {'ECHEC ' + r['echec'] if 'echec' in r else 'fait'} en {r['secondes']} s ; total {t_total / 60:.0f} min")
log("fini")
