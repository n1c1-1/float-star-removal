# ======================================================================
# tuile_cpu.py
# Rôle     : Etapes processeur d'UNE tuile (apres la piste 1 faite sur la carte) : etape A, etape B, passes (methode_validee.py --reutiliser-piste1), juge v33 (Juge v33 – 23 nov/construire_paire.py), paire « debruitee », paire « brute » si l'e
# Étape    : 2. Production des paires v33 (23 nov) (voir scripts/LISEZMOI.md)
# Entrées  : tuiles choisies, chaîne v33/v33b
# Sorties  : paires _debruitee/_brute .npz
# Exemple  : usage : tuile_cpu.py <image> <tx> <ty>      |      tuile_cpu.py --brute-seul <image> <tx> <ty>
# ======================================================================
"""Etapes processeur d'UNE tuile (apres la piste 1 faite sur la carte) : etape A, etape B, passes (methode_validee.py --reutiliser-piste1),
juge v33 (Juge v33 – 23 nov/construire_paire.py), paire « debruitee », paire « brute » si l'entree brute existe, nettoyage des sorties de travail.
usage : tuile_cpu.py <image> <tx> <ty>      |      tuile_cpu.py --brute-seul <image> <tx> <ty>"""
import json, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
V33 = (Path(__import__("os").environ.get("RACINE_SCRIPTS") or Path(__file__).resolve().parents[1]) / "Methode v33 production – 23 nov")
JUGE = (Path(__import__("os").environ.get("RACINE_SCRIPTS") or Path(__file__).resolve().parents[1]) / "Juge v33 – 23 nov")
SORTIE = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Paires étoiles v33 – 23 nov")
PY = r"python"
brute_seul = sys.argv[1] == "--brute-seul"
image, tx, ty = (sys.argv[2:5] if brute_seul else sys.argv[1:4]); tx, ty = int(tx), int(ty)
nom = f"{image}_g{tx}_{ty}"
fin = SORTIE / f"{image}_g{tx}_{ty}"


def brute(z):
    f = SORTIE / "entrees_brutes" / f"{image}_brut.npy"
    if not f.exists():
        return None
    raw = np.load(f, mmap_mode="r")[:, ty:ty + 512, tx:tx + 512].astype(np.float32)
    calque = z["input"] - z["target"]                 # les etoiles trouvees sur la version debruitee
    tb = raw - calque
    assert raw.shape == z["input"].shape and np.isfinite(tb).all()
    np.savez_compressed(str(fin) + "_brute.npz", input=raw, target=tb.astype(np.float32), weight=z["weight"], classes=z["classes"])
    return float(np.abs(raw - z["input"]).mean())


t0 = time.time()
if brute_seul:
    z = dict(np.load(str(fin) + "_debruitee.npz"))
    print(json.dumps(dict(nom=nom, brute=brute(z))), flush=True)
    sys.exit(0)
crop = f"{tx - 128},{ty - 128},{tx + 640},{ty + 640}"
r = subprocess.run([PY, "methode_validee.py", crop, "--nom", nom, "--image", image, "--sans-vues", "--reutiliser-piste1"], cwd=V33,
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
(SORTIE / "journaux").mkdir(parents=True, exist_ok=True)
with open(SORTIE / "journaux" / f"{image}.log", "a", encoding="utf-8") as f:
    f.write(f"== {nom} rc={r.returncode}\n{r.stdout[-3000:]}\n{r.stderr[-1500:]}\n")
ok = r.returncode == 0
if ok:
    tmp = SORTIE / "_travail" / nom; tmp.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(JUGE))
    import construire_paire as CP
    CP.OUT = tmp
    sys.argv = ["construire_paire.py", str(V33), image, nom, str(tx - 128), str(ty - 128)]
    CP.main()
    z = dict(np.load(tmp / f"{image}_{nom}.npz"))
    np.savez_compressed(str(fin) + "_debruitee.npz", **z)
    zones = json.loads((tmp / f"{image}_{nom}_zones.json").read_text(encoding="utf-8"))
    (SORTIE / "zones").mkdir(exist_ok=True)
    shutil.copy2(tmp / f"{image}_{nom}_zones.json", SORTIE / "zones" / f"{nom}.json")
    W = z["weight"]
    res = dict(nom=nom, image=image, tx=tx, ty=ty, poids_0=round(float((W == 0).mean()), 4), poids_05=round(float((W == 0.5).mean()), 4),
               poids_1=round(float((W == 1).mean()), 4), brute=brute(z) is not None, secondes_cpu=round(time.time() - t0))
    for zz in zones:
        if "taches_plus_claires" in zz:
            res["taches_plus_claires"], res["etoiles_restees"] = zz["taches_plus_claires"], zz["etoiles_restees"]
    print("RESULTAT " + json.dumps(res, ensure_ascii=False), flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
else:
    print("ECHEC", nom, (r.stderr or r.stdout)[-400:], flush=True)
shutil.rmtree(V33 / "sorties" / nom, ignore_errors=True)
shutil.rmtree(V33 / "Piste1" / "sorties" / image / f"mv_{nom}", ignore_errors=True)
sys.exit(0 if ok else 3)
