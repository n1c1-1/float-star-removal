# ======================================================================
# produire_v33b_reste.py
# Rôle     : Production des paires v33 (nuit du 23 nov) : tuiles de tuiles_finales.csv, 39 images, alternees.
# Étape    : 2. Production des paires v33 (23 nov) (voir scripts/LISEZMOI.md)
# Entrées  : tuiles choisies, chaîne v33/v33b
# Sorties  : paires _debruitee/_brute .npz
# Exemple  : usage : produire_v33.py [--essai]
# ======================================================================
"""Production des paires v33 (nuit du 23 nov) : tuiles de tuiles_finales.csv, 39 images, alternees.
 - 1 travailleur CARTE (piste 1, 2 fils, coeurs 13-14) : une tuile a la fois, en avance d'au plus 6 tuiles sur le processeur ;
 - 3 travailleurs PROCESSEUR (3 fils, coeurs 4-6 / 7-9 / 10-12) : entrees brutes d'abord, puis etapes A, B, passes, juge, paires ;
 - tout processus passe par lancer_borne.ps1 (priorite basse, 4060 Ti seule), puis coeurs reduits par PID ; garde de charge : > 85 % du processeur
   pendant 2 min ou memoire libre < 10 Go -> un travailleur processeur suspendu (repris quand la charge retombe) ;
 - reprise : une tuile dont la paire « debruitee » existe est sautee ; arret par PID de l'arbre seulement.
usage : produire_v33.py [--essai]"""
import csv, json, os, subprocess, sys, threading, time, queue
from pathlib import Path
import numpy as np, psutil
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ICI = Path(__file__).resolve().parent
V33 = (Path(__import__("os").environ.get("RACINE_SCRIPTS") or Path(__file__).resolve().parents[1]) / "Methode v33b – 23 nov")
SR = (Path(__import__("os").environ.get("RACINE_SCRIPTS") or Path(__file__).resolve().parents[1]))
SORTIE = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Paires étoiles v33 – 23 nov")
PY = r"python"
BORNE = SR / "lancer_borne.ps1"
JOUR = ICI / "journaux_v33b_reste"; JOUR.mkdir(exist_ok=True)
os.environ.update(ASTRO_GPU_OFFLINE="1", ASTRO_GPU_GAIA_EXTRACTS="<CACHE_FLOAT>/cache/gaia_extracts", PYTHONIOENCODING="utf-8",
                  PYTHONDONTWRITEBYTECODE="1")
COEURS = {"gpu": [13, 14], 0: [4, 5, 6], 1: [7, 8, 9], 2: [10, 11, 12]}
TUILES = ['bubble_a7s:2200:1800','heart_a7s:1152:1152','ic443_40da:2176:896','ic443_40da:2432:1152']
RAW = ["pleiades_a7s", "m45_asi1600", "dentelles_asi1600", "ngc6946_asi1600", "ic1318_stellina", "bubble_a7s", "xmas_a7s",
       "pleiades_g2014", "m101_a7s", "sorciere_a7s", "ngc7000_northamerica", "ic5070", "ngc7000_tiers1", "ngc7000_gfx"]
lock = threading.Lock()
actifs = {}          # travailleur -> pid racine
suspendus = set()
arret = threading.Event()
resultats = []


def log(s):
    m = f"{time.strftime('%H:%M:%S')} {s}"
    print(m, flush=True)
    with open(JOUR / "orchestrateur.log", "a", encoding="utf-8") as f:
        f.write(m + "\n")


def lance(w, args, logf, fils):
    """Un processus borne par lancer_borne.ps1, coeurs reduits, attendu jusqu'a sa fin."""
    pr = subprocess.Popen(["powershell", "-NoProfile", "-File", str(BORNE), "-Fils", str(fils), "-Log", str(logf), "-Exe", PY,
                           "-Args", args, "-Dossier", str(ICI)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                          encoding="utf-8", errors="replace")
    ligne = pr.stdout.readline()                     # « PID n fils=... » : le lanceur rend la main, l'enfant continue
    if "PID" not in ligne:
        log(f"lancement impossible {args[:80]} : {ligne[-200:]}")
        return None
    pid = int(ligne.split("PID")[1].split()[0])
    try:
        p = psutil.Process(pid)
        p.cpu_affinity(COEURS[w])
    except Exception as e:
        log(f"affinite {w} : {e}")
        p = None
    with lock:
        actifs[w] = pid
    try:
        code = p.wait() if p else None
    except Exception:
        code = None
    with lock:
        actifs.pop(w, None)
        suspendus.discard(w)
    return code


def garde():
    hist = []
    while not arret.is_set():
        time.sleep(15)
        hist.append(psutil.cpu_percent(interval=None)); hist = hist[-8:]       # 8 x 15 s = 2 min
        libre = psutil.virtual_memory().available / 2 ** 30
        with lock:
            actifs_cpu = [w for w in (0, 1, 2) if w in actifs and w not in suspendus]
            if (len(hist) == 8 and min(hist) > 85) or libre < 10:
                if actifs_cpu:
                    w = actifs_cpu[-1]
                    try:
                        for q in [psutil.Process(actifs[w])] + psutil.Process(actifs[w]).children(recursive=True):
                            try: q.suspend()
                            except Exception: pass
                        suspendus.add(w); hist.clear()
                        log(f"GARDE : travailleur {w} suspendu (charge {psutil.cpu_percent():.0f} %, memoire libre {libre:.1f} Go)")
                    except Exception as e:
                        log(f"GARDE erreur {e}")
            elif suspendus and len(hist) == 8 and max(hist) < 70 and libre > 14:
                w = suspendus.pop()
                if w in actifs:
                    try:
                        for q in [psutil.Process(actifs[w])] + psutil.Process(actifs[w]).children(recursive=True):
                            try: q.resume()
                            except Exception: pass
                    except Exception:
                        pass
                log(f"GARDE : travailleur {w} repris")


def taille(image):
    sys.path.insert(0, str(V33 / "Piste1"))
    import images
    f = images.IMAGES[image][0]
    if f.suffix == ".npy":
        return np.load(f, mmap_mode="r").shape[1:]
    from astropy.io import fits
    h = fits.getheader(f)
    return h["NAXIS2"], h["NAXIS1"]


def liste_tuiles():
    rows = list(csv.DictReader(open(r"<DONNEES>\Choix des tuiles v2 – 23 nov\tuiles_finales.csv", encoding="utf-8-sig"), delimiter=";"))
    par, passees = {}, []
    sys.path.insert(0, str(V33 / "Piste1"))
    import images
    for r in rows:
        im = r["image"]
        if not r["x"].strip() or not r["y"].strip():
            passees.append((im, "tuile sans coordonnees (image ecartee)")); continue
        par.setdefault(im, []).append((int(r["x"]), int(r["y"])))
    ok = {}
    for im, ts in par.items():
        why = None
        if im not in images.IMAGES:
            why = "image inconnue de la chaine"
        elif not (V33 / "Piste1" / "sorties" / im / "astrometrie.npz").exists():
            why = "astrometrie absente"
        elif not images.IMAGES[im][0].exists():
            why = "entree absente"
        elif not (V33 / "Catalogue Gaia profond – 20 nov" / f"gaia_dr3_profond_{im}.npz").exists():
            why = "catalogue Gaia profond absent"
        if why:
            passees += [(im, why)] * len(ts)
            continue
        H, W = taille(im)
        good = []
        for tx, ty in ts:
            if tx - 128 < 0 or ty - 128 < 0 or tx + 640 > W or ty + 640 > H:
                passees.append((im, f"tuile {tx},{ty} hors de l'image {W}x{H}"))
                continue
            good.append((tx, ty))
        ok[im] = good
    ordre = []
    for k in range(max(len(v) for v in ok.values())):
        for im in ok:
            if k < len(ok[im]):
                ordre.append((im, *ok[im][k]))
    return ordre, passees


def fait(im, tx, ty):
    return (SORTIE / f"{im}_g{tx}_{ty}_debruitee.npz").exists()


def main():
    SORTIE.mkdir(parents=True, exist_ok=True)
    ordre = [(a, int(b), int(c)) for a, b, c in (s.split(':') for s in TUILES)]; passees = []
    (ICI / "v33b2_tuiles_passees.csv").write_text("image;raison\n" + "\n".join(f"{a};{b}" for a, b in passees), encoding="utf-8")
    log(f"{len(ordre)} tuiles a produire, {len(passees)} passees : {sorted(set(passees))[:6]}")
    if "--essai" in sys.argv:
        return
    a_faire = [t for t in ordre if not fait(*t)]
    qcpu = queue.Queue()
    for im in RAW:
        if not (SORTIE / "entrees_brutes" / f"{im}_brut.npy").exists() and any(t[0] == im for t in a_faire):
            qcpu.put(("brut", im))
    t_debut = time.time()
    etat = dict(piste1=0, fait=0)
    echecs = []
    gpu_fini = threading.Event()

    def travailleur_gpu():
        for im, tx, ty in a_faire:
            if arret.is_set():
                break
            while qcpu.qsize() >= 6 + len(RAW) and not arret.is_set():
                time.sleep(10)
            nom = f"{im}_g{tx}_{ty}"; t0 = time.time()
            args = f'"{V33}\\etape_piste1_gpu_2f.py" mv_{nom} {tx - 128},{ty - 128},{tx + 640},{ty + 640} {im}'
            lance("gpu", args, JOUR / f"p1_{nom}.log", 2)
            if (V33 / "Piste1" / "sorties" / im / f"mv_{nom}" / "resume.json").exists():
                qcpu.put(("tuile", im, tx, ty, time.time() - t0))
                etat["piste1"] += 1
            else:
                echecs.append((nom, "piste 1")); log(f"ECHEC piste 1 {nom}")
        gpu_fini.set()

    def travailleur_cpu(w):
        while not arret.is_set():
            try:
                job = qcpu.get(timeout=20)
            except queue.Empty:
                if gpu_fini.is_set() and qcpu.empty():
                    return
                continue
            if job[0] == "brut":
                im = job[1]
                lance(w, f'"{ICI}\\entree_brute.py" {im}', JOUR / f"brut_{im}.log", 3)
                log(f"entree brute {im} : " + ("faite" if (SORTIE / "entrees_brutes" / f"{im}_brut.npy").exists() else "ECHEC"))
                continue
            _, im, tx, ty, dp1 = job
            nom = f"{im}_g{tx}_{ty}"; t0 = time.time()
            lance(w, f'"{ICI}\\tuile_cpu_v33b.py" {im} {tx} {ty}', JOUR / f"cpu_{nom}.log", 3)
            dur = time.time() - t0
            if fait(im, tx, ty):
                etat["fait"] += 1
                res = {}
                for l in open(JOUR / f"cpu_{nom}.log", encoding="utf-8", errors="replace"):
                    if l.startswith("RESULTAT "):
                        res = json.loads(l[9:])
                res.update(piste1_s=round(dp1), cpu_s=round(dur), fin=time.strftime("%H:%M:%S"))
                with lock:
                    resultats.append(res)
                    with open(SORTIE / "paires.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps(res, ensure_ascii=False) + "\n")
                n = etat["fait"]; el = time.time() - t_debut
                log(f"{n}/{len(a_faire)} {nom} fait (p1 {dp1:.0f} s + cpu {dur:.0f} s) poids0 {res.get('poids_0')}")
                if False:
                    mean = el / 6; restant = len(a_faire) - 6
                    fin = time.time() + restant * mean
                    msg = f"6 premieres tuiles : {mean:.0f} s par tuile en parallele (mesure), fin estimee {time.strftime('%d/%m %H:%M', time.localtime(fin))}"
                    log("RAPPORT 2 : " + msg); (ICI / "v33b_rapport2.txt").write_text(msg, encoding="utf-8")
            else:
                echecs.append((nom, "cpu")); log(f"ECHEC cpu {nom}")

    th = [threading.Thread(target=garde, daemon=True), threading.Thread(target=travailleur_gpu)] + \
         [threading.Thread(target=travailleur_cpu, args=(w,)) for w in (0, 1, 2)]
    for t in th:
        t.start()
    for t in th[1:]:
        t.join()
    arret.set()
    # tuiles dont l'entree brute n'etait pas prete au moment de la paire
    for im, tx, ty in ordre:
        if fait(im, tx, ty) and not (SORTIE / f"{im}_g{tx}_{ty}_brute.npz").exists() and (SORTIE / "entrees_brutes" / f"{im}_brut.npy").exists():
            lance(0, f'"{ICI}\\tuile_cpu_v33b.py" --brute-seul {im} {tx} {ty}', JOUR / f"brute_{im}_{tx}_{ty}.log", 3)
    (ICI / "v33b2_echecs.csv").write_text("tuile;etape\n" + "\n".join(f"{a};{b}" for a, b in echecs), encoding="utf-8")
    log(f"production terminee : {sum(fait(*t) for t in ordre)}/{len(ordre)} tuiles, {len(echecs)} echecs")


if __name__ == "__main__":
    main()
