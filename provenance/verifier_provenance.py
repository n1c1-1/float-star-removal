#!/usr/bin/env python
# -*- coding: ascii -*-
"""verifier_provenance.py - rebuilds the target of ONE training pair with the published chain and compares it with the file.

ROLE
  A training pair is an .npz file (input, target, weight, classes). Its target is made by the classical chain:
  subtraction guided by the Gaia catalogue (star and halo model) + refill + passes, then a judge (weights). No neural
  network, no output of third-party software enters it. This script shows it on ONE pair: it replays the chain on the
  input tile and measures the gap with the pair file.

MODES (option --mode)
  liens   (fast, default): sha256 fingerprint of the pair compared with the manifest; relations between neighbouring
          versions (v33 -> v33d: identical target, weights only set to 0; v33g: target + diffuse light L, smooth; v33h:
          weights only; raw pair: target = raw - (denoised input - denoised target), bit for bit).
  g       : rebuilds the step v33d -> v33g (preparer_g_h.py, processor, fast) and compares it with the v33g pair.
  rejeu   : rebuilds the complete chain of the tile (track 1, step A, step B, passes, judge) and compares it with the base
          pair (v33). Takes a few minutes to over half an hour depending on the tile. Not for raw pairs (the raw one is
          deduced from the denoised one: mode liens).
  tout    : liens + g + rejeu.

INPUTS
  PAIRE             path of a pair file (e.g. <image>_g<tx>_<ty>_debruitee_v33g.npz or the version without suffix).
  RACINE_SCRIPTS    (environment variable; default: the repository's scripts folder, next to provenance/) scripts of the chain.
  RACINE_ORIGINE    (optional; default: RACINE_SCRIPTS) folder holding the UNPUBLISHED files of the chain: astrometry
                    and star model of each image (Piste1/sorties/<image>), deep Gaia catalogue, list of the images of set 2,
                    old import folder (images.py, juger.py). Without them, the replay stops and says so.
  DONNEES           (required for rejeu) data folder (pairs, inputs, APOD inputs).
  ESSAIS, RACINE_FLOAT, CACHE_FLOAT (optional): other roots. If the published scripts carry root tokens
  (<SCRIPTS>, <DONNEES>, <ESSAIS>, <FLOAT>, <CACHE_FLOAT>), they are replaced in the working copy by these values.
  The scripts of the chain are COPIED into a work folder (--travail): nothing is written in RACINE_SCRIPTS.
  REMPLACEMENTS_COPIE (optional): JSON list [[old, new], ...] of texts replaced in the working copy (local adaptation,
  e.g. an engine module name different from that of the published scripts).
  What requires FLOAT (engine): offline astrometry and the Gaia extracts (RACINE_FLOAT, CACHE_FLOAT).

OUTPUTS
  A report on the standard output and, with --rapport, a JSON file. For the replay: per channel, on the pixels of weight
  > 0 of the base pair, the maximum and mean absolute gap (in sky noises, the images' unit) and the share of identical
  pixels; share of identical weights and classes. Exit code 0 if everything asked for is bit-identical, 1 if there is a
  gap (to be read in the report), 2 on error.

LIMITS (stated plainly)
  - The original production ran track 1 on the graphics card (torch, 4060 Ti); its result is not guaranteed bit for bit
    from one machine or library to another. By default this script redoes it on the processor (--device cpu); on the
    tiles tried the result is bit-identical, but a small gap remains possible elsewhere: it would be quantified.
  - The kept inputs are needed (source image, astrometry and deep Gaia catalogue of the image, offline Gaia extracts).
    If one is missing, the replay stops and says so; the modes liens and g remain possible.

EXAMPLES
  python verifier_provenance.py PAIRE.npz
  DONNEES=/path/data RACINE_ORIGINE=/path/origin python verifier_provenance.py PAIRE.npz --mode rejeu --device cpu --travail /tmp/rejeu
  RACINE_SCRIPTS=/path/scripts python verifier_provenance.py PAIRE.npz --mode g
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
import numpy as np  # noqa: E402

NOM = re.compile(r"^(?P<image>.+)_g(?P<tx>\d+)_(?P<ty>\d+)_(?P<typ>debruitee|brute)(?:_(?P<ver>v33[a-z]))?\.npz$")
# tuiles faites par la variante v33b de la chaine (piste 1 corrigee pour 4 plantages) en plus des images apod_*
V33B = {("bubble_a7s", 2000, 1367), ("bubble_a7s", 1446, 1310), ("bubble_a7s", 2200, 1800), ("heart_a7s", 128, 128),
        ("heart_a7s", 384, 384), ("heart_a7s", 1152, 1152), ("ic443_40da", 3200, 384), ("ic443_40da", 2432, 640),
        ("ic443_40da", 2176, 896), ("ic443_40da", 2432, 1152), ("ic1318_stellina", 2328, 1500),
        ("ic1318_stellina", 896, 1148), ("ngc7000_northamerica", 896, 1408)}


RACINE0 = []   # racine donnee au lancement (les modes g et rejeu changent la variable d'environnement pour leurs sous-processus)


def racine_scripts():
    """RACINE_SCRIPTS, par defaut le dossier scripts du depot (a cote de provenance/)."""
    if RACINE0:
        return RACINE0[0]
    r = os.environ.get("RACINE_SCRIPTS")
    return Path(r) if r else Path(__file__).resolve().parent.parent / "scripts"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def trouver(racine, motif):
    r = sorted(Path(racine).glob(motif))
    if not r:
        raise SystemExit("introuvable sous %s : %s" % (racine, motif))
    return r[0]


def charger(p):
    z = np.load(p)
    return {k: z[k] for k in z.files}


def stats(a, b, m, titre):
    """Ecart entre deux (C, H, W) sur le masque m (H, W) : par canal max, moyenne, part identique."""
    a = np.nan_to_num(np.asarray(a, np.float64))
    b = np.nan_to_num(np.asarray(b, np.float64))
    out = {"titre": titre, "pixels": int(m.sum()), "canaux": []}
    for c in range(a.shape[0]):
        d = np.abs(a[c] - b[c])[m]
        out["canaux"].append({"ecart_max": float(d.max()) if d.size else 0.0, "ecart_moyen": float(d.mean()) if d.size else 0.0,
                              "part_identique": float((d == 0).mean()) if d.size else 1.0})
    out["identique_au_bit"] = all(c["part_identique"] == 1.0 for c in out["canaux"])
    return out


def ligne(s):
    print(s, flush=True)


def afficher(st):
    ligne("  %s (%d pixels) : %s" % (st["titre"], st["pixels"], "IDENTIQUE au bit" if st["identique_au_bit"] else "ecart"))
    for i, c in enumerate(st["canaux"]):
        ligne("    canal %d : ecart max %.6g, ecart moyen %.6g, part identique %.6f" % (i, c["ecart_max"], c["ecart_moyen"], c["part_identique"]))


def lire_manifeste(arg, ici):
    p = Path(arg) if arg else ici / "MANIFESTE_PAIRES.csv"
    if not p.exists():
        return None, None
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig"), delimiter=";"))
    return rows, p


# ---------------------------------------------------------------------------------------------------- mode liens
def mode_liens(paire, info, rapport, manifeste):
    ligne("== liens : empreinte et relations entre versions")
    sha = sha256(paire)
    rapport["sha256_paire"] = sha
    ligne("  sha256 : " + sha)
    if manifeste is not None:
        r = [x for x in manifeste if x["sha256_paire"] == sha]
        if r:
            ligne("  manifeste : ligne trouvee (%s ; modeles %s ; role %s ; version %s)" % (r[0]["nom_fichier"], r[0]["modeles"], r[0]["role"], r[0]["version_cible"]))
            rapport["manifeste"] = "trouve"
        else:
            ligne("  manifeste : empreinte ABSENTE (fichier modifie, ou paire non lue par N5/N7)")
            rapport["manifeste"] = "absent"
    d = paire.parent
    base = "%s_g%s_%s_%s" % (info["image"], info["tx"], info["ty"], info["typ"])
    f = {v: d / (base + ("_" + v if v else "") + ".npz") for v in ("", "v33c", "v33d", "v33g", "v33h")}
    ok = True
    if info["typ"] == "brute":
        deb = d / ("%s_g%s_%s_debruitee.npz" % (info["image"], info["tx"], info["ty"]))
        if f[""].exists() and deb.exists():
            B, D = charger(f[""]), charger(deb)
            tb = (B["input"].astype(np.float32) - (D["input"] - D["target"])).astype(np.float32)
            egal = np.array_equal(np.nan_to_num(tb), np.nan_to_num(B["target"]))
            ligne("  brute (version de base) : cible == brute - (entree debruitee - cible debruitee) : " + ("OUI, au bit" if egal else "NON"))
            rapport["brute_formule"] = bool(egal)
            ok &= egal
        else:
            ligne("  brute : fichiers de base absents, formule non verifiee")
    if f[""].exists() and f["v33d"].exists():
        A, Dd = charger(f[""]), charger(f["v33d"])
        t_egal = np.array_equal(np.nan_to_num(A["target"]), np.nan_to_num(Dd["target"]))
        wa, wd = A["weight"], Dd["weight"]
        seulement_zero = bool(np.all((wd == wa) | (wd == 0)))
        ligne("  v33 -> v33d : cible identique : %s ; poids seulement mis a 0 : %s (poids changes : %.3f %% des pixels)" %
              ("OUI" if t_egal else "NON", "OUI" if seulement_zero else "NON", 100.0 * float((wd != wa).mean())))
        rapport["v33_v33d"] = {"cible_identique": bool(t_egal), "poids_seulement_a_zero": seulement_zero}
        ok &= t_egal and seulement_zero
    if f["v33d"].exists() and f["v33g"].exists():
        Dd, G = charger(f["v33d"]), charger(f["v33g"])
        L = np.nan_to_num(G["target"].astype(np.float64)) - np.nan_to_num(Dd["target"].astype(np.float64))
        lum = L.mean(0)
        gy, gx = np.gradient(lum)
        autres = all(np.array_equal(Dd[k], G[k]) for k in ("input", "weight", "classes"))
        ligne("  v33d -> v33g : entree, poids, classes identiques : %s ; (cible v33g - cible v33d) = lumiere diffuse L : moyenne %.4f, "
              "min %.4f, max %.4f bruits, pente max %.4f par pixel (lisse : sigma 40 px)" %
              ("OUI" if autres else "NON", float(lum.mean()), float(lum.min()), float(lum.max()), float(np.hypot(gx, gy).max())))
        rapport["v33d_v33g"] = {"autres_champs_identiques": bool(autres), "L_moyen": float(lum.mean()), "L_pente_max": float(np.hypot(gx, gy).max())}
        ok &= autres
    if f["v33g"].exists() and f["v33h"].exists():
        G, H = charger(f["v33g"]), charger(f["v33h"])
        t_egal = np.array_equal(np.nan_to_num(G["target"]), np.nan_to_num(H["target"]))
        ligne("  v33g -> v33h : cible identique : %s (seuls les poids changent)" % ("OUI" if t_egal else "NON"))
        rapport["v33g_v33h"] = {"cible_identique": bool(t_egal)}
        ok &= t_egal
    rapport["liens_ok"] = bool(ok)
    return ok


# ---------------------------------------------------------------------------------------------------- mode g
def module_g(racine, info, a):
    """Copie de travail de preparer_g_h.py et de diag_paires.py (jetons adaptes) ; rend le module importe."""
    origine = Path(os.environ.get("RACINE_ORIGINE") or racine)
    trav = Path(a.travail) if a.travail else Path(tempfile.mkdtemp(prefix="verifier_provenance_g_"))
    rg = trav / "racine_g"
    if rg.exists():
        shutil.rmtree(rg)
    rg.mkdir(parents=True)
    ign = shutil.ignore_patterns("__pycache__", "*.pyc")
    for motif in ("Tours N5 N6*", "Diagnostic *carts*"):
        src = trouver(racine, motif)
        shutil.copytree(src, rg / src.name, ignore=ign)
    # fichiers non publies lus par diag_paires.py : catalogue Gaia profond et modele d'etoile de l'image
    for motif in ("Methode v33b*", "Methode v33 production*"):
        for o in sorted(Path(origine).glob(motif)):
            for sub in (o / "Catalogue Gaia profond*",):
                for cat in sorted(o.glob("Catalogue Gaia profond*")):
                    g = cat / ("gaia_dr3_profond_%s.npz" % info["image"])
                    if g.exists():
                        (rg / o.name / cat.name).mkdir(parents=True, exist_ok=True)
                        shutil.copy2(g, rg / o.name / cat.name / g.name)
            m = o / "Piste1" / "sorties" / info["image"] / "modele_psf_image.json"
            if m.exists():
                (rg / o.name / "Piste1" / "sorties" / info["image"]).mkdir(parents=True, exist_ok=True)
                shutil.copy2(m, rg / o.name / "Piste1" / "sorties" / info["image"] / m.name)
    os.environ["RACINE_SCRIPTS"] = str(rg)
    adapter_chemins(rg)
    prep = trouver(rg, "Tours N5 N6*") / "preparer_g_h.py"
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "4"
    spec = importlib.util.spec_from_file_location("preparer_g_h", prep)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(prep.parent))
    spec.loader.exec_module(mod)
    return mod


def mode_g(paire, info, rapport, racine, a):
    ligne("== g : refait l'etape v33d -> v33g (preparer_g_h.py, processeur)")
    d = paire.parent
    base = "%s_g%s_%s_%s" % (info["image"], info["tx"], info["ty"], info["typ"])
    f33d, f33g = d / (base + "_v33d.npz"), d / (base + "_v33g.npz")
    if not (f33d.exists() and f33g.exists()):
        ligne("  fichiers v33d / v33g absents a cote de la paire : etape non refaite")
        rapport["g"] = "non_refait"
        return True
    mod = module_g(racine, info, a)
    t0 = time.time()
    z2, inf, I, T, Tg = mod.etape_g(f33d, "apod" if info["image"].startswith("apod_") else "v33")
    G = charger(f33g)
    st = stats(G["target"], z2["target"], np.ones(G["target"].shape[1:], bool), "cible v33g du fichier / refaite")
    afficher(st)
    ligne("  duree %.0f s ; etoiles G < 8 ecartees : %d ; correction L %s" % (time.time() - t0, inf["etoiles_g8"], "absente (M < 4000 px)" if inf["sans_correction"] else "calculee"))
    rapport["g"] = st
    return st["identique_au_bit"]


# ---------------------------------------------------------------------------------------------------- mode rejeu
SOUS_ENV = {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
JETONS = (("<SCRIPTS>", "RACINE_SCRIPTS"), ("<DONNEES>", "DONNEES"), ("<ESSAIS>", "ESSAIS"), ("<FLOAT>", "RACINE_FLOAT"),
          ("<CACHE_FLOAT>", "CACHE_FLOAT"))


def adapter_chemins(dossier):
    """Dans la COPIE de travail seulement : remplace les jetons de racine ecrits dans les scripts publies (<SCRIPTS>, <DONNEES>,
    <ESSAIS>, <FLOAT>, <CACHE_FLOAT>) par les dossiers donnes en variables d'environnement. Les scripts qui portent encore
    des chemins complets (copie locale d'origine) ne sont pas modifies. Signale les jetons restes sans valeur."""
    restes = set()
    subst = json.loads(os.environ.get("REMPLACEMENTS_COPIE", "[]"))
    for p in Path(dossier).rglob("*.py"):
        t = p.read_text(encoding="utf-8")
        t2 = t
        for ancien, nouveau in subst:
            t2 = t2.replace(ancien, nouveau)
        for jeton, var in JETONS:
            val = os.environ.get(var) or (str(racine_scripts()) if var == "RACINE_SCRIPTS" else "")
            if val:
                t2 = t2.replace(jeton, val.replace("\\", "/"))
        restes |= set(re.findall(r"<(?:SCRIPTS|DONNEES|ESSAIS|FLOAT|CACHE_FLOAT|PROJETS|BANQUE|OUTILS|DISQUE|HOME|TRAVAIL)>", t2))
        if t2 != t:
            p.write_text(t2, encoding="utf-8")
    if restes:
        ligne("  ATTENTION : jetons sans valeur dans les scripts : %s (definir la variable d'environnement correspondante)" % sorted(restes))


def copier_chaine(src, dst, image, orig=None):
    """Copie le code de la chaine (sans les sorties) et les seuls fichiers propres a l'image.
    Les fichiers non publies (astrometrie, modele, catalogue Gaia profond de l'image) sont pris sous orig (RACINE_ORIGINE)."""
    orig = orig or src
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.is_file() and p.suffix == ".py":
            shutil.copy2(p, dst / p.name)
    for sub in ("Piste1", "Piste3", "Passes"):
        (dst / sub).mkdir(exist_ok=True)
        for p in (src / sub).iterdir():
            if p.is_file() and p.suffix == ".py":
                shutil.copy2(p, dst / sub / p.name)
    s_im = orig / "Piste1" / "sorties" / image
    if not s_im.exists():
        raise SystemExit("astrometrie / modele de l'image absents (RACINE_ORIGINE ?) : %s" % s_im)
    (dst / "Piste1" / "sorties" / image).mkdir(parents=True, exist_ok=True)
    for p in s_im.iterdir():
        if p.is_file():
            shutil.copy2(p, dst / "Piste1" / "sorties" / image / p.name)
    cat = trouver(orig, "Catalogue Gaia profond*")
    (dst / cat.name).mkdir(exist_ok=True)
    g = cat / ("gaia_dr3_profond_%s.npz" % image)
    if g.exists():
        shutil.copy2(g, dst / cat.name / g.name)
    else:
        ligne("  (catalogue Gaia profond absent pour cette image : meme comportement que la production)")


def preparer_racine(racine, origine, trav, chaine_nom, image):
    """Construit trav/racine : chaine, juge, ancien dossier d'import et liste des images du jeu 2, puis adapte les jetons."""
    rt = trav / "racine"
    if rt.exists():
        shutil.rmtree(rt)
    rt.mkdir(parents=True)
    chaine = trouver(racine, chaine_nom)
    chaine_orig = trouver(origine, chaine_nom) if Path(origine) != Path(racine) else chaine
    copier_chaine(chaine, rt / chaine.name, image, chaine_orig)
    juge = trouver(racine, "Juge v33 *")
    shutil.copytree(juge, rt / juge.name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # dossier d'import ancien lu par Piste3/mesure.py (images.py, juger.py) : celui de l'origine s'il existe, sinon ceux de la piste 1 publiee
    anc = sorted(Path(origine).glob("Retrait *toiles par mod*")) or sorted(Path(racine).glob("Retrait *toiles par mod*"))
    dst_anc = rt / ("Retrait " + chr(233) + "toiles par mod" + chr(232) + "le " + chr(8211) + " 20 nov")
    dst_anc.mkdir()
    for n in ("images.py", "juger.py"):
        src_n = (anc[0] / n) if anc and (anc[0] / n).exists() else (rt / chaine.name / "Piste1" / n)
        shutil.copy2(src_n, dst_anc / n)
    sel = sorted(Path(origine).glob("Paires jeu 2*/selection_jeu2.csv")) or sorted(Path(racine).glob("Paires jeu 2*/selection_jeu2.csv"))
    if sel:
        d2 = rt / sel[0].parent.name
        d2.mkdir()
        shutil.copy2(sel[0], d2 / sel[0].name)
    adapter_chemins(rt)
    ajouter_hook(rt)
    return rt, rt / chaine.name, rt / juge.name


HOOK = """

# --- ajout de verifier_provenance.py (copie de travail seulement) : entrees recoupees (ENTREES_REJEU) ---
import os as _os
if _os.environ.get("ENTREES_REJEU"):
    import json as _j
    from pathlib import Path as _P
    _ER = _P(_os.environ["ENTREES_REJEU"])
    _load0, _meta0 = load, meta

    def load(name):
        f = _ER / name / "input.npz"
        if f.exists():
            return np.ascontiguousarray(np.load(f)["input"], np.float32)
        return _load0(name)

    def meta(name):
        f = _ER / name / "meta.json"
        return _j.loads(f.read_text(encoding="utf-8")) if f.exists() else _meta0(name)

    for _d in sorted(_ER.glob("*")):
        if (_d / "input.npz").exists():
            if _d.name in IMAGES:
                IMAGES[_d.name] = (_d / "input.npz",) + tuple(IMAGES[_d.name][1:])
            else:
                _t = _j.loads((_d / "type.json").read_text(encoding="utf-8"))
                IMAGES[_d.name] = (_d / "input.npz", _t["objet"], _t["genre"])
"""


def ajouter_hook(rt):
    """Ajoute le crochet ENTREES_REJEU a chaque images.py de la copie de travail (sans effet si la variable n'est pas definie)."""
    for p in Path(rt).rglob("images.py"):
        t = p.read_text(encoding="utf-8")
        if "ENTREES_REJEU" not in t:
            p.write_text(t + HOOK, encoding="utf-8")


def lancer(cmd, cwd, env_plus, journal):
    env = dict(os.environ)
    env.update(SOUS_ENV)
    env.update(env_plus)
    t = time.time()
    r = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    Path(journal).write_text("rc=%d\n%s\n%s" % (r.returncode, r.stdout[-6000:], r.stderr[-3000:]), encoding="utf-8")
    return r.returncode, time.time() - t


def mode_rejeu(paire, info, rapport, a):
    ligne("== rejeu : refait la chaine complete de la tuile")
    if info["typ"] == "brute":
        ligne("  paire brute : la cible se deduit de la debruitee (mode liens) ; rejeu a faire sur la paire debruitee")
        return True
    racine = racine_scripts()
    origine = Path(os.environ.get("RACINE_ORIGINE") or racine)
    image, tx, ty = info["image"], int(info["tx"]), int(info["ty"])
    nom = "%s_g%d_%d" % (image, tx, ty)
    v33b = image.startswith("apod_") or (image, tx, ty) in V33B or a.chaine == "v33b"
    if a.chaine == "v33":
        v33b = False
    chaine_nom = "Methode v33b*" if v33b else "Methode v33 production*"
    if not v33b and not list(Path(racine).glob(chaine_nom)):
        ligne("  (dossier de la chaine v33 absent des scripts publies : la chaine v33b est utilisee ; ecart possible, voir LISEZMOI)")
        chaine_nom = "Methode v33b*"
    ligne("  chaine : %s ; tuile %s ; crop %d,%d,%d,%d ; piste 1 sur %s" % (chaine_nom, nom, tx - 128, ty - 128, tx + 640, ty + 640, a.device))
    trav = Path(a.travail) if a.travail else Path(tempfile.mkdtemp(prefix="verifier_provenance_"))
    trav.mkdir(parents=True, exist_ok=True)
    rt, meth, juge = preparer_racine(racine, origine, trav, chaine_nom, image)
    os.environ["RACINE_SCRIPTS"] = str(rt)
    # entree : la tuile 512 de l'image, lue comme la chaine la lit (images.load)
    sys.path.insert(0, str(meth / "Piste1"))
    import images  # noqa: E402
    if image not in images.IMAGES or not Path(images.IMAGES[image][0]).exists():
        raise SystemExit("image source absente : %s" % image)
    full = images.load(image)
    I = np.array(full[:, ty:ty + 512, tx:tx + 512], np.float32)
    P = charger(paire)
    egal_e = bool(np.array_equal(I, P["input"].astype(np.float32)))
    ligne("  entree (tuile lue dans l'image source) identique a la paire : %s" % ("OUI, au bit" if egal_e else "NON"))
    rapport["entree_identique"] = egal_e
    crop = "%d,%d,%d,%d" % (tx - 128, ty - 128, tx + 640, ty + 640)
    fils = str(min(max(a.fils, 1), 4))
    env = {"RACINE_SCRIPTS": str(rt), "OMP_NUM_THREADS": fils, "MKL_NUM_THREADS": fils, "OPENBLAS_NUM_THREADS": fils,
           "NUMBA_NUM_THREADS": fils, "CUDA_VISIBLE_DEVICES": "-1" if a.device == "cpu" else "0"}
    if a.device == "cuda":
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    t0 = time.time()
    rc, s = lancer([sys.executable, "etape_piste1_gpu_2f.py", "mv_" + nom, crop, image, a.device], meth, env, trav / "journal_piste1.txt")
    ligne("  piste 1 : rc=%d en %.0f s" % (rc, s))
    if rc != 0:
        raise SystemExit("piste 1 a echoue (voir %s)" % (trav / "journal_piste1.txt"))
    rc, s = lancer([sys.executable, "methode_validee.py", crop, "--nom", nom, "--image", image, "--sans-vues", "--reutiliser-piste1"],
                   meth, dict(env, CUDA_VISIBLE_DEVICES="-1"), trav / "journal_A_B.txt")
    ligne("  etapes A et B (soustraction guidee par Gaia, rebouchage, passes) : rc=%d en %.0f s" % (rc, s))
    if rc != 0:
        raise SystemExit("etapes A/B ont echoue (voir %s)" % (trav / "journal_A_B.txt"))
    # juge : meme appel que la production (construire_paire.py, regles_v33.py)
    sys.path.insert(0, str(juge))
    import regles_v33  # noqa: E402
    regles_v33.DEEP = next(iter(sorted(meth.glob("Catalogue Gaia profond*"))))
    regles_v33.SORT = meth / "Piste1" / "sorties"
    import construire_paire as CP  # noqa: E402
    tmp = trav / "paire"
    tmp.mkdir(exist_ok=True)
    CP.OUT = tmp
    argv0 = sys.argv
    sys.argv = ["construire_paire.py", str(meth), image, nom, str(tx - 128), str(ty - 128)]
    CP.main()
    sys.argv = argv0
    R = charger(tmp / ("%s_%s.npz" % (image, nom)))
    ligne("  juge : paire refaite en %.0f s au total" % (time.time() - t0))
    # reference : la paire de base (v33) a cote du fichier donne, sinon le fichier donne
    ref = paire.parent / (nom + "_debruitee.npz")
    if ref.exists():
        B = charger(ref)
        ligne("  comparaison avec la paire de base : " + ref.name)
    else:
        B = P
        ligne("  paire de base absente : comparaison avec le fichier donne (peut etre une version ulterieure)")
    if info["ver"] in ("v33g", "v33h") and not ref.exists():
        # fichier publie v33g/v33h sans la paire de base : v33g = cible de la chaine + lumiere diffuse L (preparer_g_h.etape_g), refait ici
        # a partir de la paire REFAITE ; v33h a la meme cible ; les poids du fichier sont ceux de la chaine, avec des zones mises a 0 (v33d) ou changees (v33h)
        modg = module_g(racine, info, a)
        fr = trav / ("%s_g%d_%d_%s_v33d.npz" % (image, tx, ty, "debruitee"))
        np.savez(fr, **R)
        z2, inf, _I, _T, _Tg = modg.etape_g(fr, "apod" if image.startswith("apod_") else "v33")
        st_g = stats(P["target"], z2["target"], np.ones(P["target"].shape[1:], bool), "cible v33g du fichier / chaine refaite + lumiere diffuse refaite")
        afficher(st_g)
        w_ok = bool(np.all((P["weight"] == R["weight"]) | (P["weight"] == 0))) if info["ver"] == "v33g" else None
        c_ok = bool(np.array_equal(P["classes"], R["classes"]))
        i_ok = bool(np.array_equal(P["input"], R["input"]))
        ligne("  classes identiques : %s ; entree identique : %s ; poids du fichier = poids de la chaine ou 0 (v33d) : %s" % (c_ok, i_ok, w_ok if w_ok is not None else "non verifie (v33h change des poids)"))
        rapport["rejeu_v33g"] = {"cible": st_g, "classes": c_ok, "entree": i_ok, "poids_ok": w_ok, "device_piste1": a.device, "secondes": round(time.time() - t0)}
        ok = st_g["identique_au_bit"] and c_ok and i_ok and (w_ok is not False)
        if not a.conserver and not a.travail:
            shutil.rmtree(trav, ignore_errors=True)
        return ok
    m = B["weight"] > 0
    st_t = stats(B["target"], R["target"], m, "cible, pixels de poids > 0")
    st_all = stats(B["target"], R["target"], np.ones_like(m), "cible, tous les pixels")
    afficher(st_t)
    afficher(st_all)
    fb = paire.parent / (nom + "_brute.npz")
    if fb.exists():
        # paire brute de la meme tuile : cible = brute - (entree debruitee - cible debruitee), avec la cible REFAITE
        Bz = charger(fb)
        tb = (Bz["input"].astype(np.float32) - (R["input"] - R["target"])).astype(np.float32)
        st_b = stats(Bz["target"], tb, Bz["weight"] > 0, "cible brute (refaite a partir de la cible debruitee refaite)")
        afficher(st_b)
        rapport["rejeu_brute"] = st_b
    egal_w = float((B["weight"] == R["weight"]).mean())
    egal_c = float((B["classes"] == R["classes"]).mean())
    egal_i = bool(np.array_equal(B["input"], R["input"]))
    ligne("  poids identiques : %.6f des pixels ; classes identiques : %.6f ; entree identique : %s" % (egal_w, egal_c, "OUI" if egal_i else "NON"))
    rapport["rejeu"] = {"cible_poids_sup_0": st_t, "cible_tous": st_all, "poids_identiques": egal_w, "classes_identiques": egal_c, "entree_identique": egal_i,
                        "device_piste1": a.device, "secondes": round(time.time() - t0)}
    ok = st_t["identique_au_bit"] and egal_w == 1.0 and egal_c == 1.0
    if "rejeu_brute" in rapport:
        ok = ok and rapport["rejeu_brute"]["identique_au_bit"]
    if not a.conserver and not a.travail:
        shutil.rmtree(trav, ignore_errors=True)
    else:
        ligne("  dossier de travail conserve : %s" % trav)
    return ok


def main():
    ap = argparse.ArgumentParser(description="Verifie la provenance d'une paire d'entrainement (voir l'en-tete du script).")
    ap.add_argument("paire")
    ap.add_argument("--mode", default="liens", choices=("liens", "g", "rejeu", "tout"))
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"), help="piste 1 : processeur (defaut) ou carte 0")
    ap.add_argument("--chaine", default="auto", choices=("auto", "v33", "v33b"))
    ap.add_argument("--travail", default="", help="dossier de travail (copie de la chaine, journaux) ; defaut : temporaire")
    ap.add_argument("--conserver", action="store_true")
    ap.add_argument("--fils", type=int, default=4, help="fils de calcul au plus (maximum 4)")
    ap.add_argument("--manifeste", default="")
    ap.add_argument("--rapport", default="")
    a = ap.parse_args()
    RACINE0.append(racine_scripts())
    paire = Path(a.paire).resolve()
    m = NOM.match(paire.name)
    if not m:
        raise SystemExit("nom de paire non reconnu : " + paire.name)
    info = m.groupdict()
    rapport = {"paire": paire.name, "mode": a.mode}
    manifeste, _ = lire_manifeste(a.manifeste, Path(__file__).resolve().parent)
    ok = True
    try:
        if a.mode in ("liens", "tout"):
            ok &= mode_liens(paire, info, rapport, manifeste)
        if a.mode in ("g", "tout"):
            ok &= mode_g(paire, info, rapport, racine_scripts(), a)
        if a.mode in ("rejeu", "tout"):
            ok &= mode_rejeu(paire, info, rapport, a)
    except KeyError as e:
        ligne("variable d'environnement manquante : %s (DONNEES est obligatoire pour le rejeu)" % e)
        return 2
    rapport["tout_identique"] = bool(ok)
    ligne("== bilan : " + ("tout ce qui a ete demande est identique au bit" if ok else "au moins un ecart : lire le rapport ci-dessus"))
    if a.rapport:
        Path(a.rapport).write_text(json.dumps(rapport, indent=1), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
