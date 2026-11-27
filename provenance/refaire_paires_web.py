#!/usr/bin/env python
# -*- coding: ascii -*-
"""refaire_paires_web.py - rebuilds an APOD training pair from the published image, with the published chain.

ROLE
  The pixels of the APOD pairs are not published (authors' images). This script lets you rebuild them: for one or all
  the APOD pairs of the manifest (MANIFESTE_PAIRES.csv, columns url_image, fichier_telecharge, sha256_image_telechargee,
  delinearisation_params, decoupe) it
    1. gets the web image as published: local file if present (default, option --local), or download from
       url_image (option --telecharger, one network access per image, a 3 s pause between two);
    2. checks its sha256 (column sha256_image_telechargee);
    3. redoes the delinearisation (method etoiles_auto of the folder "Delinearisation - 23 nov" of the published scripts:
       stars' curve estimated on the image, parameters recorded in delinearisation_params);
    4. rebuilds the input (background and noise per colour on an image reduced to a quarter, unit = mean noise, as
       preparer_apod.py does);
    5. rebuilds the pair with the published chain (verifier_provenance.py --mode rejeu: track 1, steps A and B, passes,
       judge v33);
    6. compares with the manifest: sha256 of the downloaded image, gap between the rebuilt input and the kept input, gap
       of the target with the base pair, and sha256 of the rebuilt pair file against the column sha256_base_v33.
  If it is not bit-identical, the gap is quantified (maximum, mean, share of identical pixels) and the probable reason
  is given (library version, order of the floating-point operations, different downloaded image).

INPUTS
  --paire NAME      name (or start of a name) of a pair of the manifest, e.g. apod_070_g1536_1408; --toutes for all (38 pairs).
  --manifeste CSV   default: MANIFESTE_PAIRES.csv next to the script.
  --local / --telecharger   source of the web image (default --local; --telecharger is NOT run by default).
  RACINE_SCRIPTS    published scripts (default: the repository's scripts folder).
  RACINE_ORIGINE    unpublished files of the chain (astrometry, star model, deep Gaia catalogue of each apod_* image).
  DONNEES           data: "Images web APOD - 24 nov/brut" (web images), "Paires etoiles APOD - 24 nov" (inputs and
                    comparison pairs), "Delinearisation APOD - 24 nov/lineaire" (kept delinearisations, optional comparison).
  ESSAIS, RACINE_FLOAT, CACHE_FLOAT : see verifier_provenance.py.
WHAT REQUIRES FLOAT (engine): offline astrometry and the Gaia extracts (already computed and supplied in RACINE_ORIGINE for the
  38 images: astrometrie.npz, deep catalogue); the denoising network is NOT used (images already published, no denoising).
  Supplied by the published scripts: delinearisation, input normalisation, track 1, steps A and B, passes, judge.

OUTPUTS
  Report on the standard output, and --rapport JSON. Work folder (--travail): downloaded image, rebuilt input, logs.
  Exit code 0 if everything is bit-identical, 1 otherwise, 2 on error.

EXAMPLE
  DONNEES=/donnees RACINE_ORIGINE=/origine python refaire_paires_web.py --paire apod_070_g1536_1408 --local --travail /tmp/web
  (add --telecharger to download the image again from url_image)
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

ICI = Path(__file__).resolve().parent


def sha256_octets(b):
    return hashlib.sha256(b).hexdigest()


def sha256_fichier(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def ligne(s):
    print(s, flush=True)


def trouver(racine, motif):
    r = sorted(Path(racine).glob(motif))
    if not r:
        raise SystemExit("introuvable sous %s : %s" % (racine, motif))
    return r[0]


def stats(a, b):
    d = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))
    return {"ecart_max": float(d.max()), "ecart_moyen": float(d.mean()), "part_identique": float((d == 0).mean())}


def obtenir_image(row, donnees, trav, telecharger):
    fic = row["fichier_telecharge"]
    if telecharger:
        dest = trav / "telecharge" / fic
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(row["url_image"], headers={"User-Agent": "client-generique"})
        with urllib.request.urlopen(req, timeout=120) as r:
            dest.write_bytes(r.read())
        time.sleep(3)
        return dest
    p = trouver(donnees, "Images web APOD*") / "brut" / fic
    if not p.exists():
        raise SystemExit("image web absente en local : %s (utiliser --telecharger)" % p)
    return p


def delineariser(fichier, racine):
    """Meme lecture et meme methode que delineariser.py : etoiles_auto sur l'image RVB 0-1."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    m23 = trouver(racine, "Delinearisation *23 nov")
    sys.path.insert(0, str(m23))
    import methodes  # noqa: E402
    y = np.moveaxis(np.asarray(Image.open(fichier).convert("RGB")).astype(np.float32) / 255.0, -1, 0)
    D, info = methodes.etoiles_auto(y)
    return np.asarray(D, np.float32), info


def entree(D):
    """Comme preparer_apod.py : fond et bruit par couleur sur D[:, ::4, ::4] (pixels non nuls), unite = bruit moyen."""
    x4 = D[:, ::4, ::4]
    bk = np.array([np.median(x4[c][x4[c] != 0]) for c in range(3)], np.float32)
    sg = np.array([1.4826 * np.median(np.abs(x4[c] - bk[c])[x4[c] != 0]) for c in range(3)], np.float32)
    unit = float(sg.mean())
    return ((D - bk[:, None, None]) / unit).astype(np.float32), bk, sg, unit


def refaire(row, a, donnees, racine, rapports):
    nom = row["nom_fichier"].replace("_debruitee_v33g.npz", "")
    nom_base = nom
    image = row["image_source"]
    ligne("== %s (%s)" % (row["nom_fichier"], row["apod_info"]))
    trav = Path(a.travail) / nom if a.travail else Path(tempfile.mkdtemp(prefix="refaire_web_"))
    trav.mkdir(parents=True, exist_ok=True)
    rap = {"paire": row["nom_fichier"]}
    f = obtenir_image(row, donnees, trav, a.telecharger)
    sha = sha256_fichier(f)
    rap["sha256_image"] = {"attendu": row["sha256_image_telechargee"], "obtenu": sha, "identique": sha == row["sha256_image_telechargee"]}
    ligne("  image web : %s : sha256 %s" % (f.name, "IDENTIQUE au manifeste" if rap["sha256_image"]["identique"] else "DIFFERENT (image modifiee cote serveur ?)"))
    t0 = time.time()
    D, info = delineariser(f, racine)
    par = json.loads(row["delinearisation_params"])
    ligne("  delinearisation : %.0f s ; courbe m_float %s (manifeste %s), etoiles utilisees %s (manifeste %s), couleur %s (manifeste %s)" %
          (time.time() - t0, info.get("m_float"), par.get("m_float"), info.get("etoiles_utilisees"), par.get("etoiles_utilisees"), info.get("couleur"), par.get("couleur")))
    img, bk, sg, unit = entree(D)
    d_apod = trouver(donnees, "Paires *toiles APOD*")
    conserve = d_apod / "entrees" / image / "input.npy"
    if conserve.exists():
        s = stats(np.load(conserve), img)
        rap["entree"] = s
        ligne("  entree refaite / conservee : ecart max %.6g, moyen %.6g, part identique %.6f %s" % (s["ecart_max"], s["ecart_moyen"], s["part_identique"], "(IDENTIQUE au bit)" if s["part_identique"] == 1.0 else ""))
    # dossier de donnees de travail : l'entree refaite remplace l'entree conservee (le reste de la chaine la lit comme d'habitude)
    dw = trav / "donnees"
    ent = dw / d_apod.name / "entrees" / image
    ent.mkdir(parents=True, exist_ok=True)
    np.save(ent / "input.npy", img)
    (ent / "meta.json").write_text(json.dumps({"name": image, "background": bk.tolist(), "noise": sg.tolist(), "unit": unit}), encoding="utf-8")
    paire = d_apod / ("%s_debruitee_v33g.npz" % nom)
    env = dict(os.environ, DONNEES=str(dw), PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, str(ICI / "verifier_provenance.py"), str(paire), "--mode", "rejeu", "--device", a.device, "--fils", str(a.fils),
           "--travail", str(trav / "rejeu"), "--rapport", str(trav / "rejeu.json")]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout, flush=True)
    if r.returncode == 2 or not (trav / "rejeu.json").exists():
        ligne("  rejeu impossible : " + (r.stderr or r.stdout)[-400:])
        rap["rejeu"] = "impossible"
        rapports.append(rap)
        return False
    rj = json.loads((trav / "rejeu.json").read_text(encoding="utf-8"))
    rap["rejeu"] = rj.get("rejeu")
    refait = trav / "rejeu" / "paire" / ("%s_%s.npz" % (image, nom))
    ok_sha = False
    if refait.exists():
        sh = sha256_fichier(refait)
        ok_sha = sh == row["sha256_base_v33"]
        rap["sha256_paire_base"] = {"attendu": row["sha256_base_v33"], "obtenu": sh, "identique": ok_sha}
        ligne("  fichier de paire de base refait : sha256 %s au manifeste (colonne sha256_base_v33)" % ("IDENTIQUE" if ok_sha else "DIFFERENT (a lire avec l'ecart des tableaux ci-dessus : le fichier zip peut differer meme si les tableaux sont egaux)"))
    ok = bool(rj.get("tout_identique")) and rap["sha256_image"]["identique"] and rap.get("entree", {}).get("part_identique", 1.0) == 1.0
    rap["identique_au_bit"] = ok
    rapports.append(rap)
    ligne("  bilan : %s" % ("identique au bit (image, entree, cible, poids, classes)" if ok else "ecart : voir les lignes ci-dessus"))
    return ok


def main():
    ap = argparse.ArgumentParser(description="Refait des paires APOD depuis l'image publiee (voir l'en-tete du script).")
    ap.add_argument("--paire", default="")
    ap.add_argument("--toutes", action="store_true")
    ap.add_argument("--manifeste", default=str(ICI / "MANIFESTE_PAIRES.csv"))
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--local", action="store_true", help="utiliser l'image deja sur le disque (defaut)")
    g.add_argument("--telecharger", action="store_true", help="retelecharger l'image depuis url_image (acces reseau)")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--fils", type=int, default=4)
    ap.add_argument("--travail", default="")
    ap.add_argument("--rapport", default="")
    a = ap.parse_args()
    if "DONNEES" not in os.environ:
        raise SystemExit("variable d'environnement DONNEES obligatoire")
    donnees = Path(os.environ["DONNEES"])
    racine = Path(os.environ.get("RACINE_SCRIPTS") or ICI.parent / "scripts")
    rows = [r for r in csv.DictReader(open(a.manifeste, encoding="utf-8-sig"), delimiter=";") if r["type"] == "apod"]
    if not a.toutes:
        if not a.paire:
            raise SystemExit("donner --paire NOM ou --toutes")
        rows = [r for r in rows if r["nom_fichier"].startswith(a.paire)]
    if not rows:
        raise SystemExit("aucune paire APOD du manifeste ne correspond")
    rapports = []
    ok = True
    for r in rows:
        ok &= refaire(r, a, donnees, racine, rapports)
    ligne("== %d paire(s) : %s" % (len(rows), "toutes identiques au bit" if ok else "au moins un ecart"))
    if a.rapport:
        Path(a.rapport).write_text(json.dumps(rapports, indent=1), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
