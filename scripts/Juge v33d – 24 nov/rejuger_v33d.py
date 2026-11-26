# ======================================================================
# rejuger_v33d.py
# Rôle     : Rejuge v33d : recalcule les poids des paires (v33c + regles H halo et F faibles), ne fait que mettre des poids a 0.
# Étape    : 2. Juge v33c/v33d : halos et étoiles faibles restées (24 nov) (voir scripts/LISEZMOI.md)
# Entrées  : paires v33
# Sorties  : poids mis à 0 (fichiers _v33d)
# Exemple  : usage : rejuger_v33d.py [--dossier D] [--ecrire] [--complet] [--deep D] [--sorties D] [--planches nom,nom,...] [noms...]
# ======================================================================
"""Rejuge v33d : recalcule les poids des paires (v33c + regles H halo et F faibles), ne fait que mettre des poids a 0.
usage : rejuger_v33d.py [--dossier D] [--ecrire] [--complet] [--deep D] [--sorties D] [--planches nom,nom,...] [noms...]
  --dossier  : dossier des paires (defaut : paires v33 du 23 nov). Pour chaque tuile on part du fichier _v33c s'il existe, sinon de l'ancien.
  --complet  : pour des paires neuves (jamais jugees v33c) : applique d'abord les regles v33 (a, b) et v33c, puis v33d.
  --deep/--sorties : dossiers du catalogue profond et des sorties de la chaine (FWHM, astrometrie) si ce sont d'autres images.
  --ecrire   : ecrit TOUTES les paires gardees en <nom>_debruitee_v33d.npz et <nom>_brute_v33d.npz (meme sans changement),
               controle_v33d/poids0_v33d.txt, details json et planches.
Les paires « lachee » de revue_orchestrateur.csv (si present dans le dossier) sont ecartees."""
import json, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import regles_v33c as C
import regles_v33d as D
import vue_v33d
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def lacher(P):
    lach = []
    f = P / "revue_orchestrateur.csv"
    if f.exists():
        for l in open(f, encoding="utf-8"):
            a = l.strip().split(";")
            if len(a) >= 3 and a[2] == "lâchée":
                lach.append((a[0], a[1]))
    return lambda im, t: any(i == im and (g == "*" or g == t) for i, g in lach)


def choisir(P, nom, genre):
    """Fichier de depart d'une tuile : v33c s'il existe, sinon l'ancien."""
    for suf in ("_v33c.npz", ".npz"):
        f = P / f"{nom}_{genre}{suf}"
        if f.exists():
            return f
    return None


def main(argv):
    args = list(argv)
    def opt(n, d=None):
        if n in args:
            k = args.index(n); v = args[k + 1]; del args[k:k + 2]; return v
        return d
    P = Path(opt("--dossier", r"<DONNEES>\Paires étoiles v33 – 23 nov"))
    if opt_deep := opt("--deep"):
        C.DEEP = Path(opt_deep)
    if opt_s := opt("--sorties"):
        C.SORT = Path(opt_s)
    plan = (opt("--planches", "") or "").split(",") if "--planches" in argv else []
    ecrire = "--ecrire" in args; complet = "--complet" in args
    args = [a for a in args if a not in ("--ecrire", "--complet")]
    lach = lacher(P)
    noms = args or sorted({re.sub(r"_(debruitee|brute)(_v33c)?\.npz$", "", f.name) for f in P.glob("*.npz")
                           if re.search(r"_(debruitee|brute)(_v33c)?\.npz$", f.name)})
    out_dir = P / "controle_v33d"
    lignes, rows = [], {}
    for nom in noms:
        m = re.match(r"^(.+)_(g\d+_\d+)$", nom)
        if not m:
            print("nom ignore :", nom); continue
        im, t = m.group(1), m.group(2)
        if lach(im, t):
            continue
        fd = choisir(P, nom, "debruitee")
        if fd is None:
            continue
        z = dict(np.load(fd))
        tx, ty = map(int, t[1:].split("_"))
        W_orig = np.load(P / f"{nom}_debruitee.npz")["weight"] if (P / f"{nom}_debruitee.npz").exists() else z["weight"]
        W0 = z["weight"].copy(); W = W0.copy()
        if complet:
            C.appliquer(z["input"], z["target"], W, z["classes"], im, tx, ty)
            C.appliquer_v33c(z["input"], z["target"], W, z["classes"], im, tx, ty)
        W1 = W.copy()
        det = D.appliquer_v33d(z["input"], z["target"], W, im, tx, ty)
        fw = C.fwhm_image(im)
        a0, a1 = float((W1 == 0).mean()) * 100, float((W == 0).mean()) * 100
        lignes.append((nom, float((W_orig == 0).mean()) * 100, a0, a1, det["halos"], det["faibles"]))
        if ecrire:
            z["weight"] = W
            np.savez_compressed(P / f"{nom}_debruitee_v33d.npz", **z)
            fb = choisir(P, nom, "brute")
            if fb is not None:
                zb = dict(np.load(fb)); zb["weight"] = np.minimum(zb["weight"], W)
                np.savez_compressed(P / f"{nom}_brute_v33d.npz", **zb)
            (out_dir / "details").mkdir(parents=True, exist_ok=True)
            (out_dir / "details" / f"{nom}.json").write_text(json.dumps(det, default=float), encoding="utf-8")
        if (not plan or nom in plan) and "--planches" in argv:
            rows.setdefault(nom, vue_v33d.rangee(z["input"], z["target"], W1, W, det, fw))
    sortie = ["paire ; poids0 v33 % ; poids0 v33c (avant) % ; poids0 v33d (apres) % ; gain points ; halos ; etoiles faibles"]
    for nom, a, b, c, nh, nf in lignes:
        sortie.append(f"{nom} ; {a:.2f} ; {b:.2f} ; {c:.2f} ; {c - b:+.2f} ; {nh} ; {nf}")
    g = np.array([l[3] - l[2] for l in lignes])
    sortie.append(f"MOYENNE v33 {np.mean([l[1] for l in lignes]):.2f} ; v33c {np.mean([l[2] for l in lignes]):.2f} ; v33d {np.mean([l[3] for l in lignes]):.2f} ; "
                  f"gain moyen {g.mean():+.3f} ; max {g.max():+.2f} ({lignes[int(g.argmax())][0]}) ; paires > +3 : {int((g > 3).sum())} ; "
                  f"> +8 : {int((g > 8).sum())} ; changees {int((g > 0).sum())}/{len(g)} ; halos {sum(l[4] for l in lignes)} ; faibles {sum(l[5] for l in lignes)}")
    print("\n".join(sortie))
    if ecrire:
        out_dir.mkdir(exist_ok=True)
        (out_dir / "poids0_v33d.txt").write_text("\n".join(sortie), encoding="utf-8")
        (P / "poids0_v33d.txt").write_text("\n".join(sortie), encoding="utf-8")
    if rows:
        import cv2
        out_dir.mkdir(exist_ok=True)
        cv2.imencode(".jpg", np.concatenate(list(rows.values()), 0), [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(str(out_dir / "planche_v33d.jpg"))
        (out_dir / "planche_v33d.txt").write_text("\n".join(rows), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
