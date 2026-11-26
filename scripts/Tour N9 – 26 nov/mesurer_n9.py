# ======================================================================
# mesurer_n9.py
# Rôle     : Mesure N9 pas 10000 ET N7 pas 10000 (V2, 13 images + IC 1848, cœurs, validation) ; tableau_n9.md (lignes de tableau_n8l.md + colonne M32 60D) ; planches cœur M31 et M32 Canon 60D.
# Étape    : 10. Essai écarté N9 (cœurs compacts, 26 nov) (voir scripts/LISEZMOI.md)
# Entrées  : paires v33g
# Sorties  : instantanés, tableau_n9.md
# Exemple  : python mesurer_n9.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
r"""Mesure N9 pas 10000 ET N7 pas 10000 (V2, 13 images + IC 1848, cœurs, validation) ; tableau_n9.md (lignes de tableau_n8l.md + colonne M32 60D) ; planches cœur M31 et M32 Canon 60D.
Copie de mesurer_n8l.py (chemins r"..." et compilation vérifiés). Ancien : Mesure N8L (tour long).
Copie de mesurer_n8_10000.py. Ancien : Mesure N8 au pas 10000 (copie de mesurer_n8_5000.py ; N8 pas 5000 et N7 pas 5000 déjà mesurés, lus dans Resultats et coeurs). Ancien :
Mesure N8 et N7 au pas 5000, UN seul processus, memoire surveillee (copie de mesurer_n8.py ; originaux inchanges).
Lance seulement si > 8 Go libres ; avant chaque image (inferer_v2_mem.py) et chaque coeur : arret propre sous 4 Go.
"""
import sys, runpy, shutil, json, importlib.util, os
import psutil
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ICI = Path(__file__).resolve().parent
SORTIE = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles N9 – 26 nov")
N7 = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles N7 – 25 nov")
N8T = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles N8 – 26 nov" / "tableau_n8.md")
MEL = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Melanges N5 N6 N7 – 26 nov" / "tableau_melanges.md")
TAB8L = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles N8L – 26 nov" / "tableau_n8l.md")
N8L = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles N8L – 26 nov")
N8 = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles N8 – 26 nov")
RES = SORTIE / "Resultats"
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"; os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def libre():
    return psutil.virtual_memory().available / 2**30


def log(m):
    print(m, flush=True)


def lancer(script, *args):
    if libre() < 4:
        log(f"ARRET PROPRE : memoire libre {libre():.1f} Go < 4 Go avant {Path(script).name}"); raise SystemExit(3)
    old = sys.argv
    sys.argv = [str(ICI / script)] + [str(a) for a in args]
    try:
        runpy.run_path(str(ICI / script), run_name="__main__")
    except SystemExit as e:
        if e.code not in (None, 0):
            raise
    finally:
        sys.argv = old


def snap(tag, ck):
    log(f"{tag} : {ck} (libre {libre():.1f} Go)")
    lancer("inferer_v2_mem.py", ck, tag)
    if not (SORTIE / "coeurs" / f"coeurs_{tag}.json").exists():
        lancer("coeurs_mesure.py", "mesurer", tag, ck, "--device", "cuda:0")
    lancer("valider_n7.py", str(SORTIE / f"starless_{tag}"), "--tag", tag)
    shutil.rmtree(SORTIE / f"starless_{tag}", ignore_errors=True)


N7_CK = N7 / "N7" / "snap" / "pas_10000.pt"
N9_CK = SORTIE / "N9" / "snap" / "pas_10000.pt"


def m32(f):
    """Rapport du seul M32 Canon 60D (m31_canon60d_objectif_filee 3094,1704) dans un coeurs_*.json ; '' si absent."""
    try:
        for c in json.loads(Path(f).read_text(encoding="utf-8"))["coeurs"]:
            if c["image"] == "m31_canon60d_objectif_filee" and (c["x"], c["y"]) == (3094, 1704):
                return f"{c['rapport']:.3f}"
    except Exception:
        pass
    return ""


def ligne_tag(M, nom, tag):
    r = M.ligne(tag)
    c = json.loads((SORTIE / "coeurs" / f"coeurs_{tag}.json").read_text(encoding="utf-8"))["moyenne"]
    return f"| {nom} | {r[1]} | " + " | ".join(f"{x:.3f}" for x in r[2]) + f" | {r[3]:.4f} | {r[4]:.2f} | {r[5]:.3f} | {r[6]:.4f} | {c:.3f} | {m32(SORTIE / 'coeurs' / f'coeurs_{tag}.json')} |"


def m32_ancienne(nom):
    """M32 60D d'une ligne recopiée de tableau_n8l.md (retrouvée par le nom de l'essai dans les dossiers de cœurs)."""
    ch = {"N5": N7 / "coeurs" / "coeurs_N5.json", "N7": N7 / "coeurs" / "coeurs_N7.json", "RefExt": SORTIE / "coeurs" / "coeurs_RefExt.json"}
    if nom in ch:
        return m32(ch[nom])
    import re
    r = re.match(r"N8L .*?(\d+)\)?$", nom)
    if r:
        return m32(N8L / "coeurs" / f"coeurs_N8L_{r.group(1)}.json")
    r = re.match(r"N8 pas (\d+)$", nom)
    if r:
        return m32(N8 / "coeurs" / f"coeurs_N8_{r.group(1)}.json")
    return ""


def tableau():
    _s = importlib.util.spec_from_file_location("mesurer_v33c", ICI.parent / "Entrainement v33c – 24 nov" / "mesurer_v33c.py")
    M = importlib.util.module_from_spec(_s); _s.loader.exec_module(M); M.R = RES
    lignes = [l for l in TAB8L.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    for i, l in enumerate(lignes):                    # lignes de tableau_n8l.md recopiées, colonne M32 60D ajoutée
        if i == 0:
            out.append(l + " M32 60D |")
        elif i == 1:
            out.append(l + "---|")
        else:
            nom = [x.strip() for x in l.strip().strip("|").split("|")][0]
            out.append(l + f" {m32_ancienne(nom)} |")
    out.append(ligne_tag(M, "N7 pas 10000 (mesuré ici)", "N7_10000"))
    out.append(ligne_tag(M, "N9 pas 10000", "N9_10000"))
    txt = "\n".join(out)
    (SORTIE / "tableau_n9.md").write_text(txt + "\n", encoding="utf-8"); log(txt)


def planche_leo():
    import numpy as np, cv2
    nom = "leotriplet_canon6d_newton"
    cols, sn, ent = [], None, None
    for lab, f in [("N7 10000", SORTIE / "crops" / f"N7_10000_{nom}.npz"), ("N9 10000", SORTIE / "crops" / f"N9_10000_{nom}.npz")]:
        if f.exists():
            z = np.load(f)
            if ent is None:
                ent = z["entree"]
            cols.append((lab, z["sortie"])); sn = z["refext"]
    if ent is None:
        return
    cols = [("entree", ent)] + cols + [("RefExt", sn)]
    top = float(np.arcsinh(np.percentile(cols[0][1].mean(0), 99.5) / 60)) + 1e-6
    tl = []
    for l, a in cols:
        v = np.clip(np.arcsinh(np.clip(a.astype(np.float32), 0, None) / 60) / top, 0, 1)
        im = cv2.resize((v.transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8), (460, 460), interpolation=cv2.INTER_AREA).copy()
        cv2.putText(im, l, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        tl.append(im)
    _, buf = cv2.imencode(".jpg", np.hstack(tl), [cv2.IMWRITE_JPEG_QUALITY, 90])
    buf.tofile(str(SORTIE / "planche_leo_triplet.jpg"))


if __name__ == "__main__":
    if libre() <= 8:
        log(f"Pas de lancement : memoire libre {libre():.1f} Go <= 8 Go"); sys.exit(2)
    (SORTIE / "coeurs").mkdir(exist_ok=True)
    if not (SORTIE / "coeurs" / "coeurs_RefExt.json").exists():       # RefExt (starless de référence) : fenêtres M31 et M32 60D
        lancer("coeurs_mesure.py", "mesurer", "RefExt", "refext")
    for tag, ck in (("N7_10000", N7_CK), ("N9_10000", N9_CK)):
        if not (RES / tag / "resume.json").exists():
            snap(tag, ck)
        else:
            log(f"{tag} : déjà mesuré")
    tableau()
    lancer("coeurs_mesure.py", "planche")
    planche_leo()
    log("termine")
