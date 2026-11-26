# ======================================================================
# chaine_melanges.py
# Rôle     : Mélanges de poids N5 / N6 / N7 (moyenne pondérée a*A + (1-a)*B des paramètres flottants, tampons entiers pris de A), puis mesure identique à celle de la chaîne N7 (inférence deux échelles V2, cœurs, validation).
# Étape    : 11. Mélanges de poids (26 nov) (voir scripts/LISEZMOI.md)
# Entrées  : instantanés N5, N6, N7
# Sorties  : poids mélangés .pt, tableau_melanges.md
# Exemple  : lancer_borne.ps1 -Fils 4 -Log ...\chaine.out -Exe python -Args "-I chaine_melanges.py"
# ======================================================================
r"""Mélanges de poids N5 / N6 / N7 (moyenne pondérée a*A + (1-a)*B des paramètres flottants, tampons entiers pris de A), puis mesure identique à
celle de la chaîne N7 (inférence deux échelles V2, cœurs, validation). Aucun entraînement. Reprenable : repère poids\<nom>.pt et etapes\<nom>.fini.

    lancer_borne.ps1 -Fils 4 -Log ...\chaine.out -Exe python -Args "-I chaine_melanges.py"
"""
import glob, importlib.util, json, subprocess, sys, os, time, shutil
from pathlib import Path
import numpy as np, cv2, torch

ICI = Path(__file__).resolve().parent
SORTIE = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Melanges N5 N6 N7 – 26 nov")
N56 = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles N5 N6 – 25 nov")
N7D = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles N7 – 25 nov")
PY = r"python"
_s = importlib.util.spec_from_file_location("mesurer_v33c", ICI.parent / "Entrainement v33c – 24 nov" / "mesurer_v33c.py")
M = importlib.util.module_from_spec(_s); _s.loader.exec_module(M)
ENV = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", NUMBA_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
           CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0", PYTHONIOENCODING="utf-8")


def pas(f):
    return int(Path(f).read_text().split()[0])


SRC = {"N5": N56 / "N5" / "snap" / f"pas_{pas(N56 / 'N5_meilleur_pas.txt')}.pt",
       "N6": N56 / "N6" / "snap" / f"pas_{pas(N56 / 'N6_meilleur_pas.txt')}.pt",
       "N7": N7D / "N7" / "snap" / f"pas_{pas(N7D / 'N7_meilleur_pas.txt')}.pt"}
MEL = [("N7N5_075", "N7", "N5", 0.75), ("N7N5_050", "N7", "N5", 0.5), ("N7N6_050", "N7", "N6", 0.5),
       ("N5N6_050", "N5", "N6", 0.5), ("N5N6_075", "N5", "N6", 0.75)]


def log(m):
    l = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {m}"
    print(l, flush=True)
    with open(SORTIE / "chaine.log", "a", encoding="utf-8") as f:
        f.write(l + "\n")


def melanger(nom, A, B, a):
    out = SORTIE / "poids" / f"{nom}.pt"
    if out.exists():
        return out
    ca = torch.load(SRC[A], map_location="cpu", weights_only=False); cb = torch.load(SRC[B], map_location="cpu", weights_only=False)
    na, nb = ca["net"], cb["net"]
    if list(na) != list(nb) or any(na[k].shape != nb[k].shape for k in na):
        raise SystemExit(f"clés ou formes différentes entre {A} et {B}")
    net = {}
    for k in na:
        if na[k].is_floating_point():
            net[k] = (a * na[k].double() + (1 - a) * nb[k].double()).to(na[k].dtype)
        else:
            net[k] = na[k].clone()
    (SORTIE / "poids").mkdir(exist_ok=True)
    torch.save({"net": net, "step": ca.get("step"), "melange": f"{a}*{A}({SRC[A].name}) + {1 - a}*{B}({SRC[B].name})"}, out)
    log(f"{nom} : poids écrits ({len(net)} tenseurs)")
    return out


def lancer(args):
    r = subprocess.run([PY, "-I"] + args, env=ENV)
    if r.returncode:
        raise SystemExit(f"échec {args[0]} code {r.returncode}")


def mesurer(nom, ck):
    rep = SORTIE / "etapes" / f"{nom}.fini"
    if rep.exists():
        return
    t0 = time.time()
    lancer([str(ICI / "inferer_v2.py"), str(ck), nom])
    lancer([str(ICI / "coeurs_mesure.py"), "mesurer", nom, str(ck), "--device", "cuda:0"])
    lancer([str(ICI / "valider_n7.py"), str(SORTIE / f"starless_{nom}"), "--tag", nom])
    shutil.rmtree(SORTIE / f"starless_{nom}", ignore_errors=True)
    rep.write_text(time.strftime("%H:%M:%S") + "\n")
    log(f"{nom} : mesuré en {(time.time() - t0) / 60:.0f} min")


def coeurs_de(root, tag):
    f = root / "coeurs" / f"coeurs_{tag}.json"
    return json.loads(f.read_text(encoding="utf-8"))["moyenne"] if f.exists() else float("nan")


def tableau():
    lignes = (N7D / "tableau_n7.md").read_text(encoding="utf-8").splitlines()
    ref = {l.split("|")[1].strip(): l for l in lignes[2:]}
    out = lignes[:2] + [ref[k] for k in ("N5", "N6", "N7")]
    for nom, A, B, a in MEL:
        if not glob.glob(str(SORTIE / "Resultats" / nom / "*.json")):
            continue
        M.R = SORTIE / "Resultats"
        r = M.ligne(nom)
        c = coeurs_de(SORTIE, nom)
        out.append(f"| {A}/{B} a={a} | {r[1]} | " + " | ".join(f"{x:.3f}" for x in r[2]) + f" | {r[3]:.4f} | {r[4]:.2f} | {r[5]:.3f} | {r[6]:.4f} | "
                   + ("" if c != c else f"{c:.3f}") + " |")
    out += [ref["modèle FLOAT"], ref["RefExt"]]
    txt = "\n".join(out) + "\n"
    (SORTIE / "tableau_melanges.md").write_text(txt, encoding="utf-8")
    print(txt, flush=True)


def vignette(a, top, lab, taille):
    v = np.clip(np.arcsinh(np.clip(a.astype(np.float32), 0, None) / 60) / top, 0, 1)
    im = cv2.resize((v.transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8), (taille, taille), interpolation=cv2.INTER_AREA).copy()
    cv2.putText(im, lab, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return im


def planches(meilleurs):
    """entrée | N7 | deux meilleurs mélanges | RefExt ; même étirement (arcsinh 60) pour toutes les vignettes d'une planche."""
    cols = [("entree", np.load(N7D / "coeurs" / "fen_entree.npy").astype(np.float32)), ("N7", np.load(N7D / "coeurs" / "fen_N7.npy").astype(np.float32))]
    cols += [(m, np.load(SORTIE / "coeurs" / f"fen_{m}.npy").astype(np.float32)) for m in meilleurs]
    cols.append(("RefExt", np.load(N7D / "coeurs" / "fen_RefExt.npy").astype(np.float32)))
    top = float(np.arcsinh(np.percentile(cols[0][1].mean(0), 99.9) / 60)) + 1e-6
    _, buf = cv2.imencode(".jpg", np.hstack([vignette(a, top, l, 420) for l, a in cols]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    buf.tofile(str(SORTIE / "planche_coeur_m31.jpg"))
    nom = "leotriplet_canon6d_newton"
    z7 = np.load(N7D / "crops" / f"N7_{nom}.npz")
    cols = [("entree", z7["entree"]), ("N7", z7["sortie"])]
    for m in meilleurs:
        cols.append((m, np.load(SORTIE / "crops" / f"{m}_{nom}.npz")["sortie"]))
    cols.append(("RefExt", z7["refext"]))
    top = float(np.arcsinh(np.percentile(cols[0][1].mean(0), 99.5) / 60)) + 1e-6
    _, buf = cv2.imencode(".jpg", np.hstack([vignette(a, top, l, 460) for l, a in cols]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    buf.tofile(str(SORTIE / "planche_leo_triplet.jpg"))


def classement():
    """Critère : cœurs >= 0,8 d'abord, puis restantes G<8 + G 11-14, puis fond médian."""
    lig = []
    M.R = SORTIE / "Resultats"
    for nom, A, B, a in MEL:
        if not glob.glob(str(SORTIE / "Resultats" / nom / "*.json")):
            continue
        r = M.ligne(nom)
        lig.append((nom, coeurs_de(SORTIE, nom), r[2][0], r[2][2], r[3]))
    ok = [l for l in lig if l[1] >= 0.8]
    if not ok:      # aucun mélange n'atteint 0,8 : on classe d'abord par cœurs décroissants
        return sorted(lig, key=lambda l: -l[1])
    return sorted(ok, key=lambda l: (l[2] + l[3], l[4]))


if __name__ == "__main__":
    (SORTIE / "etapes").mkdir(parents=True, exist_ok=True)
    for nom, A, B, a in MEL:
        mesurer(nom, melanger(nom, A, B, a))
    tableau()
    cl = classement()
    log("classement : " + " ; ".join(f"{l[0]} coeurs {l[1]:.3f} G<8 {l[2]:.3f} G11-14 {l[3]:.3f} fond {l[4]:.4f}" for l in cl))
    planches([l[0] for l in cl[:2]])
    log("chaîne terminée")
