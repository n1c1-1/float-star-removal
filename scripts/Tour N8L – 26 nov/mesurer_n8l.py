# ======================================================================
# mesurer_n8l.py
# Role     : N8L measurement (long round) at the best validation step AND at step 24000, tableau_n8l.md, M31 core sheet and Leo sheet.
# Stage    : 10. Discarded trial N8L (cores 10 %, 24,000 steps, 26 Nov) (see scripts/LISEZMOI.md)
# Inputs   : v33g pairs
# Outputs  : snapshots, table
# Example  : python mesurer_n8l.py [arguments]  (see the docstring; paths via variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
r"""N8L measurement (long round) at the best validation step AND at step 24000, tableau_n8l.md, M31 core sheet and Leo sheet.
Copy of mesurer_n8_10000.py. Former: N8 measurement at step 10000 (copy of mesurer_n8_5000.py; N8 step 5000 and N7 step 5000 already measured, read from Resultats and coeurs). Former:
N8 and N7 measurement at step 5000, ONE single process, memory monitored (copy of mesurer_n8.py; originals unchanged).
Starts only if > 8 GB free; before each image (inferer_v2_mem.py) and each core: clean stop under 4 GB.
"""
import sys, runpy, shutil, json, importlib.util, os
import psutil
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ICI = Path(__file__).resolve().parent
SORTIE = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles N8L – 26 nov")
N7 = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles N7 – 25 nov")
N8T = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles N8 – 26 nov" / "tableau_n8.md")
MEL = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Melanges N5 N6 N7 – 26 nov" / "tableau_melanges.md")
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


def pas_choisis():
    hist = json.loads((SORTIE / "N8L" / "history.json").read_text(encoding="utf-8"))
    best = min(hist, key=lambda h: h["val_etoiles"])["pas"]
    fin = max(h["pas"] for h in hist)
    return best, fin


def ligne_tag(M, nom, tag):
    r = M.ligne(tag)
    c = json.loads((SORTIE / "coeurs" / f"coeurs_{tag}.json").read_text(encoding="utf-8"))["moyenne"]
    return f"| {nom} | {r[1]} | " + " | ".join(f"{x:.3f}" for x in r[2]) + f" | {r[3]:.4f} | {r[4]:.2f} | {r[5]:.3f} | {r[6]:.4f} | {c:.3f} |"


def tableau(pas):
    _s = importlib.util.spec_from_file_location("mesurer_v33c", ICI.parent / "Entrainement v33c – 24 nov" / "mesurer_v33c.py")
    M = importlib.util.module_from_spec(_s); _s.loader.exec_module(M); M.R = RES
    lignes = [l for l in N8T.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = lignes[:]            # lignes de tableau_n8.md recopiées telles quelles
    best, fin = pas
    out.append(ligne_tag(M, f"N8L meilleur pas ({best})", f"N8L_{best}"))
    if fin != best:
        out.append(ligne_tag(M, f"N8L pas {fin}", f"N8L_{fin}"))
    txt = "\n".join(out)
    (SORTIE / "tableau_n8l.md").write_text(txt + "\n", encoding="utf-8"); log(txt)


def planche_leo(pas):
    import numpy as np, cv2
    nom = "leotriplet_canon6d_newton"
    best, fin = pas
    cols, sn = [], None
    ent = None
    for lab, f in [("N7", N7 / "crops" / f"N7_{nom}.npz")] + [(f"N8L {p}", SORTIE / "crops" / f"N8L_{p}_{nom}.npz") for p in sorted({best, fin})]:
        if f.exists():
            z = np.load(f)
            if ent is None:
                ent = z["entree"]
            cols.append((lab, z["sortie"])); sn = z["refext"]
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
    for f in ("fen_entree.npy", "fen_N7.npy", "fen_RefExt.npy"):
        if not (SORTIE / "coeurs" / f).exists(): shutil.copy2(N7 / "coeurs" / f, SORTIE / "coeurs" / f)
    pas = pas_choisis()
    log(f"meilleur pas de validation {pas[0]} ; dernier pas {pas[1]}")
    for p in sorted({pas[0], pas[1]}):
        if not (RES / f"N8L_{p}" / "resume.json").exists():
            snap(f"N8L_{p}", SORTIE / "N8L" / "snap" / f"pas_{p}.pt")
    tableau(pas)
    lancer("coeurs_mesure.py", "planche")
    planche_leo(pas)
    log("termine")
