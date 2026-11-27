# ======================================================================
# mesurer_rl.py
# Role     : Measurement: table N5 / S40 / S80 + references (tableau_n56.md: FLOAT model, RefExt) with a cores column, and contact sheets.
# Stage    : 10. Discarded trial: restoring the slow part (25 Nov) (see scripts/LISEZMOI.md)
# Inputs   : N5 outputs
# Outputs  : tableau_rl.md
# Example  : mesurer_rl.py valider      -> valider_rl.py on each starless_<tag> folder (CPU)
# ======================================================================
r"""Measurement: table N5 / S40 / S80 + references (tableau_n56.md: FLOAT model, RefExt) with a cores column, and contact sheets.

    mesurer_rl.py valider      -> valider_rl.py on each starless_<tag> folder (CPU)
    mesurer_rl.py tableau      -> tableau_rl.md + planches\
"""
import glob, json, os, sys, subprocess, importlib.util
from pathlib import Path
import numpy as np, cv2
ICI = Path(__file__).resolve().parent
OUT = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Restauration lente – 25 nov")
N7 = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles N7 – 25 nov" / "coeurs")
VALR = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Validation étoiles – 22 nov" / "Resultats")
PY = r"python"
SR = ICI.parent
sys.path.insert(0, str(SR)); sys.path.insert(0, str(SR / "Validation – 22 nov"))
_s = importlib.util.spec_from_file_location("mesurer_v33c", SR / "Entrainement v33c – 24 nov" / "mesurer_v33c.py")
M = importlib.util.module_from_spec(_s); _s.loader.exec_module(M)
from liste import LISTE
COEURS = json.loads((SR / "Tour N7 – 25 nov" / "coeurs_liste.json").read_text(encoding="utf-8-sig"))
TAGS = {"N5": "N5 seul", "S40": "N5 + correction sigma 40", "S80": "N5 + correction sigma 80"}


def coeurs(tag):
    r = []
    for nom, x, y in COEURS:
        d0 = Path(LISTE[nom][0])
        ent = np.load(d0 / "input.npy", mmap_mode="r"); sor = np.load(OUT / f"starless_{tag}" / f"{nom}.npy", mmap_mode="r")
        x0, y0 = max(x - 16, 0), max(y - 16, 0)
        e = np.asarray(ent[:, y0:y + 17, x0:x + 17]).mean(0); s = np.asarray(sor[:, y0:y + 17, x0:x + 17]).mean(0)
        yy, xx = np.ogrid[y0:y0 + e.shape[0], x0:x0 + e.shape[1]]
        m = np.hypot(yy - y, xx - x) < 15
        r.append(float(s[m].mean() / e[m].mean()))
    return float(np.mean(r)), r


def valider():
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", CUDA_VISIBLE_DEVICES="", PYTHONIOENCODING="utf-8")
    for t in TAGS:
        if len(glob.glob(str(OUT / "Resultats" / t / "*.json"))) < 14:
            subprocess.run([PY, "-I", str(ICI / "valider_rl.py"), str(OUT / f"starless_{t}"), "--tag", t, "--device", "cpu"], env=env, check=True)


def tableau():
    en = ["| essai | images | " + " | ".join(f"restantes G {b}" for b in M.BANDES) + " | fond méd. | fond p99 | structure méd. | cœurs (sortie/entrée, idéal 1) |", "|---|---|" + "---|" * 8]
    for root, t, nom, cc in ((OUT / "Resultats", "N5", TAGS["N5"], "N5"), (OUT / "Resultats", "S40", TAGS["S40"], "S40"), (OUT / "Resultats", "S80", TAGS["S80"], "S80"),
                             (VALR, "livre", "modèle FLOAT", None), (VALR, "reference", "RefExt", "RefExt")):
        if not glob.glob(str(root / t / "*.json")): continue
        M.R = root
        r = M.ligne(t)
        if cc in ("N5", "S40", "S80"): c = coeurs(cc)[0]
        elif cc == "RefExt": c = json.loads((N7 / "coeurs_RefExt.json").read_text(encoding="utf-8"))["moyenne"]
        else: c = float("nan")
        en.append(f"| {nom} | {r[1]} | " + " | ".join(f"{x:.3f}" for x in r[2]) + f" | {r[3]:.4f} | {r[4]:.2f} | {r[5]:.3f} | " + ("" if c != c else f"{c:.3f}") + " |")
    txt = "\n".join(en); (OUT / "tableau_rl.md").write_text(txt + "\n", encoding="utf-8"); print(txt)
    print("cœurs par cœur:", {t: [round(v, 3) for v in coeurs(t)[1]] for t in TAGS})
    planches()


def planches():
    (OUT / "planches").mkdir(exist_ok=True)
    for nom in ("m31_a7s_newton", "leotriplet_canon6d_newton"):
        d0 = Path(LISTE[nom][0])
        ent = np.load(d0 / "input.npy", mmap_mode="r"); sn = np.load(d0 / "starless.npy", mmap_mode="r")
        C, H, W = ent.shape
        L = np.asarray(ent, np.float32).mean(0)
        lis = cv2.GaussianBlur(np.minimum(L, np.percentile(L[L != 0], 99.9)), (0, 0), 60)
        cy, cx = np.unravel_index(int(np.argmax(lis)), lis.shape)
        h = 400; Y0 = int(np.clip(cy - h, 0, H - 2 * h)); X0 = int(np.clip(cx - h, 0, W - 2 * h))
        cols = [("entree", ent)] + [(t, np.load(OUT / f"starless_{t}" / f"{nom}.npy", mmap_mode="r")) for t in TAGS] + [("RefExt", sn)]
        cr = [(l, np.asarray(a[:, Y0:Y0 + 2 * h, X0:X0 + 2 * h], np.float32)) for l, a in cols]
        top = float(np.arcsinh(np.percentile(cr[0][1].mean(0), 99.5) / 60)) + 1e-6
        tl = []
        for l, a in cr:
            v = np.clip(np.arcsinh(np.clip(a, 0, None) / 60) / top, 0, 1)
            im = cv2.resize((v.transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8), (420, 420), interpolation=cv2.INTER_AREA).copy()
            cv2.putText(im, l, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2); tl.append(im)
        # image entière réduite, même étirement : deuxième rangée
        s = 4
        full = []
        top2 = float(np.arcsinh(np.percentile(L, 99.5) / 60)) + 1e-6
        for l, a in cols:
            sm = np.stack([cv2.resize(np.asarray(a[c], np.float32), (W // s, H // s), interpolation=cv2.INTER_AREA) for c in range(3)])
            v = np.clip(np.arcsinh(np.clip(sm, 0, None) / 60) / top2, 0, 1)
            im = cv2.resize((v.transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8), (420, int(420 * H / W)), interpolation=cv2.INTER_AREA).copy()
            cv2.putText(im, l, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2); full.append(im)
        _, buf = cv2.imencode(".jpg", np.vstack([np.hstack(tl), np.hstack(full)]), [cv2.IMWRITE_JPEG_QUALITY, 90])
        buf.tofile(str(OUT / "planches" / f"{nom}.jpg"))


if __name__ == "__main__":
    {"valider": valider, "tableau": tableau}[sys.argv[1]]()
