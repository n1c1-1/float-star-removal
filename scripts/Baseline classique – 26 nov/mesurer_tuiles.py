# ======================================================================
# mesurer_tuiles.py
# Role     : Measures ANY starless source on the baseline tiles with the validation's own functions; also runs the CPU network inference on the tiles.
# Stage    : 15. Classical baseline versus networks on 512 px tiles (26-27 Nov) (see scripts/README.md)
# Inputs   : classique_<image>.npy and candidats\<source>_<image>.npy in F:\float_ia\Article – 26 nov\baseline\, validation's mesures_entree.json
# Outputs  : resultats_tuiles.json, tableau_tuiles.md, seuils.json, planche_<image>.jpg
# Example  : python -I mesurer_tuiles.py [--inferer E3 N2 N5 N7 N7N5] [--images name ...] [--sans-planches]
# ======================================================================
r"""Measurement of ANY starless source on the baseline tiles, with the SAME functions as valider_reseau.py (imported, not copied:
etoiles_restantes, ouverture, masque_loin, degats_fond; mesures_entree read from the validation's mesures_entree.json).

Pseudo-whole image: the input whose tile (the 512 px CORE, not the margin) is replaced by the candidate; only the Gaia stars and the pixels
of the tile core are scored. The "far from stars" mask is computed on the WHOLE image (as in the validation) then cropped; the threshold of
"bright structure" (90th percentile of the input smoothed with sigma 6, valid pixels) is that of the whole image (cached in seuils.json). The apertures
are read in a window of core +/- 256 px (input outside the core), so a star at the edge of the core sees an unprocessed neighbourhood: counted as is.

Candidates (3, 512, 512):
  classique: F:\float_ia\Article – 26 nov\baseline\classique_<image>.npy (launcher of the chain v33b)
  networks : F:\float_ia\Article – 26 nov\baseline\candidats\<source>_<image>.npy: inference on a 1280 px window around the tile (clipped to the
             image), astro_gpu.stages.star_removal.apply(net, window, cpu, None, tile=1024, overlap=256), one scale, without the
             two-scale correction, then cropping of the core. CPU only.
usage: mesurer_tuiles.py [--inferer E3 N2 N5 N7 N7N5] [--images name ...] [--sans-planches]
Writes baseline\resultats_tuiles.json and baseline\tableau_tuiles.md (over the images that have a classical candidate)."""
import os, sys, json, time, hashlib
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "4"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
import numpy as np
import cv2
SR = Path(r"C:\Users\Nicolas\sandbox\starRemoval")
ICI = Path(__file__).resolve().parent
OUT = Path(r"F:\float_ia\Article – 26 nov\baseline")
sys.path.insert(0, str(SR / "Validation – 22 nov")); sys.path.insert(0, str(SR)); sys.path.insert(0, r"C:\Users\Nicolas\sandbox\astro")
import valider_reseau as VR            # (positionne CUDA_VISIBLE_DEVICES=0 a l'import : remis a vide juste apres ; torch n'est importe que plus bas)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"       # (vide = variable supprimee sous Windows : "-1")
os.environ["ASTRO_GPU_OFFLINE"] = "1"
from liste import LISTE

T = 512
MODELES = {
    "E3": Path(r"F:\astro_essais\starRemoval\train\E3\snap\step_20000.pt"),
    "N2": Path(r"F:\float_ia\Entrainement étoiles nuit – 24 nov\N2\snap\pas_26000.pt"),
    "N5": Path(r"F:\float_ia\Entrainement étoiles N5 N6 – 25 nov\N5\snap\pas_23000.pt"),
    "N7": Path(r"F:\float_ia\Entrainement étoiles N7 – 25 nov\N7\snap\pas_22000.pt"),
    "N7N5": Path(r"F:\float_ia\Melanges N5 N6 N7 – 26 nov\poids\N7N5_075.pt"),
}
SHA_N7N5 = "9c88c524212b4523849115fefeb07a02c5289424d8d3a50578ad8fec36e2199b"
SOURCES = ["classique"] + list(MODELES)
tuiles = json.loads((ICI / "tuiles.json").read_text(encoding="utf-8"))
(OUT / "candidats").mkdir(parents=True, exist_ok=True)


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


def candidat_path(src, nom):
    return OUT / ("classique_%s.npy" % nom) if src == "classique" else OUT / "candidats" / f"{src}_{nom}.npy"


def noms_tuiles(images=None):
    return [n for n in LISTE if n in tuiles and tuiles[n].get("x0") is not None and (not images or n in images)]


# ----------------------------------------------------------------------------------------------------- reseaux
def inferer(sources, noms):
    import torch
    torch.set_num_threads(4)
    assert not torch.cuda.is_available(), "le GPU ne doit pas etre visible"
    dev = torch.device("cpu")
    for s in sources:
        if s == "N7N5":
            h = hashlib.sha256(MODELES[s].read_bytes()).hexdigest()
            assert h == SHA_N7N5, f"sha256 du melange {h}"
            log("N7N5 : sha256 conforme au fichier publie")
        net, info = VR.charger_reseau(MODELES[s], dev)
        log(f"{s} charge : {info}")
        for nom in noms:
            f = candidat_path(s, nom)
            if f.exists():
                continue
            x0, y0 = tuiles[nom]["x0"], tuiles[nom]["y0"]
            a = np.load(Path(LISTE[nom][0]) / "input.npy", mmap_mode="r")
            C, H, W = a.shape
            wx0, wx1, wy0, wy1 = max(0, x0 - 384), min(W, x0 + T + 384), max(0, y0 - 384), min(H, y0 + T + 384)
            win = np.ascontiguousarray(a[:, wy0:wy1, wx0:wx1], np.float32)
            t0 = time.time()
            out = VR.inference(net, win, dev)
            core = np.ascontiguousarray(out[:, y0 - wy0:y0 - wy0 + T, x0 - wx0:x0 - wx0 + T], np.float32)
            np.save(f, core)
            dt = time.time() - t0
            log(f"  {s} {nom} fenetre {win.shape} en {dt:.0f} s")
            with (OUT / "candidats" / "temps_inference.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(dict(source=s, image=nom, fenetre=list(win.shape), secondes=round(dt, 1)), ensure_ascii=False) + "\n")
        del net


# ----------------------------------------------------------------------------------------------------- mesure
def seuil_structure(nom, a):
    f = OUT / "seuils.json"
    d = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    if nom not in d:
        C, H, W = a.shape
        L = np.zeros((H, W), np.float32)
        for y in range(0, H, 512):
            L[y:y + 512] = np.asarray(a[:, y:y + 512], np.float32).mean(0)
        lis = cv2.GaussianBlur(L, (0, 0), 6)
        d[nom] = float(np.percentile(lis[L != 0], 90))
        f.write_text(json.dumps(d, indent=1), encoding="utf-8")
    return d[nom]


_CTX = {}


def contexte(nom):
    if nom in _CTX:
        return _CTX[nom]
    x0, y0 = tuiles[nom]["x0"], tuiles[nom]["y0"]
    a = np.load(Path(LISTE[nom][0]) / "input.npy", mmap_mode="r")
    C, H, W = a.shape
    gx, gy, gg = VR.charger_gaia(nom, (H, W))
    me = json.loads((VR.DATA / nom / "mesures_entree.json").read_text(encoding="utf-8"))
    fw, bruit = me["fwhm_px"], me["bruit_pixel"]
    g_det = min(me["g_detectable"], 19.0)
    far, fac = VR.masque_loin((H, W), gx, gy, gg, fw, g_det)
    P = 256
    wx0, wx1, wy0, wy1 = max(0, x0 - P), min(W, x0 + T + P), max(0, y0 - P), min(H, y0 + T + P)
    win = np.ascontiguousarray(a[:, wy0:wy1, wx0:wx1], np.float32)
    k = (gx >= x0) & (gx < x0 + T) & (gy >= y0) & (gy < y0 + T)           # etoiles du coeur seulement
    c = dict(x0=x0, y0=y0, fw=fw, bruit=bruit, fac=fac, win=win, off=(wx0, wy0), seuil=seuil_structure(nom, a), shape=(C, H, W),
             far=far[y0:y0 + T, x0:x0 + T] & (win[:, y0 - wy0:y0 - wy0 + T, x0 - wx0:x0 - wx0 + T].mean(0) != 0),
             sx=gx[k] - wx0, sy=gy[k] - wy0, sg=gg[k], n_gaia_coeur=int(k.sum()))
    _CTX.clear()
    _CTX[nom] = c
    return c


def mesurer(nom, core):
    c = contexte(nom)
    wx0, wy0 = c["off"]
    x0, y0 = c["x0"], c["y0"]
    sl = (slice(None), slice(y0 - wy0, y0 - wy0 + T), slice(x0 - wx0, x0 + T - wx0))
    out = c["win"].copy()
    out[sl] = core
    L_in, L_out = c["win"].mean(0), out.mean(0)
    r = {"image": nom, "tuile": [x0, y0], "n_gaia_coeur_tous_G": c["n_gaia_coeur"]}
    r["etoiles_restantes"] = VR.etoiles_restantes(L_in, L_out, c["sx"], c["sy"], c["sg"], c["fw"], c["bruit"])
    lis = cv2.GaussianBlur(L_in, (0, 0), 6)[sl[1:]]
    D = np.abs(core - c["win"][sl]).mean(0)
    r["fond"] = {"part_loin": round(float(c["far"].mean()), 4), **VR.degats_fond(D, c["far"], lis > c["seuil"])}
    return r


def contact(nom, noms_src, dest):
    c = contexte(nom)
    wx0, wy0 = c["off"]
    x0, y0 = c["x0"], c["y0"]
    inp = c["win"][:, y0 - wy0:y0 - wy0 + T, x0 - wx0:x0 - wx0 + T]
    cand = [np.load(candidat_path(s, nom)) for s in noms_src]
    top = float(np.percentile(np.arcsinh(np.clip(inp.mean(0), 0, None) / 3.0), 99.97)) + 1e-6

    def rgb(a):
        return (np.clip(np.arcsinh(np.clip(a, 0, None) / 3.0) / top, 0, 1).transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)

    sheet = np.hstack([rgb(inp)] + [rgb(x) for x in cand])
    sheet = np.vstack([np.zeros((30, sheet.shape[1], 3), np.uint8), sheet])
    cv2.putText(sheet, f"{nom} tuile ({x0},{y0}) : entree | " + " | ".join(noms_src), (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    for q in (92, 88, 84, 80, 75, 70, 60):
        ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, q])
        if len(buf) <= 400 * 1024:
            break
    buf.tofile(str(dest))
    return q, len(buf)


def agreger(res):
    """Medianes sur les images (mediane par image d'abord) ; seules les images ayant un classique, donc memes tuiles pour toutes les sources."""
    agg = {}
    for s in SOURCES:
        rows = [r[s] for r in res.values() if s in r]
        a = {"images": len(rows), "bandes": {}}
        for b, *_ in VR.BANDES:
            v = [x["etoiles_restantes"][b]["mediane"] for x in rows if "mediane" in x["etoiles_restantes"].get(b, {})]
            a["bandes"][b] = {"mediane_des_images": round(float(np.median(v)), 4) if v else None, "images": len(v)}
        for key in ("fond", "structure_brillante"):
            m = [x["fond"][key]["mediane"] for x in rows if "mediane" in x["fond"].get(key, {})]
            p = [x["fond"][key]["p99"] for x in rows if "p99" in x["fond"].get(key, {})]
            a[key] = {"mediane_des_medianes": round(float(np.median(m)), 4) if m else None,
                      "p99_median_des_images": round(float(np.median(p)), 4) if p else None, "images": len(m)}
        agg[s] = a
    return agg


def tableaux(res, agg, noms):
    f = lambda v, d=3: "-" if v is None else f"{v:.{d}f}"
    md = ["# Classical baseline versus networks: same tiles (512 px core)", "",
          f"{len(noms)} images/tiles measured: " + ", ".join(noms) + ".", "",
          "Ratio = remaining flux / original flux at the Gaia positions of the tile core (aperture 2 FWHM, annulus 3-5 FWHM), median per image then median over images.",
          "Background = |output - input| (3 colours averaged, noise units) far from stars; structure = same on the bright structure.", "",
          "## Aggregate (median over images)", "",
          "| source | " + " | ".join(b[0] for b in VR.BANDES) + " | background med | background p99 | structure med |", "|---|" + "---|" * (len(VR.BANDES) + 3)]
    for s in SOURCES:
        a = agg[s]
        if not a["images"]:
            continue
        md.append(f"| {s} | " + " | ".join(f"{f(a['bandes'][b[0]]['mediane_des_images'])} (n_img={a['bandes'][b[0]]['images']})" for b in VR.BANDES)
                  + f" | {f(a['fond']['mediane_des_medianes'])} | {f(a['fond']['p99_median_des_images'], 2)} | {f(a['structure_brillante']['mediane_des_medianes'])} |")
    md += ["", "## Per image: median remaining ratio (n stars)", ""]
    for s in SOURCES:
        md += [f"### {s}", "", "| image | " + " | ".join(b[0] for b in VR.BANDES) + " |", "|---|" + "---|" * len(VR.BANDES)]
        for n in noms:
            r = res[n].get(s)
            if not r:
                continue
            cells = []
            for b, *_ in VR.BANDES:
                e = r["etoiles_restantes"].get(b, {})
                cells.append(f"{e['mediane']:.3f} (n={e['n']})" if "mediane" in e else f"n={e.get('n', 0)}")
            md.append(f"| {n} | " + " | ".join(cells) + " |")
        md.append("")
    md += ["## Per image: background and structure (median / p99)", "", "| image | source | background | structure |", "|---|---|---|---|"]
    fmt = lambda e: f"{e['mediane']:.3f} / {e['p99']:.2f}" if "mediane" in e else f"pixels={e.get('pixels', 0)}"
    for n in noms:
        for s in SOURCES:
            r = res[n].get(s)
            if r:
                md.append(f"| {n} | {s} | {fmt(r['fond']['fond'])} | {fmt(r['fond']['structure_brillante'])} |")
    return "\n".join(md)


def main():
    images = None
    if "--images" in sys.argv:
        images = []
        for a in sys.argv[sys.argv.index("--images") + 1:]:
            if a.startswith("--"):
                break
            images.append(a)
    noms = [n for n in noms_tuiles(images) if candidat_path("classique", n).exists()]
    if "--inferer" in sys.argv:
        i = sys.argv.index("--inferer") + 1
        srcs = []
        while i < len(sys.argv) and not sys.argv[i].startswith("--"):
            srcs.append(sys.argv[i])
            i += 1
        inferer(srcs, noms)
    res = {}
    for nom in noms:
        res[nom] = {}
        for s in SOURCES:
            f = candidat_path(s, nom)
            if f.exists():
                z = np.load(f)
                assert z.shape == (3, T, T), (s, nom, z.shape)
                res[nom][s] = mesurer(nom, z.astype(np.float32))
        log(f"{nom} mesure : {list(res[nom])}")
    agg = agreger(res)
    meta = dict(regle="choisir_tuiles.py", tuiles={n: tuiles[n] for n in noms}, modeles={k: str(v) for k, v in MODELES.items()})
    (ICI / "resultats_tuiles.json").write_text(json.dumps(dict(meta=meta, images=res, agregat=agg), indent=1, ensure_ascii=False), encoding="utf-8")
    (ICI / "tableau_tuiles.md").write_text(tableaux(res, agg, noms), encoding="utf-8")
    if "--sans-planches" not in sys.argv:
        J = json.loads((OUT / "journal.json").read_text(encoding="utf-8")) if (OUT / "journal.json").exists() else {}
        ordre = [n for n in noms_tuiles() if n in J and "echec" not in J[n]][:4]       # regle : les 4 premieres tuiles faites
        for nom in ordre:
            if "N7N5" in res.get(nom, {}) and "classique" in res[nom]:
                q, sz = contact(nom, ["classique", "N7N5"], OUT / f"planche_{nom}.jpg")
                c = contexte(nom)
                x0, y0 = c["x0"], c["y0"]
                wx0, wy0 = c["off"]
                np.savez_compressed(OUT / f"crop_figure_{nom}.npz", entree=c["win"][:, y0 - wy0:y0 - wy0 + T, x0 - wx0:x0 - wx0 + T],
                                    classique=np.load(candidat_path("classique", nom)), N7N5=np.load(candidat_path("N7N5", nom)), x0=x0, y0=y0)
                log(f"planche {nom} q={q} {sz / 1024:.0f} Ko")
    log("tableaux ecrits")


if __name__ == "__main__":
    main()
