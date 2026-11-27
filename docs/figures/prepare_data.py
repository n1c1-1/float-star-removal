# ======================================================================
# prepare_data.py
# Role     : Collects, from the working disks of the author, the small data files that make_figures.py plots
#            (validation summaries, training histories, core / control measurements, small image cut-outs
#            of Nicolas's own images). The collected files are published in docs/figures/data/.
# Inputs   : result folders on the author's disks (paths below; third parties do not need to run this script)
# Outputs  : docs/figures/data/*.json, docs/figures/data/*.npz
# Run      : python prepare_data.py     (CPU only, a few seconds; reads only, writes only docs/figures/data)
# ======================================================================
import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = Path(__file__).resolve().parent
OUT = HERE / "data"
OUT.mkdir(exist_ok=True)
F = Path(r"F:\float_ia")
VAL = F / "Validation étoiles – 22 nov" / "Resultats"
BANDS = ["<8", "8-11", "11-14", "14-17", ">17"]

# label -> result folder (one json per image + resume.json)
MODELS = {
    "E3": VAL / "E3_V2",
    "N2": F / "Portée M27 – 25 nov" / "Resultats" / "N2_v2",
    "N5": F / "Entrainement étoiles N5 N6 – 25 nov" / "Resultats" / "N5",
    "N6": F / "Entrainement étoiles N5 N6 – 25 nov" / "Resultats" / "N6",
    "N7": F / "Entrainement étoiles N7 – 25 nov" / "Resultats" / "N7",
    "N7/N5 0.75": F / "Melanges N5 N6 N7 – 26 nov" / "Resultats" / "N7N5_075",
    "N7/N5 0.5": F / "Melanges N5 N6 N7 – 26 nov" / "Resultats" / "N7N5_050",
    "N7/N6 0.5": F / "Melanges N5 N6 N7 – 26 nov" / "Resultats" / "N7N6_050",
    "N5/N6 0.5": F / "Melanges N5 N6 N7 – 26 nov" / "Resultats" / "N5N6_050",
    "N5/N6 0.75": F / "Melanges N5 N6 N7 – 26 nov" / "Resultats" / "N5N6_075",
    "N8 step 5000": F / "Entrainement étoiles N8 – 26 nov" / "Resultats" / "N8_5000",
    "N8 step 10000": F / "Entrainement étoiles N8 – 26 nov" / "Resultats" / "N8_10000",
    "N8L step 13000": F / "Entrainement étoiles N8L – 26 nov" / "Resultats" / "N8L_13000",
    "N8L step 24000": F / "Entrainement étoiles N8L – 26 nov" / "Resultats" / "N8L_24000",
    "N7 step 10000": F / "Entrainement étoiles N9 – 26 nov" / "Resultats" / "N7_10000",
    "N9 step 10000": F / "Entrainement étoiles N9 – 26 nov" / "Resultats" / "N9_10000",
    "N10 step 10000": F / "Entrainement étoiles N10 – 26 nov" / "Resultats" / "N10_10000",
    "N5 + slow correction sigma 40": F / "Restauration lente – 25 nov" / "Resultats" / "S40",
    "N5 + slow correction sigma 80": F / "Restauration lente – 25 nov" / "Resultats" / "S80",
}
HISTORIES = {
    "N2": F / "Entrainement étoiles nuit – 24 nov" / "N2" / "history.json",
    "N5": F / "Entrainement étoiles N5 N6 – 25 nov" / "N5" / "history.json",
    "N6": F / "Entrainement étoiles N5 N6 – 25 nov" / "N6" / "history.json",
    "N7": F / "Entrainement étoiles N7 – 25 nov" / "N7" / "history.json",
    "N8L": F / "Entrainement étoiles N8L – 26 nov" / "N8L" / "history.json",
    "N9": F / "Entrainement étoiles N9 – 26 nov" / "N9" / "history.json",
    "N10": F / "Entrainement étoiles N10 – 26 nov" / "N10" / "history.json",
}
CORES = {
    "N5": F / "Entrainement étoiles N7 – 25 nov" / "coeurs" / "coeurs_N5.json",
    "N6": F / "Entrainement étoiles N7 – 25 nov" / "coeurs" / "coeurs_N6.json",
    "N7": F / "Entrainement étoiles N7 – 25 nov" / "coeurs" / "coeurs_N7.json",
    "N7/N5 0.75": F / "Melanges N5 N6 N7 – 26 nov" / "coeurs" / "coeurs_N7N5_075.json",
    "N8L step 24000": F / "Entrainement étoiles N8L – 26 nov" / "coeurs" / "coeurs_N8L_24000.json",
    "N9 step 10000": F / "Entrainement étoiles N9 – 26 nov" / "coeurs" / "coeurs_N9_10000.json",
    "N10 step 10000": F / "Entrainement étoiles N10 – 26 nov" / "coeurs" / "coeurs_N10_10000.json",
    "N7/N5 0.5": F / "Melanges N5 N6 N7 – 26 nov" / "coeurs" / "coeurs_N7N5_050.json",
    "N7/N6 0.5": F / "Melanges N5 N6 N7 – 26 nov" / "coeurs" / "coeurs_N7N6_050.json",
    "N5/N6 0.5": F / "Melanges N5 N6 N7 – 26 nov" / "coeurs" / "coeurs_N5N6_050.json",
    "N5/N6 0.75": F / "Melanges N5 N6 N7 – 26 nov" / "coeurs" / "coeurs_N5N6_075.json",
    "N8 step 5000": F / "Entrainement étoiles N8 – 26 nov" / "coeurs" / "coeurs_N8_5000.json",
    "N8 step 10000": F / "Entrainement étoiles N8 – 26 nov" / "coeurs" / "coeurs_N8_10000.json",
    "N8L step 13000": F / "Entrainement étoiles N8L – 26 nov" / "coeurs" / "coeurs_N8L_13000.json",
    "N7 step 10000": F / "Entrainement étoiles N9 – 26 nov" / "coeurs" / "coeurs_N7_10000.json",
    "E3": F / "Article – 26 nov" / "coeurs" / "coeurs_E3.json",
}
CONTROL = {
    "N7": F / "Controle galaxies compactes – 26 nov" / "Resultats" / "N7_pas22000.json",
    "N7/N5 0.75": F / "Controle galaxies compactes – 26 nov" / "Resultats" / "melange_N7N5_075.json",
    "N9 step 10000": F / "Controle galaxies compactes – 26 nov" / "Resultats" / "N9.json",
    "N10 step 10000": F / "Controle galaxies compactes – 26 nov" / "Resultats" / "N10.json",
}


def scrub(text):
    """Removes drive paths of the author's machine from the published json files."""
    import re
    return re.sub(r'[A-Za-z]:\\\\[^"]*', "<path>", text)


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def aggregate(images):
    """The rule of the published tables: per band, the mean over images of the share of stars keeping > 10 % of their flux
    (images with at least 5 stars in the band); background and structure damage: median over images."""
    row = {}
    for b in BANDS:
        v = [im["etoiles_restantes"][b]["part_gt_0.1"] for im in images
             if im["etoiles_restantes"].get(b, {}).get("n", 0) >= 5 and "part_gt_0.1" in im["etoiles_restantes"][b]]
        row[b] = float(np.mean(v)) if v else None
        m = [im["etoiles_restantes"][b]["mediane"] for im in images
             if im["etoiles_restantes"].get(b, {}).get("n", 0) >= 5 and "mediane" in im["etoiles_restantes"][b]]
        row["median_ratio " + b] = float(np.median(m)) if m else None
    row["background_median"] = float(np.median([im["fond"]["fond"]["mediane"] for im in images]))
    row["background_p99"] = float(np.median([im["fond"]["fond"]["p99"] for im in images]))
    sb = [im["fond"]["structure_brillante"]["mediane"] for im in images
          if im["fond"].get("structure_brillante") and "mediane" in im["fond"]["structure_brillante"]]
    row["structure_median"] = float(np.median(sb)) if sb else None
    row["n_images"] = len(images)
    return row


def validation():
    out = {}
    for label, folder in MODELS.items():
        images = [load(f) for f in sorted(glob.glob(str(folder / "*.json"))) if not f.endswith("resume.json")]
        images = [d for d in images if "etoiles_restantes" in d]
        slim = {}
        for im in images:
            slim[im["image"]] = {
                "fwhm_px": im.get("fwhm_px"),
                "bands": {b: im["etoiles_restantes"].get(b, {}) for b in BANDS},
                "background": im["fond"]["fond"],
                "structure": im["fond"].get("structure_brillante"),
            }
        out[label] = {"source_folder": folder.name, "aggregate": aggregate(images), "images": slim}
    (OUT / "validation.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("validation.json", {k: v["aggregate"]["n_images"] for k, v in out.items()})


def histories():
    out = {k: load(p) for k, p in HISTORIES.items()}
    (OUT / "histories.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def cores():
    out = {}
    for k, p in CORES.items():
        d = load(p)
        out[k] = {"mean": d["moyenne"], "cores": [{"image": c["image"], "ratio": c["rapport"], "flux_in": c["flux_entree"]} for c in d["coeurs"]]}
    (OUT / "cores.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def control():
    ctl = load(F / "Controle galaxies compactes – 26 nov" / "controle.json")
    # numbers only: no pixel of these third-party images is published
    out = {"galaxies": [{"id": g["id"], "fwhm_ratio": g["rapport_fwhm"], "peak_noise": g["crete_bruits"]} for g in ctl["galaxies"]], "models": {}}
    for k, p in CONTROL.items():
        d = load(p)
        out["models"][k] = {"summary": d["resume"],
                            "galaxy_3px": [g["galaxie"]["rapport_3px"] for g in d["galaxies"]],
                            "star_3px": [g["etoile"]["rapport_3px"] for g in d["galaxies"]],
                            "galaxy_8px": [g["galaxie"]["rapport_8px"] for g in d["galaxies"]],
                            "star_8px": [g["etoile"]["rapport_8px"] for g in d["galaxies"]]}
    (OUT / "control.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def horsehead():
    """Cut-outs (256 px) of Nicolas's own Horsehead stack and of the one-scale outputs of E3, N5, N7 and the 0.75 mix."""
    D = F / "Diagnostic Tete de cheval – 26 nov" / "travail"
    sel = {"G6.18": 2, "moyenne B": 4, "moyenne D": 6}
    star = {}
    for name, i in sel.items():
        z = np.load(D / f"crop_{i}.npz")
        cx, cy = int(z["c"][0]), int(z["c"][1])      # c = (x, y) of the star in the crop
        y0, x0 = max(cy - 128, 0), max(cx - 128, 0)
        for k in ("entree", "x1_N5", "x1_N7", "x1_melange"):
            star[f"{name}|{k}"] = z[k][:, y0:y0 + 256, x0:x0 + 256].astype(np.float16)
    np.savez_compressed(OUT / "horsehead_cutouts.npz", **star)
    r = load(F / "Diagnostic Tete de cheval – 26 nov" / "resultats.json")
    r["tableau"] = {k: {c: v for c, v in row.items() if "FLOAT" not in c} if isinstance(row, dict) else row for k, row in r["tableau"].items()} \
        if isinstance(r["tableau"], dict) else r["tableau"]
    (OUT / "horsehead_results.json").write_text(scrub(json.dumps(r, indent=1, ensure_ascii=False)), encoding="utf-8")


def pair_example():
    z = np.load(F / "Paires étoiles v33 – 23 nov" / "horsehead_g3574_1666_debruitee.npz")
    np.savez_compressed(OUT / "pair_example.npz", input=z["input"][:, 128:384, 128:384].astype(np.float16),
                        target=z["target"][:, 128:384, 128:384].astype(np.float16), weight=z["weight"][128:384, 128:384].astype(np.float16))


def sheets():
    """Whole-image contact sheets (input | output | difference) of the 0.75 mix on two of Nicolas's own images, and the
    512 px tile crops (input | classical chain | mix) of the same two images."""
    for im in ("m31_a7s_newton", "ic405_a7s_newton"):
        shutil.copy(F / "Melanges N5 N6 N7 – 26 nov" / "Resultats" / "N7N5_075" / f"{im}_planche.jpg", OUT / f"sheet_{im}.jpg")
        z = np.load(F / "Article – 26 nov" / "baseline" / f"crop_figure_{im}.npz")
        np.savez_compressed(OUT / f"tile_{im}.npz", **{k: z[k].astype(np.float16) for k in ("entree", "classique", "N7N5")})


def baseline():
    b = Path(r"C:\Users\Nicolas\sandbox\starRemoval\Article – 26 nov\baseline")
    for src, dst in (("resultats_tuiles.json", "baseline_tiles.json"), ("tuiles.json", "baseline_tile_choice.json")):
        (OUT / dst).write_text(scrub((b / src).read_text(encoding="utf-8")), encoding="utf-8")


if __name__ == "__main__":
    validation(); histories(); cores(); control(); horsehead(); pair_example(); sheets(); baseline()
