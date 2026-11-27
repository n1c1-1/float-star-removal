# ======================================================================
# make_figures.py
# Role     : Draws every figure of the README (docs/figures/*.png) from the data published in docs/figures/data/
#            (made by prepare_data.py from the result folders) and from docs/images/planche_synthetique.jpg.
# Inputs   : docs/figures/data/*.json, *.npz, *.jpg ; docs/images/planche_synthetique.jpg
# Outputs  : docs/figures/*.png (one consistent style, each file <= 400 KB)
# Run      : python docs/figures/make_figures.py        (CPU only, about 1 minute; needs numpy, matplotlib, Pillow)
# ======================================================================
import io
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
IMAGES = HERE.parent / "images"
BANDS = ["<8", "8-11", "11-14", "14-17", ">17"]
LIMIT = 400 * 1024

# ---- one sober style: light background, one palette (colour-blind safe), same fonts everywhere
BG, FG, GRID = "#fbfaf7", "#222222", "#dcdad3"
COL = {"E3": "#8a8a8a", "N2": "#2a9d8f", "N5": "#0072b2", "N6": "#7b6fb0", "N7": "#e69f00", "mix": "#c4402f", "classical": "#111111",
       "N8L": "#56b4e9", "N9": "#a05a2c", "N10": "#009e73"}
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "axes.edgecolor": FG, "axes.labelcolor": FG,
    "xtick.color": FG, "ytick.color": FG, "text.color": FG, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 100, "font.family": "DejaVu Sans",
})


def save(fig, name, dpi=150):
    """Saves a PNG; if it is above 400 KB, shrinks it (palette, then size) until it fits."""
    path = HERE / name
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    scale = 1.0
    while True:
        cur = im if scale == 1.0 else im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
        out = io.BytesIO()
        cur.save(out, format="PNG", optimize=True)
        if out.tell() > LIMIT:
            out = io.BytesIO()
            cur.quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(out, format="PNG", optimize=True)
        if out.tell() <= LIMIT or scale < 0.4:
            break
        scale *= 0.9
    path.write_bytes(out.getvalue())
    print(f"{name}: {out.tell() / 1024:.0f} KB {cur.size}")


def fwd(v):
    return np.arcsinh(np.asarray(v, np.float32) / 3.0) / 10.0


def disp(v, gain=4.0, black=0.04):
    """The display function of the example images: the network's stretch, a fixed gain and black (docs/images/CAPTIONS.md)."""
    return np.clip(fwd(v) * gain + black, 0, 1)


def rgb(a, gain=4.0):
    return np.moveaxis(disp(np.asarray(a, np.float32), gain), 0, -1)


def J(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def box(ax, x, y, w, h, text, fc="#ffffff", ec=FG, fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, lw=1.0, color=FG))


# ------------------------------------------------------------------------------------------------------------------ 1
def fig_overview():
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0.02, 0.76, 0.26, 0.18, "1  Synthetic generator\nbackground first, then stars\n(target = background, exact)", "#e8f1f8")
    box(ax, 0.36, 0.76, 0.20, 0.18, "2  Network E3\n(synthetic data only)", "#eef0f2")
    box(ax, 0.02, 0.44, 0.26, 0.20, "3  Real images + Gaia\nclassical chain v33b:\nsubtract, refill", "#f9ecd6")
    box(ax, 0.02, 0.14, 0.26, 0.20, "4  Judge: weight 0 where\nthe chain failed\n(+ v33g: diffuse light back)", "#f9ecd6")
    box(ax, 0.36, 0.28, 0.24, 0.30, "5  Fine-tuning from E3\non real pairs (weighted L1)\n+ realistic synthetic stars\n+ synthetic galaxy cores (N7)", "#f4dede")
    box(ax, 0.68, 0.28, 0.20, 0.30, "6  Weight average\nN7 x 0.75 + N5 x 0.25", "#f4dede")
    box(ax, 0.68, 0.04, 0.30, 0.20, "7  Two-scale inference\nwhole image by tiles;\nhalo zones of G < 6.5 at 1/2 scale", "#dff0e6")
    box(ax, 0.36, 0.06, 0.24, 0.16, "Output: starless image\n+ stars layer (linear)", "#ffffff", bold=True)
    arrow(ax, 0.28, 0.85, 0.36, 0.85)
    arrow(ax, 0.15, 0.44, 0.15, 0.34)
    arrow(ax, 0.28, 0.24, 0.36, 0.36)
    arrow(ax, 0.46, 0.76, 0.46, 0.58)
    arrow(ax, 0.60, 0.43, 0.68, 0.43)
    arrow(ax, 0.78, 0.28, 0.78, 0.24)
    arrow(ax, 0.68, 0.14, 0.60, 0.14)
    ax.text(0.02, 0.985, "Gaia catalogue positions every real-image subtraction (3) and every halo-zone decision (7).", fontsize=7.5, va="top", style="italic")
    fig.tight_layout()
    save(fig, "fig01_overview.png")


# ------------------------------------------------------------------------------------------------------------------ 2
def fig_representation():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    x = np.linspace(-5, 3000, 4000)
    a.plot(x, fwd(x), color=COL["N5"], lw=1.6, label=r"$y=\mathrm{asinh}(v/3)/10$")
    a.plot(x, x / 30.0, color=GRID, lw=1.0, ls="--", label="linear, same slope at 0")
    a.set_xscale("symlog", linthresh=3); a.set_ylim(-0.05, 0.95)
    a.set_xlabel("linear value $v$ (sky-noise units, sky = 0)"); a.set_ylabel("network input $y$")
    a.set_title("(a) Fixed training stretch"); a.legend(fontsize=7.5, loc="upper left")
    r = np.logspace(-0.3, 2.6, 400)                       # px
    fwhm, beta = 3.0, 3.0
    alpha = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
    core = (1 + (r / alpha) ** 2) ** (-beta)
    r0, gam = 12.0, 2.0
    halo = 0.02 * (1 + (r / r0) ** 2) ** (-gam / 2)
    prof = 3000 * (0.98 * core + halo / halo[0] * halo[0])
    tot = np.minimum(prof, 600)
    b.loglog(r, prof, color=COL["N7"], lw=1.4, label="unclipped star")
    b.loglog(r, 3000 * 0.98 * core, color=GRID, lw=1.0, label="core (Moffat)")
    b.loglog(r, 3000 * halo, color=COL["N2"], lw=1.0, ls="--", label=r"aureole $\propto(1+(r/r_0)^2)^{-\gamma/2}$")
    b.loglog(r, tot, color=COL["mix"], lw=1.4, label="after clipping (saturation)")
    b.axhline(1, color=FG, lw=0.6, ls=":"); b.text(1.4, 1.2, "1 noise unit", fontsize=7)
    b.set_ylim(0.1, 6000); b.set_xlabel("radius (px)"); b.set_ylabel("value (noise units)")
    b.set_title("(b) Synthetic star profile (illustrative)"); b.legend(fontsize=6.5, loc="upper right")
    fig.tight_layout()
    save(fig, "fig02_representation.png")


# ------------------------------------------------------------------------------------------------------------------ 3
def fig_synthetic():
    im = Image.open(IMAGES / "planche_synthetique.jpg").convert("RGB")
    h = im.height // 2
    sheet = Image.new("RGB", (im.width * 2 + 16, h), (251, 250, 247))     # the six rows laid out as two columns of three
    sheet.paste(im.crop((0, 0, im.width, h)), (0, 0)); sheet.paste(im.crop((0, h, im.width, im.height)), (im.width + 16, 0))
    fig, ax = plt.subplots(figsize=(7.4, 7.4 * sheet.height / sheet.width + 0.45))
    ax.imshow(sheet); ax.axis("off"); ax.grid(False)
    ax.set_title("Six synthetic examples. Each row: input | target (the background itself, exact) | stars removed (difference)", fontsize=8)
    fig.tight_layout()
    save(fig, "fig03_synthetic_examples.png", dpi=200)


# ------------------------------------------------------------------------------------------------------------------ 4
def fig_chain():
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    xs = [0.01, 0.21, 0.41, 0.61, 0.81]
    labels = ["Image sheet\n(measured once on the\nwhole image):\nastrometry, Gaia G<21,\nPSF model, saturation,\nspike angles",
              "Track 1: forced\nphotometry\nfree flux at each Gaia\nposition; subtract the\nstar model. No filling.",
              "Step A: big stars\nown radial profile,\nspike templates,\nsaturated core\nre-created from rim",
              "Step B: final passes\nrefill leftovers near\ncatalogue stars and\nspikes; isolated pixels\nhealed; grain restored",
              "Judge: weights\n1 good, 0.5 repaired,\n0 failed zone\n(never edits the target)"]
    cols = ["#eef0f2", "#f9ecd6", "#f9ecd6", "#f9ecd6", "#dff0e6"]
    for x, t, c in zip(xs, labels, cols):
        box(ax, x, 0.30, 0.17, 0.60, t, c, fs=7.2)
    for x in xs[:-1]:
        arrow(ax, x + 0.17, 0.60, x + 0.20, 0.60)
    ax.text(0.5, 0.14, "Guard per zone: light removed / light expected from the catalogue stars must lie in 0.7-1.5, else the zone is dropped.\n"
            "Cost: about 4 minutes per 512 px tile (13 minutes on average on CPU in our baseline run, 3-44 minutes).", ha="center", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig04_classical_chain.png")


# ------------------------------------------------------------------------------------------------------------------ 5
def fig_tiles():
    fig, axs = plt.subplots(2, 3, figsize=(7.4, 5.1))
    for row, im in enumerate(("m31_a7s_newton", "ic405_a7s_newton")):
        z = np.load(DATA / f"tile_{im}.npz")
        for col, (k, t) in enumerate((("entree", "input"), ("classique", "classical chain v33b"), ("N7N5", "network, N7/N5 0.75 mix"))):
            ax = axs[row, col]
            ax.imshow(rgb(z[k], 8.0)); ax.axis("off"); ax.grid(False)
            if row == 0:
                ax.set_title(t, fontsize=8.5)
        axs[row, 0].text(0.02, 0.97, "M31, A7S + Newton" if row == 0 else "IC 405, A7S + Newton", transform=axs[row, 0].transAxes, color="white", fontsize=7.5, va="top")
    fig.suptitle("The same 512 px tiles of Nicolas's images, same strong stretch (gain 8)", fontsize=9)
    fig.tight_layout()
    save(fig, "fig05_classical_vs_network_tiles.png")


# ------------------------------------------------------------------------------------------------------------------ 6
def fig_pair():
    z = np.load(DATA / "pair_example.npz")
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.8))
    axs[0].imshow(rgb(z["input"], 2.5)); axs[0].set_title("input")
    axs[1].imshow(rgb(z["target"], 2.5)); axs[1].set_title("target")
    w = np.asarray(z["weight"], np.float32)
    axs[2].imshow(w, cmap="gray", vmin=0, vmax=1); axs[2].set_title("weight (1 white, 0 black)")
    for a in axs:
        a.axis("off"); a.grid(False)
    fig.suptitle("A training pair from Nicolas's Horsehead stack (256 px cut-out of a 512 px tile)", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig06_training_pair_weights.png")


# ------------------------------------------------------------------------------------------------------------------ 7
def fig_training():
    H = J("histories.json")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for k, c, best in (("N5", COL["N5"], 23000), ("N7", COL["N7"], 22000), ("N8L", COL["N8L"], None), ("N6", COL["N6"], 20000)):
        h = H[k]
        st = [x["pas"] for x in h]
        a.plot(st, [x["val_etoiles"] for x in h], color=c, lw=1.4, label=k)
        b.plot(st, [x["val"] for x in h], color=c, lw=1.4, label=k)
        if best:
            y = [x["val_etoiles"] for x in h if x["pas"] == best][0]
            a.plot([best], [y], "o", color=c, ms=5, mfc="white")
    a.set_ylim(0.012, 0.03)
    a.set_xlabel("training step"); a.set_ylabel("validation loss under stars (L1, stretched units)")
    a.set_title("(a) Held-out tiles, on the star footprint"); a.legend(fontsize=7.5)
    b.set_xlabel("training step"); b.set_ylabel("validation loss, whole tile")
    b.set_ylim(0.0028, 0.0042); b.set_title("(b) Held-out tiles, whole tile")
    fig.text(0.01, 0.005, "Open circles: the snapshot kept (lowest loss under stars). 6 held-out tiles; batch 8, 34 500 steps (N8L: 24 000), about 5 h on one 16 GB GPU.", fontsize=6.8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "fig07_training_curves.png")


# ------------------------------------------------------------------------------------------------------------------ 8
def band_curve(agg):
    return [agg[b] if agg.get(b) is not None else np.nan for b in BANDS]


def tile_aggregate(label):
    T = J("baseline_tiles.json")["images"]
    out = {}
    for b in BANDS:
        v = [im[label]["etoiles_restantes"][b]["part_gt_0.1"] for im in T.values()
             if im[label]["etoiles_restantes"].get(b, {}).get("n", 0) >= 5 and "part_gt_0.1" in im[label]["etoiles_restantes"][b]]
        out[b] = float(np.mean(v)) if v else None
    return out


def fig_residual():
    V = J("validation.json")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
    xs = np.arange(len(BANDS))
    for k, c, key in (("E3", COL["E3"], "E3"), ("N2", COL["N2"], "N2"), ("N5", COL["N5"], "N5"), ("N7", COL["N7"], "N7"), ("N7/N5 0.75", COL["mix"], "N7/N5 0.75")):
        a.plot(xs, band_curve(V[key]["aggregate"]), "-o", color=c, ms=3.5, lw=1.4, label=k)
    a.set_title("(a) 13 validation images")
    for label, c, name in (("classique", COL["classical"], "classical chain v33b"), ("E3", COL["E3"], "E3"), ("N5", COL["N5"], "N5"), ("N7", COL["N7"], "N7"), ("N7N5", COL["mix"], "N7/N5 0.75")):
        y = np.array(band_curve(tile_aggregate(label)), float)
        y[:2] = np.nan                      # G < 8: no tile with 5 stars; G 8-11: 3 tiles only, drawn apart below
        b.plot(xs, y, "-o" if label != "classique" else "--s", color=c, ms=3.5, lw=1.4, label=name)
        y8 = band_curve(tile_aggregate(label))[1]
        b.plot([1], [y8], "o" if label != "classique" else "s", mfc="white", color=c, ms=4.5)
    b.set_title("(b) 512 px tiles of 9 images")
    b.text(0.02, 0.66, "open markers: G 8-11, only 3 tiles\n(58 stars), not joined", transform=b.transAxes, fontsize=6.8, style="italic")
    for ax in (a, b):
        ax.set_xticks(xs); ax.set_xticklabels(BANDS); ax.set_xlabel("Gaia G magnitude band (fainter to the right)")
        ax.legend(fontsize=6.8, loc="upper left")
    a.set_ylabel("share of stars keeping > 10 % of their flux")
    b.text(0.02, 0.60, "G < 8: no tile with 5 stars", transform=b.transAxes, fontsize=6.8, style="italic")
    fig.tight_layout()
    save(fig, "fig08_residual_vs_magnitude.png")


# ------------------------------------------------------------------------------------------------------------------ 9
def fig_cores():
    C = J("cores.json")
    names = ["M31 (A7S)", "M32 (A7S)", "M31 (60D)", "M32 (60D)", "Leo core 1", "Leo core 2", "mean"]
    models = [("E3", COL["E3"]), ("N5", COL["N5"]), ("N7", COL["N7"]), ("N7/N5 0.75", COL["mix"]), ("N9 step 10000", COL["N9"])]
    models = [m for m in models if m[0] in C]
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    w = 0.8 / len(models)
    for i, (m, c) in enumerate(models):
        vals = [x["ratio"] for x in C[m]["cores"]] + [C[m]["mean"]]
        ax.bar(np.arange(7) + (i - len(models) / 2 + 0.5) * w, vals, w, color=c, label=m.replace(" step 10000", " (10 000 steps)"))
    ax.axhline(1, color=FG, lw=0.8, ls=":")
    ax.set_xticks(range(7)); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("flux out / flux in, 15 px disc (ideal 1)"); ax.set_ylim(0, 1.15)
    ax.set_title("Galaxy-core preservation, six cores of three validation images")
    ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "fig09_core_preservation.png")


# ------------------------------------------------------------------------------------------------------------------ 10
def fig_control():
    C = J("control.json")
    ms = [m for m in ("N7", "N7/N5 0.75", "N9 step 10000", "N10 step 10000") if m in C["models"]]
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    rng = np.random.default_rng(1)
    for i, m in enumerate(ms):
        g, s = np.array(C["models"][m]["galaxy_3px"]), np.array(C["models"][m]["star_3px"])
        ax.scatter(i - 0.17 + rng.uniform(-0.07, 0.07, len(g)), g, s=14, color=COL["mix"], alpha=0.8, label="galaxy" if i == 0 else None)
        ax.scatter(i + 0.17 + rng.uniform(-0.07, 0.07, len(s)), s, s=14, color=COL["E3"], alpha=0.8, label="star of the same brightness" if i == 0 else None)
        ax.hlines(np.median(g), i - 0.3, i - 0.04, color=COL["mix"]); ax.hlines(np.median(s), i + 0.04, i + 0.3, color=COL["E3"])
    ax.axhline(1, color=FG, lw=0.7, ls=":"); ax.axhline(0, color=FG, lw=0.7, ls=":")
    ax.set_xticks(range(len(ms))); ax.set_xticklabels([m.replace(" step 10000", "\n(10 000 steps)") for m in ms])
    ax.set_ylabel("flux out / flux in, 3 px disc"); ax.set_ylim(-0.15, 1.25)
    ax.set_title("Compact-galaxy control (18 galaxies): ideal is 1 for galaxies, 0 for stars")
    ax.legend(fontsize=7.5, loc="center right")
    fig.tight_layout()
    save(fig, "fig10_compact_galaxy_control.png")


# ------------------------------------------------------------------------------------------------------------------ 11
def fig_tradeoff():
    V, C = J("validation.json"), J("cores.json")
    keys = [k for k in V if k in C]
    short = {"N7/N5 0.75": "mix 0.75", "N7/N5 0.5": "mix 0.5", "N7/N6 0.5": "N7/N6", "N5/N6 0.5": "N5/N6 .5", "N5/N6 0.75": "N5/N6 .75",
             "N8 step 5000": "N8 5k", "N8 step 10000": "N8 10k", "N8L step 13000": "N8L 13k", "N8L step 24000": "N8L 24k", "N7 step 10000": "N7 10k",
             "N9 step 10000": "N9", "N10 step 10000": "N10"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.8))
    offs = {}
    LABELED = {"E3", "N5", "N6", "N7", "N7/N5 0.75", "N7/N5 0.5", "N9 step 10000", "N10 step 10000", "N8L step 24000", "N7 step 10000"}
    for k in keys:
        x = C[k]["mean"]
        c = COL["mix"] if k == "N7/N5 0.75" else (COL["N5"] if k.startswith("N5") else COL["N7"] if k.startswith("N7") else COL["N9"] if k.startswith("N9") else "#555555")
        for ax, band in ((a, "<8"), (b, "14-17")):
            y = V[k]["aggregate"][band]
            ax.scatter(x, y, s=22, color=c, zorder=3)
            if k not in LABELED:
                continue
            cell = (ax, round(x / 0.12), round(y / (0.02 if ax is a else 0.012)))
            n = offs.get(cell, 0)
            offs[cell] = n + 1
            ax.annotate(short.get(k, k), (x, y), fontsize=6.4, xytext=(4, 3 + 8 * n), textcoords="offset points")
    a.set_ylabel("stars G < 8 keeping > 10 % of flux"); b.set_ylabel("stars G 14-17 keeping > 10 % of flux")
    for ax in (a, b):
        ax.set_xlabel("core preservation (ideal 1)")
    a.set_title("(a) bright stars"); b.set_title("(b) faint stars")
    fig.suptitle("Every variant trades core preservation against stars left in (unlabelled grey points: other N8 / N8L / mix variants)", fontsize=8)
    fig.tight_layout()
    save(fig, "fig11_core_star_tradeoff.png")


# ------------------------------------------------------------------------------------------------------------------ 12
def fig_horsehead():
    z = np.load(DATA / "horsehead_cutouts.npz")
    rows = [("G6.18", "bright star G 6.2"), ("moyenne B", "medium star B"), ("moyenne D", "medium star D")]
    cols = [("entree", "input"), ("x1_N5", "N5"), ("x1_N7", "N7"), ("x1_melange", "N7/N5 0.75 mix")]
    fig, axs = plt.subplots(3, 4, figsize=(7.4, 5.8))
    for i, (rk, rl) in enumerate(rows):
        for j, (ck, cl) in enumerate(cols):
            ax = axs[i, j]
            ax.imshow(rgb(z[f"{rk}|{ck}"], 3.0)); ax.axis("off"); ax.grid(False)
            if i == 0:
                ax.set_title(cl, fontsize=9)
        axs[i, 0].text(0.03, 0.96, rl, transform=axs[i, 0].transAxes, color="white", fontsize=7.5, va="top")
    fig.suptitle("Horsehead, Nicolas's 704 x 15 s stack: saturated faint stars (flat top at about 100 noise units), one scale", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig12_horsehead_failure.png")


# ------------------------------------------------------------------------------------------------------------------ 13
def fig_sheet(im, title, name):
    a = Image.open(DATA / f"sheet_{im}.jpg").convert("RGB")
    a = a.crop((0, 36, a.width, a.height))
    fig, ax = plt.subplots(figsize=(7.4, 7.4 * a.height / a.width + 0.55))
    ax.imshow(a); ax.axis("off"); ax.grid(False)
    w = a.width / 3
    for j, t in enumerate(("input", "output (N7/N5 0.75 mix, two scales)", "input - output")):
        ax.text(w * (j + 0.5), -10, t, ha="center", va="bottom", fontsize=8.5)
    ax.set_title(title, fontsize=9, pad=18)
    fig.tight_layout()
    save(fig, name)


def main():
    fig_overview(); fig_representation(); fig_synthetic(); fig_chain(); fig_tiles(); fig_pair(); fig_training()
    fig_residual(); fig_cores(); fig_control(); fig_tradeoff(); fig_horsehead()
    fig_sheet("m31_a7s_newton", "M31, Sony A7S + Newton 130/900 (Nicolas): whole image, then two enlargements", "fig13_sheet_m31.png")
    fig_sheet("ic405_a7s_newton", "IC 405, Sony A7S + Newton 130/900 (Nicolas): whole image, then two enlargements", "fig14_sheet_ic405.png")


if __name__ == "__main__":
    main()
