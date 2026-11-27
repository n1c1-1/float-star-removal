# Sources of the numbers and figures of the README

Everything under `docs/figures/data/` is published. It was collected from the author's result folders by `prepare_data.py`
(the paths of those folders are written at the top of that script; a third party does not need them to redraw the figures).
`python docs/figures/make_figures.py` redraws every PNG from `data/` and from `docs/images/planche_synthetique.jpg`.

## Tables

| README item | file | how |
|---|---|---|
| Main table 5.1 (E3, N2, N5, N7, mix: star shares, background, structure) | `data/validation.json`, key `aggregate` of each model | share of stars keeping more than 10 % of their flux, mean over images with at least 5 stars in the band; background and structure: median over images. Same rule as `scripts/Entrainement v33c – 24 nov/mesurer_v33c.py` (`ligne`); the N5, N7, mix and N2 rows reproduce the published tables in `resultats/`. The E3 row is new (26-27 Nov): E3 step 20 000, two-scale inference on CPU, `scripts/Mesure E3 – 26 nov/` |
| Cores column (5.1, Fig. 9, Fig. 11) | `data/cores.json` | flux out / flux in in a 15 px disc, six cores, mean; E3 measured with `scripts/Mesure E3 – 26 nov/coeurs_cpu.py` (one-scale, as for the other models) |
| Star counts per band (147, 2 205, 9 997, 12 177, 8 409) | `data/validation.json`, field `bands.<band>.n` summed over the 13 images | |
| FWHM column of the validation table | `data/validation.json`, field `fwhm_px` | |
| Section 5.2 (classical baseline and same-tile networks) | `data/baseline_tiles.json` (full per-image, per-band results), `data/baseline_tile_choice.json` (tiles), `scripts/Baseline classique – 26 nov/` | aggregate recomputed in `make_figures.py` (`tile_aggregate`) with the same rule as the main table. Time: 3 h 40 total, 2 h 01 for the chain alone (`scripts/Baseline classique – 26 nov/README.md`) |
| Ablation table 5.3 | `data/validation.json` (all models), `data/cores.json` | rows N3 and N4 come from the one-scale tables of the 24-25 Nov runs (`resultats/`, `docs/LESSONS.md`) |
| Compact-galaxy control 5.4 | `data/control.json` | medians over 18 galaxies, 3 px ratio |
| Horsehead table (Section 6) | `data/horsehead_results.json` (the original `resultats.json` of the diagnostic) | residual flux in a 2 FWHM disc, one scale; N5 two-scale values from the same run |
| Training hyper-parameters (batch 8, lr 5e-4, 34 500 steps, 5.3 h, 60 % synthetic) | `scripts/Tour N7 – 25 nov/train_n7.py` and the `config.json` of the run | |
| Training pairs (103 + 35 + 38, 6 held out; 182 files) | `provenance/MANIFESTE_PAIRES.csv` | |
| Provenance verdicts, bit-identical replay | `docs/PROVENANCE.md`, `provenance/` | |
| Target correction v33g (+0.12 to −0.05 noise unit) | `docs/METHOD.md` step 3 | |
| Two-scale inference (12.6 % of the image for M42) | `docs/METHOD.md` step 8 | |
| Speed: 13 min per classical tile, 45 s per network and tile (4 threads); 13 s per whole image on a graphics card | `scripts/Baseline classique – 26 nov/README.md`; the 13 s figure comes from the application's own measurement on M45 (1024 px tiles) and is not reproduced in this repository | |

## Figures

| figure | data |
|---|---|
| fig01 overview, fig04 classical chain | diagrams, no data |
| fig02 representation | the formulas of the README (illustrative profile, not a measurement) |
| fig03 synthetic examples | `docs/images/planche_synthetique.jpg`, regenerated bit-exact from seeds by `exemples/generer_exemples.py` (array hashes unchanged, labels now in English) |
| fig05 classical versus network tiles | `data/tile_m31_a7s_newton.npz`, `data/tile_ic405_a7s_newton.npz` (Nicolas's images, 512 px tiles, float16) |
| fig06 training pair and weights | `data/pair_example.npz` (Nicolas's Horsehead stack, 256 px cut-out) |
| fig07 training curves | `data/histories.json` (history of each run) |
| fig08 residual flux versus magnitude | `data/validation.json`, `data/baseline_tiles.json` |
| fig09 cores | `data/cores.json` |
| fig10 control | `data/control.json` |
| fig11 trade-off | `data/validation.json`, `data/cores.json` |
| fig12 Horsehead | `data/horsehead_cutouts.npz` (Nicolas's image; one-scale outputs of N5, N7, mix; 256 px cut-outs) |
| fig13, fig14 whole images | `data/sheet_m31_a7s_newton.jpg`, `data/sheet_ic405_a7s_newton.jpg` (contact sheets of the 0.75 mix on Nicolas's images) |

No pixel of a third-party image is published (the compact-galaxy control is reported with numbers only).
