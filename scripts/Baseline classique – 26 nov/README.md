Folder names keep the original French and date for traceability.

# Baseline: classical Gaia-guided star removal (chain v33b) on a documented subset of crops

Run on 26-27 Nov 2026. Everything here is a copy or a new script; no validated script was modified.

The scripts hard-code the paths of the author's machine (`C:\Users\...`, `F:\float_ia\...`): they are given for traceability and must be adapted before any run.

## What is where

- Code (this folder): `choisir_tuiles.py` (tile rule), `tuiles.json` (chosen tiles), `lancer_chaine_tuiles.py` (CPU orchestrator: onboarding + chain + copy of the result),
  `mesurer_tuiles.py` (measurement of any starless source, also runs the network inference), `suivre_inference.py` (runs the network inference while the chain runs),
  `_lancer.ps1` / `_lancer2.ps1` (the exact launch lines), `chaine_v33b/` (copy of the chain code, not published here; see below), `resultats_tuiles.json`, `tableau_tuiles.md`.
- Outputs (`F:\float_ia\Article – 26 nov\baseline\`): `classique_<image>.npy` (3x512x512 final result of the chain on the tile core, plus `_carte.npy`, the fill-class map),
  `candidats\<model>_<image>.npy` (network results, same crops), `journal.json` (times, return codes), `journaux\` (chain and track 1 logs), `lanceur.log`, `inference.log`,
  `candidats\temps_inference.jsonl`, `seuils.json`, `planche_<image>.jpg` (input | classical | N7/N5 mix, <= 400 KB) and `crop_figure_<image>.npz`
  (keys `entree`, `classique`, `N7N5`, `x0`, `y0`) for the 4 first tiles done: m31_a7s_newton, ic405_a7s_newton, m33_osc_lunette, m27_mono_nb.

## The chain copy

`chaine_v33b/` (not republished in this repository: the published chain is `Methode v33b – 23 nov`) = copy of its code only (top-level scripts, `Piste1/*.py`, `Piste3/`, `Passes/`, `outils/`); not copied: its `sorties` (big outputs) and the 893 MB catalogue folder.
Only change: `Piste1/images.py` got a block at the end (published as `images_validation_block.py`) registering the 13 validation images (input read in place, same `input.npy` as the validation; kind "filtres" for the two mono narrow-band
images, "couleur" otherwise, same rule as set 2). Per-image onboarding as the set 2 images were: `Piste1/sorties/<image>/astrometrie.npz` and
`Catalogue Gaia profond – 20 nov/gaia_dr3_profond_<image>.npz` are plain copies of the validation's `astrometrie.npz` and `gaia_dr3_profond.npz` (same format);
the per-image "sheet" (calibration + star model, `etalonnage_image.npz`, `modele_psf_image.npz`) is built by the chain's own `batir_etalonnage_image.py` / `batir_modele_image.py` on the whole image.
The chain's own failure zones are not hidden (see caveats).

## Tile rule (written before any run, deterministic, not tuned on results)

For each validation image of LISTE that has `gaia_dr3_profond.npz` (13 images, LISTE order): one 512x512 tile whose origin (x0, y0) is on the 128 px grid, with
128 <= x0 and x0+512 <= W-128 (same for y), so the chain crop (tile +/- 128) is inside the image; at least 95 % of non-zero pixels (3-channel any != 0);
maximising the number of Gaia stars (deep extract) with 8 <= G < 17 in the tile; ties by lowest (y, x). Result in `tuiles.json`.

## Done and not done

Budget rule: orchestrator takes the images in LISTE order and does not start a new image once the elapsed time (onboarding + chain) is above 3 h.

Done (9 of 13 images, 9 tiles): m31_a7s_newton, ic405_a7s_newton, m33_osc_lunette, m27_mono_nb, ngc3372_mono_nb, m16_seestar_s50, ngc6934_stellina, leotriplet_canon6d_newton, m31_canon60d_objectif_filee.
No onboarding or chain failure on any of them.

Not done (4): m42_canon40da_lunette, ic5146_canon60da_lunette, m42_dobson_grande_fwhm, ic1848_stellina. Reason: time budget only (never attempted, so nothing is known about their onboarding).

Wall time per image (seconds; onboarding = image sheet, chain = the tile):

| image | onboarding | chain | total |
|---|---|---|---|
| m31_a7s_newton | 1023 | 382 | 1410 |
| ic405_a7s_newton | 1018 | 507 | 1531 |
| m33_osc_lunette | 1192 | 1316 | 2513 |
| m27_mono_nb | 1002 | 829 | 1836 |
| ngc3372_mono_nb | 283 | 256 | 544 |
| m16_seestar_s50 | 282 | 749 | 1037 |
| ngc6934_stellina | 283 | 457 | 745 |
| leotriplet_canon6d_newton | 659 | 166 | 831 |
| m31_canon60d_objectif_filee | 488 | 2623 | 3116 |
| sum | 5930 (1 h 39) | 7285 (2 h 01) | 13183 (3 h 40) |

The chain alone costs 2 h 01 for the 9 tiles (about 13 min per tile on average, 3-44 min); with the one-off per-image onboarding the total is 3 h 40, i.e. 40 min above the 3 h budget (the last image started at
2 h 54 and took 52 min: dense field with elongated stars, slow track 1). Whole-image onboarding is the dominant cost of a new image (up to 20 min), and happens once per image.
Network inference ran at the same time as the chain (4 threads each, 8 of 16 threads): about 45 s per model per 1280 px window.

## Measurement

`mesurer_tuiles.py` imports `valider_reseau.py` (the validation's own functions: `etoiles_restantes`, `ouverture`, `masque_loin`, `degats_fond`, `charger_gaia`, `charger_reseau`, `inference`)
and the validation's `mesures_entree.json` (FWHM, noise, detectable G). Pseudo full image = input with the 512 core replaced by the candidate; stars and pixels restricted to the core.
"Far from stars" mask: computed on the whole image like the validation, then cropped. "Bright structure": threshold = 90th percentile of the whole input smoothed (sigma 6), cached in `seuils.json`.
Networks: E3 `step_20000`, N2 `pas_26000`, N5 `pas_23000`, N7 `pas_22000`, N7/N5 0.75 mix = the existing file `F:\float_ia\Melanges N5 N6 N7 – 26 nov\poids\N7N5_075.pt`
(its sha256 is the published 9c88c524...2199b, checked in the script before use; it was not rebuilt), on a 1280 px window around the tile (clipped to the image), `astro_gpu.stages.star_removal.apply`
tile=1024, overlap=256, CPU, single scale, no two-scale correction, then cropped to the core.

Aggregate on the same 9 tiles (median over images of the per-image median; full per-image tables in `tableau_tuiles.md`):

| source | <8 | 8-11 | 11-14 | 14-17 | >17 | background med | background p99 | structure med |
|---|---|---|---|---|---|---|---|---|
| classical v33b | - (no image with >= 3 stars) | 0.011 (4 img) | -0.001 | 0.005 | 0.059 | 0.001 | 0.10 | 0.004 |
| E3 | - | -0.000 | 0.000 | 0.122 | 0.362 | 0.021 | 0.05 | 0.009 |
| N2 | - | 0.001 | 0.003 | 0.030 | 0.071 | 0.003 | 0.06 | 0.005 |
| N5 | - | -0.001 | -0.000 | 0.003 | 0.050 | 0.004 | 0.06 | 0.008 |
| N7 | - | -0.003 | -0.001 | 0.026 | 0.056 | 0.003 | 0.02 | 0.006 |
| N7/N5 0.75 | - | -0.001 | -0.000 | 0.039 | 0.250 | 0.004 | 0.02 | 0.005 |

## Commands

Launch lines are in `_lancer.ps1` (chain) and `_lancer2.ps1` (inference follower): `lancer_borne.ps1 -Fils 4 -Log <log> -Exe <venv python> -Args '-I "<script>" ...'`.
Then, once everything is done (the measurement is a few seconds of CPU):

    python -I mesurer_tuiles.py            # measures every source present; add  --inferer E3 N2 N5 N7 N7N5  to run missing network inferences (CPU)
    python -I choisir_tuiles.py            # (already run) writes tuiles.json

## Caveats

- Networks are single-scale and run on crops (1280 px window); the validation tables are whole images with the two-scale correction for very bright stars. Numbers are not interchangeable with
  the full-image tables. In these tiles almost no G < 8 star exists (fewer than 3 usable stars in the "<8" band in every image), so the band is empty and the two-scale correction matters little here.
- Only 9 tiles of 512 px, one per image, chosen by star count: that favours dense fields. Per-image medians in the faint bands (>17) rest on hundreds of stars, the bright bands on very few (n is in the tables).
- Tile margin: the aperture of a star near the edge of the core reads the unprocessed margin (input), for every source alike.
- The classical chain is not helped: its weight-0 zones (ignored when it is used as a training target) are measured like all other pixels; every Gaia star of the tile core counts.
  ngc3372_mono_nb is poor for every source (and the chain: 0.52 / 0.62 in 14-17 / >17): crowded field, noted, not fixed.
- Chain crop = tile +/- 128; the image sheet (star model, calibration) comes from the whole image, as in the production runs.
- RAM: the guard of the orchestrator was broken (it read 0 GB all along: it passed a subprocess handle to a psutil call). A manual check during m33_osc_lunette (4233x6216) showed the track 1 process at 11.1 GB resident, above the 8 GB cap
  (15 GB was free on the machine, no problem occurred). The two other A7S images (4272x6384) were probably in the same range, not measured. The smaller images were not checked.
- CUDA: `CUDA_VISIBLE_DEVICES=""` deletes the variable on Windows (the 4060 Ti would stay visible); `-1` is used instead (the chain's own steps already set -1; `torch.cuda.is_available()` was asserted False before inference).
  No GPU was used.
- The orchestrator process and the E3 job (`chaine_e3.py`, log `e3.log`) did not overlap: the chain started after "valider rc" appeared in `e3.log` (that job's validation step ended with `valider rc 1`; not looked at here).
