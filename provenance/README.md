# Provenance of the star-removal network's training data

Question asked: has the star-removal network (NAFNet) ever been trained on the output of another star-removal
program? This folder establishes, file by file, where the data read by the models **E3**, **N5**, **N7** and by the
**N7/N5 0.75** blend come from, and makes it possible to redo a target from its input.

## What the folder contains

| file | role |
|---|---|
| `MANIFESTE_PAIRES.csv` | one line per pair file read by the training of N5 and N7 (182 files: 103 denoised + 35 raw + 38 APOD in training, 6 denoised in validation): sha256 checksums, tile, source image, production scripts and commits |
| `MANIFEST_E3.md` | proof that E3 (starting point of N5 and N7) is trained on **generated** pairs (checksums of the pairs manifest, seeds, code) |
| `STATIC_ANALYSIS.md` | for each training and production script: what it reads (file, line, function) and the proof that nothing in it comes from a third-party star-removal program |
| `verifier_provenance.py` | redoes the target of a pair with the published chain and compares it to the file (bit-level difference) |
| `refaire_paires_web.py` | redoes an APOD pair from the published web image (optional download, `--telecharger`), delinearisation included, then compares it to the manifest |
| `SHA256SUMS_PAIRES_NICOLAS.txt` | sha256, name (with alias) and size of the 54 pairs from Nicolas's personal tree, copied outside the repository for the release (`Depot public – paires/paires/`, no `.npz` in the repository) |

The CSV can be read by a spreadsheet (separator ";"). The `commit_*` columns give the commit of the scripts repository that contained the
script at the time of production ("incertain" if the script was not yet committed; the first following commit is then
given); `scripts_non_modifies_depuis_prod` says whether the chain's files have been left untouched since production.
Names containing a third party's first name are replaced by the alias `tiers1`; the verification finds the line by the file's checksum.

Columns of `MANIFESTE_PAIRES.csv`: `nom_fichier`, `modeles` (N5+N7: read by both), `role` (training or validation), `type` (debruitee,
brute, apod), `image_source`, `tx`, `ty` (top-left corner of the 512 px tile in the source image), `sha256_paire`, `taille_octets`,
`sha256_image_source` and `chemin_image_source` (the input read: the image, or the non-denoised version for a raw pair),
`chemin_entree_debruitee` and `sha256_entree_debruitee`, `pile_origine` (folder of the original stack, from the image's `meta.json`),
`script_production`, `commit_production`, `chaine_methode` and `commit_chaine`, `juge` and `commit_juge`, `version_cible` (v33g), `filiation`,
`scripts_non_modifies_depuis_prod`, `juge_v33d_non_modifie_depuis`, `preparer_g_h_non_modifie_depuis`, `script_derivation`, `commit_derivation`,
`sha256_base_v33` and `sha256_v33d` (checksums of earlier versions of the same file), `date_base_v33`, `poids_zero_pct` (share of weight-0 pixels),
`apod_info` (number, date, title, APOD address) and `sha256_image_web_origine` (checksum of the downloaded image file, for APOD).

## Who trained what, with which data

| model | start | data read |
|---|---|---|
| E, E2, E3 (7-9 Nov) | random weights, then E, then E2 | **generated** pairs `pairs_v2/v3/v4` + stars drawn on the fly; no image (`MANIFEST_E3.md`) |
| N5 (25 Nov) | E3, step 20000 | 182 manifest files (version v33g) + synthetic backgrounds and stars (see below) |
| N7 (25 Nov) | E3, step 20000 | **the same** 182 files (v33g) + synthetic cores (Sérsic) added on 25 % of the synthetic tiles |
| N7/N5 0.75 blend | — | **no data**: weight average 0.75 × N7 (step 22000) + 0.25 × N5 (step 23000), same architecture (log of 26 Nov) |

N5 **does not resume** N2 (night of 24 Nov): N5 starts from E3, step 20000. N2, N1 and the following rounds read the same tiles in version
v33d (checksums in the `sha256_v33d` column).

The "backgrounds" of the synthetic star generator are the targets or the inputs of the real and APOD pairs themselves (half and
half), drawn from the batch; the star shapes come from banks of real bright stars measured in
images with their stars (`STATIC_ANALYSIS.md`, §2.2 and §2.3). No background and no shape comes from a starless image
made by other software.

## Table: dataset -> inputs -> target -> third-party software

| dataset | where the inputs come from | how the target is made | third-party software outputs: none (proof) |
|---|---|---|---|
| generated pairs `pairs_v2/v3/v4` (E, E2, E3) | no image: backgrounds, defects, stars and noise drawn from a seed | the target is the background without the drawn stars (by construction) | manifest `reads_no_image: true`, 4000 checksums, code identical to that of the manifest (`MANIFEST_E3.md`) |
| real "denoised" pairs (103 + 6) | image with its stars: stack minus its gradient, background and noise normalised, denoised by the application's **denoising** network, white balance (the image's `input.npy`) | **input minus a star layer**: track 1 (star-and-halo model fitted on the image, positions and magnitudes from the Gaia catalogue), stage A (refilling by patches taken from the image), stage B (three passes), then judge (weights); v33d: weights set to 0 only; v33g: + smoothed diffuse light L | no reading (`STATIC_ANALYSIS.md` §3 and §4); bit-exact replay on 3 tiles (m1, m45, bubble: below) |
| real "raw" pairs (35) | the same tile, **not denoised** (`entree_brute.py`: normalised half-stacks, no network) | raw − (denoised input − denoised target): the same star layer is removed | formula checked bit for bit on 3 raw pairs (`liens` mode: `bubble_a7s_g1446_1310`; replay: `m45_asi1600_g2650_1450`, `bubble_a7s_g2000_1367`) |
| APOD pairs (38 kept out of 65 produced, 13 images) | web images published by APOD, **delinearised** (inverse stretch estimated on the stars), without new denoising; the list (number, date, title, APOD address) is in the `apod_info` column | same chain (variant v33b) and same judge as above; 9 images "dropped" at the by-eye review (27 tiles); 7 images could not be produced (track 1 crashed: 20 tiles) | no reading; bit-exact replay on 1 APOD tile (below) |
| star-shape banks and style cache | real isolated bright stars, medians of 5 to 8 stars, measured in images **with** stars (`input.npy`, `input.fits`, `original.npy`) | do not enter a target: used to draw stars added on the backgrounds | `STATIC_ANALYSIS.md` §2.3 (no reading of `starless`) |

### What is established for the APOD pairs

The code (`Production APOD – 24 nov`) prepares each input (`preparer_apod.py`: median and dispersion of the background per colour, unit = mean noise,
"no network, no download"), then launches track 1 on the card, then stage A, stage B and the judge on the CPU
(`tuile_cpu_apod.py`): the same as for Nicolas's images, in variant v33b of the chain (fixes for 4 track-1 crashes).
The 65 base files are exactly the 65 lines of `paires.jsonl`. **Doubt**: the web images are images already processed by
their authors with unknown software (including, perhaps, a star-reduction tool); that processing is in the **input**; the
**target** is still computed by our chain from that input, without reading the output of any removal program.

## How to run the verification

Environment variables: `RACINE_SCRIPTS` (folder of the chain's scripts) and `DONNEES` (data folder). The script copies the
chain into a working folder; it writes nothing into `RACINE_SCRIPTS`. At most four compute threads.

```
# 1. quick: checksum in the manifest, relations between versions (target v33 = v33d, v33g = v33d + L, raw = formula)
python verifier_provenance.py <pair>.npz

# 2. redo the step v33d -> v33g (a few seconds)
python verifier_provenance.py <pair>.npz --mode g

# 3. redo the target from the input (track 1 on CPU; 1 to a few minutes on the tiles tried, much more on the slow tiles)
python verifier_provenance.py <pair>.npz --mode rejeu --device cpu --travail <folder>

# everything
python verifier_provenance.py <pair>.npz --mode tout --rapport rapport.json
```

The replay compares to the base pair (`<image>_g<x>_<y>_debruitee.npz`): per channel, on the pixels of weight > 0, maximum and
mean absolute difference (in sky noise units), share of identical pixels; weights, classes and input are compared too.

### Trial results (26 Nov 2026, CPU, 4 threads)

| pair | chain | mode | result |
|---|---|---|---|
| `m1_g896_896_debruitee` (non-APOD) | v33 | replay (track 1 on CPU: 132 s; A + B: 34 s; 169 s in total) | target, weights, classes, input: **bit-identical** (max difference 0 over 258458 pixels of weight > 0) |
| `m45_asi1600_g2650_1450_debruitee` and `_brute` | v33 | liens + g + replay (102 s) | denoised target, redone **raw** target, weights, classes: **bit-identical**; v33g redone identically |
| `apod_070_g1536_1408_debruitee` (APOD) | v33b | liens + g + replay (74 s) | target, weights, classes: **bit-identical**; v33g redone identically |
| `bubble_a7s_g1446_1310_brute` | — | liens | raw target = raw − (denoised input − denoised target): **bit-identical**; v33 -> v33d: identical target, weights only set to 0 |
| `bubble_a7s_g2000_1367_debruitee` and `_brute` (v33b) | v33b | liens + g + replay (track 1 on CPU: 803 s; A + B: 50 s; 861 s in total) | denoised target, redone raw target, weights, classes: **bit-identical**; v33g redone identically |

Even the replay of track 1 on the **CPU** gives, on these 6 tiles, the same bits as the production, which ran track 1 on
the graphics card: the batched fit runs in `float64`. This does not say that the same holds for all tiles nor for
another library version: only these replays are established. Stages A, B, passes and judge gave no difference.

## What is established, what remains uncertain

**Established**

- N5 and N7 read exactly the same 182 pair files (N5's and N7's `config.json` identical for the lists); all are
  outputs of `produire_v33.py` / `produire_v33b*.py` / `produire_apod.py` (the 124 + 65 base files are the lines of `paires.jsonl`).
- The targets of the pairs (real, raw, APOD) are outputs of the classical chain; no training or production script opens
  a file from third-party star-removal software (`STATIC_ANALYSIS.md`).
- E3 is trained on generated data (`MANIFEST_E3.md`): manifest `reads_no_image: true`, code identical to that of the manifest, and the
  check regeneration (`make_dataset.py --verify 1`, 26 Nov) gives back the noisy files of both sets **byte for byte**.

**Uncertain or to be decided (by name)**

1. The commits: the scripts repository dates from 21 Nov; E3 (8-9 Nov) has no commit; the v33 pairs were written before the commit
   of the production script for 14 CSV lines out of 182 (column `commit_production`, value "incertain"; the chain's files were nevertheless not modified since the pairs for 168 lines, column `scripts_non_modifies_depuis_prod`); `preparer_g_h.py` was modified after the v33g
   pairs were written (the `g` mode shows that the current version reproduces them bit for bit for the 3 pairs tried; column `preparer_g_h_non_modifie_depuis` = no).
2. `train_e.py` was modified after E3's training (`MANIFEST_E3.md`, §5).
3. The APOD images may have been processed by their authors with any software before publication (see above).
4. Several "non-APOD" images come from image banks, datasets or stacks from online telescopes: the code does not say
   who took them (see the folder's `RESUME.txt`, outside the repository).
5. E3, "denoised" files: the check regeneration does **not** give back the manifest's denoised files (max difference 216 and 88 in float16;
   probable cause: the code of the engine's denoising stage has changed since 9 Nov; not established). The noisy files are identical:
   the difference concerns the denoising network, not star removal.
6. The full replay was done on only 6 tiles (m1, m45, 58_hoo_g3456_896, bubble_a7s_g2000_1367, bubble_a7s_g2200_1800, apod_070); the 182 checksums can be verified,
   the replays of the other tiles were not done. A trial on `heart_a7s_g128_128` (production: 24 min of track 1 on the card) was stopped
   after 9 minutes of track 1 on CPU: no result.
7. The engine's catalogue reader (`astro_gpu.core.<reader>`) carried a third-party software name in its module name: it reads
   extracts of the Gaia catalogue, not images; to be renamed before publication.

## Additions of 26 Nov (published scripts, APOD pairs, release pairs)

- **Published scripts.** `verifier_provenance.py` takes as default `RACINE_SCRIPTS` = the repository's `scripts/`. The unpublished files of the chain
  (astrometry and star model of each image, deep Gaia catalogue, list of the images of set 2, old import folder) are given by
  `RACINE_ORIGINE`; the tokens `<DONNEES>`, `<ESSAIS>`, `<FLOAT>`, `<CACHE_FLOAT>`, `<SCRIPTS>` of the published scripts are replaced in the working
  copy by `DONNEES`, `ESSAIS`, `RACINE_FLOAT`, `CACHE_FLOAT`, `RACINE_SCRIPTS`. The published scripts do not contain the folder of the chain
  "v33 production": the non-APOD v33 pairs are therefore replayed with chain v33b (it differs from it only by guards on the image edges).
- **Trials with the published scripts** (CPU, 4 threads): `apod_070_g1536_1408` (chain v33b) and `m1_g896_896` (non-APOD): target, weights, classes and input
  **bit-identical** (max difference 0); step v33d -> v33g redone identically for the APOD pair.
  **A published script does not run as is**: `scripts/Methode v33b – 23 nov/Piste1/astrometrie.py` line 30 does `from astro_gpu.core import asterion, catalogue_local`
  (and uses `catalogue_local.` at lines 27, 66, 94-155: `locate`, `astro_cone`, `SCALE`, `_sep_deg`, `XP_WAVELENGTHS`); the cleaning renamed the engine's module,
  but in the host application's engine the module still carries its old name: `ImportError: cannot import name 'catalogue_local'`. `scripts/` was not corrected;
  for the trial the variable `REMPLACEMENTS_COPIE` was used (text replacement in the working copy only). Either the module must be renamed in the engine, or
  the old name kept in the published script.
- **APOD pairs: columns added to `MANIFESTE_PAIRES.csv`**: `url_page_apod`, `url_image`, `fichier_telecharge`, `sha256_image_telechargee` (image as downloaded, 38 out of 38 on disk),
  `delinearisation_params` (compact JSON: method `etoiles_auto`, chosen colour, `m_depart`, `m_float`, stars seen and used, cost, background normalisation, noise and unit),
  `decoupe` (tx, ty, 512 px, chain v33b, chain window). **Warning**: for the 38 lines, `url_image` and `fichier_telecharge` contain the photographer's family
  name (file names of the site): to be decided before publication (removing them takes away the possibility of re-downloading without going through the APOD page).
- **`refaire_paires_web.py`**: tested with `--local` on `apod_070_g1536_1408`: web image (sha256), delinearisation redone (7 s: curve, number of stars and colour identical to the manifest),
  input redone **bit-identical** to the kept input, pair redone (published chain) **bit-identical**, and the base pair **file** redone has the same sha256 as
  the `sha256_base_v33` column. `--telecharger` was not run (no network access).
  What requires the host application: offline astrometry and the Gaia extracts (already computed for the 38 images, supplied through `RACINE_ORIGINE`); the denoising network is not used.
  Not verified: the other 37 APOD pairs, nor after re-downloading (the server may serve a different file: the `sha256_image_telechargee` column would detect it).
- **Release pairs**: 54 files, 297.7 MB, copied into `Depot public – paires/paires/`; checksums in `SHA256SUMS_PAIRES_NICOLAS.txt`, rechecked after copying.
  The 90 files (488.1 MB) coming from banks, datasets, LCO stacks or "to be checked" are not copied: list in `RESUME.txt` (outside the repository).

## Decisive check of the targets (26 Nov, evening): verdict for the 182 lines

**Verdict: established for the 182 lines** (columns `origine_cible`, `preuve_cible`, `verdict_cible` of `MANIFESTE_PAIRES.csv`). The target of each pair read by N5 and N7 comes from
the classical chain (131 lines: chain v33; 51 lines: chain v33b, including the 38 APOD), then v33d (weights only) and v33g (smoothed diffuse light L); for the 35 raw ones, from the formula
raw - (denoised input - denoised target). None comes from the synthetic generator (the synthetic tiles are drawn on the fly, they are not pair files), from a network, from a retouch
or from third-party software: no target is unestablished. Proofs, line by line:

1. **Production logs**: each of the 124 + 65 base tiles is in `paires.jsonl`; its track-1 log (`p1_<tile>.log`, `no_fill` settings, working folder `mv_<tile>`) and its end log
   (`cpu_<tile>.log`, `RESULTAT` line) are in `Production v33 – 23 nov/journaux*` or `Production APOD – 24 nov/journaux`. The only exception: `58_hoo_g3456_896`, produced in the initial trial
   (log `journaux/58_hoo.log`, "== 58_hoo_g3456_896 rc=0", without separate `p1_`/`cpu_`); it is established by the bit-exact replay (below).
2. **Only the input is read.** `Piste1/images.py` (`load`) opens the image's input file (column `chemin_entree_debruitee`: `input.npy`, `original.npy` or `input.fits`); no script of the chain names the files
   `starless*.npy` / `cible.npy` (static analysis); `git log -S` on the folders of the chain and of the production finds no reading of these files nor of `Comparaison référence` in any commit
   (`starless_reseau` appears only in `vues_mesures_m8_sho.py`, a sheet script never called).
3. **Numerical check against the neighbouring files.** For the 60 non-APOD tiles whose image folder contains a `starless*.npy` or `cible.npy` file (images m1, m13, m82, m97, ngc6543, ngc6888,
   ngc7662, 58_hoo, m8_sho, horsehead, horsehead_40da, ic443_40da, m45_40da, m51_origin, m51_seestar_s30, m57, ngc869, rosette_40da, wizard_a7s), we took the pixels where that file differs from the input by more than
   5 noise units (at least 142 pixels per tile; typically thousands): the target equals (to within 1e-4) the neighbouring file there on **at most 0.42 % of the pixels** (a derived target would give 100 %).
   The other images (Nicolas's stacks, set 1, set 2, benches, APOD) have no starless or target file in their folder: there was nothing to read.
4. **Version relations verified on the 182 lines**: target v33 = target v33d bit for bit; v33g - v33d = smooth diffuse light L (max slope given in `preuve_cible`); raw = formula, bit for bit.
5. **Bit-exact replay of the chain** on 6 tiles, including 3 tiles (5 lines) of the 14 lines with an "uncertain" commit (`58_hoo_g3456_896`, `bubble_a7s_g2000_1367`, `bubble_a7s_g2200_1800`):
   the target redone by the chain is identical to the file's target. For the 14 lines with an uncertain commit, the git history further shows that no committed version of the scripts (v32, v33, v33b) reads a reference file.
   Not replayed (production: 24 to 53 minutes of track 1 per tile): `bubble_a7s_g1446_1310`, `heart_a7s` (3 tiles), `ic443_40da` (4 tiles); their image folders contain no starless file (heart, bubble) or the check of point 3 applies (ic443).

## Full replay of Nicolas's published pairs (folder outside the repository `Depot public – paires/rejeu_entrees/`)

- Contents (469.6 MB, 132 files, checksums in `SHA256SUMS_REJEU_NICOLAS.txt`): for the 12 images of Nicolas whose pairs are published (27 tiles): `entrees/<image>/input.npz` (the whole image as zeros except the
  768 px windows read by the chain around each tile and the maximum of each channel, which the judge uses), `meta.json`, `type.json`; `origine/…/Piste1/sorties/<image>/` (astrometry, star model, calibration) and the deep Gaia extract of each image.
- Use: `RACINE_ORIGINE=<rejeu_entrees>/origine ENTREES_REJEU=<rejeu_entrees>/entrees python verifier_provenance.py <pair>.npz --mode rejeu` (plus `RACINE_FLOAT` and `CACHE_FLOAT`). The script publishes its replay from these files alone; a pair published in v33g format is
  compared by redoing the chain's target and then the diffuse light L (no base pair is needed).
- Trial (CPU, 4 threads, published scripts, only files of `rejeu_entrees`): `m101_a7s_g2085_883_debruitee_v33g.npz`: input identical to the pair, v33g target redone **bit-identical**, classes and input identical, weights conforming (219 s).
- Size: 469.6 MB for the replay + 297.7 MB of pairs = 767 MB in total (under 1 GB: reasonable for a release attachment). No APOD image is copied.
- Still requires the host application's engine (offline astrometry, Gaia extracts) and the fix to the catalogue reader's module name pointed out above.
