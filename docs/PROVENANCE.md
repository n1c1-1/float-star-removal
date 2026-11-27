# Provenance: the network was never trained on the output of another piece of software

**What this document lets you state, and check for yourself**: the published models (E3, N5, N7 and their N7/N5 0.75 blend) were trained only on (1) pairs **generated** by code and seeds, (2) **real** pairs whose target is made by our own Gaia-guided subtraction chain, (3) synthetic stars and cores placed on backgrounds that are themselves targets from (1) or (2). No file read by the training is the output of another star-removal program.

The technical detail is in `../provenance/` (manifests, verifiers, static analysis). This document explains it in plain terms.

## 1. The four models and their data

| model | starting point | data read by the training |
|---|---|---|
| **E3** (8-9 Nov) | E2, itself trained from random weights | **generated** pairs `pairs_v4` (500 pairs, seed 1, "clean" and "defects" sets) + stars drawn on the fly; no image |
| **N5** (25 Nov) | E3, step 20,000 | 182 files of real pairs (version v33g) listed in `provenance/MANIFESTE_PAIRES.csv` + synthetic backgrounds and stars |
| **N7** (25 Nov) | E3, step 20,000 | **the same** 182 files + synthetic galaxy cores (25 % of the synthetic tiles) |
| **blend N7/N5 0.75** (26 Nov) | — | **no data**: 0.75 × N7 (step 22,000) + 0.25 × N5 (step 23,000), weights averaged |

E, E2 and E3 start from **random weights**; no pre-existing network weights are read by the training.

## 2. How each target is made

| dataset | where the inputs come from | how the target is made |
|---|---|---|
| **generated** pairs (E2, E3) | nothing: backgrounds, defects, stars and noise drawn from a seed | the target **is** the background without the drawn stars: exact by construction |
| **real denoised** pairs (103 training + 6 held out) | Nicolas's image with its stars: stack minus its gradient, normalised in noise units, denoised | **input minus a star layer**: track 1 (star-and-halo model fitted on the image, positions and brightnesses from the Gaia catalogue), stage A (refilling by patches taken from the image), stage B (passes), judge (weights). v33d: weights set to 0 only; v33g: + smoothed diffuse light |
| **real raw** pairs (35) | the same tile, not denoised | raw − (denoised input − denoised target): the same star layer is removed |
| **web (APOD)** pairs (38 tiles from 13 images) | images published by APOD, **delinearised** (inverse stretch estimated on the stars), not re-denoised | **same chain** (variant v33b) and same judge as above |
| backgrounds and shapes of the N5/N7 star generator | the backgrounds are **targets or inputs of real pairs** (half and half); the star shapes come from banks of real bright stars measured in images **with** their stars | enter no target of a real pair; used to draw added stars |

The APOD pairs: checked in the code (`scripts/Production APOD – 24 nov/`) and the logs. 65 tiles were produced from 22 images, 9 images were discarded at the by-eye review (27 tiles), 7 images could not be produced (20 tiles); 38 tiles from 13 images remain. The input is prepared with no network and no download (`preparer_apod.py`); the target comes from the same chain as for Nicolas's images. **A doubt to be aware of**: the web images were already processed by their authors with software we do not know (possibly with a star-reduction tool); that processing is in the **input**; the **target** is still computed by our chain from that input.

## 2 bis. The verdict for the 182 pairs read by N5 and N7

**Established for the 182 lines** of `provenance/MANIFESTE_PAIRES.csv` (columns `origine_cible`, `preuve_cible`, `verdict_cible`): the target of each pair comes from the **classical chain** (131 lines: chain v33; 51 lines: chain v33b, including the 38 web pairs), then v33d (weights set to 0 only) and v33g (smoothed diffuse light); for the 35 raw pairs, from the formula "raw − (denoised input − denoised target)". No target comes from a network output, from a hand retouch, or from third-party software. No target is "not established". (The synthetic tiles of N5 and N7 are drawn on the fly; they are not pair files.)

The three proofs, in plain terms:

1. **The production logs.** Each tile has its track-1 log ("no filling" settings, working folder specific to the tile) and its end log, and appears in the list of produced pairs (`paires.jsonl`). The chain reads only the **input image** of each source image; no script names the "starless" or "target" files that exist next to some images, and the git history of the chain's folders shows no reading of such files. A single tile has no separate log (`58_hoo_g3456_896`, initial trial): it is established by the replay.
2. **A numerical check.** For the 60 non-web tiles whose image folder contains a "starless" or "target" file (stacks from online telescopes, datasets…), the pair's target was compared to that neighbouring file on the pixels where it differs from the input by more than 5 noise units: the target equals it on **at most 0.42 % of the pixels** (a target derived from that file would give 100 %). The other images have no such file: there was nothing to read.
3. **The bit-exact replay.** The published chain redoes the target from the input and finds it **bit-identical** on 6 tiles, including 3 tiles (5 lines) among the 14 lines whose production commit is "uncertain". For the 14 uncertain lines, the git history also shows that no committed version of the scripts ever read a reference file.

Each CSV line gives its proof (`preuve_cible`). Not replayed (24 to 53 minutes of track 1 per tile): `bubble_a7s_g1446_1310`, `heart_a7s` (3 tiles), `ic443_40da` (4 tiles); their folders have no "starless" file, or the numerical check applies.

## 3. The two ways to check

### Nicolas's images: the pairs are published (release attachment)

The pairs derived from Nicolas's images and read by N5 and N7 (54 files, 298 MB; + 470 MB of replay files) are provided as an attachment on the repository's Releases page (outside git: too large). Their list, with checksum and size, is in `provenance/SHA256SUMS_PAIRES_NICOLAS.txt`. A third party can:
1. check that the files are intact (`sha256sum -c`);
2. check the **relations** between files (target v33g = target v33d + smoothed diffuse light; raw pair = raw − star layer; weights only set to 0 between versions): `python provenance/verifier_provenance.py <pair>.npz --mode g`;
3. rerun the published chain and find the target **bit for bit**: `python provenance/verifier_provenance.py <pair>.npz --mode rejeu` (see `provenance/README.md`). **For the full replay**, the chain starts from the original image (768 px crop around the tile, astrometry, star model, deep Gaia extract). These input files are prepared for the 54 pairs, in a separate release attachment (`rejeu_entrees`, 469.6 MB, 132 files, checksums in `provenance/SHA256SUMS_REJEU_NICOLAS.txt`): `RACINE_ORIGINE=<rejeu_entrees>/origine ENTREES_REJEU=<rejeu_entrees>/entrees python provenance/verifier_provenance.py <pair>.npz --mode rejeu` (see `provenance/README.md`). Tried with these files alone: `m101_a7s_g2085_883` (v33g target redone bit-identical). Without this attachment, one checks the checksums and the relations (modes `liens` and `g`).

### Web images: only the means to redo them are published

The web images belong to their authors: **their pixels, and the pairs derived from them, are not published**. Instead, for each of the 38 pairs, `provenance/MANIFESTE_PAIRES.csv` gives: the address of the APOD page and of the image, the sha256 checksum of the image file as downloaded at the time, the exact delinearisation parameters, the crop (tile position and size), and the checksum of the pair file. The script `provenance/refaire_paires_web.py` downloads the image again from its address (option `--telecharger`), checks its checksum, redoes the delinearisation, the input and the pair with the published chain, and compares the checksum with the manifest's.

## 4. What was really tried (26 Nov)

Replay of the target from the input, with the **published** chain or its originals, on the CPU:

| pair | result |
|---|---|
| `m1_g896_896_debruitee` (Nicolas's image, chain v33) | target, weights, classes, input **bit-identical** (maximum difference 0 over 258,458 pixels of weight > 0) |
| `m45_asi1600_g2650_1450` and its raw version (v33) | denoised and **raw** targets bit-identical; step v33d → v33g redone identically |
| `bubble_a7s_g2000_1367` and its raw version (v33b) | bit-identical (track 1 on CPU: 803 s) |
| `apod_070_g1536_1408` (web, v33b) | bit-identical; with `refaire_paires_web.py --local`: image checksum, delinearisation, input and pair redone identically, and the redone pair **file** has the same sha256 as the manifest |

The production ran track 1 on the graphics card; its replay on CPU nonetheless gives the same bits (the fit runs in `float64`). **This is established only for these tiles**: we do not promise the same for the other tiles, nor with another library version. Re-downloading from the network was not tested (no network access during preparation).

## 5. Static analysis: what the training scripts read

Method (detail and line numbers: `provenance/STATIC_ANALYSIS.md`): we compute the **import closure** of each training and target-making script, and search, in that whole closure, for the name of any third-party star-removal software, the words `starless`, `Comparaison`, the reference folders, and any reading of network weights. Folders actually read:

| script | folders read | content |
|---|---|---|
| `train_e.py` (+ `day_e3.py`) | `pairs_v4/` (+ validation set drawn from these pairs), E2 weights | generated pairs; weights trained by this same script |
| `train_n56.py` and `train_n7.py` | `Paires étoiles v33 – 23 nov/` (files `*_v33g.npz`), `Paires étoiles APOD – 24 nov/` (`*_v33g.npz`), E3 step 20,000 weights | outputs of **our** chain; E3 generated only |
| `paires_synthetiques_n2.py`, `_n7.py`, `etoiles2.py` | the arrays of the pairs above (backgrounds), the star banks (shapes) | targets of our chain; real stars **with** their stars (shape measurement) |
| `produire_*.py`, `tuile_cpu*.py`, `methode_validee.py`, `etape_A/B.py`, track 1, track 3, passes, judges, `preparer_g_h.py` | the input image, its astrometry, the Gaia catalogue, the `starless.npy` of **our** track 1 | no output of another piece of software |

**Result of the search**: the name of third-party star-removal software appears in no closure as data read. The word `starless` there designates only the output of our own track 1. The 30 scripts that read outputs of other software or of trained networks (measurement, sheets, diagnostics) are in **none** of the closures: they are measurement scripts, never called by the training or by the making of targets. **The comparison folders are read only by measurement scripts.**

## 6. What remains uncertain (written down, not hidden)

1. **E3 has no commit** (the scripts repository dates from 21 Nov): its provenance rests on the checksum manifest of the generated pairs (`reads_no_image: true`, 4,000 checksums, code identical to that of the manifest) and on the logs. `train_e.py` was modified after E3 ended (9 Nov, 15:24).
2. **E3, "denoised" files**: the check regeneration (`make_dataset.py --verify`) gives back the noisy files **byte for byte**, but not the denoised ones (maximum difference 216 on the input, 88 on the target). Probable cause: the application's denoising code has changed since 9 Nov; not established. This does not affect the origin of the pairs.
3. **Production commits**: 14 manifest lines out of 182 have an "uncertain" commit (the production script was committed after the first pairs); `preparer_g_h.py` was modified after the v33g pairs were written, but the verifier's `g` mode reproduces them bit for bit on the 3 pairs tried.
4. **Web images**: processed by their authors before publication (see §2).
5. **Ownership of some non-APOD images**: several come from image banks, datasets or stacks from online telescopes; the code does not say who took them. The pairs in the release attachment are limited to those from Nicolas's personal tree; the other 90 files (488 MB) stay unpublished.
6. **Full replay** done on 6 tiles; the 182 checksums and relations are verified, the other replays were not done (§2 bis).
7. **Application dependencies**: the offline astrometry and the application's denoising network are not supplied (`scripts/README.md`).
