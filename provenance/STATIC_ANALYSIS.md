# Static analysis: what each script reads

Goal: to show, by reading the code, that neither the training of E3, N5 and N7, nor the making of the targets of their
pairs reads any output of third-party star-removal software. Each line cites the file, the line of
code and the function. Roots: `<SCRIPTS>` (scripts), `<DONNEES>` (data), `<ESSAIS>` (trials), `<FLOAT>` (the application's engine).
The line numbers are those of the files as of 26 November 2026.

## 0. Method and what is proven

1. We start from the training scripts (`train_e.py` + `day_e3.py` for E3; `train_n56.py` for N5; `train_n7.py` for N7)
   and from those that make the targets (`tuile_cpu*.py`, `produire_*.py`, `methode_validee.py`, `etape_A.py`, `etape_B.py`,
   track 1, track 3, passes, judges, `preparer_g_h.py`).
2. We compute their **import closure** (Python syntactic analysis of the `import`s, following the folders of the
   path variable that each script writes itself): 12 files for E3, 20 for N5, 13 for N7, 24 for the v33 production
   chain, 24 for the v33b/APOD chain, 8 for the judges and the g/h step.
3. In the whole closure we search for: the name of third-party software (all spellings, with or without separator), `starless`,
   `sans_etoiles`, `Comparaison`, `stars_dataset`, `paires_nb`, `_revue`, and any reading of network weights
   (`torch.load`, `load_state_dict`).
4. In **all** the scripts of `<SCRIPTS>` we search for those that mention both `starless` and a reference folder
   (`stars_dataset`, `paires_nb`, `Comparaison référence`) or a file `starless_fixed` / `starless_reseau`:
   30 files (including `diag_validation.py`, `part_b.py`, `coeur_m31.py` and 22 copies of `vues_mesures_m8_sho.py` stored in the
   method's version folders, including v33 and v33b: they are stored there but never imported or launched by the
   chain). **None belongs to any of the closures above** (checked by program). These are measurement and sheet scripts
   (the scripts `inferer_v2.py`, `valider_*.py`, `coeurs_mesure.py`, `mesurer_*.py` read outputs of already trained networks
   in order to measure them; they are not in a closure either).
5. Result of the search (point 3) in the closures: the third-party software name appears nowhere as data read;
   `starless` there designates only `starless.npy`, the output of **our** track 1 (folder `sorties/<image>/mv_<tile>/`,
   `retrait_modele.py` l. 2204, `retrait_rapide.py` l. 2851; read by `piste3.py` l. 83 and `juger.py` l. 89), and comments.
   Git history (`git log -S`) on the folders of the chain and of the production: no commit added or removed a reading of
   `starless_fixed`, `cible.npy`, `Comparaison référence` or `REF_STARLESS`; `starless_reseau` appears only in the copies of
   `vues_mesures_m8_sho.py` (a sheet script, never imported or launched by the chain; commits 17b4d20 and c24bd7a).
   E, E2 and E3 start from random weights; no pre-existing network weights are read by the training.

What is NOT established by reading the code alone: the physical origin of the input images (who took them) and the fact
that the current files are those that were read at the time (see `MANIFESTE_PAIRES.csv`: checksums, dates, commits).

## 1. E3 (and E, E2): trained on generated data only

| script | line | function | what is read | output of third-party software? |
|---|---|---|---|---|
| `day_e3.py` | 17-19 | constants | `<ESSAIS>/starRemoval/pairs_v4`, `diagnose/valset_pairs_v4.npz`, `train/E2/best.pt` | no: generated set, validation set drawn from the generated pairs, E2's weights |
| `day_e3.py` | ~33 | `main` | launches `make_dataset.py --seed 1 --n 500` (generation) if the manifest is missing | no |
| `train_e.py` | 44-46 | constants | `pairs_v2` (default), `train/`, `valset_pairs_v2.npz`: replaced by the arguments of `day_e3.py` | no |
| `train_e.py` | 134, 144 | `Tiles.__init__`, `Tiles.arrays` | `info.json`, `input_denoised.npy`, `target_denoised.npy` of the pair folders | no: files of `make_dataset.py` (checksums in the manifest) |
| `train_e.py` | 229 | `val_set` | the validation file (`np.load(path)`) | no: cut from the held-out pairs (`diagnose.py` l. 222-247) |
| `train_e.py` | 174-181 | `RealTiles.__init__` | real images and corrected targets | **not executed**: requires `--real`; E3's `config.json` has no `real` key, `log.txt` does not contain "real images" |
| `train_e.py` | 60-124 | `synthetic_stars` | nothing (draws analytic PSFs) | no |
| `train_e.py` | 296-297 | `main` | `torch.load(a.init)`: starting weights (E2 for E3) | no: E2 was trained by this same script |
| `models.py` | 230 | `chain_nafnet` | **the architecture** only (`astro_gpu.stages.star_removal.NAFNet()`, random weights), no weight file read | no |
| `make_dataset.py` | 58-61, 78, 144 | modules, `sha256`, `_denoiser` | `catalogue.json`, the application's denoising network | no (denoising, not star removal) |
| `backgrounds.py`, `defects.py`, `stars.py`, `psf.py`, `instruments.py` | — | — | no image: no `np.load` (except `instruments.py` l. 45: `catalogue.json`) | no |
| `synth.py` | 61 | `Synth.__init__` | `psf_bank.npz`, PSF bank computed by `synth.py build` (optical model, seed 2718) | no |

E, E2 and E3 start from random weights (E's `log.txt`: "random weights"); no pre-existing network weights are read by
the training. Only the architecture (the network class) is taken from the application.

## 2. N5 and N7: where the pairs read, the synthetic backgrounds and the star shapes come from

### 2.1 The training scripts

| script | line | function | what is read |
|---|---|---|---|
| `Tours N5 N6/train_n56.py` | 41-43 | constants | `PAIRES` = `<DONNEES>/Paires étoiles v33 – 23 nov`, `APOD` = `<DONNEES>/Paires étoiles APOD – 24 nov` |
| `train_n56.py` | 50-63 | `lister` | `PAIRES.glob("*_v33g.npz")` (real, denoised and raw) and `APOD.glob("*_v33g.npz")`; the 6 held-out tiles (`TENUES`) go to validation |
| `train_n56.py` | 181 | `main` | `T1.Paires(ten, 1)`: reading of the 6 validation tiles |
| `train_n56.py` | 198 | `main` | `torch.load(a.depart)`: `<ESSAIS>/starRemoval/train/E3/snap/step_20000.pt` (E3, generated only) |
| `train_n56.py` | 116, 124-126 | `Lots.__iter__` | `paires_synthetiques_n2.paire_synthetique(fond, ...)`; **`fond = t[i]` (the target) or `x[i]` (the input) of a real or APOD pair drawn in the batches**, one time out of two |
| `train_n56.py` | 76-90, 125 | `extras_nebuleuse` | places very bright stars on the weight-1 pixels of the target: reads only the pair's arrays |
| `Tour N7/train_n7.py` | 44-45, 53-66, 189, 206 | same | same readings (same config: 103 + 35 + 38 pairs, 6 held out) |
| `train_n7.py` | 119, 127-135 | `Lots.__iter__` | `paires_synthetiques_n7.paire_synthetique_coeurs`: the same + synthetic cores (Sérsic) added to the target |
| `Entrainement jeu 1/train_jeu1.py` | 36-43 | `Paires.__init__` | `np.load(f)` of the files it is given (used by N5/N7 to read the pairs, never for its own folder) |
| `train_jeu1.py` | 31, 150 | constant, `main` | `Paires étoiles jeu 1 – 22 nov/*.npz`: **only in `T1.main`, never called by N5/N7** (they import `Paires` and `valider`) |
| `train_jeu1.py` | 85 | `LotsSynth.__iter__` | `from paires_synthetiques import paire_synthetique` (original generator): class not used by N5/N7 |

The **exact** lists of the files read are in the `config.json` of `N5` and of `N7` (keys `debruitees`, `brutes`, `apod`,
`tenues_fichiers`): 103 denoised + 35 raw + 38 APOD for training, 6 tiles for validation, identical
for N5 and N7. They are copied, with checksums, into `MANIFESTE_PAIRES.csv`. N5 does **not** resume N2: N5 starts from
E3 step 20000 (N5's log and `config.json`: `depart`). The pairs of the night of 24 Nov (N1, N2) are the same tiles
in version `v33d` (checksums in the `sha256_v33d` column).

### 2.2 The "backgrounds" of the synthetic star generator

Precise answer: the **backgrounds** are the 256 px tiles of the real and APOD pairs themselves, drawn from the current batch
(`train_n56.py` l. 124: `fond = t[i] if rng.random() < 0.5 else x[i]`), hence **the target of our chain** or the **input** of the
pair. They are neither images from other software nor another folder. The generator adds drawn stars
(`paire_synthetique`, 60 % of the tiles); the target remains that of the pair (the pair's weights unchanged).
The only other reading of backgrounds is `vitesse()` (`paires_synthetiques_n2.py` l. 361-368, `paires_synthetiques_n7.py` l. 442-449,
`paires_synthetiques.py` l. 355-361): a speed measurement from the command line, which reads `sans_etoiles_tuile.npy` of two
tiles of **our** chain (M57 and M8 SHO); never called by the training.

### 2.3 The star shapes (`cache_styles.npz`, banks)

- `paires_synthetiques_n2.py` l. 45 / l. 213 and `paires_synthetiques_n7.py` l. 48 / l. 216: `np.load(cache_styles.npz)`.
  This file (4141488 bytes, 22 Nov 19:52, sha256 0e270ba4b22bb15a6cb60d83b547d8e8472333f8f0f95c5ceb1230b7723fc3f8)
  is built once by `construire_cache()` (l. 156 / 159, called only if it is missing) with 32 "styles"
  drawn by `etoiles2.tirer_image` (seed 2026): Fourier optics (`psf.py`), catalogue instruments
  (`instruments.py`), and 11 banks of real shapes.
- `etoiles2.py` l. 397-410 (`banques_dispo`): reads `Etoiles synthetiques – 22 nov/banque/banque2_*.npz`.
  Constants `BPRP` l. 65: percentiles of the Gaia colours of 4 fields.
- The banks are built by `banque_bibliotheque.py` (l. 57: `lire_image(Path(chemin.split(" (")[0]))`),
  `banque_individuelle.py` (l. 104: `A.IM.load(nom)`), `banque_formes.py` (l. 85: `A.IM.load(nom)`): each time **the input
  image with its stars** (`input.npy`, `input.fits`, `original.npy`); real bright isolated stars are median-combined there
  (detection `banque_detection.py`, or Gaia positions from `astrometrie.npz`). The mention "(+ starless.npy)" in
  the library inventory is text: the code cuts the path before that parenthesis and never reads that file.
  `grep` of the four bank scripts and of `etoiles2.py`: no occurrence of `starless` or `sans_etoiles`.
- Banks really used by the 32 styles of the cache: rosette (2), m57 (2), m8_sho (1), horsehead (1) = our test images
  (`Piste1/images.py`); E065, E062 (image bank), E010 (the author's project), S023, S007 (`stars_dataset`),
  W004, W008 (`stars_dataset/_revue_pn`); 16 styles without a bank. The 28 `banque2_*` files of `banque/` are loaded
  by `banques_dispo`, but only the styles above were drawn into the cache.
- **Doubt for Nicolas to lift**: several banks come from images of datasets or banks
  (`stars_dataset/set_*`, `_revue_pn`, `astro_banque`) whose author is not established by the code. These are images
  **with** stars; none is the output of a star-removal program.

## 3. Making the targets (classical chain): everything the chain reads

A pair's target is `sorties/<tile>/B/t3/sans_etoiles_tuile.npy` (`construire_paire.py` l. 149), the final output of:
track 1 → stage A → stage B (3 passes) → judge (weights). Readings, by step:

| step (script, line, function) | reads | nature |
|---|---|---|
| `images.py` l. 45 `load` (via `construire_paire.py` l. 142 and track 1) | the input image: `input.npy`/`input.fits` of the image (list `IMAGES` l. 14-30; `selection_jeu2.csv` l. 76-82 `_jeu2`; `entrees/apod_*` l. 89-92 `_apod`) | image **with** stars (denoised), not a removal output |
| `Piste1/astrometrie.py` l. 18-66, 94-155 | offline Gaia DR3 extracts (`gaia_extracts`) via the engine's catalogue reader (`astro_gpu.core.<reader>`) | star catalogue (positions, magnitudes, XP spectra) |
| `Piste1/retrait_rapide.py` (`run`, batched fit in `float64`, `ajust_lot.py`) | `astrometrie.npz`, `etalonnage_image.*`, `modele_psf_image.*`, `psf_etendue.*` of the image, the input image | star-and-halo model fitted **on the image itself**, subtraction; writes `starless.npy` (l. 2851), `etoiles.csv` |
| `etape_A.py` l. 30-60 + `Piste3/piste3.py` | `starless.npy` and `etoiles.csv` from track 1 (l. 83, 137, 203), `gaia_dr3_profond_<image>.npz`, the image; "patch" filler (`remplir.py`, `patchfill.py`): the patches come **from the image itself** | refilling of cores and hard zones |
| `etape_B.py` + `Passes/passe_finale_p.py`, `_t2_p.py`, `_t3_p.py` | `carte_remplissage_tuile.npy`, `etoiles.csv`, deep catalogue | passes on star remnants, dark spots, defects |
| `Juge v33/construire_paire.py` l. 142-205, `regles_v33.py` | the target (`sans_etoiles_tuile.npy`), `carte_remplissage_tuile.npy`, the input, the deep catalogue | computes the **weights**; never modifies the target |
| `Juge v33c`, `Juge v33d` (`rejuger_v33c.py`, `rejuger_v33d.py`) | v33 pairs | "only set weights to 0" (checked: target v33 = target v33d bit for bit, `liens` mode of `verifier_provenance.py`) |
| `Tours N5 N6/preparer_g_h.py` `etape_g`, `etape_h` | v33d pairs, Gaia catalogue (`diag_paires.py`) | v33g = target v33d + diffuse light L (normalised Gaussian smoothing of σ 40 px of the input − target difference over the untouched sky); v33h = weights only |
| raw: `tuile_cpu.py` l. 24-31 `brute` | `entrees_brutes/<image>_brut.npy` (made by `entree_brute.py`: normalised half-stacks, no network) | raw target = raw − (denoised input − denoised target), bit for bit |

Precise points:

- The module `Piste3/mesure.py` (l. 12-14) puts `Retrait étoiles par modèle – 20 nov` at the head of the import path: for stage A,
  `images` and `juger` are read **from that old folder** (`juger.py` only does measurements; `images.py` is replaced by the
  copy of `Piste1/images.py` by `etape_A.py` l. 33-40 for all images other than Rosette). Checked by import:
  `images` and `juger` do come from that folder; `juger.py` l. 89 rereads track 1's `starless.npy`.
- A network intervenes at only one place in the code: `Piste3/piste3.py` l. 759-766, `e3_local` (network E3, generated only);
  it is called only if `filler == "e3"` (l. 619, 719). **The production calls `P3.run(..., filler="patch")`**
  (`etape_A.py`, last line of code): this path is not taken.
- No reading of a weight file in the production chain nor in the training closures other than `E3/snap/step_20000.pt` (start of N5 and N7) and `E2/best.pt` (start of E3).
  The measurement scripts (`inferer_v2.py`, `valider_n56.py`, `valider_n7.py`, `coeurs_mesure.py`, `diag_validation.py`) read already trained networks in order to measure them; they are in no closure.
- The denoising of the inputs (`preparer_entree.py`, `preparer_entree_cpu.py`, `stars_dataset.generate` for the old sets) uses
  the application's denoising network (`astro_gpu.stages.denoise`), which is not a star-removal network. The script
  `stars_dataset.py` also contains the function `generate`, which calls a third-party program; it writes `input.npy` **before**
  the call (l. 286 of that file, `np.save(d / "input.npy", ...)`; the program call: l. 288) and `starless.npy` after (l. 295); the chain imports only
  `valid_of` and `sky_variation_of` (`entree_brute.py` l. 17, `preparer_entree*.py` l. 21-22) and never reads `starless.npy`.

## 4. Comparison folders: read by measurement scripts only

- `<DONNEES>/Comparaison référence – 23 nov`: read by `Diagnostic écarts – 25 nov/diag_validation.py` l. 32 (constant `REF`), l. 152
  (`outs["reference"]`). No training or production script mentions it.
- `starless*.npy` of the `stars_dataset/*` folders (starless outputs of other methods, corrected or not): read only by
  measurement or sheet scripts: `Cœur M31/coeur_m31.py`, `Diagnostic cœurs/diag_coeurs.py`,
  `Controle galaxies compactes/mesurer_controle.py`, `Tour N7/coeurs_mesure.py`, `Tour N7/valider_n7.py`, `Melanges N5 N6 N7/*`
  (copies), `Bibliotheque images/scripts/part_b.py`, `Methode v*/vues_mesures_m8_sho.py`.
  Files `starless_fixed.npy`, `starless_reseau.npy` and `cible.npy` exist beside the `input.npy` / `original.npy` of some images
  (`paires_nb/<image>`, `_revue_combi/<image>`, `stars_dataset/<set>`). **They are not read**: `images.py` opens the input file by name,
  no script of the chain names them, and the numerical check on the 90 tiles concerned (see `MANIFESTE_PAIRES.csv`, column `preuve_cible`) shows that the target
  equals none of them, even on the pixels where they differ strongly from the input.

## 5. Doubts, stated plainly

1. **APOD images**: these are published images, already processed by their authors with unknown software. The visual sorting discarded
   the "reduced stars" and "starless" images, but we cannot prove that none underwent star processing by
   third-party software before publication. That processing is **earlier** and is part of the input; the **target** is still computed by
   our chain from that input (nothing from any other removal program is read). The input is a delinearised image
   (inverse stretch estimated on the stars, `Delinearisation APOD – 24 nov`), not a star removal.
2. **Ownership of the non-APOD images**: the code does not say who took each image (see the folder's `RESUME.txt`, outside the repository).
3. **Exact versions of the code at the time**: see the `commit_*` and `scripts_non_modifies_*` columns of `MANIFESTE_PAIRES.csv`.
   For the base v33 pairs, the git repository contains the production script only after the first pairs; the orchestrator
   has not changed since its files (modification date earlier than the pairs for 168 lines out of 182).
   `preparer_g_h.py` was modified (part h) after the v33g pairs were written: the `g` mode of `verifier_provenance.py` redoes
   step g with the current version and compares it to the files (result in `provenance/README.md`).
4. The engine's catalogue reader (`astro_gpu.core.<reader>`, used by `Piste1/astrometrie.py`) carried a third-party software name in
   its module name: it is a **Gaia catalogue** reader, not an image reader; to be renamed before publication if one wants to avoid the
   question.
5. `train_e.py` was modified after E3's training (see `MANIFEST_E3.md`, section 5).
