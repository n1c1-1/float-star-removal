# The scripts: which script does which step

These scripts are **the ones that were actually used** (copied from the workspace on 26 November 2026). The code is kept as is; only these changed: the paths, a header at the top of each file (role, step, inputs, outputs, example), and the names that contained a third party's name (see "Cleaning" below).

**Folder names keep the original French and date, on purpose.** The scripts call each other by these names, and the dated names (e.g. `Tour N7 – 25 nov`) keep each step traceable to the working log. The same applies to `modeles/`, `exemples/`, `provenance/`, `resultats/` and `prompts/`. Identifiers inside the code (function and file names such as `inferer_une_image.py`, `tuile_cpu*.py`) are also left in French.

## Paths: three variables

| variable | role | default value |
|---|---|---|
| `RACINE_SCRIPTS` | the `scripts/` folder | computed from the file's location |
| `DONNEES` | data and outputs (pairs, trainings, measurements) | `donnees/` |
| `ESSAIS` | older trials (E2/E3, image sets) | `essais/` |

In text strings (docstrings, hard-coded paths in lists), the tokens `<DONNEES>`, `<ESSAIS>`, `<FLOAT>`, `<SCRIPTS>`, `<DISQUE>` replace the old absolute paths: they are **templates to adapt**, not paths that work as they are.

## The steps

| step (see `docs/METHOD.md`) | folder | main scripts |
|---|---|---|
| 1. E2 / E3, synthetic data only | root of `scripts/` | `make_dataset.py` (generated pairs + manifest), `backgrounds.py`, `defects.py`, `stars.py`, `psf.py`, `instruments.py` + `catalogue.json`, `train_e.py` (training), `day.py`, `day_e2.py`, `day_e3.py` (orchestration), `diagnose.py`, `models.py` (networks and stretch), `measure.py`, `star_measure.py`, `donut_measure.py` (measurements that fix the generator's ranges) |
| 2. classical chain v33b | `Methode v33b – 23 nov/` | `etape_piste1_gpu_2f.py` (track 1), `etape_A.py`, `etape_B.py`, `methode_validee.py` (chains everything), folders `Piste1/` (star model and subtraction), `Passes/` (final passes), `Piste3/` (stage A); `Catalogue Gaia profond – 20 nov/` (Gaia extract) |
| 2. production and judges | `Production v33 – 23 nov/`, `Juge v33 – 23 nov/`, `Juge v33d – 24 nov/`, `Choix des tuiles – 23 nov/` | `produire_v33.py`, `produire_v33b.py`, `tuile_cpu*.py`, `construire_paire.py`, `regles_v33*.py`, `rejuger_v33d.py`, `choisir.py` |
| 3. web images | `Images web APOD – 24 nov/`, `Delinearisation – 23 nov/`, `Delinearisation APOD – 24 nov/`, `Production APOD – 24 nov/` | `moisson.py`, `telecharge.py` (collection), `methodes.py` and `courbe_etoiles2.py` (delinearisation "étoiles_auto"), `delineariser.py`, `resoudre.py`, `telecharger_gaia_apod.py`, `preparer_apod.py`, `produire_apod.py`, `tuile_cpu_apod.py` |
| 4. realistic synthetic stars | `Etoiles synthetiques – 22 nov/` | `etoiles2.py` (generator v2), `paires_synthetiques.py`, `images_mesurees.py`, `banque_*.py`, `aigrettes_banque.py`, `aveugle.py` + `score_aveugle.py` (blind test) |
| 6. training on real pairs | `Entrainement jeu 1 – 22 nov/` (`train_jeu1.py`, common base), `Entrainement v33c – 24 nov/`, `Entrainement nuit – 24 nov/` (N1, N2, `paires_synthetiques_n2.py`) | — |
| 7-8. N5 (targets v33g), N6 | `Tours N5 N6 – 25 nov/` | `preparer_g_h.py` (v33g and v33h), `train_n56.py`, `enchainer_n56.py`, `mesurer_n56.py` |
| 9. N7 (synthetic cores) | `Tour N7 – 25 nov/` | `paires_synthetiques_n7.py`, `train_n7.py`, `coeurs_mesure.py`, `mesurer_n7.py` |
| 10. discarded trials | `Entrainement N3 – 25 nov/`, `Entrainement N4 – 25 nov/`, `Tour N8L – 26 nov/`, `Tour N9 – 26 nov/`, `Restauration lente – 25 nov/` | `train_n3.py`, `train_n4.py`, `train_n8l.py`, `train_n9.py`, `restaurer.py` |
| 11. weight blends | `Melanges N5 N6 N7 – 26 nov/` | `chaine_melanges.py` |
| 12. two-scale inference V2 | `Portée M27 – 25 nov/` | `deux_echelles_v1v2.py`, `portee_m27.py`; copy used by the measurements: `inferer_v2.py` in the folders `Tours N5 N6`, `Tour N7`, `Melanges` |
| 13. measurement and controls | `Validation – 22 nov/`, `Controle galaxies compactes – 26 nov/`, `Diagnostic *`, `Cœur M31 – 25 nov/`, `Tete de cheval – 26 nov/` | `valider_reseau.py` + `liste.py`, `mesurer_controle.py`, `chercher_galaxies.py`, `diag_coeurs.py`, `b_tests.py` |
| 14. using a network | root of `scripts/` | `inferer_une_image.py` (remove the stars of your own image), `astro_gpu/` (excerpt of the host application: the network and its tiled application) |
| 15. classical baseline on tiles | `Baseline classique – 26 nov/` | `choisir_tuiles.py`, `lancer_chaine_tuiles.py`, `mesurer_tuiles.py`, `suivre_inference.py`: measurement of the classical chain v33b on one 512 px tile of 9 of the 13 validation images, and of the networks on the same tiles (`tableau_tuiles.md`); see its `README.md` |
| 16. E3 measurement on CPU | `Mesure E3 – 26 nov/` | `inferer_v2_cpu.py`, `valider_cpu.py`, `coeurs_cpu.py`: CPU two-scale inference, validation and core measurement of the E3 network (step 20000) used for the E3 rows of the README tables (copies of `Portée M27/inferer_v2.py`, `Validation/valider_reseau.py` and `Tour N7/coeurs_mesure.py` adapted to CPU-only, nothing else changed) |
| figures of the README | `docs/figures/` | `prepare_data.py` (collects the small data files from the author's result folders into `docs/figures/data/`), `make_figures.py` (draws `docs/figures/*.png` from them) |

`lancer_borne.ps1` launches a command at low priority, on cores 4 to 15, with fixed thread counts: this is what was used for all the computations of these days.

## Why three copies of `inferer_v2.py`, `valider_n7.py`, `coeurs_mesure.py`?

Rule followed throughout the work: **never modify a validated script; copy it and change one thing**. The copies of a same script are therefore successive versions (see their docstring: "copie de ..."). The most recent one is in the folder of the last round.

## Chain v33 or v33b?

The training pairs come from both: v33 for the production of 23 Nov, v33b (which fixes four track-1 crashes and differs by guards on the edges) for four images rerun and for all the web images. Only v33b is published here (the `Production v33` folder calls `Methode v33b`); the verification replays of the v33 pairs (`provenance/`) are **bit-identical** with chain v33b on the tiles tried.

## What depends on the host application (FLOAT) and is not supplied

| component | used by |
|---|---|
| offline astrometric solving (`astro_gpu.core.asterion`) and local catalogue reader (imported here under the name `catalogue_local`; it has another name in the application) | `Piste1/astrometrie.py`, `resoudre.py`, `essai_astrometrie.py`, `preparer_apod.py`, `entree_brute.py`, `portee_m27.py` |
| denoising network (`astro_gpu.stages.denoise`) | `make_dataset.py` ("denoised" files), `entree_brute.py` |
| Gaia extracts and astrometry already computed for each image | variable `ASTRO_GPU_GAIA_EXTRACTS`; derived files not published (see `docs/TUTORIAL.md`) |

**Supplied**: `astro_gpu/stages/star_removal.py`, an excerpt of the application (NAFNet network, stretch, tiled application) that is enough for inference and for all the training and measurement scripts.

## Cleaning applied to the copies

- absolute paths replaced (variables above);
- identifiers that named third-party software are replaced by the neutral token `refext` / `RefExt` (variables, output files);
- the first names of third parties in image names are replaced by `tiers1`, `tiers2` (e.g. `ngc7000_tiers1`): the real names of these data files differ;
- the matrix product operator is written `operator.matmul(a, b)` and decorators are written as function calls (same behaviour);
- no image, no `.npz`, no star-bank cache is copied. For the generators that use `cache_styles.npz`, it is regenerated: `python paires_synthetiques.py cache 32`.

All scripts compile (`python -m py_compile`). Many launch long computations or read unpublished data: read the header before running them.
