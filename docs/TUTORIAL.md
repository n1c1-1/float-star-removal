# Step-by-step appendix: remove the stars of your own image

Three routes, from the shortest to the longest, from end to end. For each: what you do, what you should see, the time and the resources. **What was really run to write this tutorial is marked "tried"**; the rest is written from the scripts and the logs, and says clearly what remains to be confirmed.

Reference hardware: Windows 11, 16-thread processor, RTX 4060 Ti card with 16 GB for training; the CPU alone is enough for routes 1 and 2.

## Route 1 — Try a network in 5 minutes (CPU, no image to download)

**Setup.**
```
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
```
Get `E3_step_20000.pt` (Releases page; 117 MB) and check it: `sha256sum -c modeles/SHA256SUMS`.

**1. Make the toy image** (flat sky of value 1000, noise 5, four Gaussian stars with peaks of 60, 200, 800 and 3000 times the noise):
```
python exemples/image_jouet.py image_jouet.npy
```

**2. Remove the stars:**
```
python scripts/inferer_une_image.py image_jouet.npy --poids E3_step_20000.pt --sortie sortie_jouet --carte cpu
```
*What you should see*: the console prints the sky (≈ 997) and the noise (≈ 5.0) estimated per colour, then "écrit dans sortie_jouet". `sortie_jouet/apercu.jpg` shows three thumbnails (input | starless | stars): the four stars are now only in the right-hand thumbnail. **Tried**: at the four positions, the output value in noise units is −0.13, −0.13, 0.05 and 0.45 (the faintest, 13 noise units, leaves 0.45 noise), against 599, 161, 40 and 13 in the input.
*Time*: of the order of a minute for 512 × 512 on CPU (not precisely timed), less on the card (`--carte cuda`).

**3. See what the network has learned:** look at `docs/images/planche_synthetique.jpg` (input | target | difference) and redo these images identically with `python exemples/generer_exemples.py` (≈ 2 minutes, CPU, 4 threads; the sha256 checksums in `exemples/EMPREINTES.txt` must be identical: **tried**, three byte-identical runs on the reference machine).

## Route 2 — A public image (APOD), from the URL to the training pair

Chosen image: "Orion and the Running Man", APOD of 10 March 2023, `https://apod.nasa.gov/apod/ap230310.html` (number 070 of our list, `docs/WEB_SOURCES.md`). The image belongs to its author: it is not redistributed here; you download it from the page.

The path is the one that made the web pairs of the training:

| step | what you do | script | what you should see |
|---|---|---|---|
| 1. image | download the page's image file (column `url_image` of `provenance/MANIFESTE_PAIRES.csv`); its sha256 must be that of `sha256_image_telechargee` | `provenance/refaire_paires_web.py --telecharger` (or by hand) | a stretched (non-linear) image, ≈ 2 MB |
| 2. delinearise | recover an "as if linear" image: the response curve is estimated on the stars (method `etoiles_auto`) | `scripts/Delinearisation – 23 nov/methodes.py`, `scripts/Delinearisation APOD – 24 nov/delineariser.py 70` | a floating-point array (3, H, W); the curve, the number of stars seen and the chosen colour are written in the log (**tried**: ≈ 7 s, values identical to the manifest) |
| 3. astrometry | find the sky coordinates of each pixel (TAN-SIP solution) | `scripts/Delinearisation APOD – 24 nov/resoudre.py 70` (uses the application's local solver) | a `070_astrometrie.npz`; failing that, any other solver that writes the same format |
| 4. deep Gaia extract | download the Gaia DR3 stars (G < 21) of the field, **with the supplied script**, once, outside any computation | `scripts/Delinearisation APOD – 24 nov/telecharger_gaia_apod.py 70` | `gaia_dr3_profond_apod_070.npz` (one request to the public Gaia data service; 3 s wait between requests) |
| 5. choose a tile | pick 512 px tiles on content | `scripts/Delinearisation APOD – 24 nov/choisir_tuiles.py 70` | a numbered mosaic; the example's tile is at (1536, 1408) |
| 6. prepare the input | background and noise per colour, unit = mean noise | `scripts/Production APOD – 24 nov/preparer_apod.py` | `entrees/apod_070/input.npy` |
| 7. redo the pair | track 1 (Gaia-guided subtraction) → stage A → stage B → judge | `scripts/Production APOD – 24 nov/tuile_cpu_apod.py apod_070 1536 1408` after track 1 (`etape_piste1_gpu_2f.py`); or all in one: `provenance/verifier_provenance.py ... --mode rejeu` | a `.npz` pair: input, target, weights; **tried**: ≈ 74 s on CPU, target **bit-identical** to the training one |

**What is really tried, and what is not.** Tried on 26 Nov, on CPU: `provenance/refaire_paires_web.py --paire apod_070_g1536_1408 --local` (local image, sha256 checked, delinearisation, input and pair redone, **everything bit-identical**). **Not tried end to end by a third party**: the download (`--telecharger`, no network access during preparation) and steps 3 and 4 (astrometry and Gaia extract) from a blank image. Today, `refaire_paires_web.py` needs, for each image, unpublished **derived** files (astrometry, deep Gaia extract, star model: variable `RACINE_ORIGINE`): publishing them as a release attachment is still to be decided (they are coordinate and catalogue files, not image pixels).

**Resources.** Without a graphics card: track 1 takes 1 to 14 minutes per 512 px tile on 4 threads (803 s for a dense tile in our trials); with the card (original production) about 5 times less. Memory: a few GB. Disk: the image and its intermediates, a few hundred MB.

Then apply the network to the whole delinearised image (as in route 1): `python scripts/inferer_une_image.py 070.npy --poids modeles/N7N5_075.pt`. For very big stars (G < 6.5), two-scale inference (`scripts/Portée M27 – 25 nov/deux_echelles_v1v2.py`) needs the Gaia extract from step 4.

## Route 3 — Redo a training

**3a. Synthetic data and E3 (no image).**
```
python scripts/make_dataset.py --n 500 --seed 1 --sets clean,defects --workers 6 --out donnees/pairs_v4
python scripts/make_dataset.py --verify 3 --out donnees/pairs_v4
python scripts/day_e3.py --hours 6.5
```
*What you should see*: "verified: all identical" for the noisy files. Note: the "denoised" files require the application's denoising network (not supplied); the script needs it for these files, and our measurements show that the regeneration of the denoised files is not identical to that of 2026-11-09 (`PROVENANCE.md`, uncertainty no. 2). *Resources*: one pair ≈ 10 s on 14 cores, about 100 MB per pair; E3 took 6.5 h on the 16 GB card.

**3b. Fine-tuning on real pairs: N5 then N7.** You need real `*_v33g.npz` pairs (input, target, weights): (i) those of the attachment (Nicolas's images), (ii) those of the web images redone with `refaire_paires_web.py`. Put them in `<DONNEES>/Paires étoiles v33 – 23 nov/` and `<DONNEES>/Paires étoiles APOD – 24 nov/` (variable `DONNEES`).
```
python "scripts/Tours N5 N6 – 25 nov/train_n56.py" --tag N5 --suffixe v33g --n2 --pas 34500 --depart E3_step_20000.pt
python "scripts/Tour N7 – 25 nov/train_n7.py"    --tag N7 --suffixe v33g --n2 --coeurs 0.25 --depart E3_step_20000.pt
```
*Measured resources*: 34,500 steps ≈ 5 h (≈ 9 minutes per 1,000 steps), batch of 8 tiles of 256 px, 16 GB card, 2 loading processes; the training stops cleanly below 3.5 GB of free memory and resumes with `--reprendre`. An intermediate measurement is written every 1,000 steps (`history.json`); **choose the step with the smallest error "under the stars"** (N5: 23,000; N7: 22,000).

**3c. Measure and blend.**
```
python "scripts/Validation – 22 nov/valider_reseau.py" <weights.pt> --tag N5
python "scripts/Melanges N5 N6 N7 – 26 nov/chaine_melanges.py"      # N7/N5 0.75
```
The validation requires the 13 images of `docs/VALIDATION.md` (not published: replace them with your own images, never seen in training). Short run to test an idea: 10,000 steps ≈ 1 h 30; stopping rule written before launch; do not tune on the control set of small galaxies.

## If something goes wrong

- `ModuleNotFoundError: astro_gpu...`: run from the repository root (`scripts/` contains an excerpt of the host application, `scripts/astro_gpu/`); the chain scripts that do astrometry or denoising need the complete host application (`scripts/README.md`).
- Very different result after a library version change: the "bit-for-bit" identities were checked with Python 3.12.13, numpy 2.5.3, OpenCV 5.0.0, torch 2.11.0.
