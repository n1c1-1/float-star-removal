# The method, step by step

Each step says **why** it exists, **how** it is done, what the **key parameters** are and what **measured effect** changing them had. Dates are those of the work (Nicolas's PC clock). The folder names under `scripts/` keep the original date (e.g. "Tour N7 – 25 nov").
The measurement figures come from `resultats/` (protocol: `VALIDATION.md`). What failed is in `LESSONS.md`.

Overview: we want a network that receives a linear image in "noise units" (each colour minus its sky, divided by the sky noise) and returns the same image without stars. The network sees a fixed stretch, `asinh(v / 3) / 10`, undone exactly at the output.

---

## Step 1 — E2 and E3: trained on synthetic data only (7 to 9 November)

**Why.** A first starting point that is **entirely ours**: no image, hence no question of origin. A background is fabricated, stars are placed on it, and the target is the background.

**How.**
- `scripts/make_dataset.py` builds the set from code and seeds only. It reads no image; it reads only `catalogue.json` (public data sheets of 43 telescopes/lenses and 21 cameras). Its manifest gives the sha256 of every file, and `--verify` regenerates pairs to compare.
- Backgrounds come from `backgrounds.py` (emission nebulae, Sérsic galaxies, reflection nebulae, dust) and `defects.py` (mottling, vignetting, gradient, donuts, stacking edges).
- Stars come from `stars.py` and `psf.py`: a shape computed by Fourier optics (pupil with obstruction and spider, Zernike aberrations, turbulence, tracking, pixels), an r⁻² halo, diffraction spikes, reflections, colour-by-colour saturation.
- `scripts/train_e.py` trains the chain's network (NAFNet, width 32, encoder 2-2-4-8, middle 12, decoder 2-2-2-2, 256 px tiles, batch of 8, L1 loss on the fixed stretch). In addition, 60 % of the tiles receive stars drawn on the fly.
- E2 (night of 7 to 8 Nov): pairs regenerated with the saturation colours and the galaxy cores. E3 (8-9 Nov, `day_e3.py`): "pairs_v4" pairs in which giant stars (Alnitak class) are **kept whole** in the target, with saturation and bright tail calibrated on our measurements; resumed from E2's best weights, for 6.5 h, learning rate 3.7e-4, 950 training pairs and 25 held out.

**Key parameters and effects.** The share of stars drawn on the fly (60 %) and the choice of step: E3 step 20,000 is the one kept (it is the starting point of N5 and N7). Result: on **real** images these networks were 3 to 4 times worse than on synthetic data, hollowed out galaxy cores, and the synthetic score misled the choice of snapshot (`LESSONS.md`). Hence the following steps.

---

## Step 2 — The classical chain v33: Gaia-guided subtraction, refilling, passes (20 to 23 November)

**Why.** We need **real** pairs (real images, real defects) whose target is fabricated **without any third-party network**. We start from Nicolas's idea: "training images that can be generated automatically by the hundred, even if it takes time". The classical chain removes the stars by calculation: we know where they are (Gaia), we fit a star-and-halo model and subtract it. It is slow (≈ 4 minutes per 512 px tile) but depends on nothing else. The folders `scripts/Methode v33b – 23 nov`, `Production v33 – 23 nov`, `Juge v33 – 23 nov`, `Juge v33d – 24 nov` hold its code.

**How, in order.**

1. *Image sheet*, measured once on the whole image and then frozen: astrometry and deep Gaia catalogue (G < 21) offline; per colour, star width, scale of the bursts, saturation plateau, star model (at least 30 isolated stars, radius 28 px, wing taken from the brightest ones), spike angles of the instrument.
2. *Track 1 — subtraction.* At the Gaia positions, only the brightness of each star is free (forced photometry). A star is removed if its flux exceeds 1.5 times its error; the position is specific to each colour if the peak exceeds 20 noise units; two stars are fitted together only if less than 2.5 magnitudes apart. **Track 1 fills nothing**: it also returns the list of doubtful zones.
3. *Stage A — big stars.* Radial profile specific to each star; extended image shape only for 7 ≤ G ≤ 10.5; spikes at the instrument's angles (±2°), one template per colour and per filter, followed down to ≈ 0.45 noise; saturated cores recreated (smooth level of the surroundings + sky grain).
4. *Stage B — last pass ("hybrid").* Refill what remains, bright or dark, near the catalogue stars and along the known spikes; isolated pixels of 1 to 3 px always healed; grain of 2 to 4 px put back only in the changed pixels. Fine nebula filaments are protected (an elongated residual is refused, except within the band of a known spike).
5. *Automatic guard per zone.* Ratio "light removed / light expected from the catalogue stars": outside 0.7 to 1.5, the zone is discarded.

**The judge and the weights.** A pair is `input`, `target`, `weight`. Weight 1 = real pixels or a zone redone well; 0.5 = repair judged good; **0 = badly redone zone**, which the network must not learn (rings, wrong level, islands, star remnants, failed halos). Local rules of judge v33: the target must never be brighter than the input (blob ≥ 25 px or difference > 5 noise units → weight 0, dilated by 3 px); a star left in the target (compact peak > 20 noise units at a Gaia position) → weight 0 on a disc of radius max(2.5 FWHM, measured remnant, cap 4 FWHM). Judge v33d (24 Nov): weight 0 on the halo (radius measured by the radial profile, cap 40 px) and on the faint stars left. The judge only **sets weights to 0**: it never modifies the target.

**Tiles.** No grid: 3 to 5 tiles of 512 px **chosen** per image on content (nebula, galaxy, stars of all sizes), never black sky; weight 0 where the chain fails. The big stars (G ≲ 7) stay at weight 0 in the real pairs: they are learned "to perfection" from the synthesis. Two versions of each input: denoised and raw (raw = raw input minus the same star layer).

**Key parameters and measured effects (log of trials v1 to v33).**
- Deep Gaia catalogue (G < 21) instead of detecting the faint ones in the residual: faint stars left from 22 % to 2-3 %.
- Star model built on the whole image (instead of 9 stars per crop): corrects the wing, makes the rings disappear; a flux 11 times too strong avoided by the frozen sheet.
- Pairs fitted only within 2.5 mag: the ring of the G 12-14 disappears; "residual core" zones refilled: dipoles from 57 to 7.
- Last pass within the spike band (v21): the dotted spike remnants disappear, filaments and nebula unchanged.
- v33 (23 Nov): blue discs, red and magenta patches removed by a scaled grain; production 247 s per tile (1 GPU worker for track 1 + 3 CPU workers).
- Production on Nicolas's images: about a hundred tiles, each sheet reviewed by eye (15 pairs dropped); 109 denoised and 36 raw pairs kept at production (the training reads 103 denoised, 35 raw and 6 held-out denoised files: why one raw pair is not read is not established in the logs); on average ≈ 21 % of pixels at weight 0.
- **Web images** (APOD), 24 Nov: delinearised (inverse stretch estimated on the stars, published method "radiometric response curve"), astrometry, deep Gaia, then the same chain (variant v33b): 65 tiles produced, 38 kept from 13 images.

Exact provenance of each target: `PROVENANCE.md`.

---

## Step 3 — Targets v33g: putting back the slow diffuse light (25 November)

**Why.** Diagnosis of 25 Nov: track 1 subtracts the model of **all** the stars of the deep catalogue, wings included out to 150 px, even in the "untouched" zones. The targets were therefore darker than the input in the diffuse sky; the network learned to erase smooth diffuse light (M31's disc, the core of M65/M66).

**How.** `scripts/Tours N5 N6 – 25 nov/preparer_g_h.py g`: D = input − target; M = class-0 pixels (untouched) outside the discs of stars G < 8; L = normalised Gaussian smoothing of D over M (sigma 40 px, per channel) with iterative clipping (4 rounds, pixels rejected beyond 2.5 robust deviations); target v33g = target + L.

**Effect.** Median of (input − target) over M: +0.12 → −0.05 noise. N5 (targets v33g) versus N2 (targets v33d): stars G 14-17 0.313 → 0.265, background 0.0015 → 0.0011, background p99 0.12 → 0.07; structure 0.005 → 0.007 (slightly higher).

---

## Step 4 — Realistic synthetic stars and the N2 recipe (22 to 25 November)

**Why.** We need real pairs **and** synthetic stars placed on real backgrounds. The synthetic stars must be **very realistic**, because under a saturated core the truth exists in no real image.

**How.**
- `scripts/Etoiles synthetiques – 22 nov/etoiles2.py`: generator v2, 8 tuning rounds (22 Nov) with a blind test (a judge had to tell real from fake thumbnails; the goal is a judge error rate close to 50 %, i.e. indistinguishable stars; the notes of each round are in `scripts/Etoiles synthetiques – 22 nov/score_aveugle.py` and its score files, and the lesson from this test is in `LESSONS.md`); banks of real bright-star shapes (medians of 5 to 8 stars measured in images **with** their stars); 19 ms per 256 px tile.
- N2 recipe (24 Nov, `train_nuit.py --n2`): weights of 0.5 raised to 1; on 50 % of the synthetic tiles, 1 or 2 imposed very bright stars (peak 300 to 300,000 noise units, big halo, spikes) placed on a real nebula background (where the target has weight 1 on a 40 px disc); draw 50 % denoised, 25 % raw, 25 % web; start from E3 step 20,000; batch 8, learning rate 5e-4, 1,000-step ramp then cosine; 60 % of tiles with synthetic stars.

**Key parameters.** Share of synthetic tiles (60 %), share of imposed very bright stars (50 % of the synthetic ones), weight (0.5 → 1: no visible effect on faint stars, N1 = N2 at 0.312 / 0.313).

---

## Step 5 — Synthetic galaxy cores (N7, 25-26 November)

**Why.** N5 erases galaxy cores: the flux at M31's core goes from 369 to 5-7 (cores 0.19). Working hypothesis: the network takes a galaxy core or a planetary nebula for a star, for lack of any example of a compact extended object **kept** in the target.

**How.** `scripts/Tour N7 – 25 nov/paires_synthetiques_n7.py` (`coeur_synthetique`): on 25 % of the synthetic tiles, 1 or 2 cores (Sérsic n 1-4, effective radius 4-60 px, ellipticity 0-0.6, peak 50-5,000 noise units, 30 % clipped, yellow-white galaxy or blue-green planetary, full width at half maximum ≥ 2.5 times that of the tile's stars) present **identically in the input and the target**, weight 1.

**Effect.** Cores 0.19 → 0.81. Cost: stars G < 8 0 → 0.021; G 11-14 0.072 → 0.094; G 14-17 0.265 → 0.283; median background 0.0011 → 0.0023. Gains: background p99 0.07 → 0.03, structure 0.007 → 0.004. **Side effect found on 26 Nov**: because the synthetic cores include clipped cores of 50 to 5,000 noise units, N7 also keeps saturated stars of low signal-to-noise ratio (flat top at ≈ 100 noise units): a "saturated star / compact core" conflict (`IMPROVING.md`).

---

## Step 6 — Fine-tuning from E3

**Why.** Starting from zero on the real pairs is weak (24 Nov: a from-scratch trial was worse than from E3); E3 already knows the general shape of a star. **How.** All trainings N1 to N9 start from the E3 step 20,000 weights (`--depart`). N5/N6: 34,500 steps (≈ 5 h, 9 minutes per 1,000 steps), snapshot every 1,000 steps, **best step = smallest error "under the stars"** on 6 held-out tiles (N5: step 23,000; N6: 20,000; N7: 22,000). Validation becomes flat from about 22,000 steps: a longer run is useless (N8L, 24,000 steps, did no better than N7).

---

## Step 7 — Weight blending between networks with the same start

**Why.** Each training has different flaws; the average of their weights is one more network, **free** (no data, no training). **How.** `scripts/Melanges N5 N6 N7 – 26 nov/chaine_melanges.py`: a·A + (1 − a)·B over the 664 floating-point tensors (same architecture, same keys and shapes, integer buffers taken from A). **Rule**: blend only networks that started from the same point (E3); across different starts it does not work.

**Effect** (see `resultats/tableau_melanges.md`): N7/N5 0.75 → cores 0.798, stars G < 8 0.021, background 0.0018, structure 0.003 (slight gain over N7 alone: background 0.0023 and structure 0.004) at the price of slightly higher G 14-17 (0.295 versus 0.283). N7/N5 0.5: G < 8 at 0.000 but cores 0.653. The N5/N6 blends keep the core erased (0.13 to 0.16).

---

## Step 8 — Two-scale inference V2 (25 November)

**Why.** On M27, the very big star (G 5.6, halo of 285 px) became a **flat grey-blue disc**: the network's reach (≈ 180 px) is smaller than the halo, and larger tiles change nothing (tried: identical residual). Shrinking the whole image ×2 makes the disc disappear but damages the rest.

**How.** `scripts/Portée M27 – 25 nov/deux_echelles_v1v2.py` and `scripts/Tours N5 N6 – 25 nov/inferer_v2.py`: for each Gaia star with G < 6.5, the halo radius is measured in the input (first radius where the median radial profile minus the local background falls below 2 noise units; background = median of the 400-500 px ring, otherwise 300-400; cap 300 px). Within this radius, the output is that of the network applied to the image reduced ×2 and enlarged again; blending: full up to 70 % of the radius, cosine up to 100 %.

**Effect.** Remaining stars G < 8 0.012 → 0.000; background and structure unchanged (structure 0.005 → 0.009 in the fixed-formula version, restored to 0.005 with the measured V2 radius); M27 clean. Share of the image replaced: 12.6 % for M42 (V2) versus 49.5 % with the flux-square-root radius formula.

---

## What was not kept

See `LESSONS.md`: N3/N4 (large synthetic halos), N6 (weights v33h), putting the slow part back at inference, N8/N8L (cores at 10 %), N9 (compact cores), N5/N6 blends.
