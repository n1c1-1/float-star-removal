# Captions of the example images

These images are **entirely synthetic**: no real image is read to make them. They are redone
exactly (bit for bit) by `exemples/generer_exemples.py` (the seeds are written at the top of the script; the sha256
checksums are in `exemples/EMPREINTES.txt`, the drawn parameters in `exemples/PARAMETRES_TIRES.txt`).

## How to read an image

Each image has three panels: **Input (with stars) | Target (without stars) | Difference = stars removed**.

- The three panels have the **same stretch**, the network's: `asinh(v / 3) / 10`, where `v` is the linear value in
  sky noise units (the sky is 0, its noise is 1), multiplied for display by a fixed gain (4.0) and offset
  by a fixed black (0.04). The "difference" panel is displayed with the same function, applied to input minus target.
- The sheet `planche_synthetique.jpg` assembles the six examples (one row per example, three columns); its tiles are
  reduced to 250 px on a side (block averaging), which smooths the noise a little compared with the individual images.

## Why the target is exact by construction

1. **The background is known before the stars are placed on it.** The background (the sky without stars, with its noise) is made
   first. The stars are then added on top: the input is "background + stars".
2. **The target is this background, copied as is.** The script checks it at every run (`cible == fond`, array equality,
   no tolerance; for example 4, `cible == fond + galaxy core`, the core being added before the stars).
3. **The background noise is the same in the input and in the target**: the sky noise (correlated between neighbouring pixels) and the
   photon noise of the background light are part of the background. Only the stars bring their own photon noise,
   and it is only in the input. The difference input minus target therefore contains only the stars (core, halo,
   spikes, clipping plateau, photon noise of their light).
4. **The target is never computed by other software**: no star removal, no manual correction, no background estimate. It
   is the starting datum, not a result.

## What is generated (common to the six examples)

- **Background**: `scripts/backgrounds.py`, function `background`, generator `numpy.random.default_rng([background_seed, 0])`
  (nebulae, galaxies, dust, sky level and slight gradient), then the sky noise (generator
  `[background_seed, 3]`): noise correlated at 0.09 to 0.42 between neighbouring pixels, divided by the square root of 2 (full stack
  of two half-stacks), plus the photon noise of the background light.
- **Stars**: `scripts/Etoiles synthetiques – 22 nov/etoiles2.py` (image styles: instrument drawn from
  `scripts/catalogue.json`, Fourier optics, halo, spikes, per-star clipping plateau, colours, photon noise)
  cached in 16 styles (seed 2026), then the fast draw `paire_synthetique` (power-law counts, 50 to 6000
  stars per million pixels above one noise unit, colours of our fields, cores by convolution, halo and spikes of
  stars above 20 noise units).
- **No bank of measured shapes.** The generator can also borrow star shapes measured on real
  images; these banks are not in the repository. These examples therefore use **only the parametric model**: the
  very bright ones have clipping plateaus in the shape of a rounded square and spikes drawn by formula, less varied than
  in a real image.
- **Noisy version only**: the "denoised pairs" path (which requires the application's denoiser) is not used.
- Computed on CPU, 4 threads; no image read, no network access.

## The six images

### exemple_01_champ_lisse.jpg — star field on a smooth background
- Tile 256 x 256 px, 3 colours. Background seed 1114, star seed 1114.
- Background: very faint emission nebula (99.9th percentile level 1.5 noise units), gradient 0.26; correlated noise 0.125, photon_k 0.108.
- Stars: image style no. 2 (full width at half maximum 3.29 px, 2 spikes, colour), density drawn by the seed, no imposed star.
- To see: dozens of sharp stars, from faint to bright, on an almost empty sky; the target keeps only the faint nebula.

### exemple_02_tres_brillantes.jpg — very bright stars on a nebula
- Tile 384 x 384 px. Background seed 1203, star seed 1203, module `paires_synthetiques_n7.py`.
- Background: emission nebula (99.9th percentile 27.8 noise units), ionisation front, correlated noise 0.367, photon_k 0.272.
- Stars: style no. 15 (3.74 px, 4 spikes); two very bright stars **imposed** (parameter `extras`): peak 30,000 noise units at
  (110, 120) and peak 9,000 noise units at (290, 90); the field of free stars is drawn by the seed.
- To see: the two clipped stars (rounded-square plateau) placed on the nebula; the target shows the nebula intact beneath them.

### exemple_03_halo_aigrettes.jpg — star with a large halo and spikes
- Tile 384 x 384 px. Background seed 1304, star seed 1304, module `paires_synthetiques_n3.py` (large halos).
- Background: emission nebula with filaments, absorbing dust (tau 1.73 at the 99th percentile), correlated noise 0.246, photon_k 0.148.
- Stars: style no. 15 (3.74 px, 4 spikes); one star imposed at the centre (192, 192), peak 37,000 noise units, with a halo
  at the scale of the biggest stars in our images: 38 noise units at 60 px, 1 noise unit at 285 px (slope 2.33), four spikes
  carried out to 1.25 x 285 px.
- To see: the halo covers almost the whole tile; the target recovers the nebula hidden beneath (this is the hardest example).

### exemple_04_coeur_galaxie.jpg — galaxy core kept
- Tile 384 x 384 px. Background seed 1407, star seed 1407, `paire_synthetique_coeurs` with `p_coeurs = 1` (a core always present).
- Background: very faint reflection nebulae (99.9th percentile 1.2 noise units), correlated noise 0.345, photon_k 0.023.
- Synthetic core (present identically in the input AND in the target): Sérsic profile n = 2.61, effective radius 13.9 px,
  ellipticity 0.345, central brightness 123 noise units, galaxy colour (yellow-white), not clipped, full width at half maximum 10.6 px.
- Stars: style no. 10 (3.61 px, no spikes, narrow-band mode), dense field drawn by the seed.
- To see: the core, compact and bright like a star but extended, stays in the target; the difference contains only
  stars (no trace of the core). This is the case that a network that has never seen one would confuse with a star.

### exemple_05_champ_dense.jpg — dense field
- Tile 256 x 256 px. Background seed 1501, star seed 1501.
- Background: small emission nebula with an oxygen component (99.9th percentile 5.4 noise units), correlated noise 0.194, photon_k 0.051.
- Stars: style no. 15 (3.74 px, 4 spikes), high density of medium-to-faint stars drawn by the seed.
- To see: dozens of stars sometimes overlap; the target keeps the nebula and its noise identically.

### exemple_06_fond_defauts.jpg — background with stacking defects (set 2)
- Tile 256 x 256 px. Background seed 1611 (the defects: `default_rng([1611, 1])`), star seed 1611.
- Background: faint emission nebula plus the defect layer of `scripts/defects.py`: grain 0.010; mottling of 8.5 px
  amplitude 0.107; large scale 0.091 (gradient, vignetting, glow); 1 dust ring (donut); 2 stacking edges (a
  bluer band on the right edge); shifted sky tint (R - G = -0.32, B - G = -0.10).
- Stars: style no. 11 (4.95 px, no spikes, colour), a few stars drawn by the seed, including a bright one.
- To see: the defects are part of the background, hence of the target: a network must remove the stars without smoothing or correcting the sky.

## Limits

- The star shapes come from the parametric model alone (no bank of measured shapes, no denoised version).
- The stretch is a display stretch; the original arrays are in noise units, linear, 32-bit floating point.
