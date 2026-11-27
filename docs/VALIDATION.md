# Validation: how we measure, honestly

An improvement is declared only on images **that the network has never seen** and that nobody has used to tune anything. This document describes: the measurement protocol, the list of validation images (without the images), the control set of small galaxies, the saturated-star safeguard, and the pitfalls already met.

## 1. The measurement protocol (`scripts/Validation – 22 nov/valider_reseau.py`)

For each whole image of the list (inference by overlapping tiles, network = `chain_nafnet`, stretch `asinh(v/3)/10`), the script measures:

1. **Remaining stars**: remaining flux / original flux at the Gaia catalogue positions (aperture of 2 FWHM, background = median of the 3-5 FWHM ring of the same image), per-image median ratio and 90th percentile per G magnitude band: < 8, 8-11, 11-14, 14-17, > 17 (0 = removed, 1 = intact), and the **share of stars with a ratio above 0.1** (`part_gt_0.1`). The tables of this repository and of the README report this share (mean over the images having at least 5 stars in the band), not the median ratio, which is near 0 for the first three bands.
2. **Damage to the background**: |output − input| (mean of the 3 colours, in noise units of the input) on the pixels far from any detectable Gaia star (radius = max(4 FWHM, radius specific to the magnitude)): median ("background median") and 99th percentile ("background p99"), then the same measure on the "bright structure" (smoothed input above its 90th percentile: "structure median"). **Unit: 0.001 is a thousand times below the noise, invisible.**
3. **Exact error under synthetic stars**: 20 tiles of 256 px far from bright stars, `paire_synthetique(fond, graine)` (target = background), L1 on the asinh stretch over the footprint of the stars, per brightness class. The only measure where the truth is exact.
4. **Sheet**: input | output | difference at the same stretch (not published: the images are those of the list).
5. **Cores** (`coeurs_mesure.py`, added on 25 Nov): ratio of output flux / input flux in a 15 px disc around 6 galaxy cores (M31 and M32 on two images, M65 and M66 on a third; ideal 1). The "M32 60D" column isolates the compact core of M32 in the photo from a Canon 60D.

The "two-scale V2" inference (step 8 of `METHOD.md`) is applied to all measurements from 25 Nov to 26 Nov. The script also accepts a **folder of starless outputs (any method)** (`<name>.npy`, same units as the input) and scores it the same way. Such outputs never enter a training set (see `PROVENANCE.md`).

Command: `python "scripts/Validation – 22 nov/valider_reseau.py" <weights.pt> --tag <name>`.

## 2. The validation images (list; no image is published)

Preparation: same chain as the training (gradient removed, noise units, background 0, denoising network, white balance). None of these images has a tile in the training pairs (no training target).

| name | type | instrument (sheet) | scale (″/px) | what the image tests |
|---|---|---|---|---|
| m31_a7s_newton | galaxy with bright core | full-frame hybrid camera, filter removed, Newton 130/900 | 1.28 | spider spikes, very bright galaxy core |
| ic405_a7s_newton | nebula | same | 1.28 | spikes on bright stars in a nebula |
| m33_osc_lunette | diffuse galaxy | ASI294MC Pro colour camera, 350 mm refractor | 1.81 | stars without spikes, HII regions |
| m27_mono_nb | compact planetary, assembled narrow band | ASI1600MM mono camera, Newton 1325 mm, Ha/OIII/SII filters | 0.59 | very contrasted compact planetary, 3 filters |
| ngc3372_mono_nb | rich field, narrow band (Ha+OIII) | 0.4 m telescope of an online network | 0.74 | dense Milky Way, nebula, very bright star |
| m16_seestar_s50 | smart telescope | Seestar S50, 250 mm | 1.58 | small instrument, short stack |
| ngc6934_stellina | globular cluster | Stellina, 400 mm | 1.24 | very tight stars, saturated core |
| leotriplet_canon6d_newton | field of faint galaxies | Canon 6D DSLR, Newton 200/1000 | 1.35 | spikes, FWHM of 8 px, M65/M66 cores |
| m31_canon60d_objectif_filee | trailed stars | Canon 60D DSLR, 200 mm lens, unguided mount | 4.56 | elongated stars, wide galaxy, M31/M32 cores |
| m42_canon40da_lunette | Orion's core | Canon 40Da DSLR, 72ED refractor | 2.73 | Trapezium, halos of bright stars |
| ic5146_canon60da_lunette | dark and bright nebula | Canon 60Da DSLR, refractor | 2.0 | star field + diffuse nebula |
| m42_dobson_grande_fwhm | large FWHM (10 px) | ASI533MC Pro, 12-inch Dobsonian | 0.51 | very wide stars: everything changes scale |
| ngc7822_a7s_lunette | faint extended nebula | hybrid camera, filter removed, refractor | 4.25 | few bright structures (replaced by the next one: the catalogue of a 7.5° field was too slow) |
| ic1848_stellina | faint nebula, smart telescope | Stellina, 400 mm | 1.24 | faint nebula in a rich field, FWHM 4 px |

The tables count **13 images**: the entries of `scripts/Validation – 22 nov/liste.py` that have a Gaia extract. The 14th entry of the list, NGC 7822, was replaced by IC 1848 (the Gaia query for a 7.5° field was too slow) and has no extract.

## 3. The control set of small galaxies (26 Nov): used only for judging

**Goal.** To know whether a network **keeps** small compact galaxies without having specialised on M32. This set is used only for judging: never for tuning, never for training.

**Construction** (`scripts/Controle galaxies compactes – 26 nov/`):
- 5 published web images (address list: `WEB_SOURCES.md`, "control" section) already delinearised and plate-solved with the offline deep Gaia catalogue;
- detection by `chercher_galaxies.py`: source **absent** from the Gaia extract (nothing within 1.2 star FWHM), FWHM ≥ 1.2 times that of the 15 nearest reference stars, peak ≥ 20 noise units, no Gaia star G < 15 within 10 FWHM, no neighbour more than 30 % as bright, faint ring (not inside a large galaxy), edges at 192 px;
- final choice **by eye** (`revue_candidats.py`, 64×64 thumbnails): 18 galaxies (1.2 to 2.8 star FWHM); almost all real galaxies (spirals, ellipticals, edge-on); the images whose Gaia astrometry does not fit were discarded;
- **independence**: none of the validation images; 3 images with no training tile at all, 2 images with tiles in which every galaxy is at least 64 px away.

**Measurement** (`mesurer_controle.py <weights.pt|folder of outputs> --tag name`, CPU): 384 px crop around the galaxy; ratio of output flux / input flux within 3 px and 8 px (mean of the 3 colours, background = median of the 24-40 px ring). **Ideal: 1 for the galaxy.** Safeguard: the same ratio on the Gaia star **of the same brightness** for each galaxy (ideal 0).

**Reference results (medians over 18 galaxies, galaxy / star of the same brightness):**

| network | 3 px | 8 px |
|---|---|---|
| N7 step 22,000 | −0.002 / −0.002 | −0.005 / −0.006 |
| blend N7/N5 0.75 | −0.001 / −0.002 | −0.006 / −0.004 |
| N9 step 10,000 | ≈ 0, like N7 | ≈ 0 |

All of them erase almost all the galaxies; only no. 17, which is diffuse, remains. **This set therefore defines the objective**: a network that keeps the galaxy median near 1 **without** keeping the star of the same brightness.

## 4. The saturated-star safeguard (diagnostic of 26 Nov)

A test to run in addition to the table, before announcing that a network "keeps the cores without losing anything" (scripts `scripts/Diagnostic Tete de cheval – 26 nov/`): on 1,024 px crops containing bright stars **saturated at low signal-to-noise ratio** (flat top at ≈ 100 noise units, short exposures), measure the **flux remaining** after removal within a 2 FWHM disc (minimum 6 px) and within a 25 px disc (0 = removed, 1 = intact). Six stars of the Horsehead field (G 3.79; G 6.18; four medium stars), plus two high-signal validation stars for comparison. Result: N5 with two-scale inference removes all six (0.00 to 0.02); N7 keeps five of six (0.95 to 0.98; the sixth at 0.11); the 0.75 blend keeps all six (0.95 to 0.99). At high signal (860 to 12,500 noise units), all networks remove them (0.00 to 0.02). This test fails for a network that confuses a saturated star with a compact core.

## 5. Measuring an improvement honestly: the checklist

1. Change **one** thing, in a copy, and write beforehand: "already tried" and "what differs".
2. Measure on the validation (13 images) **and** on the control set of small galaxies **and** on the saturated-star test. A gain on only one of the three is not a gain.
3. Report all the columns, including those that degrade. Example: N7 gains the cores (0.19 → 0.81) but loses the bright stars (G < 8: 0 → 0.021): this is not a net improvement, it is a shift of the trade-off.
4. Look at the image (whole square, ×6 zooms, input | output), never only the table.
5. Compare at equal step **and** at equal learning rate (a 10,000-step run is not comparable at equal step with a 34,500-step run).
6. A validation set that does not contain a defect does not rule it out: add the image that shows it (Horsehead, 26 Nov).
7. Never tune on the control set.
