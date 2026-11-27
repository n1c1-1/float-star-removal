# Improving: known limits, open leads, how to measure

This document is for anyone who wants to do better. It says **where it does not work**, **what has already been tried** (so as not to redo it: detail in `LESSONS.md`) and **how to honestly prove an improvement** (`VALIDATION.md`).

## 1. Lead no. 1: the "saturated star / compact core" conflict

**The problem.** There is no single best model, there is a **trade-off**:
- **N5** (with two-scale inference) removes all saturated stars, including those of low signal-to-noise ratio, **but erases galaxy cores** (cores 0.19; M32 in the photo from a Canon 60D at 0.03).
- **N7** and the **N7/N5 0.75** blend keep the cores (≈ 0.8) **but also keep bright saturated stars** whose top is flat at ≈ 100 times the noise (short exposures), because a synthetic galaxy core and a clipped star of low signal look alike: N7's synthetic cores precisely include clipped cores of 50 to 5,000 noise units.
- Proof (26 Nov): on a linear stack of 704 exposures of 15 s of the Horsehead (not denoised, with Alnitak), the 0.75 blend keeps five stars out of six tested (remaining flux 0.95 to 0.98; within the 2 FWHM disc); N5 with V2 removes them all (0.00 to 0.02). The large nebula areas and the Horsehead are well preserved. The other levers tried change nothing: raising the input gain (×4, ×8) repairs only three stars out of six (the faintest), denoising the input has no effect, and ×2 two-scale inference repairs only partially (one star at 0.38, two others stay at 0.95).
- Our validation set contained no saturated star at this level (860 to 12,500 noise units only): the table did not show it (`VALIDATION.md` §4).

**Hypotheses to establish** (the main cause is identified, the secondary causes are not):
1. Very bright stars **absent from the Gaia catalogue** (Alnitak is one) are outside the zones of two-scale inference and of the Gaia trigger.
2. A **non-denoised short-exposure** image is little present in the training (raw pairs make up only 25 % of the draw; the rest is denoised or web).

**The proposed lead: the Gaia map of known stars as a network input.** Saturated stars are in the catalogue; galaxy cores, planetary nebulae and small compact galaxies are not (a compact source **absent** from Gaia is a "keep" candidate). Giving the network, in addition to the image, a map of Gaia positions and brightnesses (a channel or a probability map) would let it resolve the ambiguity. To try first: a single change (the input channel), start from E3 with the first input layer adapted, measurement on the three tests of `VALIDATION.md` (validation, control of small galaxies, saturated stars). Risks: the catalogue is incomplete for the very brightest stars (Alnitak is not in it) and for fields with poor astrometric solutions; a wrong astrometry shifts the map (see the images discarded from the control set).

## 2. Small compact galaxies (control set of 18 galaxies)

**Limit.** N7, the blend and N9 erase almost all small compact galaxies (median ≈ 0 instead of 1). This defect is **much wider than M32**. The control set defines the objective.
**Already tried and failed.** More synthetic cores (25 %: N7; 10 %: N8/N8L; 40 % of *compact* cores: N9): M32 rises (0.05 → 0.26) but at the price of kept stars (G < 8: 0.065 → 0.136); **shape alone is not enough**.
**Leads.** (a) the Gaia map as input (§1); (b) a synthetic population of varied small galaxies (ellipticals, spirals, edge-on) in addition to the cores; (c) a larger control set (18 galaxies from 5 images is small); (d) do not tune on this set.

## 3. Faint stars, G 14 to 17

Between 26 % and 30 % of their flux remains after removal. Diagnosis of 25 Nov: our real targets remove these stars correctly, but **52 to 64 % of the G 14-17 stars are at weight 0** at the centre in the pairs (not taught). **Tried**: N6 (weight 1 for well-removed G 14-17 stars, 0 for remnants): G 14-17 rises to 0.351, N6 abandoned. **Leads**: supervision by *synthetic* faint stars (exact target) rather than by real data; longer training on these stars alone; checking that the "well removed" criterion does not select the easy stars.

## 4. Big stars and the network's reach

The network sees ≈ 180 px; the halos of big stars reach ≈ 285. Two-scale inference V2 fixes this (step 8). **Leads**: a network with a larger reach (Fourier convolutions, global attention); integrating V2 into the application (proposal made, not integrated); halo radius measured without Gaia for stars the catalogue does not contain. **Tried with no effect**: large tiles, giant synthetic halos (N3, N4).

## 5. Data

- **"Unavoidable" targets** (under a saturated core): no truth in the image; the target is a plausible fill. Open question: what weight to give them (zero, reduced, or normal with the best possible fill).
- **Linear or denoised input?** The star model is exact on linear data, the network works after denoising. Never settled; lead: two pairs per image (denoised and raw), already done for 35 tiles.
- **More images and instrument types** (dense fields, narrow band, very low signal): step 2 produced about a hundred real tiles and 38 web tiles; the chain costs about 4 minutes per tile.
- **Targets v33g**: the diffuse light put back is a smoothing; replace it with a finer sky model.

## 6. What has already failed: do not redo

N3, N4, N6, putting the slow part back at inference, N8/N8L, N9, N5/N6 blends, E2/E3 alone, starting from zero: see `LESSONS.md`, table A. Rule: before a trial, write "already tried" and "what differs".

## 7. Measuring an improvement honestly (summary)

1. One change per trial, in a copy.
2. Three tests, not one: validation (13 images: remaining stars per magnitude band, background, structure, cores), control of small galaxies (galaxy towards 1, star of the same brightness towards 0), saturated stars at low signal (remaining flux).
3. **Star safeguard**: any improvement of the cores or galaxies must show that the bright stars (G < 8) and the saturated stars of low signal do not come back (N7: 0 → 0.021; N9: 0.065 → 0.136: examples of gains paid for in stars).
4. Report all the columns, including those that degrade; look at the image.
5. Never tune on the control set; publish the step, the recipe and the checksums.
6. Every training datum must have a written provenance (`PROVENANCE.md`): a network trained on the output of another piece of software proves nothing.
