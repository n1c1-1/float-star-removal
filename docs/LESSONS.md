# What failed, and why

So that the same trials are not repeated. Each line says what was tried, what happened, and the lesson. Dates: November 2026. The trials are read with the rule "one change at a time, and write down beforehand what has already been tried and how the new trial differs".

## A. The networks

| trial | date | what was done | result | lesson |
|---|---|---|---|---|
| E2, E3 (synthetic only) | 7-9 Nov | training on 100 % generated pairs | 3 to 4 times worse on real stars; galaxy cores hollowed out; the best snapshot by synthetic score was not the best on real data | a synthetic set alone is not enough: real data is needed; the snapshot is chosen on held-out real images |
| A (E3 fine-tuned on real pairs) and B (from scratch), 22-23 Nov | 22-23 Nov | 2 h on ≈ 99 real pairs of set 1, B starting from random weights | A better than B (error on held-out pairs 0.00108 versus 0.00117); but background damaged (0.0033 versus 0.0005) and bright structure degraded; pairs full of defects (saturated discs, rings, holes) | start from E3; check **every** pair by eye; tiles taken "on a grid" without looking at the content give dense fields almost all at weight 0 |
| Smooth pedestal (trial C) | 23 Nov | adding a smooth pedestal identical in input and target | stopped before measurement; the defect was the target being darker than the input (see v33g) | measure the defect before "teaching" it |
| N1/N2 (night of 24 Nov) | 24 Nov | pairs v33d + web, weights 0.5 → 1, imposed very bright stars | bright stars well removed (G < 8 at 0.000), but M31's cores erased, background and structure more damaged, half of the G 14-17 stars unsupervised (weight 0) | targets darker than the input in the "untouched" sky (wings subtracted out to 150 px): fixed by v33g |
| **N3** (large synthetic halos) | 25 Nov | halo generator up to 285 px, 25 % of the synthetic tiles; judge v33e that measures the halo on the image | M27's flat disc stays identical; judge v33e took whole nebulae for halos (+49 points of weight 0 on the planetary nebula NGC 6543) | the problem is not in the large-halo data: it is the network's **reach** (see inference V2) |
| **N4** (judge by Gaia magnitude, 50 % large halos) | 25 Nov | halo radius by a law in G magnitude, safeguards | the judge barely changes the weights (+0.06 point); M27 unchanged | same conclusion; the "halo data" track is abandoned |
| Inference with large tiles, M27 | 25 Nov | the largest possible tiles, 50 % overlap | identical residual; the disc persists | the network's reach is the limit |
| Shrinking the whole image ×2 or ×3 | 25 Nov | image reduced then enlarged again | the disc disappears, but the rest is damaged (shrinking also removes signal) | use shrinking only **within the zone** of the big stars: inference V2 |
| **N6** (weights v33h: repaired G 14-17 stars → 1, remnants > 10 % → 0) | 25 Nov | N5 with the v33h weights | G 14-17 higher (0.351 versus 0.265), background 0.0027 | giving more weight to the faint stars of our targets degraded them; N6 abandoned, N5 kept |
| **Putting the slow part back at inference** | 25 Nov | N5 then adding the smoothed diffuse light (sigma 40) to the output | structure 0.007 → 0.004, but median background 0.0011 → 0.0026 and cores unchanged (M31's blue-black hole identical, M32 erased) | the correction does not touch the main defect; not to be retried without revisiting the clipping (G 8-14 wings, large structures) |
| N5/N6 blends (0.5 and 0.75) | 26 Nov | weight average | cores 0.13 to 0.16: the erased core stays erased | a blend does not give back what neither network can do |
| **N8** (cores 10 %, short run 10,000 steps) then **N8L** (24,000 steps) | 26 Nov | a single change: 10 % of cores instead of 25 % | cores ≈ 0.80 whatever the share (25 % or 10 %); stars higher in the short run; N8L no better than N7 | the share of cores is not the lever; diagnosis: a compact core (M32, Canon 60D) is taken for a star |
| **N9** (40 % compact cores) | 26 Nov | cores 1.3 to 1.5 times the star width, Sérsic n 3-6 | M32 0.05 → 0.26, cores 0.81 → 0.87, BUT stars kept: G < 8 0.065 → 0.136, G 8-11 0.053 → 0.109, G 14-17 0.253 → 0.348; small galaxies of the control set still erased | shape alone does not tell a star from a compact core and costs stars; lead: a Gaia map of the known stars as an input |

## B. The classical chain (Gaia-guided subtraction)

The same mistakes were repeated: on 22 Nov, 24 trials redid the method of 20 Nov, already judged a failure. Hence the rule of writing "already tried / what differs" before each trial.

**Faint stars.**
- Works: the deep catalogue (G < 21) instead of detecting in the residual; limit the side measurements to G 17.5 and remove down to G 21.
- Do not do again: loosening detection outside the catalogue on an image of assembled filters (nebula knots are taken for stars); tuning the depth on a measured "gain" when the catalogue covers the whole image (artefact).

**Medium stars (rings, dipoles, holes).**
- Works: star model on the whole image; position specific to each colour; pairs within 2.5 mag; PSF redone without the bright stars already removed; final pass on the "residual cores".
- Do not do again: filling alone in a dense field; one common width per brightness bin; a global colour shift; one PSF per colour; widening the filling discs against a ring (it pushes the ring outward at every pass).

**Spikes.**
- Works: instrument angles measured on the whole image, fixed to ±2°; one template per colour; tracking down to ≈ 0.45 noise; remnants subtracted rather than recreated; the final pass that refills the remnants within the band of a known spike.
- Do not do again: a box template (coloured rectangles); forcing a width; stacking a common star image (**the measurement improves while the image degrades**); measuring the spike star by star near the noise; assuming a cross at 90° and at most four spikes (wrong: M8 has six).

**Big saturated stars (G ≲ 7).**
- Do not do again: subtracting the halo and spikes with a star-specific model **and recreating only the core**: at strong stretch there remain bars, a streaked disc and a flat grey disc; shared extended profile (magenta discs); "extended source" rule on a clipped star (its truncated peak always triggers it).
- Works: the per-star profile; saturation read on the whole image; cores recreated by a harmonic background never lower than their surroundings; saturated plateau **measured per star** (each star is clipped at its own level); halo reach **measured** (up to 520 px) instead of a formula that cut it at 250-300 px.
- Limit: under a magnitude-5 star, the truth does not exist in the image. Stars of the Alnitak class are almost impossible to remove cleanly; their central disc is marked "to ignore" (weight 0) and the halo and rays beyond are kept (real pixels).
- General lesson: a rejected trial can become good again once an upstream defect is fixed: note it as "rejected under such conditions", not "never".

**Nebula.** Do not do again: recreating large zones (31 % invented, nebula / dark lane boundary lost); filling wide hollows unconditionally. Works: redoing wide hollows unless they are already dark in the input; judging a "real" remnant against the local structure, not against the sky noise.

**Speed.** Porting track 1 to the graphics card: ≈ 5 times faster, bit-identical result; two computations on the card at the same time slow each other down by 10 to 20 times.

## C. Judging and measuring: what misled us

- A **measurement that improves while the image degrades** (spikes, z3/z4). Always look at the image before believing the measurement.
- A low invented share taken for quality; the pixel-diff between versions (a single discarded hollow changes 65 % of the pixels); tolerances in sky noise measured on a nebula (they measure the nebula).
- **Reading the columns.** "background" is |output − input| in **noise units** far from the stars: 0.001 is a thousand times below the noise, invisible. It is not a gap to correct. The real gaps were the cores (0.80 for an ideal of 1) and the faint stars.
- **The blind judge** is an argument, not the truth: its *reasons* are not causes. On the synthetic stars, three corrections drawn from its remarks (thinned spikes, halo join, smoothed round shape) gave a better rate, but removed one by one each was harmful; each change must be measured alone, with a fresh judge.
- **An incomplete validation set hides defects.** On 26 Nov, a real image of the Horsehead (saturated stars at low signal-to-noise, flat top at ≈ 100 noise units) showed that N7 keeps them: the validation contained no saturated star at that level (only 860 to 12,500 noise units). Widen the validation before concluding.
- **Agents leave errors.** The N8L and N9 measurement scripts needed 3 relaunches because of undefined names: require a check for undefined names (py_compile is not enough) before any long run.
- **Hardware matters.** A training run was stopped by lack of RAM (the safeguard stops cleanly below 3.5 GB free and resumes above 8 GB: `Tour N8L – 26 nov/train_n8l.py`).

## D. The method rules that came out of it

1. One change per trial, in a copy; never edit a validated script in place.
2. Write before the trial: "already tried", "what differs", what the change risks breaking elsewhere.
3. After 3 targeted trials with no movement of the defect, step back instead of a 4th trial.
4. Short runs (≈ 10,000 steps) to choose a direction, an intermediate measurement every 5,000 steps **inside** the training process, a stopping rule written before launch, a long run only for the winning recipe (capped at ≈ 24,000 steps).
5. **Local** weight 0, never by whole category: neither "all zero" nor "all perfect".
6. Only networks that started from the same point are blended.
