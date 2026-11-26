# Diagnose piste-1 colour dipoles

- Date : 2026-11-21 à 14:49 (heure de Paris)
- Numéro du jour : 18
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Context: piste 1 (frozen code "<SCRIPTS>\Methode validee – 21 nov\Piste1\retrait_modele.py", read-only) subtracts PSF models of Gaia stars; fit_star (line ~364) fits flux per colour at a position COMMON to the colours (Gauss-Newton), with an optional per-colour offset field "choff"/measure_chroma (line ~854-920) that is OFF (setting "chroma": false since run c6; on decoupe6 it measured offsets < 0.08 px). Each removed medium star leaves in the piste 1 starless a colour DIPOLE: red on one side, blue/dark on the other (see "Methode validee – 21 nov\diagnostic_taches\zoom_x8_hd1_fort.png", panel p1). These survive the later passes as 2–3 px rainbow spots.

Data, read-only: "Methode validee – 21 nov\Piste1\sorties\rosette\mv_filaments\" and "\mv_lisse\" and "\c9_decoupe6\": starless.npy (3,768,768, the crop), etoiles.csv (positions x,y in crop coordinates — check — G, fluxes, width), journal.txt. The input crop: read how piste1_decoupe.py / etape_piste1.py load the Rosette image and the crop box (filaments crop (1466,1591,2234,2359), lisse (2016,1666,2784,2434), decoupe6 (1530,790,2298,1558)); load it the same way, read-only.

Measure, for isolated stars G 11.5–15.5 (no brighter neighbour within 8 px, not saturated, at least 10 px from crop edges), on each of the 3 crops:
1. In the INPUT: per-channel centroid (background-subtracted, 5 px radius, iterative) → offsets R−G and B−G in px. Median and scatter; plot vectors vs position in the full image (quiver PNG) to see if constant, radial from image centre, or random. Image size from the loader.
2. In the piste 1 STARLESS residual at those stars: per-channel residual first moment (dipole vector) and monopole (sum) within 4 px; are R and B dipoles opposite and aligned with the step-1 offsets? Correlation coefficient between dipole vector and centroid offset vector.
3. Same for G-channel PSF width vs R and B (second moments in input): is one channel wider (colour-dependent PSF) — which would give a ring/monopole pattern rather than a dipole.
Give the size of the effect in noise units: typical dipole amplitude / local noise.

Report in French, ≤ 18 lines: per crop, median offsets R−G, B−G (px) with scatter and number of stars; pattern (constant / radial / random); correlation dipole↔offset; width differences; conclusion in one sentence: which mechanism explains the dipoles, with the numbers. Paths of the PNGs.
