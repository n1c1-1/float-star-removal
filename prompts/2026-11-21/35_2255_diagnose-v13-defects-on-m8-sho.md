# Diagnose v13 defects on M8 SHO

- Date : 2026-11-21 à 22:55 (heure de Paris)
- Numéro du jour : 35
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

[passage retiré]

Defects seen by eye (whole tile views in <SCRIPTS>\A_REGARDER – 21 nov soir\M8_SHO_*_entree_v13_reseau*.png):
(1) a_coeur (very bright, steep nebula): dozens of circles with CYAN edges where stars were removed; a darker brownish zone around the spiked star at the top; a horizontal bright bar (spike remnant).
(2) c_champ_dense (faint background, hard stretch): many small bubbles/rings and coarser grain than the input, a pink patch at bottom-left.
(3) d_brillante: the bright star has 6 diffraction spikes (Newton with 3-vane spider); v13 leaves the 6 spikes as coloured dotted lines, a black hole with scalloped edges at the core, a green-blue zone below, bubbles.

Questions:
A. Rings/bubbles/cyan edges: first stage where they appear (input → piste 1 → A → t1 → t2 → t3) on 6 examples per tile a and c (x8 zooms input | p1 | A | t1 | t2 | t3, same stretch per row: per-channel black = tile p1, white = tile p99.5, arcsinh 40); fill class at the ring pixels; is the ring a per-channel level mismatch at the fill border (measure each channel's step across the border, in noise) or a PSF-wing residual (radial profile per channel around the star in piste 1 output)? Compare FWHM per channel (SII, Ha, OIII) measured on isolated stars of the input vs the single PSF model the code uses per channel.
B. Spikes on d: how many spike angles does piste 1 / stage A detect for this image (log lines "angles d'instrument", spike templates), which angles, vs the true 6 (measure the true angles on the input with a radial-angular profile around the bright star); which code path handles spikes and whether it assumes 4 (or a cross).
C. Core hole on d: the star's G and saturation; which treatment (v13: G < 7 → v10 own radial profile; 7 ≤ G ≤ 10.5 → v12 extended shape) and which stage makes the black core.
Report in French, ≤ 20 lines: per question the cause with file:line and numbers, and paths of the zoom PNGs. No fix.
