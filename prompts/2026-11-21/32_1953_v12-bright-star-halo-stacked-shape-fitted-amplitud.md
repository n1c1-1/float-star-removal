# v12: bright-star halo, stacked shape + fitted amplitude

- Date : 2026-11-21 à 19:53 (heure de Paris)
- Numéro du jour : 32
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

One-change experiment (design + implementation) in piste 1. COMPUTE CAP (strict, the PC saturated earlier today): ONE chain at a time, never in parallel, env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4; no bash loops forking many processes. Python: python (pandas broken). cv2.imwrite fails silently on non-ASCII paths: use cv2.imencode(...)[1].tofile(path). No downloads. <SCRIPTS> is a git repo. Never modify v1..v11 folders.

Read first: JOURNAL.md (last 6 entries), "Methode v10 – 21 nov\LISEZMOI.txt" if present (else v7's), "Methode v10 – 21 nov\Piste1\retrait_modele.py" around l.1318-1330 (bright stars modelled by their own radial profile: median per ring of the median of 8 sectors, minus the background of the ring beyond, made decreasing and ≥ 0), profile_refit, measure_halo (~l.1538), and the whole-image star model built in v7 (modele_psf_image.py / modele_psf_image.npz, radius 28 px).

Defect: crop "coeur" (centre (1620,1700), declared in CROPS) — around the bright spiked star at tile (358,81), piste 1 removes 1.5 (in image units) at 30 px and 0.46 at 77 px, and the result goes 0.2–0.3 below the neighbouring sky where the nebula behind is a dark lane → two black holes (tile ≈ (281,75) and (381,62)). [outil de comparaison] on the same tile keeps the whole halo glow; the truth is between. My reading: the star's own radial profile includes the nebula light around it (bright nebula on most sectors), and this circular model is then subtracted over the dark lane too.

The ONE change (v12), following the extended-PSF method for bright stars (Infante-Sainz, Trujillo & Román 2020, MNRAS 491, arXiv:1911.01430: build the extended PSF by stacking bright stars over the whole image, then for each bright star fit only its AMPLITUDE (and local background) in a ring where the star dominates, and subtract the scaled extended PSF): replace the per-star free radial profile of bright stars by (a) a SHAPE shared by the image — the stacked, normalised radial profile of the bright unsaturated-wing stars of the WHOLE image (outer wing out to ~100–150 px, robust stacking with sector medians and masking of other stars, joined to the v7 core model at ~20 px), built once per image and cached next to modele_psf_image.npz; (b) per star, a robust fit of amplitude + flat background in an inner ring (e.g. 2–6 FWHM, choose and justify) with neighbours masked and sigma-clipping, so the nebula's large-scale light is not absorbed into the halo; keep the core/saturation handling as it is. State the radii, the number of stars stacked, and how the profile is joined.

Do: copy "Methode v10 – 21 nov" (code + Catalogue + Piste1\sorties incl. the image models; piste 1 must RERUN; delete each crop's stage-A cache sorties\<crop>\A\sorties\base_piste1 before rerunning, it is never refreshed otherwise) to "<SCRIPTS>\Methode v12 – 21 nov\", implement, run methode_validee.py on coeur (bord intérieur) first, then decoupe6, then coeur_est — one at a time.

Measure v10 vs v12: on coeur, the two hole positions (281,75) and (381,62): result minus the median sky of a ring 100–140 px from the star (luminance, in noise units), and the radial profile of (result − input) around the star (r 10..140 px, table); fitted amplitudes vs Gaia-expected flux for each bright star; on all three crops: dark points < −5 noise (L = channel mean, 9×9 median, sig = 1.4826·MAD), rainbow/dark counts, invented share; coeur_est filament rms 3–8 px (x 40–300, y 150–350; v10 0.0354); piste 1 time.
Deliverables in <SCRIPTS>\A_REGARDER – 21 nov soir\: bord_interieur_ENTIER_entree_v10_v12.png and decoupe6_ENTIER_entree_v10_v12.png (3 panels input | v10 | v12, x2; stretch arcsinh 60, black = median of full image green [::7,::7], white = 99.5th pct of full[1,500:2500,1100:3100]; the 512 tile of each crop = central 512 of its 768 CROPS box).
