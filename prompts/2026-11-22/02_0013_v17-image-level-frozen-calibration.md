# v17: image-level frozen calibration

- Date : 2026-11-22 à 00:13 (heure de Paris)
- Numéro du jour : 02
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

One-change experiment (design + implementation) in piste 1. COMPUTE CAP (strict): ONE chain at a time, never in parallel, env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4; no bash loops forking many processes. Python: python (pandas broken). cv2.imwrite fails silently on non-ASCII paths: use cv2.imencode(...)[1].tofile(path). No downloads. Write only under <SCRIPTS>\ + JOURNAL.md + git; read the input in place on F:. Never modify existing version folders.

Read first: "<SCRIPTS>\Recul M8 – 22 nov\RECUL.md" (and its scripts), JOURNAL.md (last 6 entries), "Methode v15 SHO – 21 nov\LISEZMOI.txt".

Finding (measured): on the bright M8 core tile a_coeur, piste 1 re-calibrates per crop and the nebula fools it — calibration stars are chosen by "pixel > 10 noise" (retrait_modele.py ~l.758), which the nebula alone satisfies: crop FWHM 7.48 px vs 4.37 on the whole image; zero point 25.57 ± 2.2 mag vs 22.98 ± 0.57 (11× too much flux); S/H/O star colour 0.12/0.37/0.52 = the nebula's colour (true stars 0.32/0.33/0.35). Result: stars subtracted at 4–16× their true flux (bounds forbid less), 8–16 % of all the light of the zone removed, the core-vs-ring check fires on 73 % of empty places → 19 % of the tile refilled → the cyan discs. Tile b shows the same error, smaller; tile c is correct.

The ONE change (v17), forced photometry with image-level calibration (Lang, Hogg & Schlegel 2016, "The Tractor"; RECUL.md Q1 step 1): measure ONCE on the whole image and freeze — PSF width/shape (the v7 whole-image model, which already exists), zero point (Gaia G → flux), per-channel star colour (median S/H/O share of isolated, unsaturated, catalogue stars chosen by star criteria — PSF-like, catalogue-matched — NOT by "pixel > 10 noise"), saturation levels (v14's per-channel detection); cache them next to modele_psf_image.npz; each crop reads them and only fits per-star fluxes (and positions as now). Nothing else changes (keep v15's spikes and v14's saturation as they are; do not change the core-vs-ring check, the flux bounds or the fills in this version — just give them the right calibration).

Do: copy "Methode v15 SHO – 21 nov" (code + Catalogue + Piste1\sorties with the image models) to "<SCRIPTS>\Methode v17 SHO – 22 nov\", implement, delete sorties\<tile>\A\sorties\base_piste1 before each run, run a_coeur first, then b_filaments, c_champ_dense, d_brillante — one at a time.
Measure v15 (for a/b/c use v13 if v15 did not run them — say which) vs v17 per tile: crop-level FWHM / zero point / colour used (should now equal the image values), ratio light removed / catalogue star light per channel (target 0.8–1.2), core-vs-ring check firing rate at empty places (RECUL scripts), invented share, dark points < −5 and bright leftovers > +5 noise (L = channel mean, 9×9 median, sig = 1.4826·MAD).
Views: whole tile, x2, 3 panels input | previous | v17, stretch arcsinh 40, per-channel black = median of the whole image, white = per-channel 99.8th pct of the whole image → <SCRIPTS>\A_REGARDER – 21 nov soir\M8_SHO_<tile>_<prev>_v17.png (for c also a _fort version with per-tile p1/p99.5 stretch). Look at each yourself, one sentence each.
