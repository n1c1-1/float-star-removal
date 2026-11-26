# v13 on SHO image M8 samples

- Date : 2026-11-21 à 22:10 (heure de Paris)
- Numéro du jour : 34
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Try the frozen best method v13 on a NEW image: an SHO (Hubble palette, mono narrowband) image. COMPUTE CAP (strict): ONE chain at a time, never in parallel, env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4; no bash loops forking many processes. Python: python (pandas broken). cv2.imwrite fails silently on non-ASCII paths: use cv2.imencode(...)[1].tofile(path). NO DOWNLOAD of any kind (no Gaia/TAP query): use only catalogues already on disk (the offline Gaia extract that the code / FLOAT uses — look how astrometrie.py and the piste 1 code find the "extrait hors ligne", e.g. env ASTRO_GPU_GAIA_EXTRACTS; say which file you used and its depth). Write ONLY under <SCRIPTS>\ (new folder below) + JOURNAL.md + git; read inputs in place on F: (never write on F:). Never modify v1..v13 folders.

[passage retiré]

Steps:
1. Copy "<SCRIPTS>\Methode v13 – 21 nov" (code + Catalogue folder; not sorties) to "<SCRIPTS>\Methode v13 SHO – 21 nov\". In the copy, declare the image in Piste1\images.py (name "m8_sho", path above, object "M8", type "filtres") — a declaration, not a method change; check how the "filtres" type is handled (flux per channel = Gaia brightness × median colour) and that nothing else assumes RGB.
2. Astrometry for m8_sho with the existing code/tools (offline only), cached as the code expects. If the deep catalogue file gaia_dr3_profond_m8_sho.npz is missing, make the code fall back to the offline extract (as it does when the deep file is absent) and log the number of catalogue stars in the image and the magnitude limit.
3. Overview: a reduced view of the whole image (arcsinh stretch, black = per-channel median, white = per-channel 99.8th pct) with a grid, then choose 4 tiles of 512 px (768 crops) by rule and draw them on the overview: (a) the bright nebula core (Hourglass region, highest nebula brightness), (b) a region of fine dark lanes/filaments, (c) the densest star field, (d) the tile containing the brightest star(s) of the image. Save the overview as <SCRIPTS>\A_REGARDER – 21 nov soir\M8_SHO_0_carte_echantillons.png.
4. Run v13 (methode_validee.py with the crops added to CROPS for this image — check how the image name is passed; add the minimum plumbing in the copy) on the 4 tiles, one at a time.
5. Views for each tile: whole 512 square, x2, 3 panels input | v13 | current network (starless_reseau, same tile), stretch arcsinh 40, black = per-channel median of the whole image, white = per-channel 99.8th pct of the whole image (same for all panels) → A_REGARDER – 21 nov soir\M8_SHO_<a|b|c|d>_<short name>_entree_v13_reseau.png. Look at them yourself and describe in 1–2 sentences per tile what is wrong (leftovers, holes, colour halos (magenta/green rings typical of SHO), lost filaments), without fixing anything.
6. Measures per tile: dark points < −5 noise and bright leftovers > 5 noise (L = mean of channels, 9×9 median, sig = 1.4826·MAD), invented share, piste 1 time, number of catalogue stars used vs extra detections ("hors catalogue").
