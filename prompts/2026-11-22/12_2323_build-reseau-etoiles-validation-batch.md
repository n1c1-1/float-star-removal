# Build réseau étoiles validation batch

- Date : 2026-11-22 à 23:23 (heure de Paris)
- Numéro du jour : 12
- Type d'agent : agent général
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Goal: build a real VALIDATION BATCH for the star-removal network trained from the classical pairs ("jeu 1"), so every trained network can be measured on images it never saw. Work autonomously; report ≤ 20 lines at the end.

HARD RULES (shared machine, crashed 3 times from overload):
- Every python process sets OMP_NUM_THREADS=MKL_NUM_THREADS=NUMBA_NUM_THREADS=OPENBLAS_NUM_THREADS=4 (torch.set_num_threads(4)); one computation at a time; no process pool. Another agent runs production (4 threads + GPU): you may use the GPU (RTX 4060 Ti only, NEVER the 3070) only when no `produire_jeu1.py` / `train_jeu1.py` process is running.
- Write only in <SCRIPTS>\Validation – 22 nov\ (scripts, LISEZMOI.txt, journal) and <DONNEES>\Validation étoiles – 22 nov\ (data). Library sources are READ ONLY; never copy whole projects.
- Gaia downloads are a separate preparation step (never inside a processing run). Python: python; cv2.imencode().tofile() for non-ASCII paths; sys.stdout.reconfigure(encoding="utf-8").

1. SELECTION (12–15 whole images). Source: <SCRIPTS>\Bibliotheque images – 22 nov\inventaire.csv and BILAN.md. EXCLUDE every target used in training, whatever the instrument: Rosette, M8, M57, Horsehead/IC 434/NGC 2023, NGC 6543, NGC 869, IC 443, IC 5070, NGC 7000, M97, M1, M82, M13, NGC 7662, NGC 6888, WASP-74. Exclude Euclid and stretched exports/TIFFs, star/starless layers, synthetic sets. Prefer stacks that already exist (no new stacking). Cover these types, one image each at least: his A7S + Newton (spikes); OSC refractor; mono narrow-band assembled; smart telescope (Seestar/Dwarf/Stellina); DSLR on lens or refractor; a galaxy with a bright core (M31 or M51 if available); a compact planetary nebula (M27 is one candidate among others); a dense star field; a faint extended nebula; large FWHM; trailed stars if any. Write the list with instrument, scale, why chosen, in LISEZMOI.txt.
[passage retiré]
3. ASTROMETRY + DEEP GAIA (G<21) per image, reusing the other agent's tools in "Images jeu 1 – 22 nov" (local solver; astrometrie_cone.py fallback; telecharger_gaia_profond.py via ESA sync circle queries). Store under <DONNEES>\Validation étoiles – 22 nov\<nom>\.
4. METRICS SCRIPT `valider_reseau.py <checkpoint.pt | dossier de starless> [--images ...]` that, per image (whole image, tiled inference with overlap, the network = chain_nafnet from <SCRIPTS>\models.py with fwd/inverse as in Entrainement jeu 1 – 22 nov\train_jeu1.py and train_e.py), computes:
   (a) stars left: flux remaining / original flux at Gaia positions (aperture 2 FWHM, local annulus background), median and 90th percentile per G band (<8, 8–11, 11–14, 14–17, >17);
   (b) background damage: on pixels farther than max(4 FWHM, catalogue-star radius by magnitude) from any Gaia star, robust |output − input| in noise units (median, 99th percentile), and the same restricted to bright structure (input above the 90th percentile of the smoothed image: nebula, galaxy core);
   (c) exact error under synthetic stars: 20 tiles 256 per image, background = the input with its stars removed is NOT available, so use tiles chosen far from bright stars, add stars with <SCRIPTS>\Etoiles synthetiques – 22 nov\paires_synthetiques.py paire_synthetique(fond, graine) (target = fond), L1 on the asinh stretch over star footprints, per star class; fixed seeds so every network sees the same tiles;
   (d) a contact sheet per image: input | output | difference, same stretch.
   It must also accept a folder of already-made starless images (same names) so other tools' outputs can be scored identically. Output JSON per run + a summary table.
5. Smoke-test the script on 2 images with the network FLOAT ships today (find its checkpoint path in <FLOAT> — search for the star model loading code; read-only) and give its numbers as the first baseline. Do not train anything.
Report: the chosen list (type, instrument), what failed and why, the baseline numbers, paths, commits.
