# Fable step-back on M8 SHO blockage

- Date : 2026-11-22 à 00:05 (heure de Paris)
- Numéro du jour : 01
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Cold step-back, advice only: no code change, measurements of a few minutes at most (numpy, OMP_NUM_THREADS=4), write only in "<SCRIPTS>\Recul M8 – 22 nov\" (create; save your answer there as RECUL.md). Python: python (pandas broken). cv2.imwrite fails silently on non-ASCII paths: use cv2.imencode(...)[1].tofile(path). Read-only everywhere else; the input is on F: (read only).

Project goal (owner): training pairs (image with stars → starless target) generated automatically by the hundreds to train a star-removal network from scratch; each pair carries a map of invented pixels (excluded or down-weighted in training), a per-image confidence, and failed zones can be discarded. The owner accepts that the classical method will not work unchanged on every image: Claude tunes it per image with written rules. A first image outside the Rosette was just tried: M8 in SHO (mono narrowband SII/Ha/OIII assembled, Newton with 6-spike diffraction, very bright steep nebula core).

Read in <SCRIPTS>\: JOURNAL.md (the last ~8 entries: v13 on M8, the M8 diagnosis, v14, v15, v16, the decision), "Recul – 21 nov soir\RECUL.md" (the previous step-back on the Rosette), "Methode v13 SHO – 21 nov\LISEZMOI.txt" and "...\diagnostic_m8\" (scripts and results). Look yourself at the views in "A_REGARDER – 21 nov soir\": M8_SHO_0_carte_echantillons.png, M8_SHO_a_coeur_entree_v13_reseau.png, M8_SHO_a_coeur_v13_v16.png, M8_SHO_b_filaments_entree_v13_reseau.png, M8_SHO_c_champ_dense_entree_v13_reseau_fort.png, M8_SHO_d_brillante_v14_v15.png (third panel of the "entree_v13_reseau" views = FLOAT's current network, for comparison).

State: v13 is good on tile b (filaments), acceptable on c (dense faint field), bad on a (very bright, steep nebula core: dozens of discs with cyan (SII) edges where stars were removed — born in piste 1, step at fill borders mostly in S; S star width differs from the model by ~7–10 % and a scale factor cannot fit both core and half-height) and on d (G 5.2 clipped star with 6 spikes: core hole, spike remnants; v14 saturation per channel helps a little, v15 finds 6 spikes but leaves dark strokes, the "left whole giant" rule trips wrongly). v16 (per-channel width + non-negative wing) made a worse. Three targeted changes did not fix the main defect (cyan discs in the bright core).

Questions — one recommendation each, with the published method behind it where relevant (search; cite sources) and expected effect:
Q1. Cyan discs in a very bright, steep nebula: what is the principled cause (per-channel PSF mismatch in narrowband? background/level estimation of the fit and of the fill on a steep gradient? fill copying from the wrong place?) — use the diagnosis numbers and, if needed, one quick measurement. What is the principled fix (e.g. per-channel independent PSF models for narrowband — each filter is effectively a separate image; local background as a plane/low-order surface under each star; inpainting constrained by the per-channel low-frequency field)?
Q2. Very bright clipped stars with 6 spikes: fight them, or exclude their disc + spikes from the training target (invented-pixel map / discard zone)? How many such stars per typical image, how much of the image area, what does the network lose if they are excluded?
Q3. Per-image tuning: given M8, which parameters must become per-image (and per-channel for narrowband) knobs, and which rules are Rosette-specific and must go (e.g. "giant left whole", spike cross assumption)? Propose the short parameter sheet.
Q4. Is it more productive to process narrowband images as three separate mono images (one per filter, each with its own PSF, saturation and catalogue colour term) and recombine, rather than as an RGB image? Pros/cons.
Report in French, ≤ 30 lines, plain words, then a 3-step ordered plan.
