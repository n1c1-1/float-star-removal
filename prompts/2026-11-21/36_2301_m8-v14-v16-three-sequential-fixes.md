# M8: v14-v16 three sequential fixes

- Date : 2026-11-21 à 23:01 (heure de Paris)
- Numéro du jour : 36
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Three successive one-change experiments on the SHO image M8 (each its own version, cumulative, each measured and committed before the next). COMPUTE CAP (strict): ONE chain at a time, never in parallel, env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4; no bash loops forking many processes. Python: python (pandas broken). cv2.imwrite fails silently on non-ASCII paths: use cv2.imencode(...)[1].tofile(path). No downloads. Write only under <SCRIPTS>\ + JOURNAL.md + git; read the input in place on F: (never write on F:). Never modify existing version folders (v1..v13, "v13 SHO").

Read first: JOURNAL.md (last 3 entries: M8 run + M8 diagnosis), "Methode v13 SHO – 21 nov\LISEZMOI.txt", "...\diagnostic_m8\" scripts, and the code lines cited below.

Diagnosis (M8 SHO, tiles a_coeur, c_champ_dense, d_brillante; crops in decoupes_m8_sho.json):
- Saturation: the G 5.23 star in d is clipped (54/65/29 px ≥ 99 % of each channel's max, clip levels differ per channel) but piste 1 sees no saturation (Piste1\retrait_modele.py ~l.633-644: no plateau found on the 12 brightest → threshold inf, rs = 0); the clipped core is fitted as real → over-subtraction ring r 6–10 (O median −92, min −762) → black hole; stage A only half fixes it.
- Spikes: the image has 6 spikes at 60° (53,114,173,234,293,354°); stage A keeps 4 because Passes\mesure.py:130 and Piste3\mesure.py:104 cap `peaks(S, 2.5)[:4]` (a symmetric profile gives 2 peaks per spike pair → cap = a cross). In piste 1, spikes() (retrait_modele.py ~l.1873) detected nothing on d and spike_model is off by default (~l.1791). Also Piste3\piste3.py:134-136 leaves the G 5.23 star whole (ring 8–15 px at 10 % of peak > 4 % threshold; GIANT_G 4.5).
- Rings/cyan edges: born in piste 1; the step at the fill border is almost only in S (SII). In modele_psf_image.npz the S wing goes negative beyond 13.5 px (−0.2e-4 per px per unit flux) while H and O stay positive; measured FWHM S/H/O on 668 isolated stars 4.07/4.22/4.37 px vs model 3.78/4.22/4.18 (S 7 % too narrow).

Versions (copy the previous version folder each time: code + Catalogue + Piste1\sorties with the image models; piste 1 reruns; delete sorties\<tile>\A\sorties\base_piste1 before each run):
- v14 (from "Methode v13 SHO – 21 nov" → "Methode v14 SHO – 21 nov"): saturation detected PER CHANNEL robustly — a pixel is saturated in a channel if it is ≥ (1 − ε) of that channel's maximum over the image (ε ~0.5–1 %, state it) with a minimum plateau size; the existing saturated-core handling then applies. Run d_brillante.
- v15 (from v14): spike angles — no fixed cap of 4 peaks: keep all significant peaks of the symmetric angular profile (pairs 180° apart), so 6 spikes are found; apply in both mesure.py and check piste 1's spikes() finds them (if piste 1 needs its spike handling on for these stars, state what you enabled). Re-examine the "left whole" rule for the G 5.23 (with saturation now detected, does it still trip? if it does and the star is not a giant of the σ Ori class, report it — don't change that rule in this version). Run d_brillante.
- v16 (from v15): the per-channel star model — wing forced ≥ 0 (no negative wing) and width per channel fitted on the image's isolated stars (S/H/O), in the whole-image model builder (modele_psf_image.py) — rebuild the model. Run a_coeur, c_champ_dense, d_brillante.
After each version: measures on its tiles (dark points < −5 and bright leftovers > +5 noise, L = channel mean, 9×9 median, sig = 1.4826·MAD; invented share; for d: per-channel annulus means r 0–6, 6–12, 12–30, 30–60 minus ring 100–140 px sky; spike count found and angles; for a/c: per-channel step at fill borders as the diagnosis measured), and whole-tile views 3 panels input | previous version | new version, x2, stretch arcsinh 40 with per-channel black = median of the whole image, white = per-channel 99.8th pct of the whole image (as M8_SHO_*_entree_v13_reseau.png) → <SCRIPTS>\A_REGARDER – 21 nov soir\M8_SHO_<tile>_<prev>_<new>.png. Look at each view yourself, 1 sentence each.
