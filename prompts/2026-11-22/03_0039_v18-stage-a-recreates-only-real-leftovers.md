# v18: stage A recreates only real leftovers

- Date : 2026-11-22 à 00:39 (heure de Paris)
- Numéro du jour : 03
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

One-change experiment in stage A. COMPUTE CAP (strict): ONE chain at a time, never in parallel, env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4; no bash loops forking many processes. Python: python (pandas broken). cv2.imwrite fails silently on non-ASCII paths: use cv2.imencode(...)[1].tofile(path). No downloads. Write only under <SCRIPTS>\ + JOURNAL.md + git; read the input in place on F:. Never modify existing version folders.

Read first: JOURNAL.md (last 3 entries: M8 step-back, v17), "Recul – 21 nov soir\RECUL.md" (Q3: one residual-healing step on a mask taken from the residual), "Methode v17 SHO – 22 nov\LISEZMOI.txt", and in "Methode v17 SHO – 22 nov\Piste3\piste3.py" how the hard zones are built for bright stars (core disc r_hard / own disc ra, spike bands from mesure.py angles — 6 spikes since v15 —, `left` spike leftovers, zp1 zones from piste 1, closing/fill_holes) and filled (F.structure_texture2).

Finding (owner, M8 tile a_coeur): a dark globule with a bright star on its tip (tile region x 60–230, y 370–512; the star around tile (125,395)) is INTACT after piste 1 (star nearly removed, small ring), and DESTROYED by stage A: class 2 ("cores recreated") covers 12.9 % of that region — a multi-branch star shape plus a diagonal band along a spike direction — filled uniformly with rectangular streaks of copied patches; t1–t3 don't change it. Zoom: <SCRIPTS>\A_REGARDER – 21 nov soir\M8_SHO_a_forme_sombre_par_etape.png. Stage A decides what to recreate from a geometric template, not from what piste 1 actually left.

The ONE change (v18): in stage A, a pixel of a bright star's hard zone (core disc and spike bands) is recreated only if piste 1 left a real leftover there: |piste-1 residual − local background| > k·local noise in at least one channel (k ≈ 3; local background = a robust median at a scale of ~3·FWHM excluding the zone; local noise from the surrounding sky, per channel), the resulting mask lightly dilated (2 px) and its holes/closing as now. Pixels of the template where piste 1 is clean keep the piste 1 value. Saturated cores (v14 saturation masks) are always recreated. State k, the scales, and how many px the masks shrink per tile.

Do: copy "Methode v17 SHO – 22 nov" (code + Catalogue + Piste1\sorties incl. image calibration/models) to "<SCRIPTS>\Methode v18 SHO – 22 nov\", apply, reuse piste 1 (--reutiliser-piste1, piste 1 unchanged) and delete sorties\<tile>\A\sorties\base_piste1 before each run; run a_coeur, b_filaments, c_champ_dense, d_brillante one at a time. Then run the same version on the Rosette crops coeur_est and decoupe6 for non-regression (the Rosette image is declared in the same code; reuse or rerun piste 1 as needed — say which).
Measures v17 vs v18 per tile: class-2/4 share, invented share, dark points < −5 and bright leftovers > +5 noise, and for the globule region the mean absolute difference to piste 1 (should drop); for coeur_est the filament rms 3–8 px (x 40–300, y 150–350).
Views: M8 tiles a and d: 3 panels input | v17 | v18, x2, stretch as the previous M8 views → A_REGARDER – 21 nov soir\M8_SHO_<tile>_v17_v18.png; plus the globule stage strip like M8_SHO_a_forme_sombre_par_etape.png for v18 → M8_SHO_a_forme_sombre_v18.png; Rosette decoupe6 and coeur_est: input | v13 | v18 (x2, arcsinh 60 stretch as B_coeur_est_ENTIER_*) → <crop>_ENTIER_entree_v13_v18.png. Look at each yourself, one sentence each.
