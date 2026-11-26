# v11: two-sign star-scale test

- Date : 2026-11-21 à 19:47 (heure de Paris)
- Numéro du jour : 31
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

One-change experiment. COMPUTE CAP (strict): ONE chain at a time, never in parallel, env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4; no bash loops forking many processes. Python: python (pandas broken). cv2.imwrite fails silently on non-ASCII paths: use cv2.imencode(...)[1].tofile(path). <SCRIPTS> is a git repo. Never modify v1..v10 folders.

Read first: JOURNAL.md (last 3 entries: v8, v9, v10) and "Methode v10 – 21 nov\Passes\passe_finale_p.py" (star_leftovers_only, star_blob, dog_stack, point_is_star_leftover).

Finding: in coeur_est, medium-star leftovers from piste 1 that look like a small coloured RING with a dark centre (e.g. tile (113,109), worst remaining dark point −52 noise; also (449,497) dipole red/dark) are not healed by t1/t2/t3 in v9/v10. Cause (my reading): the star-scale blob is built from ONE sign of the DoG (signe·DoG > half peak), so a dark-centre + bright-ring leftover gives a small dark core or an annulus — the test then rejects it or heals only the centre.

The ONE change (v11): build the star-scale blob from the MAGNITUDE of the residual at star scale, both signs together: |DoG| summed over channels (R, G, B in noise units), blob = connected region where it exceeds half its peak, filled (binary_fill_holes) so a ring becomes a disc; the same tests apply (axis ratio < 2, equivalent radius ≤ 2·FWHM, centroid ≤ 6 px of a source); healed pixels = the component (as now) united with that filled blob, restricted to the star's zone disc. Nothing else changes.

Do: copy "Methode v10 – 21 nov" (code + Catalogue + Piste1\sorties) to "<SCRIPTS>\Methode v11 – 21 nov\", apply, run --reutiliser-piste1 on coeur_est, then decoupe6, then filaments (one at a time).
Measure v10 vs v11: dark points < −5 noise (L = channel mean, 9×9 median, sig = 1.4826·MAD) count and 10 worst; rainbow/dark detector counts; value at (113,109) and (449,497); filament high-pass rms 3–8 px coeur_est x 40–300, y 150–350 (v10 0.0354 — must not drop more than 5 %); invented share.
Deliverables: <SCRIPTS>\A_REGARDER – 21 nov soir\B_coeur_est_ENTIER_entree_v10_v11.png (input | v10 | v11, x2; stretch arcsinh 60, black = median of full image green [::7,::7], white = 99.5th pct of full[1,500:2500,1100:3100]; tile = full[:,1394:1906,2194:2706]); plus the same 6-row stage zoom as "B_coeur_est_points_noirs_v9_par_etape.png" redone on v11's 6 worst dark points → B_coeur_est_points_noirs_v11_par_etape.png.
