# v10: always heal isolated 1-3 px outliers

- Date : 2026-11-21 à 19:36 (heure de Paris)
- Numéro du jour : 30
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

One-change experiment. COMPUTE CAP (strict, the PC saturated earlier): run ONE chain at a time, never in parallel, with environment OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4; no bash loops that fork many processes. Python: python (pandas broken). cv2.imwrite fails silently on non-ASCII paths: use cv2.imencode(...)[1].tofile(path). <SCRIPTS> is a git repo. Never modify v1..v9 folders.

Read first: JOURNAL.md (last 2 entries: v8, v9) and "Methode v9 – 21 nov\Passes\passe_finale_p.py" (star_leftovers_only, star_blob, point_is_star_leftover) + passe_finale_t2_p.py (isolated dark pixel detector) + how t3 uses point_is_star_leftover.

Finding (owner, coeur_est whole square): black points are back in v9. Zooms by stage (<SCRIPTS>\A_REGARDER – 21 nov soir\B_coeur_est_points_noirs_v9_par_etape.png): isolated 1–3 px black pixels or short diagonal chains (down to −680 noise vs a 9×9 median), created by stage A at the edge of recreated bright-star cores (tile (271,114), (208,326), (306,126), (307,123), (304,133), (303,135), (305,130)), 10–15 px from the star centre. v7's t2 healed them (isolated dark pixel detector); since v8/v9 the star-leftover test rejects them (not within 6 px of a source / not a round PSF-scale blob).

The ONE change (v10): in t2's isolated-pixel healing (and t3's isolated peaks if they go through the same test), a component of at most 3 px whose deviation exceeds 8 local noise (vs the 5×5 median without centre, as the detector already measures) is ALWAYS healed by the neighbour median, without the star-leftover test — a filament is never a 1–3 px isolated point. Larger components keep the v9 test. Nothing else changes.

Do: copy "Methode v9 – 21 nov" (code + Catalogue + Piste1\sorties for --reutiliser-piste1) to "<SCRIPTS>\Methode v10 – 21 nov\", apply, run --reutiliser-piste1 on coeur_est, then decoupe6 (one at a time).
Measure v9 vs v10: count of dark points < −5 noise (9×9 median, as in the owner's check: L = mean of channels, sig = 1.4826·MAD(L − median9)), the 15 worst with values; rainbow/dark detector counts; filament high-pass rms 3–8 px in coeur_est x 40–300, y 150–350 (v9 final 0.036 — must not drop); invented share.
Deliverables: <SCRIPTS>\A_REGARDER – 21 nov soir\B_coeur_est_ENTIER_entree_v9_v10.png (3 panels input | v9 | v10, x2; stretch arcsinh 60, black = median of full image green [::7,::7], white = 99.5th pct of full[1,500:2500,1100:3100]; tile = full[:,1394:1906,2194:2706]) and B_coeur_est_ENTIER_entree_resultat_v10.png (input | v10).
