# v9: star-scale leftover test

- Date : 2026-11-21 à 18:31 (heure de Paris)
- Numéro du jour : 29
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

One-change experiment, continuation of v8. Python: python (pandas broken). cv2.imwrite fails silently on non-ASCII paths: use cv2.imencode(...)[1].tofile(path). <SCRIPTS> is a git repo. Never modify v1..v8 folders; another agent works in "Methode v5 GPU – 21 nov" — don't touch it.

Read first: JOURNAL.md (last 2 entries: v8), "Methode v8 – 21 nov\Passes\passe_finale_p.py" (function star_leftovers_only and where it is called in t1, t2, t3).

v8 result: the rule "heal a leftover only if round (axis ratio < 2), size ≤ star zone, centroid ≤ max(2 px, 0.7·FWHM) from a source" keeps the Rosette's fine dark filaments (good) but leaves 3× more star leftovers (rainbow spots 91→308 on coeur_est); 72–88 % of the unhealed zones are within 6 px of a source, so they ARE star leftovers, rejected because they touch a filament / nebula structure (component looks elongated) or are slightly off-centre.

The ONE change (v9): judge each candidate at STAR SCALE. Compute the roundness/size test not on the raw detected component but on its PSF-scale part: residual band-passed at the star scale (e.g. difference of Gaussians σ ≈ 0.5·FWHM minus σ ≈ 2·FWHM, or the residual minus a 2·FWHM-scale median/blur), thresholded at half its peak inside the component; the candidate is a star leftover if that star-scale blob is round (axis ratio < 2), not larger than ~2·FWHM in radius, and its centroid is within 6 px of a source (Gaia/piste 1 list). Then heal the candidate as before (the same pixels the pass would have healed in v7, restricted to within the star's zone). Elongated structures at that scale (filaments) stay untouched. State the exact scales used.

Do: copy "Methode v8 – 21 nov" (code + Catalogue + Piste1\sorties to reuse piste 1) to "<SCRIPTS>\Methode v9 – 21 nov\", apply, run --reutiliser-piste1 on coeur_est, decoupe6, filaments, lisse.

Measure v7 / v8 / v9: rainbow / dark counts, dark pits < −5 noise, invented share, filament high-pass rms 3–8 px in coeur_est region x 40–300, y 150–350 (as you/the v8 agent measured: v7 t2 0.029, v8 0.049, stage A 0.052).
Deliverables: <SCRIPTS>\A_REGARDER – 21 nov soir\B_coeur_est_ENTIER_entree_v8_v9.png (3 panels input | v8 | v9, x2, stretch: arcsinh 60, black = median of full image green [::7,::7], white = 99.5th pct of full[1,500:2500,1100:3100]; tile = full[:,1394:1906,2194:2706]) and B_coeur_est_ENTIER_entree_resultat_v9.png (input | v9); x3 quarter views input | v7 | v9 at Post stretch for decoupe6, filaments, lisse in v9 sorties\comparaison\. Also <FLOAT>\_travail\comparaison_[outil de comparaison]_21nov\B_coeur_est_ENTIER_entree_v9_[outil de comparaison].png (input | v9 | [outil de comparaison]; [outil de comparaison] tile = donnees\coeur_est_[outil de comparaison].npy[:,128:640,128:640]) — [outil de comparaison] images ONLY there.
