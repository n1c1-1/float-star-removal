# v8: passes skip elongated structures

- Date : 2026-11-21 à 18:25 (heure de Paris)
- Numéro du jour : 28
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

One-change experiment. Python: python (pandas broken). GPU 4060 Ti allowed but this code is CPU numpy. cv2.imwrite fails silently on non-ASCII paths: use cv2.imencode(...)[1].tofile(path). <SCRIPTS> is a git repo. Never modify v1..v7 folders; another agent works in "Methode v5 GPU – 21 nov" — don't touch it.

Read first: JOURNAL.md (last 3 entries), "Methode v7 – 21 nov\LISEZMOI.txt", and the three passes in "Methode v7 – 21 nov\Passes\" (passe_finale_p.py = t1, passe_finale_t2_p.py, passe_finale_t3_p.py) + etape_B.py.

Finding (owner's test crop "coeur_est", centre (2450,1650), the Rosette core with fine dark filaments crossed by many stars): the fine dark filaments (3–8 px wide, long) are intact after piste 1 and stage A, and ERASED after t2 (smoothed into soft blobs); t3 changes nothing more. t2's detectors (DoG 3–8 px colour-aware, compact spots > 8 noise ≤ 80 px, isolated dark) take filament pieces near stars as star leftovers and fill them; t1 heals leftovers of both signs near stars and may also contribute — check t1 too (compare A, t1, t2 on the filament region tile x 40–300, y 150–350).

The ONE change (v8): in the passes (t1 and t2, and t3 if it does the same), a detected leftover component is healed only if it is a star leftover: COMPACT and ROUND — elongation (ratio of the two eigenvalues of its second moments, or major/minor axis) < 2 and size ≤ the star's zone — AND its centroid is within 2 px (or 0.7×FWHM) of a catalogued star / piste 1 source; elongated or off-centre dark/bright structures are left untouched. State exactly where you apply it in each pass. Nothing else changes.

Do: copy "Methode v7 – 21 nov" (code + Catalogue folder + Piste1\sorties incl. the image PSF model, to reuse piste 1 with --reutiliser-piste1) to "<SCRIPTS>\Methode v8 – 21 nov\", apply, run methode_validee.py <crop> --reutiliser-piste1 on coeur_est first, then decoupe6, filaments, lisse.

Deliverables:
- Whole-square view for coeur_est, 2 panels input | v8 and 3 panels input | v7 | v8, x2 Lanczos, stretch exactly: arcsinh 60, black = median of the green channel of the full image (sampled [::7,::7]), white = 99.5th pct of full[1, 500:2500, 1100:3100]; the 512 tile = full image [1394:1906, 2194:2706]. Save input | v8 as <SCRIPTS>\A_REGARDER – 21 nov soir\B_coeur_est_ENTIER_entree_resultat_v8.png and the 3-panel as ...\B_coeur_est_ENTIER_entree_v7_v8.png. Also the filament-region strip (tile x 40–300, y 150–350) input | A | t1 | t2 | final for v8, x2, same stretch, as ...\B_coeur_est_filament_etapes_v8.png.
- For decoupe6, filaments, lisse: x3 quarter views input | v7 | v8 at Post stretch (vues.py style) in v8 sorties\comparaison\, plus detector counts rainbow/dark and dark pits < −5 noise (as in previous entries) and invented share, v7 vs v8 — to check no regression (leftovers that were healed before and are now left must be looked at).
- If, with v8, the [outil de comparaison] panel is wanted: <FLOAT>\_travail\comparaison_[outil de comparaison]_21nov\donnees\coeur_est_[outil de comparaison].npy (3,768,768; tile = [:,128:640,128:640]) — make input | v8 | [outil de comparaison] whole-square view and save it ONLY in <FLOAT>\_travail\comparaison_[outil de comparaison]_21nov\B_coeur_est_ENTIER_entree_v8_[outil de comparaison].png (never [outil de comparaison] images in starRemoval).
Send me a one-line message as soon as the coeur_est whole-square views exist (before the regression crops). Final report in French, ≤ 15 lines.
