# Test v4: per-colour star positions

- Date : 2026-11-21 à 14:51 (heure de Paris)
- Numéro du jour : 19
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

One-change experiment on a classical star-removal method. CPU only (CUDA_VISIBLE_DEVICES=-1; run at most 2 crops in parallel). Python: python. <SCRIPTS> is a git repo. Read first: JOURNAL.md (last 4 entries), "Methode v3 – 21 nov\LISEZMOI.txt", and "Methode v3 – 21 nov\diagnostic_dipoles\" (diag.py, results).

Current best = v3 ("<SCRIPTS>\Methode v3 – 21 nov\"). Never modify v1, v2, v1g, v3 folders.

Diagnosis: piste 1 (Piste1\retrait_modele.py, fit_star line ~364-460) fits each star's flux per colour at ONE position common to the three colours (Gauss-Newton weighted by flux; per-colour offset field "choff" off). Isolated stars G 11.5–15.5 have real per-star R−G / B−G centroid offsets of ~0.04 px rms, random in direction (no global or radial pattern); the residual dipole correlates 0.6–0.8 with these offsets and peaks at 10–19 noise per colour → rainbow 2–3 px spots that the later passes only attenuate. Also check and report whether some of these stars are fitted with fix_pos=True (position fixed at the catalogue), since the G dipole exists too.

The ONE change (v4): as in standard per-band PSF photometry (each band fitted independently with its own centroid), after the common-position fit, refine the position SEPARATELY PER COLOUR for stars whose per-colour peak S/N is high enough (choose a threshold, e.g. peak > 20 noise in that colour, state it), with Gauss-Newton on that colour only, bounded to ±0.3 px around the common position; the model of each colour is then built at its own position (use the existing choff mechanism per star: choff = per-colour offset from the common position, for that star, in fit_star and in star_model — read how choff flows through fit/refit/star_model and keep it consistent everywhere the model of that star is evaluated, including the final subtraction). Fainter stars unchanged. Nothing else changes.

Do: copy "Methode v3 – 21 nov" (code only; NOT Piste1\sorties — piste 1 must rerun) to "<SCRIPTS>\Methode v4 – 21 nov\", apply the change in the copy, run the full chain methode_validee.py <crop> (piste 1 included) on filaments, lisse, decoupe6.

Measure v3 vs v4: (a) on piste 1 starless: residual dipole peak / noise at the isolated G 11.5–15.5 stars (reuse diag.py) — median and 90th percentile per colour; (b) on the final tiles: rainbow spots and dark dots counts with the detector of "Methode validee – 21 nov\diagnostic_taches\detecteur.py" (same thresholds); (c) invented share; (d) lane/globule check as in mesures_v1_v2_v3.py (must stay as v3); (e) piste 1 run time v3 vs v4; (f) x6 zooms input | v3 | v4 at the 4 worst v3 rainbow spots of lisse and filaments (Post stretch as vues.py, plus hard), in v4 sorties\comparaison\.


Report in French, ≤ 20 lines: fix_pos finding, threshold used, tables (a)–(e), paths of zooms and of the v4 x4 quarter views (filaments, lisse, decoupe6).
