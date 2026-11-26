# Final pass on leftovers (c9)

- Date : 2026-11-21 à 08:51 (heure de Paris)
- Numéro du jour : 08
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Classical (non-neural) star removal, CPU only. Do the work, report briefly in French.

WHERE YOU MAY WRITE: only in a new folder <SCRIPTS>\Piste 3 remplissage – 20 nov\sorties\passe_finale_c9\ (create it) and a script passe_finale.py next to piste3.py in <SCRIPTS>\Piste 3 remplissage – 20 nov\. Nothing anywhere else, no temp files elsewhere, no downloads, never name or use [outil de comparaison]. CPU only: set CUDA_VISIBLE_DEVICES=-1. Do not modify existing scripts or outputs (the night patched piste3.py in place; do not rerun it).

BASE (do not recompute): the owner's validated reference of last night, <SCRIPTS>\Piste 3 remplissage – 20 nov\sorties\hybride_c9\rosette_decoupe6\:
- sans_etoiles_tuile.npy = starless result (the base to improve), carte_remplissage_tuile.npy = invented-pixel map, masque_tuile.npy, zones_dures_tuile.npy, mesures.json, vue_entree_x2.png / vue_resultat_x2.png = the views the owner saw (Post-type stretch, x2).
- Find the matching input-with-stars tile and the Gaia positions/magnitudes used (deep catalogue in <SCRIPTS>\Catalogue Gaia profond – 20 nov\) by reading how hybride_c9 was produced (lancer.py, piste3.py, patch_hybride.py, the 'Retrait étoiles par modèle – 20 nov' folder, c9 = piste 1 run c9 on the denoised Rosette input, crop "decoupe6"). Reuse its stretch code (the one that made vue_resultat_x2.png) so views are identical in stretch.

THE CHANGE (one change only): a last pass, like a operator going over leftovers with the healing tool after the bulk removal.
1. Residual map: result minus a robust local background (median in an annulus / large-scale smooth that excludes star zones), in noise units, per colour and luminance.
2. Leftovers = connected areas where |residual| exceeds a threshold (bright leftovers AND dark pits/holes, both signs), restricted to where a star is: within a radius growing with brightness around each Gaia position (also off-catalogue sources piste 1 detected if available), plus along spikes for bright stars (spike directions measurable from the input). Never touch anything far from a star (protects nebula knots and wisps).
3. Grow each leftover mask a little (2–3 px + edge feather), fill from the surrounding already-cleaned sky with the existing fill code (remplir.py: PatchMatch/Criminisi + Poisson + grain match), level and colour matched to the ring.
4. Guard: a leftover zone that is large AND sits on structured nebula (high local structure at 16–48 px) is not filled but marked "à ignorer" in a separate mask.
5. Run the pass twice (second pass on the first's output). Update the invented-pixel map (exact: result != base).
Choose thresholds yourself, a few quick tries on 2–3 zooms are fine, but keep it to one method.

OUTPUTS in passe_finale_c9\: sans_etoiles_tuile.npy, carte_remplissage_tuile.npy, masque_ignorer.npy, vue_resultat_x2.png (same stretch, same framing as c9's), cote_a_cote_x2.png = c9 result | final-pass result, same stretch, plus a hard-stretch version of that side-by-side, and zooms x3 at the 6 owner stars (positions in the 9_six_* file names of c9) c9 | new. A short mesures.json: number of leftover zones filled per pass, invented share before/after, ignored share.

Do NOT judge the result as clean or better; just describe what changed and any visible problems you notice. Report under 15 lines, with the output folder path. Budget about one hour.
