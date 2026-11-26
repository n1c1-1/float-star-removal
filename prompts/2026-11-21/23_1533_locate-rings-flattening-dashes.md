# Locate rings, flattening, dashes

- Date : 2026-11-21 à 15:33 (heure de Paris)
- Numéro du jour : 23
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Current best = v5. Stage outputs per crop: piste 1 "Methode v5 – 21 nov\Piste1\sorties\rosette\mv_<crop>\starless.npy" (768 px crop, tile = central 512, offset 128) (decoupe6: c9_decoupe6 or as LISEZMOI says), stage A "sorties\<crop>\A\sorties\hybride_c9\mv_<crop>\sans_etoiles_tuile.npy" + carte_remplissage_tuile.npy + zones_dures_tuile.npy + reference_entree_moins_modeles_tuile.npy, passes "sorties\<crop>\B\t1|t2|t3\sans_etoiles_tuile.npy" + carte_remplissage_tuile.npy (+ masque_ignorer.npy). Input tile: load as vues.py does (pf.load_all with MV_CFG).

Defects confirmed (tile px x,y):
A. Thin circular rings, radius ≈ 10 px, around removed medium stars on filaments: (257,197), (264,193), (321,233), (246,221), (146,231).
B. Faint pink nebula arcs/knots flattened (present in input, flat in result): filaments around (257,197)/(264,193) arcs, (246,221), (146,231), (65–100,30–70); decoupe6 faint pink filament (420–445,295–310), (474,303), (490–511,305–320), thin one (301,281–311).
C. Small dark horizontal dashes and steps on lisse: (100,37), step (234,100), corner (243,102), sharp-edged dark disc (81,71).
D. Grain ratio low on filaments (0.50–0.59 in squares x128–383, y128–255).

For each defect: (1) the first stage where it appears (input → piste 1 → A → t1 → t2 → t3), shown by x8 zooms input | p1 | A | t1 | t2 | t3 at Post and hard stretch (Post: arcsinh 40, black = median of green of the full image, white = 99.8th pct, as vues.py; or simply reuse vues.py's functions); (2) the fill class at those pixels (carte_remplissage: 1 halo subtracted, 2 cores recreated, 3 medium corrected in place, 4 base defects recreated, 5/7/8 t1/t2/t3 retakes); (3) for A: which star (G, R, r_core, r_hard from stage A log/mesures) and whether the ring radius matches R, 0.8R (taper start), r_hard, or the t1 disc radius; for B: whether the structure is already lost in piste 1 (i.e. subtracted inside the star models: compare input − piste1 starless = what piste 1 removed; does it contain the arc?), or replaced by a fill (which class); for C: which fill/heal step makes straight horizontal edges (look for row-wise operations, rectangular windows, patch copies); for D: whether low grain is in recreated (2/4/5/7/8) or subtracted (1/3) pixels.

Report in French, ≤ 20 lines: per defect: stage, class, mechanism (file:line when you can point it), one sentence. Paths of zooms.
