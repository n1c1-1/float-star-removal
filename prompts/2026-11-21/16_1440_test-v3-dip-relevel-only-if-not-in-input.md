# Test v3: dip relevel only if not in input

- Date : 2026-11-21 à 14:40 (heure de Paris)
- Numéro du jour : 16
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

One-change experiment, continuation of the v2 test (read <SCRIPTS>\JOURNAL.md entries 7-8 first, and "Methode validee – 21 nov\LISEZMOI.txt"). CPU only (CUDA_VISIBLE_DEVICES=-1). Python: python. starRemoval is a git repo.

Findings so far: in Piste3\piste3.py, DIPS (line 40; code ~533-548: dark components >= 40 px below a 41 px median of a 3 px blur, dilated 4 px, -> F.relevel_lowfreq) fills REAL dark nebula structures (filaments lane x255-275,y170-260; lisse globule x20-60,y30-70). But turning it off (v2, folder "Methode v2 – 21 nov") brings back wide over-subtraction dark discs on decoupe6 (visible in sorties\decoupe6\comparaison_v1_v2\zoom3_x171_y167_entree_v1_v2.png: a large faint dark disc lower half and a dark blob top-left, near a bright star). So DIPS is right in principle, its selection is wrong.

The ONE change for v3: keep DIPS = True, but drop a dip component when the dip is real structure, i.e. already dark in the INPUT. For each dip component (after the size >= 40 filter, before dilation): take the input I (the window, same coordinates as res), luminance, low-passed with stars masked (use F.lowpass_masked(I, excl, 3.0) or equivalent with excl = hard | dilated other_all star mask — use the masks available at that point; keep it simple and say which). Compute depth_res = median(ring) - median(inside) on the 3 px blurred res (ring = dilation 8 minus dilation 2 of the component) and depth_in = same on the masked input low-pass. Keep the dip for relevel only if depth_in < 0.5 * depth_res (input not dark there = artefact of subtraction); otherwise drop it (real structure). Log per crop: number of dips found, kept, dropped.

Do: copy "<SCRIPTS>\Methode validee – 21 nov\" to "<SCRIPTS>\Methode v3 – 21 nov\" (code + Piste1\sorties for reuse, not other sorties). Never modify the v1 or v2 folders. Apply the change in the copy only. Run methode_validee.py <crop> --reutiliser-piste1 on decoupe6, filaments, lisse.

Measure (reuse/copy v2's mesures_v1_v2.py into v3 and extend to v1/v2/v3): (a) lane and globule mean / p10 for input, piste 1, v1, v2, v3; (b) dark pits < -5 noise counts; (c) invented share; (d) for decoupe6: x6 zooms input | v1 | v2 | v3 at the same three places as v2's comparison (x338 y73, x272 y199, x171 y167), Post stretch as vues.py, in v3 sorties\decoupe6\comparaison\. (e) Max abs difference v1 vs v3 on decoupe6 and percent of pixels differing (goal: decoupe6 ≈ v1).


Report in French, <= 20 lines: dips found/kept/dropped per crop, tables (a)(b)(c)(e), and the paths of the v3 x4 quarter views (filaments, lisse) and the decoupe6 comparison zooms.
