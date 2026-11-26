# Test v2: wide-dip relevel off

- Date : 2026-11-21 à 14:37 (heure de Paris)
- Numéro du jour : 15
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

One-change experiment on a classical star-removal method. CPU only (set CUDA_VISIBLE_DEVICES=-1). Python: python.

Context: <SCRIPTS> is a git repo. The validated method is frozen in "<SCRIPTS>\Methode validee – 21 nov\" (tag methode-validee-v1): methode_validee.py <crop> chains piste 1 -> stage A (Piste3\piste3.py, hybrid) -> stage B (Passes\, three final passes) -> views (vues.py). Read its LISEZMOI.txt first. NEVER modify anything inside "Methode validee – 21 nov" — another agent is reading it right now; it is frozen.

Defect found: in Piste3\piste3.py, the flag DIPS = True (line 40) enables, around line 533-548, a relevel of "wide dips of the base" (dark areas >= 40 px below a 41 px median background -> F.relevel_lowfreq). On nebula crops it fills REAL dark structures: on crop "filaments" the dark S-lane at tile x 255-275, y 170-260 goes from 10th percentile 0.44 (input and piste 1) to 0.77 after stage A; on crop "lisse" the dark globule near tile (35,50) is filled. Log lines "creux larges de la base : 42, 18356 px" (filaments), "50, 22427 px" (lisse), "14, 3304 px" (decoupe6).

Task:
1. Copy the whole folder "Methode validee – 21 nov" to "<SCRIPTS>\Methode v2 – 21 nov\" EXCEPT its sorties\ outputs (but you may reuse the piste 1 outputs: copy Piste1\sorties\ so --reutiliser-piste1 works, or point to them read-only). In the copy, make exactly ONE change: DIPS = False (or an env/config switch defaulting to False). Nothing else.
2. Run the copy's chain with piste 1 reused (methode_validee.py <crop> --reutiliser-piste1, check the flag name in the script) on crops decoupe6, filaments, lisse. Outputs in the copy's sorties\.
3. Measure, on each crop, versus v1 (read-only from "Methode validee – 21 nov\sorties\<crop>\sans_etoiles_tuile.npy") and versus the input tile: (a) filaments lane region x255-275,y170-260 and lisse globule region x20-60,y30-70: mean and 10th percentile of luminance for input, piste 1 starless, v1 final, v2 final; (b) count of dark pits below -5 local noise (5x5 median-without-centre, components <= 12 px, exclude 4 px border) in v1 and v2; (c) invented share from mesures.json. (d) Make my-format views for decoupe6 (where DIPS was originally meant to help): crop x6 PIL zooms input | v1 | v2 of the 3 places where v1 and v2 differ most (Post stretch as in vues.py), saved in the copy's sorties\comparaison_v1_v2\.

Report in French, <= 20 lines: table of (a), (b), (c) for v1 vs v2, paths of the x4 quarter views of v2 for filaments and lisse and of the decoupe6 comparison zooms. No interpretation beyond the numbers and what you see.
