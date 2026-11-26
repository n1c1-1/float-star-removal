# Locate stage of colour spots

- Date : 2026-11-21 à 14:33 (heure de Paris)
- Numéro du jour : 13
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Diagnosis only, no fix. Work folder: <SCRIPTS>\Methode validee – 21 nov\ (git repo root is <SCRIPTS>, tag methode-validee-v1). Python: python. CPU only. Write ONLY inside <SCRIPTS>\Methode validee – 21 nov\diagnostic_taches\ (create it). Never modify any existing script or output (copy if needed). Read LISEZMOI.txt there first and methode_validee.py to see where each stage's output lives: sorties\<crop>\A\ (stage A = hybrid), sorties\<crop>\B\ (passes t1, t2, t3), piste 1 output, final sans_etoiles_tuile.npy.

Problem: on crop "filaments" the final result keeps small defects, seen in the x4 views sorties\filaments\grand_x4_haut_droite_entree_resultat.png and grand_x4_haut_gauche_entree_resultat.png (each view = input | result, the 512 px tile cut into 4 quarters of 256 px, x4): (a) 2–3 px "rainbow" clusters (neighbouring pixels of very different hue: magenta/green/blue/yellow) at former star positions; (b) small dark dots; (c) a thin circular ring around a removed medium star (seen at hard stretch, top-left quarter around view x≈1226-1028, y≈814 → tile ≈ (50,203)); (d) a short vertical smear. Approx positions in the result half of the views (result half starts at x≈1033; tile px ≈ (x-1033)/4 + quarter offset, y/4 + quarter offset; haut_droite offset (256,0), haut_gauche (0,0)): haut_droite (1085,400) (1983,623) (1330,1000); haut_gauche (1947,242) (2039,906).

Tasks:
1. Write a small detector (script in diagnostic_taches\) for (a) and (b) on the final tile: (a) pixel whose chroma (R−G, B−G) deviates from a 7×7 median by > 6 local noise in at least one chroma channel, components ≤ 12 px; (b) luminance below a 5×5 median-without-centre by > 5 local noise, components ≤ 12 px. Exclude the tile's outer 4 px. Count and list positions for crops filaments, lisse and decoupe6 (sorties\decoupe6 exists too).
2. For every detected spot on filaments (and the 5 listed positions), find the FIRST stage where it exists: piste 1 starless, stage A, t1, t2, t3 (same detector run on each stage, matched within 2 px). Also for (c) the ring and (d) the smear, say which stage makes them (look at the stage outputs around those points, crops x8 PNG, input | each stage).
3. For the stage that creates most spots, read that stage's code and give the exact mechanism (function, line) that produces a per-channel/colour outlier or a dark pixel there. Do not change code.
