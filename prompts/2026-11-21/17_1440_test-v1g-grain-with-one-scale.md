# Test v1g: grain with one scale

- Date : 2026-11-21 à 14:40 (heure de Paris)
- Numéro du jour : 17
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

One-change experiment on a classical star-removal method. CPU only (CUDA_VISIBLE_DEVICES=-1). Python: python. <SCRIPTS> is a git repo; read JOURNAL.md entries 7-9 and "Methode validee – 21 nov\LISEZMOI.txt" first. Another agent is working in "Methode v3 – 21 nov" — do not touch it, nor v1 ("Methode validee – 21 nov", frozen) nor v2.

Diagnosis (diagnostic_taches in the v1 folder, commit a033abb): in stage B pass t3, file Passes\passe_finale_t3_p.py lines ~237-251, step D restores 2–4 px grain in changed pixels: out[c] = R[c] + wchg*need*T[c]/sT, with `need` computed PER CHANNEL. Logged median amplitude [0.0074, 0, 0]: only red gets grain, so each restored pixel becomes an isolated R−G chroma spot (62 of 114 rainbow spots on crop filaments are created by t3).

The ONE change ("v1g"): in step D, use a single amplitude for all three channels, computed from luminance (the need of the mean of channels, same method as now, applied to the luminance of R and of the real-sky patch T), and add the real-sky RGB patch T (all channels, with its own natural colour) scaled by that single factor. Read the code first and keep everything else identical (patch choice, wchg, normalisation sT — normalise by the luminance std of the patch).

Do: copy "Methode validee – 21 nov" (code + Piste1\sorties, not other sorties, not diagnostic_taches outputs except the detector scripts detecteur.py/run_detect.py you may reuse) to "<SCRIPTS>\Methode v1g – 21 nov\", apply the change in the copy only, run methode_validee.py <crop> --reutiliser-piste1 on decoupe6, filaments, lisse.

Measure, v1 vs v1g: (a) rainbow chroma spots and dark dots counts with the diagnostic detector (same thresholds) on the final tiles of the three crops; (b) grain ratio output/input on unchanged sky and on changed pixels (luminance, and per channel R, G, B); (c) max abs difference and % of pixels differing v1 vs v1g per crop (should be limited to t3-changed pixels); (d) x6 zooms input | v1 | v1g at the 3 worst v1 rainbow spots of filaments (Post stretch as vues.py and hard stretch), in the copy's sorties\filaments\comparaison\.


Report in French, <= 18 lines: tables (a)(b)(c), paths of the zooms and of the v1g x4 quarter views for filaments.
