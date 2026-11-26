# Blind judge, round 2 result

- Date : 2026-11-21 à 11:50 (heure de Paris)
- Numéro du jour : 09
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

You are a blind, strict judge of a star-removal result (astrophotography, Rosette nebula, dense star cluster crop). Report in French, under 20 lines. CPU only (set CUDA_VISIBLE_DEVICES=-1). Write ONLY in a new folder <SCRIPTS>\Juge – essais 20 nov\juge_passe_finale_t2\ (create it). Use the Python at python (read only).

Inputs (numpy, same tile, channels-first or -last: check):
- Input with stars: find it the way <SCRIPTS>\Piste 3 remplissage – 20 nov\passe_finale_t2.py loads it (read that script only to learn paths and the stretch function; do NOT read anything about how the result was made beyond that).
- Result: <SCRIPTS>\Piste 3 remplissage – 20 nov\sorties\passe_finale_c9_t2\sans_etoiles_tuile.npy
- Checklist to apply point by point: <SCRIPTS>\Retrait étoiles par modèle – 20 nov\LISTE_CONTROLE_JUGE.txt (points 1–10; 11 skip; 12 report from mesures.json in the result folder).
- Calibration: earlier judge reports are in the sibling folders of Juge – essais 20 nov (e.g. juge_hybride_L12_final) — look at their format only.

Method: render with the same Post-type stretch as the script and a hard stretch; scan the whole tile in 128 px squares at 100 % and x2 (look at the images yourself, every square), input | result side by side. Also check: (a) grain/texture — is the result over-smoothed vs the input's star-free sky (local std at 2–4 px scale, ratio per region, and visually)? (b) nebula structure kept (blurs 16/48 px of result vs input in star-free areas); (c) colour shift of star-free sky; (d) any residual star, dot, ring, spike bar, edge of a filled zone, blotch.
Warning from past calibration: in a dense field, "Gaia ratio" and "background removed" claims can be artefacts; base verdicts on what you see, give positions (x, y tile) and save zoom crops of every defect you cite into your folder.
Final line: verdict "propre à l'étirement Post : oui/non", "propre à l'étirement fort : oui/non", and the 3 worst defects.
