# Blind judge, round 3 result

- Date : 2026-11-21 à 12:18 (heure de Paris)
- Numéro du jour : 10
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

You are a blind, strict judge of a star-removal result (astrophotography, Rosette nebula, dense star cluster crop). Report in French, under 20 lines. CPU only (CUDA_VISIBLE_DEVICES=-1). Write ONLY in a new folder <SCRIPTS>\Juge – essais 20 nov\juge_passe_finale_t3\ (create it). Python: python (read only).

Result to judge: <SCRIPTS>\Piste 3 remplissage – 20 nov\sorties\passe_finale_c9_t3\sans_etoiles_tuile.npy (and mesures.json there for point 12).
A previous judge built loaders, stretches and measures for the same tile: reuse its scripts j0_charger.py … j4_zooms.py from <SCRIPTS>\Juge – essais 20 nov\juge_passe_finale_t2\ (copy into your folder and point them at the new result). Do NOT read that judge's conclusions (mesures_juge.json, its report) before forming your own verdict — judge fresh.
Checklist, point by point: <SCRIPTS>\Retrait étoiles par modèle – 20 nov\LISTE_CONTROLE_JUGE.txt (points 1–10, 12; skip 11).
Method: whole tile in 128 px squares at 100 % and x2, Post-type stretch and hard stretch, input | result; look at every square yourself. Measure grain at 2–4 px (ratio map, blocks of 32 px) and point defects (dark/bright dots at 3 and 5 noise).
Calibration warnings: (a) in this dense cluster the input's "star-free" sky holds the glow of star halos (bluish near hot stars), so a darker / redder starless sky there is expected, not a defect, unless contrast of nebula structures is reversed in places far from stars; (b) the owner judged the faint vertical glow near (305,300) to be starlight, not nebula. Give positions (x, y tile) and save a zoom of every defect you cite.
Final line: "propre à l'étirement Post : oui/non", "propre à l'étirement fort : oui/non", the 3 worst defects.
