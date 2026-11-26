# Blind judge nebula crops

- Date : 2026-11-21 à 14:33 (heure de Paris)
- Numéro du jour : 14
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

You are a blind judge of a star-removal result. You did not make it; be strict but factual. Read-only on everything except your output folder <SCRIPTS>\Juge – essais 20 nov\juge_nebuleuse_v1\ (create it; write only there). Python: python, CPU only.

Checklist to apply, all 12 points: <SCRIPTS>\Retrait étoiles par modèle – 20 nov\LISTE_CONTROLE_JUGE.txt (read it first).

Material, two crops of the Rosette nebula: <SCRIPTS>\Methode validee – 21 nov\sorties\filaments\ and ...\sorties\lisse\. Each has the final starless tile sans_etoiles_tuile.npy, the stage folders, mesures.json, and views grand_x4_<quarter>_entree_resultat.png (input | result, x4, Post stretch) plus _fort (hard stretch). Find the matching input tile by reading <SCRIPTS>\Methode validee – 21 nov\methode_validee.py (do not run it).

Known context, do not count as defects: a darker/redder sky where many stars were removed is starlight leaving, not a defect; giant stars (σ Ori class) are meant to stay whole. Prefer looking at 128 px squares at 100 % and x2 (make your own PNG crops input | result at Post and hard stretch) over the reduced views.

Report in French, ≤ 30 lines: for each crop and each of the 12 points: verdict (OK / défaut), count, positions in tile pixels (x,y), and the path of a zoom PNG showing the worst case. End with one line per crop: "propre à l'étirement Post : oui/non" and "propre à l'étirement fort : oui/non". No proposals of fixes.
