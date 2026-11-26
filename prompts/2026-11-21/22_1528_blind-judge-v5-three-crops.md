# Blind judge v5, three crops

- Date : 2026-11-21 à 15:28 (heure de Paris)
- Numéro du jour : 22
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

You are a blind judge of a star-removal result. You did not make it; be strict but factual. Read-only on everything except your output folder <SCRIPTS>\Juge – essais 20 nov\juge_v5\ (create it; write only there). Python: python, CPU only.

Checklist to apply, all 12 points: <SCRIPTS>\Retrait étoiles par modèle – 20 nov\LISTE_CONTROLE_JUGE.txt (read it first).

Material, three 512 px tiles of the Rosette nebula: <SCRIPTS>\Methode v5 – 21 nov\sorties\decoupe6\, \filaments\, \lisse\. Each has the final starless tile sans_etoiles_tuile.npy, carte_remplissage_tuile.npy, mesures.json, and views grand_x4_<quarter>_entree_resultat.png (input | result, x4, Post stretch) plus _fort (hard stretch). The matching input tile: see how vues.py in "Methode v5 – 21 nov" loads it (pf.load_all with MV_CFG = sorties\<crop>\B\cfg.json) — do not run the chain.

Known context, do not count as defects: a darker/redder sky where many stars were removed is starlight leaving; giant stars (σ Ori class) are left whole on purpose. Point 8 (large structure): compare blurred 16 and 48 px of the result with the input after masking stars (not with the raw input, whose mean includes stars). Prefer 128 px squares at 100 % and x2/x3 (make your own PNG crops input | result at Post and hard stretch) over the reduced views; look at EVERY 128 px square of each tile at Post stretch.

Report in French, ≤ 30 lines: for each crop and each of the 12 points: verdict (OK / défaut), count, positions in tile pixels (x,y), path of a zoom PNG of the worst case. End with one line per crop: "propre à l'étirement Post : oui/non" and "propre à l'étirement fort : oui/non". No proposals of fixes.
