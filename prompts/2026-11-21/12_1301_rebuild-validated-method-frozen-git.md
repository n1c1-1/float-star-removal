# Rebuild validated method, frozen + git

- Date : 2026-11-21 à 13:01 (heure de Paris)
- Numéro du jour : 12
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Classical star removal research, folder <SCRIPTS> (now a git repository, first commit 9de7dbe = everything as is; .gitignore excludes images, npy, data). CPU only (CUDA_VISIBLE_DEVICES=-1). Python: python (read only). Write only inside <SCRIPTS>. No downloads, never name or use [outil de comparaison]. Report in French, under 15 lines.


TASK: rebuild the method the owner validated ("très propre") as a FROZEN, reproducible script.
That method = (A) stage 1 "hybride_c9": piste 1 run c9 (Gaia + PSF subtraction, denoised Rosette input, crop decoupe6) + piste 3 fill, saved output: <SCRIPTS>\Piste 3 remplissage – 20 nov\sorties\hybride_c9\rosette_decoupe6\sans_etoiles_tuile.npy (made ~23:00 on 20 Nov, PC time), then (B) passe_finale.py, passe_finale_t2.py, passe_finale_t3.py (same folder, made this morning, unchanged since).
Problem: after c9, the scripts of stage 1 (<SCRIPTS>\Retrait étoiles par modèle – 20 nov\retrait_modele.py and helpers; Piste 3 remplissage – 20 nov\piste3.py, remplir.py, soustraire.py, mesure.py, psf_commune.py…) were changed in place by patch scripts (patch_*.py in Piste 3 remplissage – 20 nov, and possibly similar in the other folder) and by direct edits. Each patch_*.py holds old/new text pairs. A previous agent's attempt (nebu_*.py, sorties\nebulosites_rosette\verif.out) found piste 1 alone differs from c9 on 10 % of pixels (max 24 noise), the full stage A on 48 %.
Steps:
1. List every change made after c9: patch scripts by file date (PC times; c9 hybrid output files date ~22:00–23:15 on 20 Nov), plus look for other edits (file modification times of retrait_modele.py etc. after that; compare with any older copies, .bak, outputs that record the code/options used: mesures.json, resume.json, logs). Note them in JOURNAL.md.
2. In a NEW folder <SCRIPTS>\Methode validee – 21 nov\ copy the needed scripts and reverse those changes there (apply each patch backwards, newest first). Commit.
3. Run stage A from the copy on decoupe6 and compare with the saved c9 result: max and median abs difference in noise units, share of pixels differing > 0.01 noise. Target: identical (or differences only at float rounding). Iterate on what remains different; journal each try. Budget: 45 min of work; if not identical by then, stop and report the exact remaining difference and its likely source.
4. Once identical: add the three final passes (copies of passe_finale*.py, with paths made parameters; t3's fixes tied to named positions of decoupe6 kept only for decoupe6, everything generic kept) into one entry script methode_validee.py <crop> that runs A then B. Run it on decoupe6 and check the final result equals sorties\passe_finale_c9_t3\sans_etoiles_tuile.npy. Commit, tag "methode-validee-v1".
5. Then run it on the two nebula crops already defined (centres (1850,1975) "filaments" and (2400,2050) "lisse", same 512 px framing as nebu_common.py) into Methode validee – 21 nov\sorties\<crop>\, with the owner's view format: crop in 4 quarters, input | result side by side, x4 Lanczos, Post-type stretch (reuse nebu_views.py). Journal + commit.
Report: whether reproduction is exact, what had changed, the tag, output folders.
