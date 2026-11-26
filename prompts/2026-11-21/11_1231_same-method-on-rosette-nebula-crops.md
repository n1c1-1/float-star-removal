# Same method on Rosette nebula crops

- Date : 2026-11-21 à 12:31 (heure de Paris)
- Numéro du jour : 11
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Classical (non-neural) star removal, CPU only (set CUDA_VISIBLE_DEVICES=-1). Report in French, under 15 lines.

WHERE YOU MAY WRITE: only a new folder <SCRIPTS>\Piste 3 remplissage – 20 nov\sorties\nebulosites_rosette\ and new scripts named nebu_*.py in <SCRIPTS>\Piste 3 remplissage – 20 nov\. Do not modify any existing script or output. No downloads (the deep Gaia catalogue is already in <SCRIPTS>\Catalogue Gaia profond – 20 nov\). Never name or use [outil de comparaison]. Python: python (read only; anaconda's is broken).

GOAL: apply the method that just gave a result the owner called "très propre" on a dense-cluster crop of the Rosette, unchanged, to crops IN THE NEBULOSITY of the same Rosette image. No tuning: same settings everywhere.

The method = (A) the hybrid "hybride_c9": piste 1 run c9 (Gaia + PSF subtraction, on the DENOISED Rosette input, not the linear "Entrée non débruitée" one) + piste 3 fill, which produced <SCRIPTS>\Piste 3 remplissage – 20 nov\sorties\hybride_c9\rosette_decoupe6\, then (B) the three final passes passe_finale.py → passe_finale_t2.py → passe_finale_t3.py (same folder), which read c9's output and produced sorties\passe_finale_c9, _t2, _t3.
Difficulty: piste1 scripts (<SCRIPTS>\Retrait étoiles par modèle – 20 nov\retrait_modele.py and its options) and piste3.py/remplir.py/soustraire.py were patched in place during the night (L-series on the linear input). Find how c9 and hybride_c9 were launched (lancer.py, piste1_*.py, patch_hybride.py, logs/json in their output folders, the options recorded in mesures.json) and reproduce that exact configuration through options/flags. STEP 1 (mandatory check): rerun A on the same crop decoupe6 and compare with the saved hybride_c9\rosette_decoupe6\sans_etoiles_tuile.npy (max and median abs diff in noise units). If not essentially identical, find why (a later patch changing defaults) and fix it via options in your own nebu_ wrapper, not by editing their files. If you cannot reproduce within ~30 min, stop and report what differs.
STEP 2: choose 2 crops of 512 px in the same Rosette denoised image (same frame as decoupe6), inside the nebula: (1) bright nebula with sharp structure — filaments, dark globules/elephant trunks, bright rims; (2) fainter, smooth nebulosity with medium stars. Avoid the very bright giant stars (owner's rule: giants left whole) and avoid overlapping decoupe6. Save a small overview PNG with the crop boxes drawn.
STEP 3: run A then B (t1, t2, t3, in order, same settings; generalise their hard-coded paths/coordinates via your wrappers; the defect-specific fixes of t3 that targeted named positions of decoupe6 must be skipped — keep only its generic parts: grain restoration in changed pixels, generic dot/spot healing, edge feathering).
OUTPUTS per crop in its subfolder: sans_etoiles_tuile.npy, carte_remplissage_tuile.npy, mesures.json (invented share by class), and the owner's view format: the crop cut into 4 quarters, each quarter input | result side by side, upscaled x4 with Lanczos (about 2040x1024 each), Post-type stretch (the same stretch function that made vue_resultat_x2.png), plus the same 4 quarters at hard stretch. Name them grand_x4_<quart>_entree_resultat.png / _fort.png.
Do NOT judge the result as clean; describe what you see changed and any visible problem. Budget about 1 h 30. Report the output folder path.
