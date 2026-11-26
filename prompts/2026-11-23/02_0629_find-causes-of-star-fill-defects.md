# Find causes of star-fill defects

- Date : 2026-11-23 à 06:29 (heure de Paris)
- Numéro du jour : 02
- Type d'agent : agent général
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Find WHY our classical star-removal chain produces gross defects in its starless images, and propose fixes at the source. This is diagnosis: do not regenerate the pair set.

CONTEXT
- The starless images become the targets of our star-removal network's training pairs.
- Nicolas wants the targets GOOD everywhere. Throwing the zones away (weight 0) is not the answer: then the network never learns big stars.
- Read first, in <SCRIPTS>\:
  - RECETTE_ET_LECONS.md;
  - PLAN_RESEAU_23nov.md (rules section);
  - the chain in `Methode v32 production – 22 nov`;
  - the pair builder in `Paires jeu 1 – 22 nov` (produire_jeu1.py, construire_paire.py; the classes array tells which treatment touched each pixel);
  - the image inputs in `Images jeu 1 – 22 nov`.
- Pairs: <DONNEES>\Paires étoiles jeu 1 – 22 nov\<name>.npz (input, target, weight, classes).
- Contact sheets: the `controle_visuel\` subfolder. Look at planche_*.jpg and lumiere_inventee_top12.jpg; each row is input | target | target with weight overlay.

DEFECTS TO EXPLAIN (most frequent first)
1. Saturated RED blobs, hundreds of px, in the target where the input has only small stars. Examples: ngc7000_tiers1_ngc7000_tiers1_g1152_640 (left edge) and _g128_128 (left edge). The target is brighter than the input there, which star removal can never legitimately do. Note: this image has an odd white balance (about 0.32 / 1 / 0.65), possibly related.
2. Saturated BLUE or white DISCS replacing stars, plus concentric BLACK RINGS around stars. Examples: ngc6543_ngc6543_g1152_128, _g640_128, _g1664_1152, m97_m97_g1152_640, m97_m97_g128_1152, ic5070_ic5070_g4224_1664 (blue disc).
3. A MAGENTA blob: ngc7000_gfx_ngc7000_gfx_g7296_5248 and ngc869_ngc869_g1152_128.
4. Dark pits and dark donuts where medium stars were: m1_m1_g2176_1152, ic5070_ic5070_g3712_1664, ngc7662_ngc7662_g1664_1152.

FOR EACH DEFECT KIND
- Find which stage and which treatment class produced it. Rerun the chain on that one tile, with intermediate outputs, in a copy.
- Find the cause in the code: clipping or saturation, a colour channel scaling, a fit that diverged, a wrong PSF amplitude, a fill using the wrong source, and so on.
- Propose a fix at the source. It must be a correction of a bug or of a wrong assumption, not a new tuned hack: check RECETTE §6 and the « already tried » notes first.
- Test the fix on the example tiles, in a COPY of the scripts; never patch the v32 or jeu-1 files. Show before and after crops in one jpg per defect kind, under <DONNEES>\Diagnostic défauts – 23 nov\.

RULES
- CPU only, at most 4 threads. Each process sets OMP_NUM_THREADS, MKL_NUM_THREADS, NUMBA_NUM_THREADS, OPENBLAS_NUM_THREADS and torch threads.
- The GPU is in use by a training run: do not touch it. Never the 3070.
- Write only in <SCRIPTS>\Diagnostic défauts – 23 nov\ (code, with LISEZMOI.txt) and <DONNEES>\Diagnostic défauts – 23 nov\ (images).
- Stop only PIDs you launched.
- No downloads. Set ASTRO_GPU_GAIA_EXTRACTS as the jeu-1 scripts do.
- Write a JOURNAL.md entry with « Déjà essayé / Ce qui diffère ». Commit in French, with no mention of AI.

REPORT
- Short: per defect kind, 3 lines (cause, fix, before/after jpg path).
- Then whether regenerating the jeu-1 pairs with the fixes is worth it.
- Report immediately if blocked.
