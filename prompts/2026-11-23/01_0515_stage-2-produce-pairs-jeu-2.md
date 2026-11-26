# Stage 2: produce pairs jeu 2

- Date : 2026-11-23 à 05:15 (heure de Paris)
- Numéro du jour : 01
- Type d'agent : agent général
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

You produce training pairs (jeu 2) for our star-removal network, on the CPU, over the next 2–3 days.

The jeu-1 network learned well on its 16 images but damages unseen images: we need many more varied images. You are an executor; report briefly and do not redesign the method.

READ FIRST, in <SCRIPTS>\:
- PLAN_RESEAU_23nov.md (stage 2 and the execution rules, which you must follow);
- RECETTE_ET_LECONS.md;
- the last entries of JOURNAL.md;
- the jeu-1 pipeline you will reuse:
  - Images jeu 1 – 22 nov\ : enchainer.py, preparer_entree.py, telecharger_gaia_profond.py, astrometrie_cone.py;
  - Paires jeu 1 – 22 nov\ : produire_jeu1.py, construire_paire.py, rejuger.py;
  - the chain in Methode v32 production – 22 nov;
- the library list: Bibliotheque images – 22 nov\inventaire.csv (columns tell ready linear stacks vs stretched images; sensor/instrument; target).

WHERE YOU WRITE (absolute):
- code in <SCRIPTS>\Paires jeu 2 – 23 nov\. Copy validated scripts there and change only the copy; never patch the jeu-1 or v32 files.
- data in <DONNEES>\Paires étoiles jeu 2 – 23 nov\ (and its working subfolders).
- French folder names; a LISEZMOI.txt in each new folder.
- Source images are read-only. Write nowhere else.

STEPS
1. Selection, within 30 min.
   - Write selection_jeu2.csv: 150–200 READY LINEAR STACKS from inventaire.csv.
   - Maximise variety: spread over sensor/instrument couples and targets, at most 2 stacks per target per couple, all kinds of fields (nebula, galaxy, cluster, wide, narrow-band, mono, OSC, DSLR).
   - EXCLUDE every target of the validation batch: M31, M33, M27, M42, M16, IC 405, IC 1848, IC 5146, NGC 3372, NGC 6934, Leo Triplet (M65/M66/NGC 3628), NGC 7822. Check Validation – 22 nov\liste.py for the exact list.
   - Jeu-1 targets are allowed only with a different setup, and come last in priority.
   - If fewer than 150 usable stacks remain, note how many and stop at what exists. Stretched images are a later step that I will brief separately.
   - Report one line (count by kind), then continue without waiting.
2. Preparation, before any processing:
   - astrometry, with astrometrie_cone.py as the fallback;
   - deep Gaia downloads: ESA synchronous CIRCLE queries, maxrec 3M; no polygons, no async jobs.
   - Drop any image that fails astrometry twice, and note it.
   - Never download inside a processing run. Set ASTRO_GPU_GAIA_EXTRACTS as in jeu 1.
3. Pilot: 3 varied images end to end, 6 tiles each, same judge as jeu 1 (current construire_paire.py, which already holds the ring, level, island and star-residue tests). Look at the pair jpgs yourself (input, target, weight panel). If something is clearly broken, fix it in your copy and say so in one line.
4. Production of all selected images, 6 tiles each (fewer if the image is small).
   - Order: diverse first, so a partial set is already varied.
   - Keep a running pairs list (paires.jsonl, like jeu 1).

MACHINE RULES (they crashed the PC before; obey exactly)
- CPU ONLY. Do not use the GPU at all: the 4060 Ti runs the network training. Run the chain without --piste1-gpu. Never touch the 3070.
- At most 8 threads in total across all your processes. Each process sets OMP_NUM_THREADS, MKL_NUM_THREADS, NUMBA_NUM_THREADS, OPENBLAS_NUM_THREADS and torch.set_num_threads.
  - Example: up to 8 single-thread processes in parallel, since the per-image sheet is single-core.
- Python: python.
- Stop processes only by the PIDs you launched, never by name.
- Clean temporary files as you go.

LOGGING
- JOURNAL.md entry at start with « Déjà essayé / Ce qui diffère » (difference from jeu 1: many more images, CPU path, 6 tiles each).

REPORTS (short, one line each)
- After selection; after preparation (images ready / dropped); after the pilot.
- Then every 25 images done, with pairs so far and the ETA as a real clock time.
- Immediately if blocked.
- Final report at the end: pairs, images, dropped with reasons, weight stats (share at 1 / 0.5 / 0).

Do not start any training.
