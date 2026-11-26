# Port piste 1 to GPU (speed)

- Date : 2026-11-21 à 17:02 (heure de Paris)
- Numéro du jour : 27
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Speed work: make piste 1 of the classical star-removal method run on the GPU. Use the RTX 4060 Ti (CUDA device 0) freely; never the RTX 3070 (device 1). Python: python (check torch + CUDA there; anaconda's Python is broken; pandas broken). No downloads of data; if a Python package is missing, say so instead of installing. <SCRIPTS> is a git repo. cv2.imwrite fails silently on non-ASCII paths: use cv2.imencode(...)[1].tofile(path).

Base = the current best method, v5: "<SCRIPTS>\Methode v5 – 21 nov\" (read LISEZMOI.txt, methode_validee.py, etape_piste1.py, Piste1\retrait_modele.py). Never modify v1..v7 folders (v7 is being run by another agent right now, and another agent runs v5 on a new crop "coeur" — do not touch their outputs). Work in a copy: "<SCRIPTS>\Methode v5 GPU – 21 nov\" (code + Piste1\sorties astrometry cache only).

Context: one 512 px tile (768 px crop) takes ~10–15 min on CPU, piste 1 is most of it (decoupe6 587–727 s, filaments ~360–470 s, lisse ~300–400 s); stages A and B take ~1–2 min. The goal is hundreds of images, so piste 1 must become much faster.

Steps:
1. PROFILE first (cProfile or timers) piste 1 on crop lisse with v5 code: where do the seconds go (per function: fit_star / refit loops, E.eval / PSF stamp interpolation, build_epsf, core_correction, measure_*, model rendering, fills). Table top 10.
2. Port the hot parts to torch on CUDA, keeping the SAME algorithm (same fits, same iterations, same order of operations as far as possible): typical wins are batching the per-star fits (all stars of a pass solved in parallel as batched small least-squares / Gauss-Newton), evaluating PSF stamps for all stars at once (grid_sample), rendering all models at once. Keep a CPU path (same code with device=cpu) — the owner requires the CPU path to stay measured too.
3. EQUIVALENCE: on lisse, filaments and decoupe6, compare GPU piste 1 starless.npy with v5's (v5 sorties are in "Methode v5 – 21 nov\Piste1\sorties\rosette\mv_<crop>\" and c9_decoupe6 or as LISEZMOI says): max abs diff, 99.9th pct abs diff in sky-noise units, % pixels differing > 0.1 noise; then run stages A+B on the GPU piste 1 output and make x4 quarter views (vues.py) — differences must be invisible at Post stretch. Float32 vs float64 differences are acceptable if < 0.1 noise; anything larger: find why.
4. TIMES: table per crop: v5 CPU, new code on CPU, new code on GPU (piste 1, stage A, stage B, total).
