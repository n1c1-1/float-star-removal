# Fable step-back on remaining defects

- Date : 2026-11-21 à 15:50 (heure de Paris)
- Numéro du jour : 25
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Cold step-back, advice only: no code change, no run longer than a few minutes, write only in <SCRIPTS>\Recul – 21 nov soir\ (create). CPU only. Python: python.

Goal of the project (owner's words): training pairs (image with stars → starless target) generated AUTOMATICALLY by the hundreds, to train FROM SCRATCH a star-removal network for astrophotos (mix of images with and without deconvolution). Each pair carries a map of invented pixels (down-weighted in training), a per-image confidence, and failed zones can be discarded. A classical, non-neural method makes the targets; the owner's founding gesture: [logiciel tiers]'s healing tool works for medium and big stars, subtraction works for small ones; hybrid = subtract, then a last pass heals what is left.

Read, in <SCRIPTS>\: JOURNAL.md (whole file, it is the trace of every run), "Methode v5 – 21 nov\LISEZMOI.txt" (the current best method: piste 1 = Gaia-catalogue PSF subtraction, per-colour positions; stage A = hybrid halo/spike subtraction + PatchMatch/Poisson fills; stage B = three final passes t1/t2/t3 healing leftovers near catalogued stars), the blind judge report on v5 (data in "Juge – essais 20 nov\juge_v5\", checklist "Retrait étoiles par modèle – 20 nov\LISTE_CONTROLE_JUGE.txt"), and "Methode v5 – 21 nov\diagnostic_structure\" (where each remaining defect is born). Look yourself at a few views: "Methode v5 – 21 nov\sorties\filaments\grand_x4_haut_droite_entree_resultat.png" and "...\haut_gauche..." and "Methode v5 – 21 nov\diagnostic_structure\t2_contre_t3\*.png".

State: v5 is clean at the Post stretch to the eye on three 512 px Rosette crops (dense cluster, filaments, smooth nebula). Remaining defects: (1) thin rings r ≈ 10 px around removed G~15 stars — the PSF wing is under-subtracted at r 6–8 px in piste 1 (its log shows halo amplitude 0), then each healing pass pushes the ring outward; (2) faint pink nebula arcs/knots near stars flattened — A, t1, t2 and t3 all contribute, ~30 % of the tile is invented; (3) small numerical dashes/steps from patch fills; (4) grain lower than input (0.75–0.8, some squares 0.5). Speed: ~10–15 min CPU per 512 px tile (piste 1 dominates).

Questions — answer each with a recommendation, the published method behind it (search the literature/web; cite sources: e.g. PSF wing modelling / extended PSF in crowded fields, inpainting that preserves faint structure, how DAOPHOT/PSFEx/photutils or survey pipelines handle wings, how training sets with masks are weighted), and the expected effect:
Q1. Rings: what is the principled fix for an under-subtracted PSF wing on faint/medium stars in a denoised image — and should the healing passes stop growing their discs?
Q2. Flattened faint nebula: is it worth fighting, given that those pixels are in the invented-pixel map that is down-weighted in training? What would a network learn from a target where faint nebula is flattened only inside masked pixels? Is there a better target for invented pixels than a fill (e.g. leaving them out of the loss entirely)?
Q3. Is the current architecture (subtract + 3 cascaded healing passes tuned on one crop) a sound basis for hundreds of images, or should the passes be collapsed into one principled residual-healing step? What would you change first, and what would you stop doing?
[passage retiré]

Report in French, ≤ 35 lines, plain words: per question one recommendation + source + expected effect; then a 3-step ordered plan. Save the same text in your folder as RECUL.md.
