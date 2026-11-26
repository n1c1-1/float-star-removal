# v7: star model from whole image

- Date : 2026-11-21 à 15:56 (heure de Paris)
- Numéro du jour : 26
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

One-change experiment (design + implementation) on piste 1 of a classical star-removal method. CPU only (CUDA_VISIBLE_DEVICES=-1; never the GPU). Python: python (anaconda's Python is broken; pandas is broken in this venv). No downloads: the Gaia catalogues are already local (see the code). <SCRIPTS> is a git repo. Write only in a new folder "<SCRIPTS>\Methode v7 – 21 nov\", plus JOURNAL.md and git. Never modify v1..v6 folders. cv2.imwrite fails silently on paths with non-ASCII characters (– é): use cv2.imencode(...)[1].tofile(path) or PIL.

Read first: "Recul – 21 nov soir\RECUL.md" and its profil_reste*.py scripts (the step-back that motivates this change), JOURNAL.md (last 5 entries), "Methode v5 – 21 nov\LISEZMOI.txt", and in "Methode v5 – 21 nov\Piste1\retrait_modele.py": class E (PSF holder, ~l.180-225), build_epsf (l.226), PSF setup (~l.660-710: R = clip(ceil(5·fwhm),10,40), regions, min_psf_stars, candidate selection), the "brillantes d'abord puis PSF refaite" block (~l.1630-1650), core_correction, halo handling (log shows halo amplitude 0), star_model.

Finding (RECUL.md, measured on v5 piste 1 outputs, isolated stars, in sky-noise units): mean residual is negative at r 2–4 px and positive at r 6–12 px (and still +0.3 at r 16–22 for G 12–13.5, beyond the ~15 px where the model stops), roughly proportional to brightness, sign flips for G > 16.5. The PSF is built on ~9 stars of the 768 px crop, forced to zero at its edge r 13–15, halo 0. This wrong shape is what the healing passes later turn into rings r ≈ 10 px.

The ONE change (v7), following DAOPHOT (analytic/empirical PSF + table of residuals, Stetson 1987) and the iterated effective PSF (Anderson & King 2000; photutils EPSFBuilder): build the star model from MANY stars, with an extended radius:
 (a) candidates from the WHOLE image (the code's image loader + the full-image astrometry cache already used by piste 1; isolated, unsaturated, good S/N stars; hundreds), not the crop;
 (b) iterate: build ePSF → fit those stars → stack their normalised residuals (per colour, in brightness bins if needed) → add the mean residual to the model → repeat until the stacked residual profile is flat to ±0.1 sky noise in each G bin (12–13.5, 13.5–14.5, 14.5–15.5, 15.5–16.5) or 4 iterations;
 (c) radius 25–30 px, with the outer wing taken from brighter stars and joined smoothly to the core (extended PSF, Infante-Sainz, Trujillo & Román 2020) — do not force the wing to zero inside that radius;
 (d) if the bins still disagree (denoised image: shape depends on brightness), allow one model per G bin and say so.
Keep the region structure if it exists (spatially varying PSF) only if there are enough stars per region over the whole image. Cache the model to disk (it is per image, built once, reused by every crop of that image). Everything downstream (fit, per-colour positions of v4, stage A, passes with v5's t1 change) unchanged.

Do: copy "Methode v5 – 21 nov" code (not sorties, not diagnostics; Piste1\sorties only the astrometry cache) to the v7 folder, implement, run methode_validee.py on filaments, lisse, decoupe6 (piste 1 must rerun).

Measure v5 vs v7: (a) stacked residual profiles per G bin per colour on piste 1 output (reuse RECUL's profil_reste scripts), table at r = 1,2,3,4,6,8,10,12,16,20 px; (b) number of stars used for the model, iterations, final flatness; (c) on the final tiles: the ring positions of filaments (257,197), (264,193), (321,233), (246,221), (146,231): ring amplitude in local noise; (d) invented share per class (stage A and t1/t2/t3 separately); (e) detector counts rainbow / dark and dark pits as in previous entries; (f) piste 1 time, plus model build time; (g) x3 quarter views input | v5 | v7 at Post stretch for all quarters of the three crops in v7 sorties\comparaison\.


Report in French, ≤ 25 lines: design choices, tables (a)–(f), paths of the comparison views. Plain words.
