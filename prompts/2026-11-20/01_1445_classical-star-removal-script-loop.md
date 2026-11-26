# Classical star removal script loop

- Date : 2026-11-20 à 14:45 (heure de Paris)
- Numéro du jour : 01
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Goal: design and iterate a classical (non-neural) star-removal script, from the state of the art, that removes the stars of an astro image better than the reference star-removal tool, on 6 varied images. Long-term aim (not now): run it on thousands of images to build a training base for FLOAT's réseau étoiles. You are both the developer and the judge: you look at the results tile by tile (you can Read PNG/JPEG images), say what remains where, fix or steer the script, and decide when to give up a zone or an image. The owner trusts your eye as much as his own.

METHOD (decided with the owner):
1. Star positions from the catalogue, not from the image: solve the image's astrometry (reuse FLOAT's own plate solving / catalogue code in <FLOAT>\astro_gpu — import read-only, find it), then Gaia stars at their positions with expected brightness. Offline catalogue only: ASTRO_GPU_GAIA_EXTRACTS=<CACHE_FLOAT>/cache/gaia_extracts; NEVER download anything; check in the logs that stars come from the offline extract. A nebula knot is not in Gaia, so it cannot be taken for a star; a planetary nebula's central star is in Gaia and must be removed while the nebula stays.
2. Subtract, don't erase: empirical PSF measured on the image, varying over the field and per colour (Anderson & King 2000; photutils EPSFBuilder / PSFPhotometry / IterativePSFPhotometry; Stetson 1987 DAOPHOT), fitted with the local background; blended stars fitted together. The background and noise under each star stay intact.
3. PER-STAR CLOSED LOOP (core of the design): after subtraction, measure the residual in annuli around each star (core to halo) against the local background just outside; correct the model by the residual's shape (central bump/dip = flux, crescent = position, ring at a few px = width, far residual = halo r^-2/r^-3, Racine 1996) and iterate until the residual is flat at noise level in every annulus (stop criterion, measurable; record each star's final residual as its quality score). Guard-rails: the shape is shared with neighbours of the same field region (only flux, position and a small width margin are free per star); flux checked against the Gaia-predicted brightness (a star wanting to be much brighter than predicted is eating nebula: block and flag); neighbours fitted together. Watch for false equilibrium (residual flat because the local background is wrong, e.g. nebula edges).
4. What the model cannot do (saturated cores, spikes, reflections, unresolved blends): when the loop does not converge, fill the zone instead: start simple (smooth local background + noise with the image's own grain and correlation); move to exemplar/texture fill (PatchMatch, Barnes et al. 2009; Criminisi et al. 2004) only if your eye sees the simple fill is not enough. Do NOT use [logiciel tiers].
5. Your loop: tiles of ~512 px; metrics pre-sort suspect tiles (Gaia stars left, image changed far from stars, holes); you look, give a located verdict (clean / what remains at (x,y) / nebula cut / hole / spike left / discard), then either steer (parameter for that zone or image type) or enrich the script (new rule for a recurring failure, rerun the images it affects), or give up (zone excluded, image dropped after 2–3 failed tries). Keep a short verdict log per image.
Read for inspiration (read only): <SCRIPTS> (README.md, real_targets.py, stars.py, psf.py, star_measure.py: the 9 Nov automatic corrected targets and fill rules), and the memory note <HOME>

THE 6 IMAGES (linear, noise units, sky subtracted, as the chain gives them; check each meta.json/units):
1. Rosette, colour DSLR emission nebula (the holes defect): <SCRIPTS>\data\test_with_stars\rosette\input.fits (+ meta.json)
[passage retiré]
3. M57 at 180 mm, small PN, fine stars: <SCRIPTS>\data\train_with_stars\set_24_dslr_astrograph_d810a_eps180_m57
4. Horsehead with Alnitak (giant star on nebula): ...\train_with_stars\user_Horsehead
5. Double Cluster NGC 869, dense field, many bright stars: ...\train_with_stars\set_38_smart_dwarfmini_ngc869
[passage retiré]
The owner's M27 is a held-out judge: do not use it.

WRITE RULE — ABSOLUTE (the owner insists, no exception):
- Script, logs, outputs: ONLY inside <SCRIPTS>\Retrait étoiles par modèle – 20 nov\ (create it with a French LISEZMOI.txt: what, which work, 2026-11-20, created by session astro-f1, deletable or not). Do not modify existing files of <SCRIPTS>.
- The owner's rule for the starRemoval folder: NO trace of the reference star-removal tool there (never name it, never read its outputs in the script). The final comparison with that tool goes ONLY in <FLOAT>\_travail\Comparaison retrait étoiles par modèle – 20 nov\ (create it with LISEZMOI). Run the reference tool for the comparison with the recipe of <HOME>\AppData\Local\Temp\claude\741409e7ccb77380\b6a53c82-929b-47e8-8ad0-898ae2aa0cca\scratchpad\step2_[outil de comparaison].py (onnx weights, CPU, venv <HOME>\AppData\Local\Temp\claude\741409e7ccb77380\b6a53c82-929b-47e8-8ad0-898ae2aa0cca\scratchpad\venv_sn\Scripts\python.exe; it needs the review asinh stretch offset=soft=3 and the exact unstretch), outputs into that _travail folder only.
- Nothing anywhere else: not F:, not other C: folders, no leftovers. Read anything you need. PYTHONDONTWRITEBYTECODE=1. cv2.imwrite fails silently on accented paths: use a safe writer and check files exist. At the end list every file you created and confirm they are all in the two folders above.
Other rules: CPU only (CUDA_VISIBLE_DEVICES=-1). Never modify FLOAT production code (<FLOAT>\astro_gpu) or any weights. Never write into the owner's projects. Stop only your own processes; never PID 73024. Python: python (photutils, sep, astropy, cv2, numba, skimage available). No git commits (starRemoval is not a repository).

[passage retiré]

[passage retiré]
