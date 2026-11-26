# Find pre-denoise Rosette stack

- Date : 2026-11-20 à 23:59 (heure de Paris)
- Numéro du jour : 07
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Read-only search, then a small extraction. Goal: get the owner's Rosette stack BEFORE denoising (linear, after gradient removal if possible) for a test of a classical star-removal script, because the current test input (<SCRIPTS>\data\test_with_stars\rosette\input.fits, meta.json says "denoised": true, stage "optics") is denoised and that breaks PSF photometry.
Where to look (READ ONLY, never write there): the FLOAT project <DISQUE>/astro_float/Rosette (project.astroproj + its store: find the stage outputs — stack, gradient, denoise, deconv; read the project file and the store layout, see the FLOAT code in <FLOAT>\astro_gpu for how stores are organised, read only), and the old project <DISQUE>/astro_gpu/projects/Rosette. Also check meta.json of the test input to know which project/stage it was made from, and match the same stack (same frame size 4256x2848, same registration) so coordinates stay identical (crop x 1530–2298, y 790–1558 must show the same stars; verify by correlation with the test input).
Output: write ONLY into <SCRIPTS>\Entrée non débruitée – 21 nov\ (create, with a French LISEZMOI.txt: what, source path/stage, date 2026-11-21, session astro-f1, deletable): the pre-denoise image in the SAME units as the test input (linear, sky subtracted, divided by sky noise per colour, as starRemoval's images.py expects — read images.py and meta.json to reproduce the normalisation), full frame as float32 .npy + a meta.json, plus a check PNG of the crop next to the denoised one. Nothing written anywhere else; never modify the owner's projects; CPU only; Python python, PYTHONDONTWRITEBYTECODE=1. If no pre-denoise stack exists, say so and list what exists.
Report short in French: source found (path, stage), how normalised, verification of alignment, noise level vs the denoised input.
