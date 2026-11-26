# Fable cold step-back on dark pits

- Date : 2026-11-20 à 23:42 (heure de Paris)
- Numéro du jour : 05
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

You are asked for a cold, independent step back on a blocked problem. Read-only: do not modify or create files except inside <SCRIPTS>\Recul Fable – 21 nov\ (create it, with a short French LISEZMOI.txt). CPU only. Do not run long computations (a few quick numpy checks are fine). Python: python, PYTHONDONTWRITEBYTECODE=1.

Context: a classical (non-neural) star-removal script for amateur astro images, goal = generate hundreds of training pairs automatically (Claude may tune parameters per image with rules). Pipeline (code: <SCRIPTS>\Retrait étoiles par modèle – 20 nov\retrait_modele.py; read it): astrometric solve, Gaia DR3 positions to G<21 (deep catalogue in <SCRIPTS>\Catalogue Gaia profond – 20 nov\), empirical PSF per region (photutils ePSF, oversampled x4, built on few isolated stars), per-star closed-loop PSF fitting with local annulus background, small elliptical freedom, per-colour flux bounds from Gaia colour, joint fit of close pairs, then subtraction; fills are done separately by another step. Data: a colour DSLR stack of the Rosette, linear, debayered (maybe drizzled), FWHM ~3 px (undersampled), dense cluster on nebula, in noise units.

THE BLOCKING DEFECT: after subtraction, the starless image has hundreds of small DARK PITS. Test crop (768 px, judged zone central 512 px): outputs in ...\sorties\rosette\c10_decoupe6, c11_decoupe6, c12_decoupe6 (starless.npy, soustraction.npy, etoiles.csv, journal.txt; views in ...\sorties\essai_tuiles\c1*_decoupe6\). Facts measured:
- pits (PSF-filtered minima below −3 sigma): 746–835 in the starless vs 172 in the input (whole zone); outside bright-star zones 522–583 vs 84.
- fitted/expected flux (median by G bin): 12–14: 1.02–1.13; 14–16: 0.92–1.00; 16–18: 0.60–0.66; 18–21: 0.29–0.43.
- tried without effect on pits: removing the 0.35x lower flux bound for faint stars (c11: 835→746), calibration on isolated stars only (c12: 792), masking neighbours in the convergence test (c10), per-channel sub-pixel offsets (measured < 0.08 px, made things worse), PSF built on 31 stars instead of 5 (no clear gain on an earlier variant).
- location of pits (c11): 191 within 2 px of a catalogue star, 223 at 2–4 px, 332 farther than 4 px from any star; nearest star median G 18.3 with normal flux ratio 0.8.
- interpretation proposed by the developer (unverified): rings over-dug around stars where the model's wings exceed the data, i.e. a PSF shape problem.

Your job: think from first principles and from the state of the art (astronomy crowded-field photometry and any other field), look at the code and a few outputs, and give: (1) the 3–5 most likely causes of the pits we have NOT considered or have dismissed too fast, each with a quick decisive test (one run or one numpy check) to confirm or reject it; (2) whether pits 4+ px from any star could be produced by the method itself (e.g. sky/background model, annulus background bias in crowded fields, noise properties of drizzled/denoised data, the pit metric itself, fits of catalogue sources that are not visible, Gaia duplicates/spurious sources in dense fields, proper motion/epoch, PSF normalisation, oversampled ePSF interpolation ringing); (3) what you would do differently overall if you started this method again. Be concrete and brief (max ~700 words, French plain words). Include links for any state-of-the-art claim.
