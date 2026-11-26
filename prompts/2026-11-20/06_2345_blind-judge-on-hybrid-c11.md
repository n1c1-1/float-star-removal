# Blind judge on hybrid c11

- Date : 2026-11-20 à 23:45 (heure de Paris)
- Numéro du jour : 06
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

You are an independent visual judge of a star-removal result on an astro image (Rosette nebula, colour DSLR, linear data in noise units). You know nothing about how it was made; judge only what you see and measure.
Files (read only):
- Result: <SCRIPTS>\Piste 3 remplissage – 20 nov\sorties\hybride_c11\rosette_decoupe6\sans_etoiles_tuile.npy (judged 512 px zone) and sans_etoiles_fenetre.npy (3×768×768 float32, whole window; the judged zone is rows/cols 128–640), with carte_remplissage_tuile.npy / carte_remplissage_fenetre.npy (uint8: 0 intact, 1 halo subtracted / low frequency redone, 2 core or spike recreated, 3 medium star corrected in place, 4 defect recreated) and fenetre.json (check it for frames and units).
- Input with stars for the same window: find it in <SCRIPTS>\Piste 3 remplissage – 20 nov\sorties\base_piste1\rosette_decoupe6\ (copied base window; it should contain the input/original array — read its files to identify) or from <SCRIPTS>\data\test_with_stars\rosette\input.fits cropped at x 1530–2298, y 790–1558 of the full image.
- Checklist (French), apply every point: <SCRIPTS>\Retrait étoiles par modèle – 20 nov\LISTE_CONTROLE_JUGE.txt. Design rules you must know: the bright core plateau of planetary nebulae is kept on purpose (not relevant here); stars of the sigma-Ori class are left whole (not relevant here).
Method (mandatory): render with the same asinh stretch for input and result (offset = soft = 3 noise, white at a high percentile of the input) and also a hard stretch for faint pits/rings; cut the judged 512 px zone into 16 squares of 128 px, enlarge ×3, and LOOK (Read tool) at every square of the result next to the same square of the input; look at the whole zone at ×2. Measure on the data: dark pits (PSF-scale minima below −3 sigma, sigma from star-free sky) in result vs input; point sources left at 3 and 5 sigma; grain ratio (high-pass std, result/input, outside stars); invented share from the map; large scale (blur 16 and 48 px) result vs input with stars masked.
Write files ONLY inside <SCRIPTS>\Juge – essais 20 nov\juge_hybride_c11\ (create). Nothing elsewhere. PYTHONDONTWRITEBYTECODE=1. Python: python. Use PIL to save PNGs (cv2 fails on accented paths). CPU only.
Report in French, plain words, short: for each of the 12 checklist points: present or not, count, positions (square and x,y in the 512 zone), severity (léger / visible / grave), with the crop file names. Then a one-line verdict (usable for a training base or not) and the 3 most serious defects. Do not soften.
