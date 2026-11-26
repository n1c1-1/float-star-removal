# Blind judge on hybrid L3c

- Date : 2026-11-21 à 00:46 (heure de Paris)
- Numéro du jour : 02
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

You are an independent visual judge of a star-removal result on an astro image (Rosette nebula, colour DSLR stack, LINEAR data, not denoised, in noise units ~1 per colour, not colour-balanced). You know nothing about how it was made; judge only what you see and measure.
Files (read only):
- Result: <SCRIPTS>\Piste 3 remplissage – 20 nov\sorties\hybride_L3c\rosette_lin_decoupe6\ — sans_etoiles_fenetre.npy (3×800×800), carte_remplissage_fenetre.npy (uint8: 0 intact, 1 halo subtracted / low frequency redone, 2 core or spike recreated, 3 medium star corrected in place, 4 defect recreated), fenetre.json (crop bounds and judged zone; judged zone = 144–656 inside the crop unless the json says otherwise).
- Input with stars: <SCRIPTS>\Entrée non débruitée – 21 nov\rosette_non_debruite.npy, crop x 1628–2428, y 1357–2157 (rows = y). meta.json there.
- Gaia deep catalogue placed on this image: look in <SCRIPTS>\Retrait étoiles par modèle – 20 nov\sorties\rosette_lin\ for the catalogue copy re-placed with this image's WCS (read files to find it); use x/y pixel columns of that copy.
- Checklist (French), apply every point: <SCRIPTS>\Retrait étoiles par modèle – 20 nov\LISTE_CONTROLE_JUGE.txt.
Measurement rules (learnt from previous judges' mistakes): very dense cluster, almost no star-free sky; do not call the removal of summed star wings "background removed". Pits: fixed threshold on the true sky noise (PSF-scale filtered, measured on the result's flattest areas), count only where the result is below its own local smooth background, and give the same count on the input. Grain: compare to the result's own star-free sky. For VIEWS ONLY, balance colours by scaling each channel to the same star-free sky median; never alter the data.
Method (mandatory): same asinh stretch for input and result (offset = soft = 3 noise) plus a hard stretch; cut the judged 512 px zone into 16 squares of 128 px ×3 and LOOK (Read tool) at every square of the result next to the input; whole zone ×2.
Write files ONLY inside <SCRIPTS>\Juge – essais 20 nov\juge_hybride_L3c\ (create). Nothing elsewhere. PYTHONDONTWRITEBYTECODE=1. Python: python. PIL for PNGs. CPU only.
Report in French, plain words, short: each of the 12 checklist points: present or not, count, positions (square and x,y in the 512 zone), severity (léger / visible / grave), zoom file names. Then the 3 most serious defects and a one-line verdict. Do not soften, do not inflate.
