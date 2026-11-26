# Blind judge on final night version

- Date : 2026-11-21 à 03:53 (heure de Paris)
- Numéro du jour : 07
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

You are an independent visual judge of a star-removal result on an astro image (Rosette nebula, colour DSLR stack, LINEAR data, not denoised, noise units ~1 per colour, not colour-balanced). You know nothing about how it was made; judge only what you see and measure.
Files (read only):
- Result: <SCRIPTS>\Piste 3 remplissage – 20 nov\sorties\hybride_L12_final\rosette_lin_decoupe6\ — sans_etoiles_fenetre.npy (3×800×800), carte_remplissage_fenetre.npy (classes, meanings in fenetre.json), fenetre.json (crop bounds, judged zone 144–656 inside the crop).
- Input with stars: <SCRIPTS>\Entrée non débruitée – 21 nov\rosette_non_debruite.npy, crop x 1628–2428, y 1357–2157 (rows = y).
- Gaia deep catalogue placed on this image: in <SCRIPTS>\Retrait étoiles par modèle – 20 nov\sorties\rosette_lin\ (find the re-placed catalogue copy; x/y pixel columns, Gmag).
- Checklist (French), apply every point: <SCRIPTS>\Retrait étoiles par modèle – 20 nov\LISTE_CONTROLE_JUGE.txt.
Measurement rules (from previous judges' mistakes): very dense cluster; summed star wings are legitimately removed — not "background removed". Pits: fixed threshold on true sky noise (PSF-scale filtered, from the result's flattest areas), only where the result is below its own local smooth background; same count on the input and at random positions. Grain: vs the result's own star-free sky. Faint stars: fraction of catalogue stars above 5 sigma per G bin (12–14, 14–15, 15–16, 16–17, 17–17.5, 17.5–18, 18–19, 19–20, 20–21) on the result, input, random positions. Dipoles on G 12–15: within 3 px of the Gaia position, min < −5 and max > +5 noise on the filtered residual. Look for large faint rings/circles and plates at the hard stretch. Views only: balance colours by star-free sky median per channel; never alter the data.
Method (mandatory): same asinh stretch for input and result (offset = soft = 3 noise) plus a hard stretch AND a usual "Post-like" stretch (what a user would see); 16 squares of 128 px ×3 of the judged zone, LOOK (Read tool) at every square of the result next to the input; whole zone ×2; whole 800 px window ×1. Also produce, for the owner, two clean side-by-side PNGs (input | result) of the judged zone at the Post-like stretch and at the hard stretch, ×2, colour-balanced, named vue_proprietaire_post_x2.png and vue_proprietaire_dur_x2.png.
Write files ONLY inside <SCRIPTS>\Juge – essais 20 nov\juge_hybride_L12_final\ (create). Nothing elsewhere. PYTHONDONTWRITEBYTECODE=1. Python: python. PIL for PNGs. CPU only. Keep scripts in that folder.
Report in French, plain words, short: each of the 12 checklist points: present or not, count, positions, severity (léger / visible / grave) AND whether it is visible at the Post-like stretch or only at the hard stretch; zoom file names. Then the 3 most serious defects and a one-line verdict. Do not soften, do not inflate.
