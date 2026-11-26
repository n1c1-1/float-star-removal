# Blind judge on hybrid L10c_z

- Date : 2026-11-21 à 02:45 (heure de Paris)
- Numéro du jour : 04
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

You are an independent visual judge of a star-removal result on an astro image (Rosette nebula, colour DSLR stack, LINEAR data, not denoised, noise units ~1 per colour, not colour-balanced). You know nothing about how it was made; judge only what you see and measure.
Files (read only):
- Result: <SCRIPTS>\Piste 3 remplissage – 20 nov\sorties\hybride_L10c_z\rosette_lin_decoupe6\ — sans_etoiles_fenetre.npy (3×800×800), carte_remplissage_fenetre.npy (classes, meanings in fenetre.json), fenetre.json (crop bounds, judged zone 144–656 inside the crop).
- Input with stars: <SCRIPTS>\Entrée non débruitée – 21 nov\rosette_non_debruite.npy, crop x 1628–2428, y 1357–2157 (rows = y).
- Gaia deep catalogue placed on this image: in <SCRIPTS>\Retrait étoiles par modèle – 20 nov\sorties\rosette_lin\ (find the re-placed catalogue copy; x/y pixel columns, Gmag).
- Checklist (French), apply every point: <SCRIPTS>\Retrait étoiles par modèle – 20 nov\LISTE_CONTROLE_JUGE.txt.
Measurement rules (from previous judges' mistakes): very dense cluster; summed star wings are legitimately removed — not "background removed". Pits: fixed threshold on true sky noise (PSF-scale filtered, from the result's flattest areas), only where the result is below its own local smooth background; same count on the input. Grain: vs the result's own star-free sky. Faint stars: fraction of catalogue stars above 5 sigma per G bin (12–14, 14–15, 15–16, 16–17, 17–17.5, 17.5–18, 18–19, 19–20, 20–21) on the result, input, random positions. Dipoles on G 12–15 stars: within 3 px of the Gaia position, min < −5 and max > +5 noise on the filtered residual; give the count. Views only: balance colours by star-free sky median per channel; never alter the data.
Method (mandatory): same asinh stretch for input and result (offset = soft = 3 noise) plus a hard stretch; 16 squares of 128 px ×3 of the judged zone, LOOK (Read tool) at every square of the result next to the input; whole zone ×2.
Write files ONLY inside <SCRIPTS>\Juge – essais 20 nov\juge_hybride_L10c_z\ (create). Nothing elsewhere. PYTHONDONTWRITEBYTECODE=1. Python: python. PIL for PNGs. CPU only. Keep your judging scripts in that folder.
Report in French, plain words, short: each of the 12 checklist points: present or not, count, positions (square and x,y in the 512 zone), severity (léger / visible / grave), zoom file names. Then the 3 most serious defects and a one-line verdict. Do not soften, do not inflate.
