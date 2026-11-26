# Blind judge on hybrid c13

- Date : 2026-11-21 à 00:10 (heure de Paris)
- Numéro du jour : 01
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

You are an independent visual judge of a star-removal result on an astro image (Rosette nebula, colour DSLR, linear data in noise units). You know nothing about how it was made; judge only what you see and measure.
Files (read only):
- Result: <SCRIPTS>\Piste 3 remplissage – 20 nov\sorties\hybride_c13\rosette_decoupe6\ — sans_etoiles_fenetre.npy (3×768×768, judged zone rows/cols 128–640) and the 512 px tile version, carte_remplissage_fenetre.npy (uint8: 0 intact, 1 halo subtracted / low frequency redone, 2 core or spike recreated, 3 medium star corrected in place, 4 defect recreated), fenetre.json (frames, units).
- Input with stars, same window: <SCRIPTS>\Retrait étoiles par modèle – 20 nov\images.py load('rosette') (use it, it gives noise units), crop x 1530–2298, y 790–1558 (rows y, cols x).
- Gaia deep catalogue (positions x_pix, y_pix in full-image pixels, Gmag): <SCRIPTS>\Catalogue Gaia profond – 20 nov\gaia_dr3_profond_rosette.npz.
- Checklist (French), apply every point: <SCRIPTS>\Retrait étoiles par modèle – 20 nov\LISTE_CONTROLE_JUGE.txt.
IMPORTANT measurement rules (learnt from a previous judge's mistakes): this is a very dense cluster — there is almost no star-free sky; do NOT claim "background removed" from comparisons with the input's starry level: the star wings' summed light is legitimately removed. Pits: use a FIXED threshold on the true sky noise (measure it on the result's flattest areas, PSF-scale filter), count only pits where the result is below the local smooth background (not noise minima), and report the same count on the input as reference with the same threshold. Grain: compare to the result's own star-free sky grain, not to the starry input.
Method (mandatory): same asinh stretch for input and result (offset = soft = 3 noise) plus a hard stretch; cut the judged 512 px zone into 16 squares of 128 px, ×3, and LOOK (Read tool) at every square of the result next to the input; whole zone ×2.
Write files ONLY inside <SCRIPTS>\Juge – essais 20 nov\juge_hybride_c13\ (create). Nothing elsewhere. PYTHONDONTWRITEBYTECODE=1. Python: python. Save PNGs with PIL. CPU only.
Report in French, plain words, short: each of the 12 checklist points: present or not, count, positions (square and x,y in the 512 zone), severity (léger / visible / grave), crop file names. Then the 3 most serious defects and a one-line verdict. Do not soften, do not inflate.
