# ======================================================================
# liste.py
# Role     : List of the validation images of the star network (copy of Validation – 22 nov/liste.py).
# Stage    : 13. Measurement and controls: the E3 rows of the README tables (26 Nov) (see scripts/README.md)
# Inputs   : none (paths of the author's machine)
# Outputs  : LISTE (dictionary)
# Example  : from liste import LISTE
# ======================================================================
"""List of the validation images of the star network (22 Nov 2026). Inputs read in place (same preparation as training:
gradient removed, noise units with background 0, denoising network of the application, white balance)."""
from pathlib import Path
SD = Path("F:/astro_essais/stars_dataset")
DATA = Path("F:/float_ia/Validation étoiles – 22 nov")
# nom: (entree en place, cible pour le centre approche, type, instrument, echelle "/px, pourquoi)
LISTE = {
 "m31_a7s_newton": (SD/"user_M31", "M31", "A7S + Newton, galaxie a coeur brillant", "Sony A7S defiltre, Newton 130/900", 1.28, "aigrettes d'araignee + coeur de galaxie tres brillant"),
 "ic405_a7s_newton": (SD/"user_FlamingStar", "IC 405", "A7S + Newton, nebuleuse", "Sony A7S defiltre, Newton 130/900", 1.28, "aigrettes sur etoiles brillantes (AE Aur) dans une nebuleuse"),
 "m33_osc_lunette": (SD/"set_36_osc_refractor_asi294mc_star71_m33", "M33", "OSC lunette", "ZWO ASI294MC Pro, lunette Star71 350 mm", 1.81, "etoiles sans aigrettes, galaxie diffuse a regions HII"),
 "m27_mono_nb": (SD/"_revue_nb/56_comb", "M27", "mono bande etroite assemble + nebuleuse planetaire compacte", "ASI1600MM mono, Newton 1325 mm, Ha/OIII/SII", 0.59, "planetaire compacte tres contrastee, 3 filtres assembles"),
 "ngc3372_mono_nb": (SD/"_revue_nb/62_comb", "NGC 3372", "mono bande etroite assemble (Ha+OIII), champ riche", "LCO 0,4 m (sq38), Ha+OIII", 0.74, "Carene : voie lactee dense + nebuleuse + etoile tres brillante"),
 "m16_seestar_s50": (SD/"set_32_smart_seestar_s50_m16", "M16", "telescope connecte (Seestar S50)", "Seestar S50, 250 mm", 1.58, "petit instrument, etoiles rondes, nebuleuse brillante, pile courte"),
 "ngc6934_stellina": (SD/"set_53_smart_stellina_ngc6934", "NGC 6934", "champ stellaire dense (amas globulaire), Stellina", "Stellina, 400 mm", 1.24, "amas globulaire : etoiles tres serrees, coeur sature"),
 "leotriplet_canon6d_newton": (SD/"set_04_dslr_newton_canon6d_leotriplet", "M66", "reflex sur Newton, galaxies", "Canon 6D, Newton 200/1000", 1.35, "champ de galaxies faibles, aigrettes, FWHM 8 px"),
 "m31_canon60d_objectif_filee": (SD/"set_16_dslr_lens_canon60d_m31_untracked", "M31", "reflex sur objectif, etoiles filees", "Canon 60D, objectif 200 mm, monture non guidee", 4.56, "etoiles allongees (filees), galaxie large"),
 "m42_canon40da_lunette": (SD/"set_05_dslr_refractor_canon40da_m42", "M42", "reflex sur lunette", "Canon 40Da, lunette 72ED", 2.73, "coeur d'Orion tres brillant, Trapeze, halos d'etoiles brillantes"),
 "ic5146_canon60da_lunette": (SD/"set_08_dslr_refractor_canon60da_cocoon", "IC 5146", "reflex sur lunette, nebuleuse sombre/brillante", "Canon 60Da, lunette ZenithStar", 2.0, "Cocon : champ d'etoiles + nebuleuse diffuse"),
 "m42_dobson_grande_fwhm": (SD/"set_18_osc_newton_asi533mc_dobson12_m42", "M42", "grande FWHM (10 px)", "ZWO ASI533MC Pro, Dobson 12 pouces", 0.51, "etoiles tres larges (FWHM 10,5 px) : tout change d'echelle"),
 "ngc7822_a7s_lunette": (SD/"user_NGC7822", "NGC 7822", "nebuleuse etendue faible", "Sony A7S defiltre, lunette TS61EDPH", 4.25, "grande nebuleuse tres faible, peu de structures brillantes"),
 "ic1848_stellina": (SD/"set_15_smart_stellina_ic1848", "IC 1848", "nebuleuse etendue faible, telescope connecte", "Stellina, 400 mm", 1.24, "Ame : nebuleuse faible dans un champ riche, FWHM 4 px, remplace NGC 7822 (Gaia du champ de 7,5 deg trop lent)"),
}
