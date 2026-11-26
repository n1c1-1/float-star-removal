# ======================================================================
# images_mesurees.py
# Rôle     : Paramètres d'image MESURÉS sur nos 4 images, pour conditionner le générateur v2 dans les planches et le test à l'aveugle (les tirages libres de la galerie ne s'en servent pas).
# Étape    : 4. Étoiles synthétiques réalistes (22 nov) (voir scripts/LISEZMOI.md)
# Entrées  : fonds sans étoiles (cibles de paires), banques de formes d'étoiles
# Sorties  : paires (entrée, cible), planches, test à l'aveugle
# Exemple  : mesure_formes.py (ellipticité et asymétrie des étoiles G 12-13,5 de chaque image) ; rayons fins et perles
# ======================================================================
"""Paramètres d'image MESURÉS sur nos 4 images, pour conditionner le générateur v2 dans les planches et le test à l'aveugle
(les tirages libres de la galerie ne s'en servent pas). Sources : audit_mesures.json (largeur des moyennes, auréole des
brillantes, décalages et largeurs par couleur, plateaux), mesure_aigrettes.json (aigrettes, médianes par image ; largeur à mi-hauteur et niveau à 8-14 px mesurés à part, 22 nov),
fiches des méthodes v21-v23 (angles) ; formes des moyennes (ellip_sigma, irreg_*, grumeaux) calées sur
mesure_formes.py (ellipticité et asymétrie des étoiles G 12-13,5 de chaque image) ; rayons fins et perles
colorées des aigrettes lus sur les vignettes du premier test à l'aveugle (Tête de Cheval, M57 : oui ; M8 : non) ;
perles irisées de Rosette, M57 et Tête de Cheval relevées au tour 2 (raisons du juge 2 : « aigrettes fines à perles »,
« aigrettes irisées ») ; s30 lu directement à 22-40 px (tour 2 : l'ajustement sur 0-330 px le sous-estimait 2-3 fois) ;
grain des halos (grain_halo) faible : mesure_halos.py donnait 0,5-0,6, mais la planche Synth_10 montre des halos
cotonneux qu'aucune vraie étoile n'a (la mesure prenait la nébuleuse et les faibles hors catalogue) : jugé à l'oeil."""
IMP = {
    "rosette": dict(grain_halo=0.1, rayons_actif=True, rayons_contraste=0.05, ellip_sigma=0.12, irreg_p=0.9, irreg_q=0.5, irreg_d=1.0, grumeaux=0.55, plateau_couleur_p=0.2, fwhm_image=3.74, fwhm_moment=3.94, angles=[3.0, 93.5, 183.0, 273.5], plateau=(0.51, 0.86), a50=8e-7, gamma=2.4,
                    s30=2.0e-6, pente=1.9, L=None, larg_sigma=1.9, s11=3.5e-5, mode="rgb", pointilles=0.3, pas=11.0,
                    decal_commun=[-0.015, 0.011, 0.002, -0.014], decal_sigma=0.032, larg_couleur=[1.019, 1.0, 1.003],
                    allongement=0.15, decentrement=0.1, champ_bprp="rosette", photon_k=0.0),
    "m8_sho": dict(grain_halo=0.1, rayons_actif=False, ellip_sigma=0.035, irreg_p=0.3, irreg_q=0.2, irreg_d=0.4, grumeaux=0.12, plateau_couleur_p=0.0, fwhm_image=4.37, fwhm_moment=5.48, angles=[55.5, 113.5, 172.0, 235.5, 293.5, 352.0], plateau=(0.96, 1.0), a50=1.5e-6,
                   gamma=2.2, s30=2.8e-6, pente=2.4, L=None, larg_sigma=3.7, s11=1.5e-5, mode="nb", pointilles=0.35, pas=12.0,
                   decal_commun=[0.008, 0.007, -0.002, -0.001], decal_sigma=0.03, larg_couleur=[0.984, 1.0, 1.005],
                   allongement=0.1, decentrement=0.1, champ_bprp="m8_sho", photon_k=0.0),
    "m57": dict(grain_halo=0.1, rayons_actif=True, rayons_contraste=0.08, ellip_sigma=0.08, irreg_p=0.6, irreg_q=0.35, irreg_d=0.7, grumeaux=0.25, plateau_couleur_p=0.1, fwhm_image=2.26, fwhm_moment=2.58, angles=[45.0, 133.0, 223.0, 313.0], plateau=(0.37, 0.63), a50=4e-7, gamma=2.4,
                s30=2.4e-6, pente=2.33, L=None, larg_sigma=1.15, s11=2.8e-5, mode="rgb", pointilles=0.4, pas=11.0,
                decal_commun=[0.007, -0.002, -0.001, 0.004], decal_sigma=0.028, larg_couleur=[1.017, 1.0, 1.04],
                allongement=0.1, decentrement=0.15, champ_bprp="m57", photon_k=0.0),
    "horsehead": dict(grain_halo=0.05, rayons_actif=True, rayons_contraste=0.12, ellip_sigma=0.17, irreg_p=1.0, irreg_q=0.6, irreg_d=1.2, grumeaux=0.9, plateau_couleur_p=0.1, fwhm_image=4.22, fwhm_moment=5.22, angles=[0.0, 90.0, 180.0, 270.0], plateau=(0.5, 0.96), a50=1.5e-6, gamma=2.0,
                      s30=4.2e-6, pente=2.8, L=80.0, larg_sigma=2.6, s11=7.5e-5, mode="rgb", pointilles=0.3, pas=12.0,
                      decal_commun=[0.171, 0.138, -0.196, -0.161], decal_sigma=0.04, larg_couleur=[1.004, 1.0, 1.006],
                      allongement=0.25, axe=0.0, decentrement=0.05, champ_bprp="horsehead", photon_k=0.0),
}
