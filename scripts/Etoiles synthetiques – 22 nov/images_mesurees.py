# ======================================================================
# images_mesurees.py
# Role     : Image parameters MEASURED on our 4 images, to condition the v2 generator in the sheets and the blind test (the gallery's free draws do not use them).
# Step     : 4. Realistic synthetic stars (22 Nov) (see scripts/LISEZMOI.md)
# Inputs   : star-free backgrounds (pair targets), star shape banks
# Outputs  : pairs (input, target), sheets, blind test
# Example  : mesure_formes.py (ellipticity and asymmetry of the G 12-13.5 stars of each image) ; fine rays and beads
# ======================================================================
"""Image parameters MEASURED on our 4 images, to condition the v2 generator in the sheets and the blind test
(the gallery's free draws do not use them). Sources: audit_mesures.json (width of the medium stars, aureole of the
bright ones, shifts and widths per colour, plateaus), mesure_aigrettes.json (spikes, medians per image; half-height width and level at 8-14 px measured separately, 22 Nov),
sheets of methods v21-v23 (angles); shapes of the medium stars (ellip_sigma, irreg_*, grumeaux) calibrated on
mesure_formes.py (ellipticity and asymmetry of the G 12-13.5 stars of each image); fine rays and coloured beads
of the spikes read on the thumbnails of the first blind test (Horsehead, M57: yes; M8: no);
iridescent beads of Rosette, M57 and Horsehead noted in round 2 (judge 2's reasons: "fine spikes with beads",
"iridescent spikes"); s30 read directly at 22-40 px (round 2: the fit over 0-330 px underestimated it 2-3 times);
halo grain (grain_halo) low: mesure_halos.py gave 0.5-0.6, but the Synth_10 sheet shows cottony halos
that no real star has (the measurement picked up the nebula and the faint off-catalogue stars): judged by eye."""
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
