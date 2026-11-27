# Classical baseline versus networks: same tiles (512 px core)

9 images/tiles measured: m31_a7s_newton, ic405_a7s_newton, m33_osc_lunette, m27_mono_nb, ngc3372_mono_nb, m16_seestar_s50, ngc6934_stellina, leotriplet_canon6d_newton, m31_canon60d_objectif_filee.

Ratio = remaining flux / original flux at the Gaia positions of the tile core (aperture 2 FWHM, annulus 3-5 FWHM), median per image then median over images.
Background = |output - input| (3 colours averaged, noise units) far from stars; structure = same on the bright structure.

## Aggregate (median over images)

| source | <8 | 8-11 | 11-14 | 14-17 | >17 | background med | background p99 | structure med |
|---|---|---|---|---|---|---|---|---|
| classical | - (n_img=0) | 0.011 (n_img=4) | -0.001 (n_img=9) | 0.005 (n_img=9) | 0.059 (n_img=9) | 0.001 | 0.10 | 0.004 |
| E3 | - (n_img=0) | -0.000 (n_img=4) | 0.000 (n_img=9) | 0.122 (n_img=9) | 0.362 (n_img=9) | 0.021 | 0.05 | 0.009 |
| N2 | - (n_img=0) | 0.001 (n_img=4) | 0.003 (n_img=9) | 0.030 (n_img=9) | 0.071 (n_img=9) | 0.003 | 0.06 | 0.005 |
| N5 | - (n_img=0) | -0.001 (n_img=4) | -0.000 (n_img=9) | 0.003 (n_img=9) | 0.050 (n_img=9) | 0.004 | 0.06 | 0.008 |
| N7 | - (n_img=0) | -0.003 (n_img=4) | -0.001 (n_img=9) | 0.026 (n_img=9) | 0.056 (n_img=9) | 0.003 | 0.02 | 0.006 |
| N7N5 | - (n_img=0) | -0.001 (n_img=4) | -0.000 (n_img=9) | 0.039 (n_img=9) | 0.250 (n_img=9) | 0.004 | 0.02 | 0.005 |

## Per image: median remaining ratio (n stars)

### classical

| image | <8 | 8-11 | 11-14 | 14-17 | >17 |
|---|---|---|---|---|---|
| m31_a7s_newton | n=0 | n=2 | -0.001 (n=14) | 0.000 (n=94) | 0.023 (n=584) |
| ic405_a7s_newton | n=0 | 0.000 (n=3) | -0.002 (n=26) | -0.005 (n=228) | 0.011 (n=1018) |
| m33_osc_lunette | n=0 | n=0 | -0.004 (n=11) | 0.018 (n=132) | 0.152 (n=1108) |
| m27_mono_nb | n=0 | n=1 | -0.030 (n=14) | 0.005 (n=141) | 0.059 (n=1180) |
| ngc3372_mono_nb | n=2 | 0.046 (n=17) | 0.125 (n=97) | 0.517 (n=186) | 0.621 (n=865) |
| m16_seestar_s50 | n=0 | -0.000 (n=20) | 0.008 (n=109) | 0.118 (n=341) | 0.396 (n=840) |
| ngc6934_stellina | n=0 | n=2 | 0.001 (n=38) | 0.003 (n=502) | 0.015 (n=1054) |
| leotriplet_canon6d_newton | n=0 | n=0 | -0.004 (n=6) | 0.044 (n=25) | 0.346 (n=52) |
| m31_canon60d_objectif_filee | n=1 | 0.021 (n=13) | 0.001 (n=136) | -0.000 (n=797) | 0.007 (n=720) |

### E3

| image | <8 | 8-11 | 11-14 | 14-17 | >17 |
|---|---|---|---|---|---|
| m31_a7s_newton | n=0 | n=2 | -0.001 (n=14) | 0.022 (n=94) | 0.074 (n=584) |
| ic405_a7s_newton | n=0 | -0.000 (n=3) | -0.001 (n=26) | 0.006 (n=228) | 0.014 (n=1018) |
| m33_osc_lunette | n=0 | n=0 | -0.002 (n=11) | 0.111 (n=132) | 0.732 (n=1108) |
| m27_mono_nb | n=0 | n=1 | 0.000 (n=14) | 0.001 (n=141) | 0.005 (n=1180) |
| ngc3372_mono_nb | n=2 | 0.014 (n=17) | 0.146 (n=97) | 0.915 (n=186) | 0.878 (n=865) |
| m16_seestar_s50 | n=0 | -0.003 (n=20) | 0.019 (n=109) | 0.240 (n=341) | 0.465 (n=840) |
| ngc6934_stellina | n=0 | n=2 | 0.024 (n=38) | 0.464 (n=502) | 0.362 (n=1054) |
| leotriplet_canon6d_newton | n=0 | n=0 | -0.000 (n=6) | 0.122 (n=25) | 0.968 (n=52) |
| m31_canon60d_objectif_filee | n=1 | 0.000 (n=13) | 0.045 (n=136) | 0.146 (n=797) | 0.119 (n=720) |

### N2

| image | <8 | 8-11 | 11-14 | 14-17 | >17 |
|---|---|---|---|---|---|
| m31_a7s_newton | n=0 | n=2 | 0.003 (n=14) | 0.009 (n=94) | 0.071 (n=584) |
| ic405_a7s_newton | n=0 | 0.000 (n=3) | 0.002 (n=26) | 0.010 (n=228) | 0.020 (n=1018) |
| m33_osc_lunette | n=0 | n=0 | -0.003 (n=11) | 0.086 (n=132) | 0.455 (n=1108) |
| m27_mono_nb | n=0 | n=1 | -0.001 (n=14) | -0.009 (n=141) | 0.052 (n=1180) |
| ngc3372_mono_nb | n=2 | -0.009 (n=17) | 0.053 (n=97) | 0.882 (n=186) | 0.871 (n=865) |
| m16_seestar_s50 | n=0 | 0.002 (n=20) | 0.030 (n=109) | 0.260 (n=341) | 0.440 (n=840) |
| ngc6934_stellina | n=0 | n=2 | 0.009 (n=38) | 0.003 (n=502) | 0.010 (n=1054) |
| leotriplet_canon6d_newton | n=0 | n=0 | -0.002 (n=6) | 0.030 (n=25) | 0.975 (n=52) |
| m31_canon60d_objectif_filee | n=1 | 0.006 (n=13) | 0.004 (n=136) | 0.074 (n=797) | 0.064 (n=720) |

### N5

| image | <8 | 8-11 | 11-14 | 14-17 | >17 |
|---|---|---|---|---|---|
| m31_a7s_newton | n=0 | n=2 | -0.001 (n=14) | -0.003 (n=94) | 0.050 (n=584) |
| ic405_a7s_newton | n=0 | -0.000 (n=3) | 0.000 (n=26) | 0.003 (n=228) | 0.011 (n=1018) |
| m33_osc_lunette | n=0 | n=0 | -0.012 (n=11) | -0.007 (n=132) | 0.343 (n=1108) |
| m27_mono_nb | n=0 | n=1 | -0.000 (n=14) | -0.004 (n=141) | 0.049 (n=1180) |
| ngc3372_mono_nb | n=2 | -0.002 (n=17) | 0.054 (n=97) | 0.860 (n=186) | 0.835 (n=865) |
| m16_seestar_s50 | n=0 | -0.001 (n=20) | 0.010 (n=109) | 0.261 (n=341) | 0.469 (n=840) |
| ngc6934_stellina | n=0 | n=2 | 0.006 (n=38) | 0.009 (n=502) | 0.017 (n=1054) |
| leotriplet_canon6d_newton | n=0 | n=0 | -0.009 (n=6) | -0.052 (n=25) | 0.993 (n=52) |
| m31_canon60d_objectif_filee | n=1 | 0.002 (n=13) | -0.029 (n=136) | 0.033 (n=797) | 0.025 (n=720) |

### N7

| image | <8 | 8-11 | 11-14 | 14-17 | >17 |
|---|---|---|---|---|---|
| m31_a7s_newton | n=0 | n=2 | -0.005 (n=14) | -0.014 (n=94) | 0.030 (n=584) |
| ic405_a7s_newton | n=0 | -0.000 (n=3) | -0.001 (n=26) | -0.001 (n=228) | 0.007 (n=1018) |
| m33_osc_lunette | n=0 | n=0 | -0.009 (n=11) | 0.026 (n=132) | 0.509 (n=1108) |
| m27_mono_nb | n=0 | n=1 | -0.001 (n=14) | -0.008 (n=141) | 0.033 (n=1180) |
| ngc3372_mono_nb | n=2 | -0.007 (n=17) | 0.034 (n=97) | 0.883 (n=186) | 0.863 (n=865) |
| m16_seestar_s50 | n=0 | -0.003 (n=20) | 0.003 (n=109) | 0.242 (n=341) | 0.455 (n=840) |
| ngc6934_stellina | n=0 | n=2 | -0.003 (n=38) | 0.034 (n=502) | 0.056 (n=1054) |
| leotriplet_canon6d_newton | n=0 | n=0 | 0.466 (n=6) | 0.489 (n=25) | 1.003 (n=52) |
| m31_canon60d_objectif_filee | n=1 | -0.003 (n=13) | -0.049 (n=136) | 0.007 (n=797) | -0.001 (n=720) |

### N7N5

| image | <8 | 8-11 | 11-14 | 14-17 | >17 |
|---|---|---|---|---|---|
| m31_a7s_newton | n=0 | n=2 | -0.001 (n=14) | -0.004 (n=94) | 0.043 (n=584) |
| ic405_a7s_newton | n=0 | -0.000 (n=3) | -0.000 (n=26) | 0.003 (n=228) | 0.011 (n=1018) |
| m33_osc_lunette | n=0 | n=0 | -0.007 (n=11) | 0.039 (n=132) | 0.504 (n=1108) |
| m27_mono_nb | n=0 | n=1 | -0.000 (n=14) | -0.005 (n=141) | 0.038 (n=1180) |
| ngc3372_mono_nb | n=2 | -0.003 (n=17) | 0.077 (n=97) | 0.913 (n=186) | 0.856 (n=865) |
| m16_seestar_s50 | n=0 | -0.004 (n=20) | 0.010 (n=109) | 0.236 (n=341) | 0.478 (n=840) |
| ngc6934_stellina | n=0 | n=2 | 0.009 (n=38) | 0.293 (n=502) | 0.250 (n=1054) |
| leotriplet_canon6d_newton | n=0 | n=0 | 0.226 (n=6) | 0.115 (n=25) | 1.001 (n=52) |
| m31_canon60d_objectif_filee | n=1 | 0.000 (n=13) | -0.024 (n=136) | 0.023 (n=797) | 0.018 (n=720) |

## Per image: background and structure (median / p99)

| image | source | background | structure |
|---|---|---|---|
| m31_a7s_newton | classical | 0.001 / 0.10 | pixels=0 |
| m31_a7s_newton | E3 | 0.026 / 0.08 | pixels=0 |
| m31_a7s_newton | N2 | 0.004 / 0.06 | pixels=0 |
| m31_a7s_newton | N5 | 0.004 / 0.06 | pixels=0 |
| m31_a7s_newton | N7 | 0.003 / 0.06 | pixels=0 |
| m31_a7s_newton | N7N5 | 0.004 / 0.07 | pixels=0 |
| ic405_a7s_newton | classical | 0.010 / 0.16 | pixels=0 |
| ic405_a7s_newton | E3 | 0.068 / 0.19 | pixels=0 |
| ic405_a7s_newton | N2 | 0.029 / 0.16 | pixels=0 |
| ic405_a7s_newton | N5 | 0.018 / 0.18 | pixels=0 |
| ic405_a7s_newton | N7 | 0.015 / 0.17 | pixels=0 |
| ic405_a7s_newton | N7N5 | 0.016 / 0.16 | pixels=0 |
| m33_osc_lunette | classical | 0.001 / 0.07 | 0.004 / 0.26 |
| m33_osc_lunette | E3 | 0.004 / 0.02 | 0.009 / 0.12 |
| m33_osc_lunette | N2 | 0.001 / 0.02 | 0.002 / 0.18 |
| m33_osc_lunette | N5 | 0.003 / 0.02 | 0.009 / 0.19 |
| m33_osc_lunette | N7 | 0.003 / 0.01 | 0.006 / 0.19 |
| m33_osc_lunette | N7N5 | 0.003 / 0.01 | 0.005 / 0.16 |
| m27_mono_nb | classical | 0.043 / 1.85 | pixels=0 |
| m27_mono_nb | E3 | 0.514 / 2.38 | pixels=0 |
| m27_mono_nb | N2 | 0.022 / 1.54 | pixels=0 |
| m27_mono_nb | N5 | 0.131 / 1.48 | pixels=0 |
| m27_mono_nb | N7 | 0.190 / 1.55 | pixels=0 |
| m27_mono_nb | N7N5 | 0.155 / 1.54 | pixels=0 |
| ngc3372_mono_nb | classical | 0.001 / 0.12 | 0.002 / 0.11 |
| ngc3372_mono_nb | E3 | 0.006 / 0.03 | 0.007 / 0.04 |
| ngc3372_mono_nb | N2 | 0.002 / 0.02 | 0.005 / 0.06 |
| ngc3372_mono_nb | N5 | 0.007 / 0.02 | 0.008 / 0.02 |
| ngc3372_mono_nb | N7 | 0.005 / 0.01 | 0.007 / 0.01 |
| ngc3372_mono_nb | N7N5 | 0.004 / 0.01 | 0.005 / 0.01 |
| m16_seestar_s50 | classical | 0.007 / 0.05 | 0.008 / 0.07 |
| m16_seestar_s50 | E3 | 0.028 / 0.42 | 0.030 / 0.55 |
| m16_seestar_s50 | N2 | 0.007 / 0.20 | 0.011 / 0.27 |
| m16_seestar_s50 | N5 | 0.004 / 0.12 | 0.006 / 0.13 |
| m16_seestar_s50 | N7 | 0.004 / 0.17 | 0.006 / 0.22 |
| m16_seestar_s50 | N7N5 | 0.004 / 0.10 | 0.005 / 0.12 |
| ngc6934_stellina | classical | 0.000 / 0.03 | pixels=0 |
| ngc6934_stellina | E3 | 0.017 / 0.05 | pixels=0 |
| ngc6934_stellina | N2 | 0.003 / 0.06 | pixels=0 |
| ngc6934_stellina | N5 | 0.001 / 0.06 | pixels=0 |
| ngc6934_stellina | N7 | 0.002 / 0.02 | pixels=0 |
| ngc6934_stellina | N7N5 | 0.002 / 0.02 | pixels=0 |
| leotriplet_canon6d_newton | classical | 0.000 / 0.01 | pixels=0 |
| leotriplet_canon6d_newton | E3 | 0.002 / 0.00 | pixels=0 |
| leotriplet_canon6d_newton | N2 | 0.001 / 0.00 | pixels=0 |
| leotriplet_canon6d_newton | N5 | 0.000 / 0.00 | pixels=0 |
| leotriplet_canon6d_newton | N7 | 0.002 / 0.00 | pixels=0 |
| leotriplet_canon6d_newton | N7N5 | 0.002 / 0.00 | pixels=0 |
| m31_canon60d_objectif_filee | classical | 0.024 / 0.15 | pixels=0 |
| m31_canon60d_objectif_filee | E3 | 0.021 / 0.04 | pixels=0 |
| m31_canon60d_objectif_filee | N2 | 0.001 / 0.01 | pixels=0 |
| m31_canon60d_objectif_filee | N5 | 0.001 / 0.02 | pixels=0 |
| m31_canon60d_objectif_filee | N7 | 0.002 / 0.01 | pixels=0 |
| m31_canon60d_objectif_filee | N7N5 | 0.002 / 0.01 | pixels=0 |