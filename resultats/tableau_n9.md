# Table N9 and reference points (26 Nov)

Reading: "remaining" = share of the stars of a G magnitude band that keep more than 10 % of their flux after removal (flux ratio at the Gaia positions, 2 FWHM aperture; mean over the images with at least 5 stars in the band; 0 = every star removed; lower is better). "background median" and "background p99": difference |output - input| far from the stars, in units of the input noise (lower is better). "structure median": the same difference on bright structure (nebula, galaxy). "cores": output flux / input flux within 15 px around galaxy cores (ideal 1: the core must stay). "M32 60D": the same ratio for the compact core of M32 alone in the Canon 60D photo. Validation: 13 never-seen images (docs/VALIDATION.md). Two-scale V2 measurements unless stated otherwise. Tables of our measurements, 25-26 November 2026.

| run | images | remaining stars G <8 | remaining stars G 8-11 | remaining stars G 11-14 | remaining stars G 14-17 | remaining stars G >17 | background median | background p99 | structure median | synthetic L1 | cores (output/input, ideal 1) | M32 60D |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N5 | 13 | 0.000 | 0.017 | 0.072 | 0.265 | 0.551 | 0.0011 | 0.07 | 0.007 | nan | 0.192 | 0.034 |
| N8 step 5000 | 13 | 0.052 | 0.040 | 0.107 | 0.298 | 0.586 | 0.0038 | 0.03 | 0.006 | nan | 0.837 | 0.024 |
| N8 step 10000 | 13 | 0.052 | 0.047 | 0.109 | 0.331 | 0.604 | 0.0011 | 0.03 | 0.004 | nan | 0.785 | 0.058 |
| N7 step 5000 | 13 | 0.090 | 0.073 | 0.126 | 0.334 | 0.599 | 0.0047 | 0.04 | 0.012 | nan | 0.813 |  |
| N7 | 13 | 0.021 | 0.020 | 0.094 | 0.283 | 0.571 | 0.0023 | 0.03 | 0.004 | nan | 0.813 | 0.035 |
| N7/N5 a=0.75 | 13 | 0.021 | 0.023 | 0.090 | 0.295 | 0.584 | 0.0018 | 0.03 | 0.003 | nan | 0.798 |  |
| N8L best step (13000) | 13 | 0.052 | 0.044 | 0.101 | 0.294 | 0.571 | 0.0027 | 0.04 | 0.004 | nan | 0.792 | 0.064 |
| N8L step 24000 | 13 | 0.052 | 0.050 | 0.097 | 0.310 | 0.594 | 0.0008 | 0.04 | 0.003 | nan | 0.807 | 0.066 |
| N7 step 10000 (measured here) | 13 | 0.065 | 0.053 | 0.105 | 0.253 | 0.540 | 0.0042 | 0.05 | 0.008 | nan | 0.812 | 0.050 |
| N9 step 10000 | 13 | 0.136 | 0.109 | 0.121 | 0.348 | 0.611 | 0.0011 | 0.03 | 0.002 | nan | 0.869 | 0.263 |
