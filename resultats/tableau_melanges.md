# Table of weight blends N5 / N6 / N7 (26 Nov)

Reading: "remaining" = share of the stars of a G magnitude band that keep more than 10 % of their flux after removal (flux ratio at the Gaia positions, 2 FWHM aperture; mean over the images with at least 5 stars in the band; 0 = every star removed; lower is better). "background median" and "background p99": difference |output - input| far from the stars, in units of the input noise (lower is better). "structure median": the same difference on bright structure (nebula, galaxy). "cores": output flux / input flux within 15 px around galaxy cores (ideal 1: the core must stay). "M32 60D": the same ratio for the compact core of M32 alone in the Canon 60D photo. Validation: 13 never-seen images (docs/VALIDATION.md). Two-scale V2 measurements unless stated otherwise. Tables of our measurements, 25-26 November 2026.

| run | images | remaining stars G <8 | remaining stars G 8-11 | remaining stars G 11-14 | remaining stars G 14-17 | remaining stars G >17 | background median | background p99 | structure median | synthetic L1 | cores (output/input, ideal 1) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| N5 | 13 | 0.000 | 0.017 | 0.072 | 0.265 | 0.551 | 0.0011 | 0.07 | 0.007 | nan | 0.192 |
| N6 | 13 | 0.000 | 0.015 | 0.086 | 0.351 | 0.626 | 0.0027 | 0.03 | 0.005 | nan | 0.104 |
| N7 | 13 | 0.021 | 0.020 | 0.094 | 0.283 | 0.571 | 0.0023 | 0.03 | 0.004 | nan | 0.813 |
| N7/N5 a=0.75 | 13 | 0.021 | 0.023 | 0.090 | 0.295 | 0.584 | 0.0018 | 0.03 | 0.003 | nan | 0.798 |
| N7/N5 a=0.5 | 13 | 0.000 | 0.018 | 0.089 | 0.296 | 0.587 | 0.0016 | 0.03 | 0.003 | nan | 0.653 |
| N7/N6 a=0.5 | 13 | 0.012 | 0.015 | 0.087 | 0.312 | 0.600 | 0.0019 | 0.03 | 0.004 | nan | 0.651 |
| N5/N6 a=0.5 | 13 | 0.000 | 0.015 | 0.086 | 0.327 | 0.603 | 0.0017 | 0.04 | 0.006 | nan | 0.133 |
| N5/N6 a=0.75 | 13 | 0.000 | 0.015 | 0.083 | 0.300 | 0.577 | 0.0011 | 0.05 | 0.005 | nan | 0.158 |
