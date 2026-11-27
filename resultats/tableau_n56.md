# Table N2, N5, N6 (25 Nov)

Reading: "remaining" = share of the stars of a G magnitude band that keep more than 10 % of their flux after removal (flux ratio at the Gaia positions, 2 FWHM aperture; mean over the images with at least 5 stars in the band; 0 = every star removed; lower is better). "background median" and "background p99": difference |output - input| far from the stars, in units of the input noise (lower is better). "structure median": the same difference on bright structure (nebula, galaxy). "cores": output flux / input flux within 15 px around galaxy cores (ideal 1: the core must stay). "M32 60D": the same ratio for the compact core of M32 alone in the Canon 60D photo. Validation: 13 never-seen images (docs/VALIDATION.md). Two-scale V2 measurements unless stated otherwise. Tables of our measurements, 25-26 November 2026.

| run | images | remaining stars G <8 | remaining stars G 8-11 | remaining stars G 11-14 | remaining stars G 14-17 | remaining stars G >17 | background median | background p99 | structure median | synthetic L1 |
|---|---|---|---|---|---|---|---|---|---|---|
| N2 (V2) | 13 | 0.000 | 0.013 | 0.091 | 0.313 | 0.581 | 0.0015 | 0.12 | 0.005 | nan |
| N5 | 13 | 0.000 | 0.017 | 0.072 | 0.265 | 0.551 | 0.0011 | 0.07 | 0.007 | nan |
| N6 | 13 | 0.000 | 0.015 | 0.086 | 0.351 | 0.626 | 0.0027 | 0.03 | 0.005 | nan |
