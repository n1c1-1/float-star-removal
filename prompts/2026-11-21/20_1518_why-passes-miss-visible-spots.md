# Why passes miss visible spots

- Date : 2026-11-21 à 15:18 (heure de Paris)
- Numéro du jour : 20
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Current best = v4. A few small spots remain VISIBLE at the Post stretch in the final result (v4 sorties\<crop>\sans_etoiles_tuile.npy, tile 512 px). Positions, read from the x4 quarter views (result half starts at view x≈1033; tile x ≈ (view_x−1033)/4·1.028 + quarter offset x, tile y ≈ view_y/4·1.028 + quarter offset y; haut_gauche offset (0,0), haut_droite (256,0)) — refine each to the exact spot by local search within ±6 px:
- lisse haut_gauche: view (1410,390) dark-red spot → tile ≈ (97,100); view (1460,240) faint spot → ≈ (110,62)
- filaments haut_droite: view (1135,880) → ≈ (282,226); (1280,965) → ≈ (319,248); (1305,985) → ≈ (326,253)
Also scan both crops yourself for any other spot with peak chroma or dark deviation > 6 local noise visible at Post stretch, up to 10 more.

For each spot: (1) its amplitude in local noise (luminance and R−G, B−G), size in px; (2) nearest catalogue star (Gaia G<18 and piste 1 sources — see how t1 builds its list) distance and G, and t1's zone radius 3×(14−G) for that star: inside or outside?; (3) t1's residual against its robust background and its thresholds (hi 7 / lo 4): would it trigger? was it in masque_ignorer?; (4) t2's detectors (isolated dark, DoG 3–8 px colour-aware, compact spots >8 noise ≤80 px): value vs threshold for each; (5) the stage where the spot appears (piste 1 starless, A, t1, t2, t3) — stage outputs are in v4 sorties\<crop>\A\ and B\ and Piste1\sorties\rosette\mv_<crop>\starless.npy.
Conclude per spot with the single reason it survives (e.g. "below t1 hi threshold: 5.2 < 7", "star G 16.3 → zone radius 0 since 14−G<0", "colour-only spot, t1 works on luminance").

Report in French, ≤ 18 lines: a table spot → amplitude, nearest star (G, distance), zone radius, why each pass missed it; then one line: the most common reason. x8 zooms input | piste1 | A | t1 | t2 | t3 for each spot saved in diagnostic_restes\.
