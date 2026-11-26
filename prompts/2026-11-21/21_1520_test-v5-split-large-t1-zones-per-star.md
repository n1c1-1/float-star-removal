# Test v5: split large t1 zones per star

- Date : 2026-11-21 à 15:20 (heure de Paris)
- Numéro du jour : 21
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

One-change experiment. CPU only (CUDA_VISIBLE_DEVICES=-1). Python: python. <SCRIPTS> is a git repo. Read first: JOURNAL.md (last 3 entries), "Methode v4 – 21 nov\LISEZMOI.txt", "Methode v4 – 21 nov\diagnostic_restes\" (diag.py + *_taches.json), and the t1 pass "Methode v4 – 21 nov\Passes\passe_finale_p.py" (+ how etape_B.py calls it: t1 with PF_HI=7 PF_LO=4 PF_ZK=3).

Current best = v4 ("Methode v4 – 21 nov"). Never modify v1..v4 folders.

Diagnosis: t1 detects the visible leftovers correctly (residuals 23–28 noise > hi 7, each within 0.3–4 px of a catalogued star G 13.7–14.5), but its guard "merged zone >= 350 px on structured sky (ratio > 2.5) → masque_ignorer" discards the WHOLE merged zone (1126 px on lisse, 2064 px on filaments); t2 and t3 then also skip masque_ignorer. In nebula many detections merge into such large zones.

The ONE change (v5): in t1, a merged zone that hits that guard is no longer ignored as a whole. Inside it, keep only small discs around catalogued stars (the same star list t1 uses) whose own residual peak within the disc exceeds the hi threshold: disc radius = the star's t1 zone radius but capped at 2×FWHM (state the FWHM value used); those discs are filled as t1 fills normal zones; the rest of the large zone stays in masque_ignorer exactly as before. Nothing else changes (thresholds, t2, t3 untouched).

Do: copy "Methode v4 – 21 nov" code + its Piste1\sorties (piste 1 unchanged, reuse it with --reutiliser-piste1) to "<SCRIPTS>\Methode v5 – 21 nov\", apply the change in the copy only, run methode_validee.py <crop> --reutiliser-piste1 on filaments, lisse, decoupe6.

Measure v4 vs v5: (a) the 5 listed spots (lisse (103,99), (116,61); filaments (284,229), (322,250), (327,248)): amplitude L / R−G / B−G in local noise in v4 and v5 (reuse diag.py); (b) detector counts (rainbow / dark) as in previous entries; (c) invented share; (d) lane and globule check (mesures script of v3/v4) — must stay; (e) number of discs added per crop and their total px; (f) x6 zooms input | v4 | v5 at the 5 spots (Post + hard stretch) in v5 sorties\comparaison\.


Report in French, ≤ 18 lines: tables (a)–(e), paths of zooms and of v5 x4 quarter views.
