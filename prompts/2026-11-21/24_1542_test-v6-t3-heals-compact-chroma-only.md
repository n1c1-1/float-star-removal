# Test v6: t3 heals compact chroma only

- Date : 2026-11-21 à 15:42 (heure de Paris)
- Numéro du jour : 24
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

One-change experiment. CPU only (CUDA_VISIBLE_DEVICES=-1). Python: python. <SCRIPTS> is a git repo. Read first: JOURNAL.md (last 3 entries), "Methode v5 – 21 nov\LISEZMOI.txt", "Methode v5 – 21 nov\diagnostic_structure\" (scripts, replay_t3.py) and "Methode v5 – 21 nov\Passes\passe_finale_t3_p.py" in full.

Current best = v5. Never modify v1..v5 folders.

Diagnosis: in t3, chroma_spots (l.86-92) flags 233 "chroma spots" on crop filaments, most of them real faint PINK NEBULA arcs/knots (extended, ≥ 10 px), which the healing (l.177-187, F.fill_mask) then erases, and step C (l.~227) removes a further 0.1–0.15 of R−G there. Real star leftovers are compact (2–5 px). On decoupe6, t3's healing is useful (it removes a dark spot and marks that t2 leaves).

The ONE change (v6): a chroma spot is healed only if it is COMPACT: its chroma excess must be a small-scale feature — e.g. chroma of a 1 px blur minus chroma of a 7×7 (or 9×9) median exceeds the threshold, AND its connected component (at half its peak excess) is ≤ 25 px; extended chroma structure is kept untouched. Step C's R−G reduction then applies only to the pixels actually healed (if it currently applies more widely — read the code, state what it did before). Choose exact scales from the code's own conventions and state them. Everything else in t3 and in all other stages unchanged.

Do: copy "Methode v5 – 21 nov" code + Piste1\sorties (reuse piste 1) to "<SCRIPTS>\Methode v6 – 21 nov\" (not other sorties, not diagnostic folders), apply the change in the copy, run methode_validee.py <crop> --reutiliser-piste1 on filaments, lisse, decoupe6.

Measure v5 vs v6: (a) number of chroma spots flagged / healed per crop; (b) pink arcs: std(R−G) and band energy at filaments (257,197), (264,193), (246,221), (146,231) in input, t2, v5, v6 (reuse the diagnostic_structure scripts); (c) lisse (100,37) dashes and (81,71) disc: present or not (amplitude in noise); (d) detector counts rainbow / dark and dark pits < −5 noise as in previous entries; (e) grain ratio per 128 px square on filaments (judge's method, scripts in "Juge – essais 20 nov\juge_v5\"); (f) invented share; (g) x3 quarter views input | v5 | v6 at Post stretch for all quarters of the three crops in v6 sorties\comparaison\ — NOTE: cv2.imwrite fails silently on paths with non-ASCII characters (– é): use cv2.imencode(".png", img)[1].tofile(path) or PIL.


Report in French, ≤ 20 lines: what step C did before, scales chosen, tables (a)–(f), paths of the comparison views.
