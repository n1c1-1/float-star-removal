# Réseau étoiles stage 1: train and measure

- Date : 2026-11-22 à 23:54 (heure de Paris)
- Numéro du jour : 13
- Type d'agent : agent général
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

You run STAGE 1 of the réseau étoiles plan. Read first, entirely: <SCRIPTS>\PLAN_RESEAU_23nov.md (plan, rules), then the « Règles d'exécution » section again. Work autonomously through the night; report in ≤ 15 lines at the end; one short line to me when each numbered step below finishes (via your final message only if you must stop).


0. WAIT until both: (a) pair production is over: no process whose command line contains produire_jeu1.py or enchainer.py for 10 consecutive minutes (check every 5 min with a PowerShell Get-CimInstance Win32_Process query; do not sleep in a tight loop); (b) <SCRIPTS>\Validation – 22 nov\PRET.txt exists (written by the validation agent: image list, exact valider_reseau.py commands, output-folder format).
1. Re-judge all pairs: `python "Paires jeu 1 – 22 nov\rejuger.py"` (from <SCRIPTS>). Count pairs per image.
[passage retiré]
3. Run A (E3 fine-tuned), GPU, 2 h: from "<SCRIPTS>\Entrainement jeu 1 – 22 nov": `train_jeu1.py --tag A_E3 --hours 2 --device cuda:0 --depart <ESSAIS>\starRemoval\train\E3\snap\step_20000.pt --tenues <one pair per image for 6 different images, list it>`. Find a python with CUDA torch (try <FLOAT>\.venv first, else the ComfyUI venv <HOME>\sandbox\ComfyUI\.venv). Smoke 2 minutes first; check the log lines.
4. Run B (from zero), same command with --tag B_zero and no --depart, same --tenues.
5. Measure A and B on the validation batch with valider_reseau.py: last.pt and best.pt, plus every 2000-step snapshot of the last hour (they swing: E3 did). Add per-run the training/held-out curve (history.json).
6. STOP and report: a table (A best snapshot, B best snapshot, FLOAT current, reference tool) of (a) stars left by G band, (b) background damage (median, p99, on bright structure), (c) exact error under synthetic stars; which snapshot you'd pick and why in 2 lines; whether the gap looks like missing data (good on held-out pairs, bad on validation) or recipe (bad on both); contact-sheet paths for 4 validation images (input | A | B | FLOAT | reference, same stretch, columns in RANDOM order with the key in a separate file key.json) for a blind judgment. Do NOT start the long training: I decide it from your report.
