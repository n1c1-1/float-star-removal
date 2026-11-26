# New library images into pair production

- Date : 2026-11-22 à 18:18 (heure de Paris)
- Numéro du jour : 11
- Type d'agent : agent général
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Goal: add ~10 new images from the library to the production of star-removal training pairs ("jeu 1"), so pairs come from many instruments (few tiles from many images). Nicolas gave his explicit go for the Gaia catalogue downloads. Work autonomously; report in ≤ 20 lines at the end, plus one short line to me after each image is done.

HARD RULES (machine shared with other sessions; it crashed 3 times from my overload):
- Every python process you start sets OMP_NUM_THREADS=MKL_NUM_THREADS=NUMBA_NUM_THREADS=OPENBLAS_NUM_THREADS=4 (and torch.set_num_threads(4)); one computation at a time; never a process pool; never two series in parallel.
- GPU: only the RTX 4060 Ti (never the 3070, it drives his screens). One GPU job at a time on the machine: before any GPU step, check that no `produire_jeu1.py` process of mine is running (my production rerun ends ~19:40), else wait.
- Write only in: <SCRIPTS>\ (the method folder below and a new subfolder "Images jeu 1 – 22 nov" for your scripts) and <DONNEES>\ (data: <DONNEES>\Images jeu 1 – 22 nov\<nom>\). Library sources (<BANQUE>, <ESSAIS>, <DISQUE>/, <DISQUE>/) are READ ONLY. Never copy whole projects; read stores in place.
- Catalogue downloads are a separate preparation step, never inside a processing run. Use python from python. Use cv2.imencode().tofile() for non-ASCII paths; sys.stdout.reconfigure(encoding="utf-8").

Images (ranking and paths in <SCRIPTS>\Bibliotheque images – 22 nov\BILAN.md section 5, full list in inventaire.csv there): IC 5070 ASI6200MC+FSQ-106 (E065), WASP-74 ASI183MM+EdgeHD (E067), NGC 7000 de tiers1 A7S (E010), M97 and M1 LCO (_revue_pn), M82 LCO sq40, NGC 7000 GFX100S+AP92 (E051), M13, NGC 7662, NGC 6888 LCO. Then if time allows, from inventaire.csv: his Cœur (Heart) and his M31 Newton 150/750, one Seestar field. Skip Euclid and anything needing a new stacking. Skip an image whose stack is clearly bad (hot pixels/colour blotches) and say why.

Per image:
[passage retiré]
2. Register it in <SCRIPTS>\Methode v32 production – 22 nov\Piste1\images.py IMAGES (name, path, object for approximate centre, "couleur" or "filtres").
3. Astrometry: produce Piste1\sorties\<nom>\astrometrie.npz the same way the existing ones (rosette, m57, 58_hoo…) were produced — find the code in that Piste1 folder that writes it, and use the offline Gaia extract (env ASTRO_GPU_GAIA_EXTRACTS as set for FLOAT runs); no online solver.
4. Deep Gaia: adapt "Methode v32 production – 22 nov\Catalogue Gaia profond – 20 nov\telecharger_gaia_profond_m8_sho.py" (SRC/OUT to the v32 production folder) and download gaia_dr3_profond_<nom>.npz (G<21) for the image. Be polite to the TAP server (sleep between queries as the script does).
5. Production: when no other produire_jeu1.py runs, from "<SCRIPTS>\Paires jeu 1 – 22 nov": `produire_jeu1.py "../Methode v32 production – 22 nov" <nom> --max 6 --contenu --gpu --graine 1`. First check the first tile: rc=0 and its contact sheet <DONNEES>\Paires étoiles jeu 1 – 22 nov\<nom>_<carré>.jpg (input | target | weight) looks sane (stars gone, no big holes); if a chain fails, read the traceback, fix only obvious plumbing (paths, missing image entries, encoding), journal + commit it; a method defect (bad removal) → stop that image and report it, do not tune the method.

Report: per image: done/skipped and why, number of pairs, mean weight-1 share, anything odd; the commits.
