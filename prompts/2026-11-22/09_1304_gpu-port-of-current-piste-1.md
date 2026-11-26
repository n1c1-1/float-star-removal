# GPU port of current piste 1

- Date : 2026-11-22 à 13:04 (heure de Paris)
- Numéro du jour : 09
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Tu portes sur la carte graphique (RTX 4060 Ti) la « piste 1 » (soustraction des étoiles par catalogue) de la méthode COURANTE du retrait d'étoiles classique, pour produire vite des centaines de paires d'entraînement. Le même travail a déjà été fait et validé le 21 nov sur une version plus ancienne (v5) : <SCRIPTS>\Methode v5 GPU – 21 nov (LISEZMOI.txt, Piste1\retrait_rapide.py, Piste1\ajust_lot.py, comparer_p1.py, et l'entrée « v5 GPU » de <SCRIPTS>\JOURNAL.md, avec ses pièges : garde « pas de trou » rendue au processeur à cause de l'ordre de sommation en float32 ; deux calculs GPU à la fois se ralentissent de 10 à 20 fois ; goulot = milliers de petites opérations torch). Depuis, la piste 1 a changé : modèle d'étoile bâti sur l'image entière (v7, Piste1\modele_psf_image.py), halos étendus des brillantes (v12/v13, psf_etendue.py), étalonnage figé sur l'image entière (v17, etalonnage_image.py), saturation par couleur (v14), 6 aigrettes (v15). Le code courant de la piste 1 est dans <SCRIPTS>\Methode v26 – 22 nov\Piste1\ (retrait_modele.py et voisins) ; il ne change plus d'ici la production (seules l'étape A et les passes changent).

TRAVAIL :
1. Copie « Methode v26 – 22 nov » vers « Methode v30 GPU – 22 nov » (sans sorties\ racine, *.log, *.err, *.jpg, *.png ; garde Piste1\sorties, qui contient les pistes 1 PROCESSEUR déjà calculées des carrés : rosette c9_decoupe6 et mv_coeur_est, m8_sho mv_a_coeur… mv_d_brillante, m57 mv_a_m57… : ce sont tes références, ne les écrase pas : écris tes sorties GPU sous un autre nom de dossier).
2. Accélère retrait_modele.py dans la copie (nouveau fichier, l'original reste) en reprenant les techniques de v5 GPU, et mêmes options cuda | cpu (lots torch) | ref.
3. Équivalence : sur au moins 3 carrés déjà calculés (rosette decoupe6, m8_sho a_coeur, m57 c_champ_dense), compare starless.npy GPU contre la référence processeur : écart max et 99,9e centile en bruits du ciel par couleur (cible : < 0,01 bruit ; v5 GPU avait 0,002). Si un écart dépasse, trouve la cause (ordre de sommation, garde, tri) avant d'aller plus loin.
4. Temps : piste 1 par carré GPU contre le temps processeur des journaux (M57 : 17-24 min par carré au processeur) ; vise le plus gros gain sans changer le résultat.
5. Un script de lancement `etape_piste1_gpu.py <image> <carré>` compatible avec methode_validee.py (pour que la chaîne entière puisse prendre la piste 1 GPU), et LISEZMOI.txt.

RAPPORT (≤ 15 lignes) : écarts par carré, temps avant/après, commande de lancement, limites.
