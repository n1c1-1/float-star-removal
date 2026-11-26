# Night pair production v33

- Date : 2026-11-23 à 16:03 (heure de Paris)
- Numéro du jour : 08
- Type d'agent : agent général
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Tu produis cette nuit les paires d'entraînement (entrée avec étoiles -> cible sans étoiles, + carte de poids) d'un réseau de retrait d'étoiles. Rapports très courts.

Matériau :
- Chaîne : <SCRIPTS>\Methode v33 production – 23 nov (LISEZMOI ; piste 1 sur la carte avec --piste1-gpu).
- Production et juge des zones : <SCRIPTS>\Paires jeu 1 – 22 nov (produire_jeu1.py, construire_paire.py, rejuger.py) ; préparation du jeu 2 (astrométrie, Gaia profond, entrées) : <SCRIPTS>\Paires jeu 2 – 23 nov (journaux\prete_<nom>). Images du jeu 1 : leurs entrées et catalogues sont déjà prêts (voir produire_jeu1.py et « Images jeu 1 – 22 nov »).
- Tuiles à produire : <DONNEES>\Choix des tuiles v2 – 23 nov\tuiles_finales.csv (125 tuiles de 512 px, 39 images ; x, y = coin haut gauche dans l'entrée).
- Rejeu de contrôle déjà fait avec v33 : <DONNEES>\Paires v33 essai – 23 nov\ (14 paires).

Étape 1 — juge v33 (copie « Juge v33 – 23 nov » de construire_paire.py/rejuger.py, UN seul ajout de deux règles locales, sans toucher au reste) :
 (a) cible plus lumineuse que l'entrée : tache ≥ 25 px où cible − entrée > 5 bruits (lissé 1,5 px) -> poids 0 sur la tache dilatée de 3 px ;
 (b) étoile restée dans la cible : à une position du catalogue, pic compact dans la cible au-dessus du fond local de plus de 10 bruits (cas vus : M97 g640_128 et g128_1152, disques cerclés gardés à poids 1) -> poids 0 sur un disque de rayon max(3 x FWHM, rayon du reste mesuré).
 Vérifier sur les 14 paires du rejeu : les cas M97 et les 5 taches (IC 5070, NGC 7000 tiers1 g3200_2176) sont pris ; donner la part de poids 0 avant/après par paire. Si une paire monte de plus de 5 points de poids 0, c'est trop large : resserrer (seuil ou rayon) et revérifier, sans aller plus loin que 2 essais ; sinon continuer. C'est un compromis : on retire la zone fautive, jamais une catégorie entière.

Étape 2 — production, organisée en parallèle :
 - entrées débruitées sur la carte (débruitage de la chaîne, comme au jeu 1) pour les images qui ne l'ont pas encore ; jamais de débruitage sur le processeur ;
 - la chaîne v33 tourne UNE fois par tuile, sur l'entrée débruitée -> paire « debruitee » (entrée débruitée, cible) ;
 - paire « brute » de la même tuile : entrée non débruitée, cible_brute = entrée_brute − (entrée_débruitée − cible), même carte de poids (le calque d'étoiles trouvé sur la version débruitée est retiré de l'entrée brute) ;
 - un seul processus sur la carte à la fois (débruitage puis piste 1, en file) ; les étapes processeur (A, B, passes, juge) de plusieurs tuiles en parallèle : 3 travailleurs à 3 fils au plus + le travailleur carte à 2 fils = 11 fils au total ; RAM < 30 Go au total ;
 - ordre : alterner les images (une tuile de chaque image avant la deuxième) pour avoir de la variété tôt ; reprise possible si arrêt (une tuile faite = ignorée).
 Sorties : <DONNEES>\Paires étoiles v33 – 23 nov\<image>_g<x>_<y>_debruitee.npz et _brute.npz (input, target, weight, classes comme au jeu 1) + LISEZMOI.txt ; une planche de contrôle par image controle\<image>.jpg (pour chaque tuile : entrée | cible | poids, arcsinh 60, 3 x 400 px) pour la revue de demain matin.

Règles absolues :
- Chaque processus via powershell -File <SCRIPTS>\lancer_borne.ps1 -Fils <n> -Log <fichier> -Args "<args python>" -Dossier <dossier> (priorité basse, cœurs 4-15, 4060 Ti seule). Garde la liste de tes PID ; arrêt seulement par tes PID et leurs enfants, jamais par nom.
- Si la charge totale du processeur dépasse 85 % plus de 2 minutes, ou si la mémoire libre passe sous 10 Go : suspends un travailleur processeur.
- Écrire seulement dans <SCRIPTS>\ et <DONNEES>\ ; effacer les sorties de travail sur C: après chaque tuile. Aucun téléchargement (Gaia hors ligne via ASTRO_GPU_GAIA_EXTRACTS comme dans v33). Si une image n'a pas son catalogue ou son astrométrie : la passer, la noter.
- Ne lance aucun entraînement.

Rapports (le moins de texte possible) : (1) dès le juge v33 vérifié : 3 lignes (cas pris, poids 0 avant/après, réglage final) ; (2) après les 6 premières tuiles : durée moyenne mesurée par tuile en parallèle et HEURE DE FIN estimée (horloge réelle), 2 lignes ; (3) fin : 6 lignes au plus (tuiles faites/passées et pourquoi, part moyenne de poids 0, chemin des planches). Tout de suite si blocage.
