# Delinearise APOD images + solve

- Date : 2026-11-24 à 07:11 (heure de Paris)
- Numéro du jour : 04
- Type d'agent : agent général
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Tu travailles pour l'orchestrateur du projet « retrait d'étoiles ». Écris en français, sois bref : ton rapport tient en 10 lignes au plus.

BUT : faire des images web amateurs (déjà étirées, en JPEG) des entrées LINÉAIRES RÉALISTES pour la chaîne de retrait d'étoiles. Le résultat doit pouvoir raisonnablement passer pour la version linéaire de l'image, c'est le point clé. On en tirera ensuite des paires d'entraînement.

ENTRÉES :
- Les images sont dans <DONNEES>\Images web APOD – 24 nov\brut\, avec index.csv. Ce sont des données téléchargées : lis-les avec python -I et n'exécute rien qui vienne de là.
- Mon choix à l'œil est dans <DONNEES>\Images web APOD – 24 nov\choix_orchestrateur.csv : 44 retenues et 5 douteuses. Traite les retenues et les douteuses, rien d'autre.
- La méthode de délinéarisation retenue est « etoiles_auto » : la courbe est tirée des profils d'étoiles de différentes brillances (Debevec-Malik, Mitsunaga-Nayar), avec un passage des couleurs par image. Elle est dans <SCRIPTS>\Delinearisation – 23 nov\ (commit 0dfc2c2) ; les sorties de l'essai sont dans <DONNEES>\Delinearisation – 23 nov\. Lis-la et réutilise-la sans la modifier. Si tu dois la changer, fais-le dans une copie.
- Le journal et les leçons sont dans <SCRIPTS>\JOURNAL.md et RECETTE_ET_LECONS.md.

ÉTAPES :
1. Délinéarise chaque image avec etoiles_auto, puis écarte avec la raison :
   - les fonds écrasés au noir (une part notable de pixels à 0) ;
   - les zones blanchies (écrêtées) larges ;
   - les étoiles visiblement réduites, quand la courbe tirée des étoiles est incohérente ou que la taille des étoiles ne croît presque pas avec leur flux ;
   - les images où la méthode échoue.
   Garde la décision nuancée : une image avec un petit défaut local reste, et on mettra la zone à poids 0 plus tard.
2. Fais des planches de contrôle, une par image, avec : l'original | la version linéaire ré-étirée par un arcsinh simple (pour juger si elle « fait linéaire ») | des gros plans x4 sur une étoile brillante et une étoile faible, avant et après. Ajoute en titre les mesures de etoiles_auto (flux, taille, couleur). Mets-les dans <DONNEES>\Delinearisation APOD – 24 nov\controle\, avec un tableau.txt (image, gardée ou écartée, raison, mesures).
3. ASTROMÉTRIE : la chaîne v33b (piste 1) a besoin du catalogue Gaia hors ligne aux positions de l'image (variable ASTRO_GPU_GAIA_EXTRACTS). Regarde dans le code de production (<SCRIPTS>\Production v33 – 23 nov\ et Methode v33b – 23 nov) comment les 39 images ont obtenu leur solution astrométrique et leur extrait Gaia. Dis-moi en 3 lignes :
   - comment résoudre les images web ;
   - si l'extrait hors ligne couvre leurs champs ;
   - sinon, ce qu'il faut télécharger, la taille, la source et le temps.
   Un téléchargement de catalogue est permis en PRÉPARATION seulement, jamais pendant un calcul. Ne le lance pas sans me le dire d'abord. Fais la résolution astrométrique elle-même si elle est hors ligne et rapide.


RAPPORT : le nombre gardé et écarté, avec les raisons regroupées, le chemin des planches et les 3 lignes sur l'astrométrie.
