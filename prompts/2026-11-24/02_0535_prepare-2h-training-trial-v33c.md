# Prepare 2h training trial v33c

- Date : 2026-11-24 à 05:35 (heure de Paris)
- Numéro du jour : 02
- Type d'agent : agent général
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Tu travailles pour l'orchestrateur du projet « retrait d'étoiles ». Écris en français, sois bref : ton rapport tient en 8 lignes au plus.

BUT : préparer un essai d'entraînement de 2 h sur les nouvelles paires réelles, dans les mêmes conditions que l'essai « A » du 22 nov. Ne LANCE PAS l'entraînement : prépare-le, puis arrête-toi. Je te relancerai quand le GPU sera libre, car une autre tâche l'utilise en ce moment.

À LIRE :
- L'essai A du 22 nov : <DONNEES>\Entrainement étoiles jeu 1 – 22 nov\A_E3\. Il part de E3, notre modèle entraîné sur données générées (<ESSAIS>\starRemoval\train\E3\snap\step_20000.pt). Relève la configuration et le script, le meilleur pas étant 8000.
- La validation : <DONNEES>\Validation étoiles – 22 nov (13 images inédites + IC 1848, mesures déjà faites pour A, pour le modèle FLOAT actuel et pour l'outil de référence).
- <SCRIPTS>\JOURNAL.md et RECETTE_ET_LECONS.md (les leçons, à relire avant tout essai).
- Les paires : <DONNEES>\Paires étoiles v33 – 23 nov\. Utilise les fichiers *_v33c.npz : *_debruitee_v33c.npz ET *_brute_v33c.npz, ce sont deux versions de la même tuile. Ne prends pas les paires marquées « lâchée » dans revue_orchestrateur.csv (fichier du même dossier, et voir ses lignes « * »). Si une paire n'existe qu'en ancienne version (sans _v33c), signale-la sans la prendre.

- Le même réseau, la même perte L1 pondérée par les poids des paires, et les mêmes 60 % de tuiles d'étoiles synthétiques posées sur des fonds propres, comme A. Garde aussi les mêmes pas, taux et taille de lot, sauf si une leçon dit le contraire. Départ : E3 step_20000. Durée : 2 h (nombre de pas calé sur la vitesse de A).
- Le tirage : débruitée et brute à parts raisonnables (les brutes n'existent que pour 14 images, ne les sur-tire pas). Pas d'ajout de bruit artificiel, c'est interdit, ça a déjà été une catastrophe.
- Des instantanés tous les 1000 pas, et à la fin la même mesure de validation que pour A, avec un tableau A8000 / nouveau meilleur pas / modèle FLOAT / outil de référence.
- Sorties dans <DONNEES>\Entrainement étoiles v33c – 24 nov\, avec un LISEZMOI.txt en français.
- La commande exacte de lancement, prête, avec : powershell -File <SCRIPTS>\lancer_borne.ps1 -Fils 4 -Log <journal> -Exe python -Args "-I <script> ...". Le chargeur de données ne doit pas lancer plus de 4 processus en tout.

INTERDIT MAINTENANT : tout usage du GPU, même un test de quelques secondes. Un test à blanc sur le CPU, de 2 ou 3 pas sur 2 paires, est permis via lancer_borne avec 4 fils.
Écris seulement dans <SCRIPTS>\ et <DONNEES>\. Pas de téléchargement. Arrête seulement tes propres PID.

RAPPORT : le nombre de paires prises (débruitées / brutes), la configuration en 3 lignes, et la commande de lancement.
