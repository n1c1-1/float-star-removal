# Judge v33c fixes on pairs

- Date : 2026-11-24 à 05:04 (heure de Paris)
- Numéro du jour : 01
- Type d'agent : agent général
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Tu travailles pour l'orchestrateur du projet « retrait d'étoiles ». Écris en français, sois bref : ton rapport final tient en 10 lignes au plus.

CONTEXTE
- La nuit a produit 120 paires d'entraînement (npz : entrée, cible, poids 1 / 0,5 / 0, classes) dans <DONNEES>\Paires étoiles v33 – 23 nov\. Les planches sont dans controle\<image>.jpg : entrée | cible | poids, en arcsinh 60.
- Le juge actuel est <SCRIPTS>\Juge v33 – 23 nov\regles_v33.py. Le code de production est dans <SCRIPTS>\Production v33 – 23 nov\ (voir aussi JOURNAL.md à la racine de starRemoval).
- Ma revue à l'œil est dans <DONNEES>\Paires étoiles v33 – 23 nov\revue_orchestrateur.csv. Lis-la d'abord.
- Une autre tâche fait tourner en ce moment 4 tuiles sur le GPU, avec le même dossier de sortie. Ne touche pas à leurs fichiers tant qu'elles n'ont pas fini : les npz bubble_a7s g2200_1800, heart_a7s g1152_1152, ic443_40da g2176_896 et g2432_1152 apparaîtront plus tard. Traite-les à la fin s'ils sont là, sinon dis-le.

TÂCHE : un juge v33c qui corrige 3 défauts des poids, sur les paires existantes. On recalcule les poids, pas la chaîne.
1. Trous dans le poids 0 : on voit des anneaux et des petits disques de poids 1 ou 0,5 enfermés dans les grands disques de poids 0 (m45_40da, rosette g1744_1026, xmas, ic443, iris). Toute île de poids > 0 entièrement entourée de poids 0 et plus petite qu'environ 400 px passe à 0.
2. Plateaux en disque : à la place d'une étoile, la cible garde un disque plat (halo aplati) gris-bleu ou violet, un peu plus clair que son voisinage, au poids 1. Exemples : ngc6888 g896_896 et g896_1408, ngc6543 g1152_640 et g1408_896, ngc7000_gfx g6741_2368, m97. Règle locale, seulement aux positions d'étoiles du catalogue où la chaîne a touché : si la cible moyenne dans le disque dépasse de plus de 3 bruits la moyenne de l'anneau juste autour, mettre ce disque à 0 (rayon de 1 à 2,5 FWHM, choisi par mesure). Fais de même pour un creux de plus de 3 bruits, s'il n'est pas déjà pris.
3. Un défaut visible au poids 0,5 doit passer à 0 si la règle 2 le touche. Exemples : ngc869 g896_384, tache verdâtre ; m57 g3424_2208, tache grise collée à M57 (vérifie si elle est prise).
GARDE-FOU : le dosage compte. Si une paire gagne plus de 3 points de poids 0 avec v33c, la règle est trop large : resserre-la. Ne bascule jamais vers « tout à 0 ».
SORTIE : écris les nouveaux poids dans des copies *_v33c.npz à côté des anciens. Mets les planches v33c dans controle_v33c\, seulement pour les images où quelque chose change. Fais un tableau du poids 0 avant et après par paire dans controle_v33c\poids0_v33c.txt. Ne traite pas les paires « lâchée » de la revue.

QUESTION ANNEXE (lecture seulement, pas de calcul long) : 25 images n'ont pas de paire « brute » (entrée non débruitée) parce que le brut n'a pas été conservé. Regarde dans le code de production d'où vient l'entrée débruitée. Peut-on recouper le brut à la même position depuis le fichier source, en quelques secondes par tuile ? Réponds en 2 lignes : oui/non, comment, coût.

RÈGLES MACHINE (obligatoires) :
- Lance tout calcul de plus de quelques secondes avec : powershell -File <SCRIPTS>\lancer_borne.ps1 -Fils 4 -Log <journal> -Exe python -Args "-I <script> ...".
- Pas de GPU, il est occupé. 4 fils au plus. Arrête seulement tes propres PID.
- Écris seulement dans <SCRIPTS>\ et <DONNEES>\. Pas de téléchargement.
