# Jeu de contrôle galaxies compactes

- Date : 2026-11-26 à 15:05 (heure de Paris)
- Numéro du jour : 08
- Type d'agent : agent général
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Projet : réseau de retrait d'étoiles (astrophotographie). Réponds en français simple. AUCUN message d'attente ; un seul rapport final de 10 lignes au plus.

BUT : constituer un JEU DE CONTRÔLE indépendant de petites galaxies compactes, pour juger si le prochain réseau (N9, en entraînement, qui apprend des cœurs synthétiques compacts) garde ces galaxies sans s'être spécialisé sur M32. Ce jeu ne sert QU'À JUGER, jamais à régler ni à entraîner.

RÈGLES D'INDÉPENDANCE (essentielles) :
- Aucune des 13 images de validation + IC 1848 (liste dans <SCRIPTS>\Validation – 22 nov\valider_reseau.py, LISTE) : elles ont servi à régler les cœurs compacts (M32 Canon 60D).
- Aucune zone vue à l'entraînement : les paires sont des TUILES choisies dans des images (<DONNEES>\Paires étoiles v33 – 23 nov\, <DONNEES>\Paires étoiles APOD – 24 nov\ ; les .npz ou leurs fichiers de choix donnent l'image source et la position des tuiles). Une galaxie dans une image d'entraînement est acceptable SEULEMENT si elle est hors de toutes les tuiles tirées de cette image (marge 64 px). Mieux : des images jamais utilisées du tout, si tu en trouves de prêtes (empilées, astrométrie faite, extrait Gaia disponible) dans <DONNEES> ou <ESSAIS>. Ne touche pas aux projets FLOAT de Nicolas (<DISQUE>/astro_float) : ni écriture, ni copie.
- 10 à 20 galaxies, sur au moins 4 images différentes et au moins 2 appareils ; tailles variées (de presque ponctuelle à ~3 fois une étoile).

COMMENT LES TROUVER (sans téléchargement) : sources détectées dans l'image, ABSENTES de l'extrait Gaia (Gaia = étoiles ; extrait hors ligne : variable ASTRO_GPU_GAIA_EXTRACTS ou fichiers gaia_dr3_profond.npz déjà préparés), et plus larges que la FWHM des étoiles voisines (FWHM ≥ 1,2 fois), assez au-dessus du bruit (crête ≥ 20 bruits), loin des étoiles brillantes. Les grandes galaxies connues (M31, M33, M51…) ne comptent pas : on veut les petites. Écarte les défauts (pixels chauds, bords, satellites).

LIVRABLES dans <DONNEES>\Controle galaxies compactes – 26 nov\ (LISEZMOI.txt en français) :
1. controle.json : pour chaque galaxie, image source, x, y, FWHM mesurée / FWHM des étoiles, crête en bruits, part du flux dans 3 px, et la preuve d'indépendance (image hors entraînement, ou distance à la tuile la plus proche).
2. planche_controle.jpg : une vignette 64×64 par galaxie (étirement asinh commun), numérotée, avec à côté une étoile Gaia de même brillance pour comparaison. Je la regarderai à l'œil pour écarter les faux.
3. Un script mesurer_controle.py qui, pour un réseau donné (instantané .pt, même chargement que la chaîne N7/N8L) ou un dossier de sorties sans étoiles, découpe 384×384 autour de chaque galaxie, applique le réseau (inférence simple par tuiles, processeur accepté) et écrit le rapport flux sortie/entrée dans 3 px et dans 8 px par galaxie + la médiane. Donne aussi, sur les mêmes découpes, le flux restant des étoiles Gaia de même brillance (garde-fou : le réseau ne doit pas se mettre à garder des étoiles).
4. Exécute-le sur N7 (<DONNEES>\Entrainement étoiles N7 – 25 nov\, meilleur pas dans N7_meilleur_pas.txt) et sur le mélange N7/N5 0,75 (<DONNEES>\Melanges N5 N6 N7 – 26 nov\poids\) : ce sont les références. N9 sera mesuré plus tard avec le même script.

RÈGLES MACHINE : la carte 4060 Ti est occupée par l'entraînement N9 : PROCESSEUR SEULEMENT (os.environ["CUDA_VISIBLE_DEVICES"]="" avant import torch). Tout calcul via <SCRIPTS>\lancer_borne.ps1 :
powershell -NoProfile -File <SCRIPTS>\lancer_borne.ps1 -Fils 4 -Log <log> -Exe python -Args "-I \"<script>\" ..."
4 fils, 8 Go de mémoire au plus, un seul calcul à la fois. Ne tue que tes PID, jamais par nom. Aucun téléchargement. Chemins Windows toujours en r"..." ; compile chaque script (py_compile) avant de lancer.

RAPPORT FINAL : nombre de galaxies, images et appareils, comment l'indépendance est garantie, médianes N7 et mélange (3 px et 8 px) et garde-fou étoiles, chemin de la planche, commit.
