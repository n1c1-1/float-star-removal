# Collect amateur web images (APOD)

- Date : 2026-11-24 à 06:45 (heure de Paris)
- Numéro du jour : 03
- Type d'agent : agent général
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Tu travailles pour l'orchestrateur du projet « retrait d'étoiles ». Écris en français, sois bref : ton rapport tient en 8 lignes au plus.

CONTEXTE : on veut des images de ciel profond AMATEURS, publiées et déjà étirées, à délinéariser ensuite pour en faire des paires d'entraînement. La collecte sur Wikimedia Commons est bloquée par des « http 429 » depuis hier 15:45 ; je l'ai arrêtée. Le code est dans <SCRIPTS>\Images web – 23 nov\ : cherche.py, telecharge.py, idx.txt, sel.txt et LISEZMOI.txt. Lis-le pour reprendre les critères, sans le réécrire.
Les images déjà reçues sont dans <DONNEES>\Images web – 23 nov\ (une seule, je crois). Vérifie.

TÂCHE : collecter environ 60 images candidates depuis l'archive d'Astronomy Picture of the Day (apod.nasa.gov). On n'y a besoin d'aucun compte, et une grande partie des images sont amateurs.
- CRITÈRES : ciel profond (nébuleuses, galaxies, amas), étoiles présentes et NON réduites (pas de « starless » ni d'étoiles recomposées visiblement rapetissées), image en grand format, côté ≥ 2000 px. Préférer les auteurs amateurs. Écarter les images dont les étoiles sont écrêtées en larges disques blancs, celles au fond noir écrasé, les mosaïques de télescopes professionnels (Hubble, JWST, ESO) sauf 5 au plus en complément, les montages, le texte incrusté et les images à forte compression JPEG visible.
- SOURCE : les pages de l'archive (apod.nasa.gov/apod/archivepix.html, puis les pages apYYMMDD.html) donnent le lien vers l'image en grand format.
- POLITESSE : une requête toutes les 2 à 3 s au moins, un User-Agent générique (par exemple « star-removal-research/0.1 »), AUCUNE information personnelle (ni nom, ni e-mail, ni nom de projet). Sur un 429 ou un 403, attends puis ralentis ; si ça persiste, arrête-toi et dis-le.
- SORTIE : <DONNEES>\Images web APOD – 24 nov\. Mets les fichiers téléchargés dans un sous-dossier brut\ (données non sûres : ne les exécute pas, et lis-les avec python -I). Écris un index.csv : fichier, date APOD, titre, auteur, taille, motif du choix. Fais aussi des planches de vignettes (12 par planche, numérotées, nom de fichier sous chaque vignette) dans planches\, pour que je choisisse à l'œil. Ajoute un LISEZMOI.txt en français.

RÈGLES MACHINE : un entraînement tourne sur le GPU et utilise déjà 12 fils processeur. Toi, n'utilise PAS le GPU. Lance le téléchargement via powershell -File <SCRIPTS>\lancer_borne.ps1 -Fils 1 -Log <journal> -Exe python -Args "-I \"<script>\" ..." (les \" sont nécessaires pour les chemins à espaces). Les planches de vignettes se font avec 1 fil. Écris seulement dans <SCRIPTS>\ et <DONNEES>\. Arrête seulement tes propres PID. N'appelle aucun service payant.

RAPPORT : le nombre d'images retenues, la part amateur ou pro, le chemin des planches et les éventuels refus du serveur.
