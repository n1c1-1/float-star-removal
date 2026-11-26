# Tutoriel : de bout en bout, pas à pas

Trois parcours, du plus court au plus long. Pour chacun : ce qu'on fait, ce qu'on doit voir, le temps et les ressources. **Ce qui a été réellement exécuté pour écrire ce tutoriel est marqué « essayé »** ; le reste est écrit d'après les scripts et les journaux, et dit clairement ce qui reste à confirmer.

Matériel de référence : Windows 11, processeur à 16 threads, carte RTX 4060 Ti de 16 Go pour l'entraînement ; le processeur seul suffit pour les parcours 1 et 2.

## Parcours 1 — Essayer un réseau en 5 minutes (processeur, aucune image à télécharger)

**Préparation.**
```
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
```
Récupérer `E3_step_20000.pt` (page Releases ; 117 Mo) et vérifier : `sha256sum -c modeles/SHA256SUMS`.

**1. Fabriquer l'image jouet** (ciel plat de valeur 1000, bruit 5, quatre étoiles gaussiennes de pics 60, 200, 800 et 3000 fois le bruit) :
```
python exemples/image_jouet.py image_jouet.npy
```

**2. Retirer les étoiles :**
```
python scripts/inferer_une_image.py image_jouet.npy --poids E3_step_20000.pt --sortie sortie_jouet --carte cpu
```
*Ce qu'on doit voir* : la console affiche le ciel (≈ 997) et le bruit (≈ 5,0) estimés par couleur, puis « écrit dans sortie_jouet ». `sortie_jouet/apercu.jpg` montre trois vignettes (entrée | sans étoiles | étoiles) : les quatre étoiles ne sont plus que dans la vignette de droite. **Essayé** : aux quatre positions, la valeur de la sortie en unités de bruit est −0,13, −0,13, 0,05 et 0,45 (la plus faible, 13 bruits, laisse 0,45 bruit), contre 599, 161, 40 et 13 en entrée.
*Temps* : de l'ordre de la minute pour 512 × 512 sur processeur (non chronométré avec précision), moins sur la carte (`--carte cuda`).

**3. Voir ce que le réseau a appris :** regarder `docs/images/planche_synthetique.jpg` (entrée | cible | différence) et refaire ces images à l'identique avec `python exemples/generer_exemples.py` (≈ 2 minutes, processeur, 4 fils ; les empreintes sha256 de `exemples/EMPREINTES.txt` doivent être identiques : **essayé**, trois exécutions identiques à l'octet sur la machine de référence).

## Parcours 2 — Une image publique (APOD), de l'URL à la paire d'entraînement

Image choisie : « Orion and the Running Man », APOD du 10 mars 2023, `https://apod.nasa.gov/apod/ap230310.html` (numéro 070 de notre liste, `docs/SOURCES_WEB.md`). L'image appartient à son auteur : elle n'est pas redistribuée ici ; on la télécharge depuis la page.

Le chemin est celui qui a fabriqué les paires web de l'entraînement :

| étape | ce qu'on fait | script | ce qu'on doit voir |
|---|---|---|---|
| 1. image | télécharger le fichier d'image de la page (colonne `url_image` de `provenance/MANIFESTE_PAIRES.csv`) ; son sha256 doit être celui de `sha256_image_telechargee` | `provenance/refaire_paires_web.py --telecharger` (ou à la main) | une image étirée (non linéaire), ≈ 2 Mo |
| 2. délinéariser | retrouver une image « comme linéaire » : la courbe de réponse est estimée sur les étoiles (méthode `etoiles_auto`) | `scripts/Delinearisation – 23 nov/methodes.py`, `scripts/Delinearisation APOD – 24 nov/delineariser.py 70` | un tableau (3, H, W) flottant ; la courbe, le nombre d'étoiles vues et la couleur choisie sont écrits dans le journal (**essayé** : ≈ 7 s, valeurs identiques au manifeste) |
| 3. astrométrie | trouver les coordonnées du ciel de chaque pixel (solution TAN-SIP) | `scripts/Delinearisation APOD – 24 nov/resoudre.py 70` (utilise le résolveur local de FLOAT) | un `070_astrometrie.npz` ; à défaut, tout autre résolveur qui écrit le même format |
| 4. extrait Gaia profond | télécharger les étoiles Gaia DR3 (G < 21) du champ, **par le script fourni**, une fois, en dehors de tout calcul | `scripts/Delinearisation APOD – 24 nov/telecharger_gaia_apod.py 70` | `gaia_dr3_profond_apod_070.npz` (une requête au service public de données Gaia ; 3 s d'attente entre requêtes) |
| 5. choisir une tuile | repérer des tuiles de 512 px sur du contenu | `scripts/Delinearisation APOD – 24 nov/choisir_tuiles.py 70` | une mosaïque numérotée ; la tuile de l'exemple est en (1536, 1408) |
| 6. préparer l'entrée | fond et bruit par couleur, unité = bruit moyen | `scripts/Production APOD – 24 nov/preparer_apod.py` | `entrees/apod_070/input.npy` |
| 7. refaire la paire | piste 1 (soustraction guidée par Gaia) → étape A → étape B → juge | `scripts/Production APOD – 24 nov/tuile_cpu_apod.py apod_070 1536 1408` après la piste 1 (`etape_piste1_gpu_2f.py`) ; ou tout en un : `provenance/verifier_provenance.py ... --mode rejeu` | une paire `.npz` : entrée, cible, poids ; **essayé** : ≈ 74 s sur processeur, cible identique **au bit** à celle de l'entraînement |

**Ce qui est vraiment essayé, et ce qui ne l'est pas.** Essayé le 26 nov, sur processeur : `provenance/refaire_paires_web.py --paire apod_070_g1536_1408 --local` (image locale, sha256 vérifié, délinéarisation, entrée et paire refaites, **tout identique au bit**). **Non essayé de bout en bout par un tiers** : le téléchargement (`--telecharger`, aucun accès réseau pendant la préparation) et les étapes 3 et 4 (astrométrie et extrait Gaia) à partir d'une image vierge. Aujourd'hui, `refaire_paires_web.py` a besoin, pour chaque image, de fichiers **dérivés** non publiés (astrométrie, extrait Gaia profond, modèle d'étoile : variable `RACINE_ORIGINE`) : leur publication en pièce jointe de version est à décider (ce sont des fichiers de coordonnées et de catalogue, pas des pixels d'image).

**Ressources.** Sans carte graphique : la piste 1 prend de 1 à 14 minutes par tuile de 512 px sur 4 fils (803 s pour une tuile dense de nos essais) ; avec la carte (production d'origine) environ 5 fois moins. Mémoire : quelques Go. Disque : l'image et ses intermédiaires, quelques centaines de Mo.

Appliquer ensuite le réseau à l'image délinéarisée entière (comme au parcours 1) : `python scripts/inferer_une_image.py 070.npy --poids modeles/N7N5_075.pt`. Pour les très grosses étoiles (G < 6,5), l'inférence à deux échelles (`scripts/Portée M27 – 25 nov/deux_echelles_v1v2.py`) a besoin de l'extrait Gaia de l'étape 4.

## Parcours 3 — Refaire un entraînement

**3a. Données synthétiques et E3 (aucune image).**
```
python scripts/make_dataset.py --n 500 --seed 1 --sets clean,defects --workers 6 --out donnees/pairs_v4
python scripts/make_dataset.py --verify 3 --out donnees/pairs_v4
python scripts/day_e3.py --hours 6.5
```
*Ce qu'on doit voir* : « verified: all identical » pour les fichiers bruités. Attention : les fichiers « débruités » exigent le réseau de débruitage de FLOAT (non fourni) ; le script en a besoin pour ces fichiers, et nos mesures montrent que la régénération des fichiers débruités n'est pas identique à celle de 2026-11-09 (`PROVENANCE.md`, incertain n° 2). *Ressources* : une paire ≈ 10 s sur 14 cœurs, environ 100 Mo par paire ; E3 a demandé 6,5 h sur la carte de 16 Go.

**3b. Affinage sur paires réelles : N5 puis N7.** Il faut des paires réelles `*_v33g.npz` (entrée, cible, poids) : (i) celles de la pièce jointe (images de Nicolas), (ii) celles des images web refaites avec `refaire_paires_web.py`. Les poser dans `<DONNEES>/Paires étoiles v33 – 23 nov/` et `<DONNEES>/Paires étoiles APOD – 24 nov/` (variable `DONNEES`).
```
python "scripts/Tours N5 N6 – 25 nov/train_n56.py" --tag N5 --suffixe v33g --n2 --pas 34500 --depart E3_step_20000.pt
python "scripts/Tour N7 – 25 nov/train_n7.py"    --tag N7 --suffixe v33g --n2 --coeurs 0.25 --depart E3_step_20000.pt
```
*Ressources mesurées* : 34 500 pas ≈ 5 h (≈ 9 minutes pour 1 000 pas), lot de 8 tuiles de 256 px, carte de 16 Go, 2 processus de chargement ; l'entraînement s'arrête proprement sous 3,5 Go de mémoire libre et reprend avec `--reprendre`. Une mesure intermédiaire est écrite tous les 1 000 pas (`history.json`) ; **choisir le pas de plus petite erreur « sous les étoiles »** (N5 : 23 000 ; N7 : 22 000).

**3c. Mesurer et mélanger.**
```
python "scripts/Validation – 22 nov/valider_reseau.py" <poids.pt> --tag N5
python "scripts/Melanges N5 N6 N7 – 26 nov/chaine_melanges.py"      # N7/N5 0,75
```
La validation demande les 13 images de `docs/VALIDATION.md` (non publiées : à remplacer par vos propres images, jamais vues à l'entraînement). Tour court pour tester une idée : 10 000 pas ≈ 1 h 30 ; règle d'arrêt écrite avant le lancement ; ne pas régler sur le jeu de contrôle des petites galaxies.

## En cas de souci

- `ModuleNotFoundError: astro_gpu...` : lancer depuis la racine du dépôt (`scripts/` contient un extrait de FLOAT, `scripts/astro_gpu/`) ; les scripts de chaîne qui font de l'astrométrie ou du débruitage ont besoin de l'application FLOAT complète (`scripts/LISEZMOI.md`).
- Résultat très différent après un changement de version de bibliothèque : les identités « au bit » ont été vérifiées avec Python 3.12.13, numpy 2.5.3, OpenCV 5.0.0, torch 2.11.0.
