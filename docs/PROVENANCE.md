# Provenance : le réseau n'a jamais été entraîné sur la sortie d'un autre logiciel

**Ce que ce document permet d'affirmer, et de vérifier soi-même** : les modèles publiés (E3, N5, N7 et leur mélange N7/N5 0,75) ont été entraînés uniquement sur (1) des paires **générées** par du code et des graines, (2) des paires **réelles** dont la cible est fabriquée par notre propre chaîne de soustraction guidée par Gaia, (3) des étoiles et cœurs synthétiques posés sur des fonds qui sont eux-mêmes des cibles de (1) ou (2). Aucun fichier lu par l'entraînement n'est la sortie d'un autre logiciel de retrait d'étoiles.

Le détail technique est dans `../provenance/` (manifestes, vérificateurs, analyse statique). Ce document l'explique en termes simples.

## 1. Les quatre modèles et leurs données

| modèle | point de départ | données lues par l'entraînement |
|---|---|---|
| **E3** (8-9 nov) | E2, lui-même entraîné depuis des poids aléatoires | paires **générées** `pairs_v4` (500 paires, graine 1, jeux « clean » et « defects ») + étoiles tirées à la volée ; aucune image |
| **N5** (25 nov) | E3, pas 20 000 | 182 fichiers de paires réelles (version v33g) listés dans `provenance/MANIFESTE_PAIRES.csv` + fonds et étoiles synthétiques |
| **N7** (25 nov) | E3, pas 20 000 | **les mêmes** 182 fichiers + cœurs de galaxie synthétiques (25 % des tuiles synthétiques) |
| **mélange N7/N5 0,75** (26 nov) | — | **aucune donnée** : 0,75 × N7 (pas 22 000) + 0,25 × N5 (pas 23 000), poids moyennés |

E, E2 et E3 partent de **poids aléatoires** ; aucun poids de réseau préexistant n'est lu par l'entraînement.

## 2. Comment chaque cible est fabriquée

| jeu de données | d'où viennent les entrées | comment la cible est fabriquée |
|---|---|---|
| paires **générées** (E2, E3) | rien : fonds, défauts, étoiles et bruit tirés d'une graine | la cible **est** le fond sans les étoiles tirées : exacte par construction |
| paires **réelles débruitées** (103 d'entraînement + 6 tenues) | image avec ses étoiles de Nicolas : pile moins son gradient, normalisée en unités de bruit, débruitée | **entrée moins un calque d'étoiles** : piste 1 (modèle d'étoile et de halo ajusté sur l'image, positions et éclats du catalogue Gaia), étape A (rebouchage par patchs pris dans l'image), étape B (passes), juge (poids). v33d : des poids mis à 0 seulement ; v33g : + lumière diffuse lissée |
| paires **réelles brutes** (35) | la même tuile, non débruitée | brute − (entrée débruitée − cible débruitée) : on retire la même couche d'étoiles |
| paires **web (APOD)** (38 tuiles de 13 images) | images publiées par APOD, **délinéarisées** (étirement inverse estimé sur les étoiles), non re-débruitées | **même chaîne** (variante v33b) et même juge que ci-dessus |
| fonds et formes du générateur d'étoiles de N5/N7 | les fonds sont des **cibles ou entrées de paires réelles** (une fois sur deux chacune) ; les formes d'étoiles viennent de banques d'étoiles brillantes réelles mesurées dans des images **avec** leurs étoiles | n'entrent dans aucune cible de paire réelle ; servent à dessiner des étoiles ajoutées |

Les paires APOD : vérifié dans le code (`scripts/Production APOD – 24 nov/`) et les journaux. 65 tuiles ont été produites sur 22 images, 9 images ont été écartées à la revue à l'œil (27 tuiles), 7 images n'ont pas pu être produites (20 tuiles) ; il reste 38 tuiles de 13 images. L'entrée est préparée sans réseau et sans téléchargement (`preparer_apod.py`) ; la cible vient de la même chaîne que pour les images de Nicolas. **Doute à connaître** : les images web ont déjà été traitées par leurs auteurs avec un logiciel que nous ignorons (peut-être avec un outil de réduction d'étoiles) ; ce traitement est dans l'**entrée** ; la **cible** reste calculée par notre chaîne à partir de cette entrée.

## 2 bis. Le verdict pour les 182 paires lues par N5 et N7

**Établi pour les 182 lignes** de `provenance/MANIFESTE_PAIRES.csv` (colonnes `origine_cible`, `preuve_cible`, `verdict_cible`) : la cible de chaque paire vient de la **chaîne classique** (131 lignes : chaîne v33 ; 51 lignes : chaîne v33b, dont les 38 paires web), puis v33d (poids mis à 0 seulement) et v33g (lumière diffuse lissée) ; pour les 35 paires brutes, de la formule « brute − (entrée débruitée − cible débruitée) ». Aucune cible ne vient d'une sortie de réseau, d'une retouche à la main, ni d'un logiciel tiers. Aucune cible n'est « non établie ». (Les tuiles synthétiques de N5 et N7 sont tirées à la volée ; ce ne sont pas des fichiers de paires.)

Les trois preuves, en termes simples :

1. **Les journaux de production.** Chaque tuile a son journal de piste 1 (réglages « sans remplissage », dossier de travail propre à la tuile) et son journal de fin, et figure dans la liste des paires produites (`paires.jsonl`). La chaîne ne lit que l'**image d'entrée** de chaque image source ; aucun script ne nomme les fichiers « sans étoiles » ou « cible » qui existent à côté de certaines images, et l'historique git des dossiers de la chaîne ne montre aucune lecture de tels fichiers. Une seule tuile n'a pas de journal séparé (`58_hoo_g3456_896`, essai initial) : elle est établie par le rejeu.
2. **Un contrôle numérique.** Pour les 60 tuiles non web dont le dossier d'image contient un fichier « sans étoiles » ou « cible » (piles de télescopes en ligne, jeux de données…), on a comparé la cible de la paire à ce fichier voisin sur les pixels où il diffère de l'entrée de plus de 5 bruits : la cible ne lui est égale que sur **0,42 % des pixels au plus** (une cible dérivée de ce fichier en ferait 100 %). Les autres images n'ont aucun tel fichier : rien n'était à lire.
3. **Le rejeu au bit près.** La chaîne publiée refait la cible à partir de l'entrée et la retrouve **identique au bit** sur 6 tuiles, dont 3 tuiles (5 lignes) parmi les 14 lignes dont le commit de production est « incertain ». Pour les 14 lignes incertaines, l'historique git montre aussi qu'aucune version commitée des scripts n'a jamais lu un fichier de référence.

Chaque ligne du CSV donne sa preuve (`preuve_cible`). Non rejouées (24 à 53 minutes de piste 1 par tuile) : `bubble_a7s_g1446_1310`, `heart_a7s` (3 tuiles), `ic443_40da` (4 tuiles) ; leurs dossiers n'ont aucun fichier « sans étoiles », ou le contrôle numérique s'applique.

## 3. Les deux façons de vérifier

### Images de Nicolas : les paires sont publiées (pièce jointe de version)

Les paires issues des images de Nicolas lues par N5 et N7 (54 fichiers, 298 Mo ; + 470 Mo de fichiers de rejeu) sont fournies en pièce jointe de la page Releases du dépôt (hors git : trop volumineuses). Leur liste, avec empreinte et taille, est dans `provenance/SHA256SUMS_PAIRES_NICOLAS.txt`. Un tiers peut :
1. vérifier que les fichiers sont intacts (`sha256sum -c`) ;
2. vérifier les **relations** entre fichiers (cible v33g = cible v33d + lumière diffuse lissée ; paire brute = brute − calque d'étoiles ; poids seulement mis à 0 entre versions) : `python provenance/verifier_provenance.py <paire>.npz --mode g` ;
3. relancer la chaîne publiée et retrouver la cible **au bit près** : `python provenance/verifier_provenance.py <paire>.npz --mode rejeu` (voir `provenance/LISEZMOI_PROVENANCE.md`). **Pour le rejeu complet**, la chaîne part de l'image d'origine (découpe de 768 px autour de la tuile, astrométrie, modèle d'étoile, extrait Gaia profond). Ces fichiers d'entrée sont préparés pour les 54 paires, en pièce jointe de version distincte (`rejeu_entrees`, 469,6 Mo, 132 fichiers, empreintes dans `provenance/SHA256SUMS_REJEU_NICOLAS.txt`) : `RACINE_ORIGINE=<rejeu_entrees>/origine ENTREES_REJEU=<rejeu_entrees>/entrees python provenance/verifier_provenance.py <paire>.npz --mode rejeu` (voir `provenance/LISEZMOI_PROVENANCE.md`). Essayé avec ces seuls fichiers : `m101_a7s_g2085_883` (cible v33g refaite identique au bit). Sans cette pièce jointe, on vérifie les empreintes et les relations (modes `liens` et `g`).

### Images web : seuls les moyens de les refaire sont publiés

Les images web appartiennent à leurs auteurs : **leurs pixels, et les paires qui en dérivent, ne sont pas publiés**. À la place, pour chacune des 38 paires, `provenance/MANIFESTE_PAIRES.csv` donne : l'adresse de la page APOD et celle de l'image, l'empreinte sha256 du fichier image tel que téléchargé à l'époque, les paramètres exacts de délinéarisation, la découpe (position et taille de la tuile), l'empreinte du fichier de paire. Le script `provenance/refaire_paires_web.py` retélécharge l'image depuis son adresse (option `--telecharger`), vérifie son empreinte, refait la délinéarisation, l'entrée et la paire avec la chaîne publiée, et compare l'empreinte à celle du manifeste.

## 4. Ce qui a été essayé pour de vrai (26 nov)

Rejeu de la cible à partir de l'entrée, avec la chaîne **publiée** ou ses originaux, sur le processeur :

| paire | résultat |
|---|---|
| `m1_g896_896_debruitee` (image de Nicolas, chaîne v33) | cible, poids, classes, entrée **identiques au bit** (écart maximal 0 sur 258 458 pixels de poids > 0) |
| `m45_asi1600_g2650_1450` et sa brute (v33) | cibles débruitée et **brute** identiques au bit ; étape v33d → v33g refaite à l'identique |
| `bubble_a7s_g2000_1367` et sa brute (v33b) | identiques au bit (piste 1 sur processeur : 803 s) |
| `apod_070_g1536_1408` (web, v33b) | identique au bit ; avec `refaire_paires_web.py --local` : empreinte de l'image, délinéarisation, entrée et paire refaites à l'identique, et le **fichier** de paire refait a le même sha256 que le manifeste |

La production avait fait la piste 1 sur la carte graphique ; son rejeu sur processeur redonne pourtant les mêmes bits (l'ajustement tourne en `float64`). **Cela n'est établi que pour ces tuiles** : on ne promet pas la même chose pour les autres tuiles, ni avec une autre version de bibliothèque. Le retéléchargement depuis le réseau n'a pas été testé (aucun accès réseau pendant la préparation).

## 5. Analyse statique : ce que lisent les scripts d'entraînement

Méthode (détail et numéros de ligne : `provenance/ANALYSE_STATIQUE.md`) : on calcule la **fermeture d'imports** de chaque script d'entraînement et de fabrication de cibles, et on cherche, dans toute cette fermeture, le nom de tout logiciel tiers de retrait d'étoiles, les mots `starless`, `Comparaison`, les dossiers de références, et toute lecture de poids de réseau. Dossiers réellement lus :

| script | dossiers lus | contenu |
|---|---|---|
| `train_e.py` (+ `day_e3.py`) | `pairs_v4/` (+ jeu de validation tiré de ces paires), poids de E2 | paires générées ; poids entraînés par ce même script |
| `train_n56.py` et `train_n7.py` | `Paires étoiles v33 – 23 nov/` (fichiers `*_v33g.npz`), `Paires étoiles APOD – 24 nov/` (`*_v33g.npz`), poids de E3 pas 20 000 | sorties de **notre** chaîne ; E3 généré seulement |
| `paires_synthetiques_n2.py`, `_n7.py`, `etoiles2.py` | les tableaux des paires ci-dessus (fonds), les banques d'étoiles (formes) | cibles de notre chaîne ; étoiles réelles **avec** leurs étoiles (mesure de forme) |
| `produire_*.py`, `tuile_cpu*.py`, `methode_validee.py`, `etape_A/B.py`, piste 1, piste 3, passes, juges, `preparer_g_h.py` | l'image d'entrée, son astrométrie, le catalogue Gaia, le `starless.npy` de **notre** piste 1 | aucune sortie d'un autre logiciel |

**Résultat de la recherche** : le nom d'un logiciel tiers de retrait d'étoiles n'apparaît dans aucune fermeture comme donnée lue. Le mot `starless` n'y désigne que la sortie de notre propre piste 1. Les 30 scripts qui lisent des sorties d'un autre logiciel ou de réseaux entraînés (mesure, planches, diagnostics) ne sont dans **aucune** des fermetures : ce sont des scripts de mesure, jamais appelés par l'entraînement ni par la fabrication des cibles. **Les dossiers de comparaison ne sont lus que par des scripts de mesure.**

## 6. Ce qui reste incertain (écrit, pas caché)

1. **E3 n'a pas de commit** (le dépôt de scripts date du 21 nov) : sa provenance repose sur le manifeste d'empreintes des paires générées (`reads_no_image: true`, 4 000 empreintes, code identique à celui du manifeste) et sur les journaux. `train_e.py` a été modifié après la fin de E3 (9 nov, 15 h 24).
2. **E3, fichiers « débruités »** : la régénération de contrôle (`make_dataset.py --verify`) redonne **à l'octet** les fichiers bruités, mais pas les débruités (écart maximal 216 sur l'entrée, 88 sur la cible). Cause probable : le code de débruitage de l'application a changé depuis le 9 nov ; non établie. Cela ne touche pas l'origine des paires.
3. **Commits de production** : 14 lignes du manifeste sur 182 ont un commit « incertain » (le script de production a été commité après les premières paires) ; `preparer_g_h.py` a été modifié après l'écriture des paires v33g, mais le mode `g` du vérificateur les reproduit au bit près sur les 3 paires essayées.
4. **Images web** : traitées par leurs auteurs avant publication (voir §2).
5. **Propriété de certaines images non APOD** : plusieurs viennent de banques d'images, de jeux de données ou de piles de télescopes en ligne ; le code ne dit pas qui les a prises. Les paires de la pièce jointe de version se limitent à celles de l'arborescence personnelle de Nicolas ; les 90 autres fichiers (488 Mo) restent hors publication.
6. **Rejeu complet** fait sur 6 tuiles ; les 182 empreintes et relations sont vérifiées, les autres rejeux n'ont pas été faits (§2 bis).
7. **Dépendances de FLOAT** : l'astrométrie hors ligne et le réseau de débruitage de l'application ne sont pas fournis (`scripts/LISEZMOI.md`).
