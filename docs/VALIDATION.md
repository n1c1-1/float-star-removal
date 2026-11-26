# Validation : comment on mesure, honnêtement

Une amélioration ne se déclare que sur des images **que le réseau n'a jamais vues** et que personne n'a utilisées pour régler quoi que ce soit. Ce document décrit : le protocole de mesure, la liste des images de validation (sans les images), le jeu de contrôle des petites galaxies, le garde-fou des étoiles saturées, et les pièges déjà rencontrés.

## 1. Le protocole de mesure (`scripts/Validation – 22 nov/valider_reseau.py`)

Pour chaque image entière de la liste (inférence par tuiles recouvertes, réseau = `chain_nafnet`, étirement `asinh(v/3)/10`), le script mesure :

1. **Étoiles restantes** : flux restant / flux d'origine aux positions du catalogue Gaia (ouverture de 2 FWHM, fond = médiane de la couronne 3-5 FWHM de la même image), médiane par bande de magnitude G : < 8, 8-11, 11-14, 14-17, > 17. 0 = retirée, 1 = intacte. Un 90e centile est aussi écrit.
2. **Dégâts sur le fond** : |sortie − entrée| (moyenne des 3 couleurs, en unités de bruit de l'entrée) sur les pixels loin de toute étoile Gaia détectable (rayon = max(4 FWHM, rayon propre à la magnitude)) : médiane (« fond méd. ») et 99e centile (« fond p99 »), puis la même mesure sur la « structure brillante » (lissé de l'entrée au-dessus de son 90e centile : « structure méd. »). **Unité : 0,001 est mille fois sous le bruit, invisible.**
3. **Erreur exacte sous des étoiles synthétiques** : 20 tuiles de 256 px loin des étoiles brillantes, `paire_synthetique(fond, graine)` (cible = fond), L1 sur l'étirement asinh sur l'emprise des étoiles, par classe d'éclat. Seule mesure où la vérité est exacte.
4. **Planche** : entrée | sortie | différence au même étirement (non publiée : les images sont celles de la liste).
5. **Cœurs** (`coeurs_mesure.py`, ajoutée le 25 nov) : rapport flux sortie / flux entrée dans un disque de 15 px autour de 6 cœurs de galaxie (M31 et M32 sur deux images, M65 et M66 sur une troisième ; idéal 1). La colonne « M32 60D » isole le cœur compact de M32 sur la photo d'un Canon 60D.

L'inférence « deux échelles V2 » (étape 8 de `METHODE.md`) est appliquée à toutes les mesures du 25 nov au 26 nov. Le script accepte aussi un **dossier de sorties d'un autre logiciel** (`<nom>.npy`, mêmes unités que l'entrée) et le note de la même façon : c'est ainsi que des références externes ont été comparées, sans jamais entrer dans un jeu d'entraînement (voir `PROVENANCE.md`).

Commande : `python "scripts/Validation – 22 nov/valider_reseau.py" <poids.pt> --tag <nom>`.

## 2. Les images de validation (liste ; aucune image n'est publiée)

Préparation : même chaîne que l'entraînement (gradient retiré, unités de bruit, fond 0, réseau de débruitage de FLOAT, balance des blancs). Aucune de ces images n'a de tuile dans les paires d'entraînement (aucune cible de l'entraînement).

| nom | type | instrument (fiche) | échelle (″/px) | ce que l'image teste |
|---|---|---|---|---|
| m31_a7s_newton | galaxie à cœur brillant | appareil hybride plein format défiltré, Newton 130/900 | 1,28 | aigrettes d'araignée, cœur de galaxie très brillant |
| ic405_a7s_newton | nébuleuse | idem | 1,28 | aigrettes sur étoiles brillantes dans une nébuleuse |
| m33_osc_lunette | galaxie diffuse | caméra couleur ASI294MC Pro, lunette de 350 mm | 1,81 | étoiles sans aigrettes, régions HII |
| m27_mono_nb | planétaire compacte, bande étroite assemblée | caméra mono ASI1600MM, Newton 1325 mm, filtres Ha/OIII/SII | 0,59 | planétaire compacte très contrastée, 3 filtres |
| ngc3372_mono_nb | champ riche, bande étroite (Ha+OIII) | télescope de 0,4 m d'un réseau en ligne | 0,74 | voie lactée dense, nébuleuse, étoile très brillante |
| m16_seestar_s50 | télescope connecté | Seestar S50, 250 mm | 1,58 | petit instrument, pile courte |
| ngc6934_stellina | amas globulaire | Stellina, 400 mm | 1,24 | étoiles très serrées, cœur saturé |
| leotriplet_canon6d_newton | champ de galaxies faibles | reflex Canon 6D, Newton 200/1000 | 1,35 | aigrettes, FWHM de 8 px, cœurs M65/M66 |
| m31_canon60d_objectif_filee | étoiles filées | reflex Canon 60D, objectif 200 mm, monture non guidée | 4,56 | étoiles allongées, galaxie large, cœurs M31/M32 |
| m42_canon40da_lunette | cœur d'Orion | reflex Canon 40Da, lunette 72ED | 2,73 | Trapèze, halos d'étoiles brillantes |
| ic5146_canon60da_lunette | nébuleuse sombre et brillante | reflex Canon 60Da, lunette | 2,0 | champ d'étoiles + nébuleuse diffuse |
| m42_dobson_grande_fwhm | grande FWHM (10 px) | ASI533MC Pro, Dobson 12 pouces | 0,51 | étoiles très larges : tout change d'échelle |
| ngc7822_a7s_lunette | nébuleuse étendue faible | appareil hybride défiltré, lunette | 4,25 | peu de structures brillantes (remplacée par la suivante : le catalogue d'un champ de 7,5° était trop lent) |
| ic1848_stellina | nébuleuse faible, télescope connecté | Stellina, 400 mm | 1,24 | nébuleuse faible dans un champ riche, FWHM 4 px |

Les tableaux comptent **13 images**. La 14e entrée de la liste (voir `scripts/Validation – 22 nov/liste.py`) est la remplaçante de NGC 7822 ; le décompte exact entre ces deux images n'est pas établi dans les journaux (« 13 images + IC 1848 » dans certains, « 13 » dans les tableaux) : à vérifier dans `liste.py` avant de comparer des chiffres à la ligne près.

## 3. Le jeu de contrôle des petites galaxies (26 nov) : ne sert qu'à juger

**But.** Savoir si un réseau **garde** les petites galaxies compactes sans s'être spécialisé sur M32. Ce jeu ne sert qu'à juger : jamais à régler, jamais à entraîner.

**Construction** (`scripts/Controle galaxies compactes – 26 nov/`) :
- 5 images web publiées (liste d'adresses : `SOURCES_WEB.md`, section « contrôle ») déjà délinéarisées et astrométrées avec le catalogue Gaia profond hors ligne ;
- détection par `chercher_galaxies.py` : source **absente** de l'extrait Gaia (rien à moins de 1,2 FWHM d'étoile), FWHM ≥ 1,2 fois celle des 15 étoiles de référence les plus proches, crête ≥ 20 bruits, aucune étoile Gaia G < 15 à moins de 10 FWHM, aucun voisin plus de 30 % aussi brillant, couronne faible (pas dans une grande galaxie), bords à 192 px ;
- choix final **à l'œil** (`revue_candidats.py`, vignettes 64×64) : 18 galaxies (1,2 à 2,8 FWHM d'étoile) ; presque toutes de vraies galaxies (spirales, elliptiques, vues par la tranche) ; les images dont l'astrométrie Gaia ne colle pas ont été écartées ;
- **indépendance** : aucune des images de validation ; 3 images sans aucune tuile d'entraînement, 2 images avec des tuiles dont chaque galaxie est à au moins 64 px.

**Mesure** (`mesurer_controle.py <poids.pt|dossier de sorties> --tag nom`, processeur) : découpe de 384 px autour de la galaxie ; rapport flux sortie / flux entrée dans 3 px et 8 px (moyenne des 3 couleurs, fond = médiane de la couronne 24-40 px). **Idéal : 1 pour la galaxie.** Garde-fou : le même rapport sur l'étoile Gaia **de même brillance** de chaque galaxie (idéal 0).

**Résultats de référence (médianes sur 18 galaxies, galaxie / étoile de même brillance) :**

| réseau | 3 px | 8 px |
|---|---|---|
| N7 pas 22 000 | −0,002 / −0,002 | −0,005 / −0,006 |
| mélange N7/N5 0,75 | −0,001 / −0,002 | −0,006 / −0,004 |
| N9 pas 10 000 | ≈ 0, comme N7 | ≈ 0 |

Tous effacent presque toutes les galaxies ; seule la n° 17, diffuse, reste. Un logiciel tiers de retrait d'étoiles, mesuré de la même façon le 26 nov, les efface aussi (médiane ≈ 0,000 à 3 px). **Ce jeu définit donc l'objectif** : un réseau qui garde la médiane galaxie près de 1 **sans** garder l'étoile de même brillance.

## 4. Le garde-fou des étoiles saturées (diagnostic du 26 nov)

Test à faire en plus du tableau, avant d'annoncer qu'un réseau « garde les cœurs sans rien perdre » (scripts `scripts/Diagnostic Tete de cheval – 26 nov/`) : sur des découpes de 1 024 px contenant des étoiles brillantes **saturées à faible rapport signal/bruit** (crête plate à ≈ 100 bruits, poses courtes), mesurer le **flux restant** après retrait dans un disque de 2 FWHM (minimum 6 px) et dans un disque de 25 px (0 = retirée, 1 = intacte). Six étoiles de la Tête de Cheval (G 3,79 ; G 6,18 ; quatre étoiles moyennes), plus deux étoiles de validation à fort signal pour comparaison. Résultat : N5 avec l'inférence à deux échelles retire les six (0,00 à 0,02) ; N7 en garde cinq sur six (0,95 à 0,98 ; la sixième à 0,11) ; le mélange 0,75 les garde toutes les six (0,95 à 0,99). À fort signal (860 à 12 500 bruits), tous les réseaux les retirent (0,00 à 0,02). Ce test échoue pour un réseau qui confond étoile saturée et cœur compact.

## 5. Mesurer honnêtement une amélioration : la liste de contrôle

1. Changer **une** chose, dans une copie, et écrire avant : « déjà essayé » et « ce qui diffère ».
2. Mesurer sur la validation (13 images) **et** sur le jeu de contrôle des petites galaxies **et** sur le test des étoiles saturées. Un gain sur un seul des trois n'en est pas un.
3. Rapporter toutes les colonnes, y compris celles qui se dégradent. Exemple : N7 gagne les cœurs (0,19 → 0,81) mais perd les étoiles brillantes (G < 8 : 0 → 0,021) : ce n'est pas un progrès net, c'est un déplacement du compromis.
4. Regarder l'image (carré entier, zooms ×6, entrée | sortie), jamais seulement le tableau.
5. Comparer à pas égal **et** à taux d'apprentissage égal (un tour de 10 000 pas n'est pas comparable à pas égal à un tour de 34 500).
6. Un jeu de validation qui ne contient pas un défaut ne l'écarte pas : ajouter l'image qui le montre (Tête de Cheval, 26 nov).
7. Ne jamais régler sur le jeu de contrôle.
