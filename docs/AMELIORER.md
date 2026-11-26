# Améliorer : limites connues, pistes ouvertes, comment mesurer

Ce document s'adresse à qui veut faire mieux. Il dit **où ça ne marche pas**, **ce qui a déjà été essayé** (pour ne pas le refaire : détail dans `LECONS.md`) et **comment prouver honnêtement une amélioration** (`VALIDATION.md`).

## 1. Piste n° 1 : le conflit « étoile saturée / cœur compact »

**Le problème.** Il n'y a pas un meilleur modèle, il y a un **compromis** :
- **N5** (avec l'inférence à deux échelles) retire toutes les étoiles saturées, y compris celles de faible rapport signal/bruit, **mais efface les cœurs de galaxie** (cœurs 0,19 ; M32 sur la photo d'un Canon 60D à 0,03).
- **N7** et le mélange **N7/N5 0,75** gardent les cœurs (≈ 0,8) **mais gardent aussi des étoiles brillantes saturées** dont la crête est plate à ≈ 100 fois le bruit (poses courtes), parce qu'un cœur de galaxie synthétique et une étoile écrêtée de faible signal se ressemblent : les cœurs synthétiques de N7 incluent justement des cœurs écrêtés de 50 à 5 000 bruits.
- Preuve (26 nov) : sur une pile linéaire de 704 poses de 15 s de la Tête de Cheval (non débruitée, avec Alnitak), le mélange 0,75 garde cinq étoiles sur six testées (flux restant 0,95 à 0,98 ; dans le disque de 2 FWHM) ; N5 avec V2 les retire toutes (0,00 à 0,02) . Les grandes zones de nébuleuse et la Tête sont bien conservées. Les autres leviers essayés n'y changent rien : augmenter le gain d'entrée (×4, ×8) ne répare que trois étoiles sur six (les plus faibles), débruiter l'entrée n'a aucun effet, et l'inférence à deux échelles en ×2 ne répare que partiellement (une étoile à 0,38, deux autres restent à 0,95).
- Notre jeu de validation ne contenait aucune étoile saturée à ce niveau (860 à 12 500 bruits seulement) : le tableau ne le montrait pas (`VALIDATION.md` §4).

**Hypothèses à établir** (la cause principale est identifiée, les causes secondaires non) :
1. Des étoiles très brillantes **absentes du catalogue Gaia** (Alnitak en est une) sont hors des zones de l'inférence à deux échelles et du déclenchement par Gaia.
2. Une image **non débruitée de pose courte** est peu présente dans l'entraînement (les paires brutes ne forment que 25 % du tirage ; le reste est débruité ou web).

**La piste proposée : la carte Gaia des étoiles connues en entrée du réseau.** Les étoiles saturées sont dans le catalogue ; les cœurs de galaxie, les nébuleuses planétaires et les petites galaxies compactes n'y sont pas (une source compacte **absente** de Gaia est un candidat « garder »). Donner au réseau, en plus de l'image, une carte des positions et éclats Gaia (un canal ou une carte de probabilité) lui permettrait de lever l'ambiguïté. À essayer en premier : un seul changement (le canal d'entrée), départ de E3 avec la première couche d'entrée adaptée, mesure sur les trois tests de `VALIDATION.md` (validation, contrôle des petites galaxies, étoiles saturées). Risques : le catalogue est incomplet pour les étoiles les plus brillantes (Alnitak n'y est pas) et pour les champs mal résolus astrométriquement ; une astrométrie fausse décale la carte (voir les images écartées du jeu de contrôle).

## 2. Petites galaxies compactes (jeu de contrôle des 18 galaxies)

**Limite.** N7, le mélange et N9 effacent presque toutes les petites galaxies compactes (médiane ≈ 0 au lieu de 1) ; un logiciel tiers de retrait d'étoiles les efface aussi. Ce défaut est **bien plus large que M32**.
**Déjà essayé et échoué.** Des cœurs synthétiques en plus (25 % : N7 ; 10 % : N8/N8L ; 40 % de cœurs *compacts* : N9) : M32 remonte (0,05 → 0,26) mais au prix d'étoiles gardées (G < 8 : 0,065 → 0,136) ; **la forme seule ne suffit pas**.
**Pistes.** (a) la carte Gaia en entrée (§1) ; (b) une population synthétique de petites galaxies variées (elliptiques, spirales, vues par la tranche) en plus des cœurs ; (c) un jeu de contrôle plus grand (18 galaxies sur 5 images est petit) ; (d) ne pas régler sur ce jeu.

## 3. Étoiles faibles, G 14 à 17

Entre 26 % et 30 % de leur flux restent après retrait. Diagnostic du 25 nov : nos cibles réelles retirent ces étoiles correctement, mais **52 à 64 % des étoiles G 14-17 sont à poids 0** au centre dans les paires (non enseignées). **Essayé** : N6 (poids 1 pour les étoiles G 14-17 bien retirées, 0 pour les restes) : G 14-17 monte à 0,351, N6 abandonné. **Pistes** : supervision par des étoiles faibles *synthétiques* (cible exacte) plutôt que par le réel ; un entraînement plus long sur ces seules étoiles ; vérifier que le critère « bien retirée » ne sélectionne pas les étoiles faciles.

## 4. Grosses étoiles et portée du réseau

Le réseau voit ≈ 180 px ; les halos des grosses étoiles en font ≈ 285. L'inférence à deux échelles V2 corrige cela (étape 8). **Pistes** : un réseau à plus grande portée (convolutions de Fourier, attention globale) ; intégrer V2 dans l'application (proposition faite, non intégrée) ; rayon du halo mesuré sans Gaia pour les étoiles que le catalogue ne contient pas. **Essayé sans effet** : grandes tuiles, halos synthétiques géants (N3, N4).

## 5. Données

- **Cibles « inévitables »** (sous un cœur saturé) : aucune vérité dans l'image ; la cible est un remplissage plausible. Question ouverte : quel poids leur donner (zéro, réduit, ou normal avec le meilleur remplissage possible).
- **Entrée linéaire ou débruitée ?** Le modèle d'étoile est exact sur le linéaire, le réseau travaille après débruitage. Jamais tranché ; piste : deux paires par image (débruitée et brute), déjà faite pour 35 tuiles.
- **Plus d'images et de types d'instruments** (champs denses, bande étroite, très faible signal) : l'étape 2 a produit une centaine de tuiles réelles et 38 tuiles web ; la chaîne coûte environ 4 minutes par tuile.
- **Cibles v33g** : la lumière diffuse remise est un lissage ; remplacer par un modèle plus fin du ciel.

## 6. Ce qui a déjà échoué : ne pas refaire

N3, N4, N6, remise de la part lente à l'inférence, N8/N8L, N9, mélanges N5/N6, E2/E3 seuls, partir de zéro : voir `LECONS.md`, tableau A. Règle : avant un essai, écrire « déjà essayé » et « ce qui diffère ».

## 7. Mesurer une amélioration honnêtement (résumé)

1. Un seul changement par essai, dans une copie.
2. Trois tests, pas un : validation (13 images : étoiles restantes par bande de magnitude, fond, structure, cœurs), contrôle des petites galaxies (galaxie vers 1, étoile de même brillance vers 0), étoiles saturées à faible signal (flux restant).
3. **Garde-fou des étoiles** : toute amélioration des cœurs ou des galaxies doit montrer que les étoiles brillantes (G < 8) et les étoiles saturées de faible signal ne remontent pas (N7 : 0 → 0,021 ; N9 : 0,065 → 0,136 : exemples de gains payés en étoiles).
4. Rapporter toutes les colonnes, y compris celles qui se dégradent ; regarder l'image.
5. Ne jamais régler sur le jeu de contrôle ; publier le pas, la recette et les empreintes.
6. Chaque donnée d'entraînement doit avoir une provenance écrite (`PROVENANCE.md`) : un réseau entraîné sur la sortie d'un autre logiciel ne démontre rien.
