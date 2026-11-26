# Légendes des images d'exemple

Ces images sont **entièrement synthétiques** : aucune image réelle n'est lue pour les fabriquer. Elles sont refaites
exactement (bit à bit) par `exemples/generer_exemples.py` (les graines sont écrites en tête du script ; les empreintes
sha256 sont dans `exemples/EMPREINTES.txt`, les paramètres tirés dans `exemples/PARAMETRES_TIRES.txt`).

## Comment lire une image

Chaque image a trois panneaux : **Entrée (avec étoiles) | Cible (sans étoiles) | Différence = étoiles retirées**.

- Les trois panneaux ont le **même étirement**, celui du réseau : `asinh(v / 3) / 10`, où `v` est la valeur linéaire en
  unités de bruit du ciel (le ciel vaut 0, son bruit vaut 1), multiplié pour l'affichage par un gain fixe (4,0) et décalé
  d'un noir fixe (0,04). Le panneau « différence » est affiché avec la même fonction, appliquée à entrée moins cible.
- La planche `planche_synthetique.jpg` assemble les six exemples (une ligne par exemple, trois colonnes) ; ses tuiles sont
  réduites à 250 px de côté (moyenne par blocs), ce qui lisse un peu le bruit par rapport aux images individuelles.

## Pourquoi la cible est exacte par construction

1. **On connaît le fond avant d'y poser les étoiles.** Le fond (le ciel sans étoile, avec son bruit) est fabriqué en
   premier. Les étoiles sont ensuite ajoutées par-dessus : l'entrée est « fond + étoiles ».
2. **La cible est ce fond, copié tel quel.** Le script le vérifie à chaque exécution (`cible == fond`, égalité des
   tableaux, sans tolérance ; pour l'exemple 4, `cible == fond + cœur de galaxie`, le cœur étant ajouté avant les étoiles).
3. **Le bruit du fond est le même dans l'entrée et dans la cible** : le bruit du ciel (corrélé entre pixels voisins) et le
   bruit de photons de la lumière du fond font partie du fond. Seules les étoiles apportent leur propre bruit de photons,
   et il n'est que dans l'entrée. La différence entrée moins cible ne contient donc que les étoiles (cœur, auréole,
   aigrettes, plateau d'écrêtage, bruit de photons de leur lumière).
4. **La cible n'est jamais calculée par un autre logiciel** : ni retrait d'étoiles, ni correction manuelle, ni estimation du fond. Elle
   est la donnée de départ, pas un résultat.

## Ce qui est généré (commun aux six exemples)

- **Fond** : `scripts/backgrounds.py`, fonction `background`, générateur `numpy.random.default_rng([graine_fond, 0])`
  (nébuleuses, galaxies, poussière, niveau du ciel et léger gradient), puis le bruit du ciel (générateur
  `[graine_fond, 3]`) : bruit corrélé de corrélation 0,09 à 0,42 entre pixels voisins, divisé par la racine de 2 (empilement
  complet de deux demi-empilements), plus le bruit de photons de la lumière du fond.
- **Étoiles** : `scripts/Etoiles synthetiques – 22 nov/etoiles2.py` (styles d'image : instrument tiré de
  `scripts/catalogue.json`, optique de Fourier, auréole, aigrettes, plateau d'écrêtage par étoile, couleurs, bruit de photons)
  mis en cache en 16 styles (graine 2026), puis la pose rapide `paire_synthetique` (comptes en loi de puissance, 50 à 6000
  étoiles par million de pixels au-dessus d'un bruit, couleurs de nos champs, cœurs par convolution, halo et aigrettes des
  étoiles de plus de 20 bruits).
- **Sans banque de formes mesurées.** Le générateur sait aussi emprunter des formes d'étoiles mesurées sur de vraies
  images ; ces banques ne sont pas dans le dépôt. Ces exemples utilisent donc **uniquement le modèle paramétrique** : les
  très brillantes ont des plateaux d'écrêtage en forme de carré arrondi et des aigrettes tirées par formule, moins variés que
  dans une vraie image.
- **Version bruitée seulement** : le chemin « paires débruitées » (qui demande le débruiteur de FLOAT) n'est pas utilisé.
- Calcul sur processeur, 4 fils ; aucune image lue, aucun accès réseau.

## Les six images

### exemple_01_champ_lisse.jpg — champ d'étoiles sur fond lisse
- Tuile 256 x 256 px, 3 couleurs. Graine du fond 1114, graine des étoiles 1114.
- Fond : nébuleuse en émission très faible (niveau du 99,9e centile 1,5 bruit), gradient 0,26 ; bruit corrélé 0,125, photon_k 0,108.
- Étoiles : style d'image n° 2 (largeur à mi-hauteur 3,29 px, 2 aigrettes, couleur), densité tirée par la graine, aucune étoile imposée.
- À voir : des dizaines d'étoiles nettes, de la faible à la brillante, sur un ciel presque vide ; la cible ne garde que la faible nébuleuse.

### exemple_02_tres_brillantes.jpg — étoiles très brillantes sur nébuleuse
- Tuile 384 x 384 px. Graine du fond 1203, graine des étoiles 1203, module `paires_synthetiques_n7.py`.
- Fond : nébuleuse en émission (99,9e centile 27,8 bruits), front d'ionisation, bruit corrélé 0,367, photon_k 0,272.
- Étoiles : style n° 15 (3,74 px, 4 aigrettes) ; deux étoiles très brillantes **imposées** (paramètre `extras`) : pic 30 000 bruits en
  (110, 120) et pic 9 000 bruits en (290, 90) ; le champ d'étoiles libres est tiré par la graine.
- À voir : les deux étoiles écrêtées (plateau en carré arrondi) posées sur la nébuleuse ; la cible montre la nébuleuse intacte sous elles.

### exemple_03_halo_aigrettes.jpg — étoile avec grand halo et aigrettes
- Tuile 384 x 384 px. Graine du fond 1304, graine des étoiles 1304, module `paires_synthetiques_n3.py` (grands halos).
- Fond : nébuleuse en émission avec filaments, poussière en absorption (tau 1,73 au 99e centile), bruit corrélé 0,246, photon_k 0,148.
- Étoiles : style n° 15 (3,74 px, 4 aigrettes) ; une étoile imposée au centre (192, 192), pic 37 000 bruits, avec un halo
  à l'échelle des plus grosses étoiles de nos images : 38 bruits à 60 px, 1 bruit à 285 px (pente 2,33), quatre aigrettes
  portées jusqu'à 1,25 x 285 px.
- À voir : le halo recouvre presque toute la tuile ; la cible retrouve la nébuleuse cachée dessous (c'est l'exemple le plus dur).

### exemple_04_coeur_galaxie.jpg — cœur de galaxie gardé
- Tuile 384 x 384 px. Graine du fond 1407, graine des étoiles 1407, `paire_synthetique_coeurs` avec `p_coeurs = 1` (un cœur toujours présent).
- Fond : nébuleuses par réflexion très faibles (99,9e centile 1,2 bruit), bruit corrélé 0,345, photon_k 0,023.
- Cœur synthétique (présent à l'identique dans l'entrée ET dans la cible) : profil de Sérsic n = 2,61, rayon effectif 13,9 px,
  ellipticité 0,345, éclat central 123 bruits, couleur de galaxie (jaune-blanc), non écrêté, largeur à mi-hauteur 10,6 px.
- Étoiles : style n° 10 (3,61 px, sans aigrette, mode bande étroite), champ dense tiré par la graine.
- À voir : le cœur, compact et brillant comme une étoile mais étendu, reste dans la cible ; la différence ne contient que des
  étoiles (aucune trace du cœur). C'est le cas qu'un réseau qui n'en a jamais vu confondrait avec une étoile.

### exemple_05_champ_dense.jpg — champ dense
- Tuile 256 x 256 px. Graine du fond 1501, graine des étoiles 1501.
- Fond : petite nébuleuse en émission avec une composante d'oxygène (99,9e centile 5,4 bruits), bruit corrélé 0,194, photon_k 0,051.
- Étoiles : style n° 15 (3,74 px, 4 aigrettes), forte densité d'étoiles de brillance moyenne à faible tirée par la graine.
- À voir : des dizaines d'étoiles se chevauchent parfois ; la cible garde la nébuleuse et son bruit à l'identique.

### exemple_06_fond_defauts.jpg — fond avec défauts d'empilement (jeu 2)
- Tuile 256 x 256 px. Graine du fond 1611 (les défauts : `default_rng([1611, 1])`), graine des étoiles 1611.
- Fond : nébuleuse en émission faible plus la couche de défauts de `scripts/defects.py` : grain 0,010 ; marbrures de 8,5 px
  d'amplitude 0,107 ; grande échelle 0,091 (gradient, vignetage, lueur) ; 1 anneau de poussière (donut) ; 2 bords d'empilement (une
  bande plus bleue sur le bord droit) ; teinte du ciel décalée (R - V = -0,32, B - V = -0,10).
- Étoiles : style n° 11 (4,95 px, sans aigrette, couleur), quelques étoiles tirées par la graine, dont une brillante.
- À voir : les défauts font partie du fond, donc de la cible : un réseau doit enlever les étoiles sans lisser ni corriger le ciel.

## Limites

- Les formes d'étoiles viennent du modèle paramétrique seul (pas de banque de formes mesurées, pas de version débruitée).
- L'étirement est un étirement d'affichage ; les tableaux d'origine sont en unités de bruit, linéaires, en virgule flottante 32 bits.
