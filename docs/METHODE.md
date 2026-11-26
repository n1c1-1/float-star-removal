# La méthode, étape par étape

Chaque étape dit **pourquoi** elle existe, **comment** elle se fait, quels sont les **paramètres clés** et quel **effet mesuré** a eu leur changement. Les dates sont celles du travail (heures du PC de Nicolas). Les noms de dossier de `scripts/` gardent la date d'origine (« Tour N7 – 25 nov »).
Les chiffres de mesure viennent de `resultats/` (protocole : `VALIDATION.md`). Ce qui a échoué est dans `LECONS.md`.

Vue d'ensemble : on veut un réseau qui reçoit une image linéaire en « unités de bruit » (chaque couleur moins son ciel, divisée par le bruit du ciel) et rend la même image sans étoiles. Le réseau voit un étirement fixe, `asinh(v / 3) / 10`, défait exactement en sortie.

---

## Étape 1 — E2 et E3 : entraînés sur données synthétiques seules (7 à 9 novembre)

**Pourquoi.** Premier point de départ **entièrement à nous** : aucune image, donc aucune question d'origine. Un fond est fabriqué, des étoiles y sont posées, la cible est le fond.

**Comment.**
- `scripts/make_dataset.py` fabrique le jeu à partir du code et de graines seulement. Il ne lit aucune image ; il lit seulement `catalogue.json` (fiches publiques de 43 télescopes/objectifs et 21 caméras). Son manifeste donne le sha256 de chaque fichier, et `--verify` refait des paires pour comparer.
- Les fonds viennent de `backgrounds.py` (nébuleuses en émission, galaxies de Sérsic, réflexion, poussière) et `defects.py` (marbrures, vignetage, gradient, donuts, bords d'empilement).
- Les étoiles viennent de `stars.py` et `psf.py` : forme calculée par l'optique de Fourier (pupille avec obstruction et araignée, aberrations de Zernike, turbulence, suivi, pixels), auréole en r⁻², aigrettes, reflets, saturation couleur par couleur.
- `scripts/train_e.py` entraîne le réseau de la chaîne (NAFNet, largeur 32, encodeur 2-2-4-8, milieu 12, décodeur 2-2-2-2, tuiles de 256 px, lot de 8, perte L1 sur l'étirement fixe). 60 % des tuiles reçoivent en plus des étoiles tirées à la volée.
- E2 (nuit du 7 au 8 nov) : paires régénérées avec les couleurs de saturation et les noyaux de galaxie. E3 (8-9 nov, `day_e3.py`) : paires « pairs_v4 » où les étoiles géantes (classe d'Alnitak) sont **gardées entières** dans la cible, saturation et extrémité brillante calées sur nos mesures ; reprise de E2 depuis ses meilleurs poids, pendant 6,5 h, taux d'apprentissage 3,7e-4, 950 paires d'entraînement et 25 tenues à l'écart.

**Paramètres clés et effets.** La part d'étoiles tirées à la volée (60 %) et le choix du pas : E3 pas 20 000 est celui retenu (c'est le point de départ de N5 et N7). Résultat : sur des images **réelles**, ces réseaux étaient 3 à 4 fois moins bons que sur le synthétique, creusaient les cœurs de galaxie, et le score synthétique trompait le choix de l'instantané (`LECONS.md`). D'où les étapes suivantes.

---

## Étape 2 — La chaîne classique v33 : soustraction guidée par Gaia, rebouchage, passes (20 au 23 novembre)

**Pourquoi.** Il faut des paires **réelles** (vraies images, vrais défauts) dont la cible soit fabriquée **sans réseau tiers**. On part de l'idée de Nicolas : « des images d'entraînement générables automatiquement par centaines, même si ça prend du temps ». La chaîne classique retire les étoiles par calcul : on sait où elles sont (Gaia), on ajuste un modèle d'étoile et de halo et on le soustrait. Elle est lente (≈ 4 minutes par tuile de 512 px) mais ne dépend de rien d'autre. Les dossiers `scripts/Methode v33b – 23 nov`, `Production v33 – 23 nov`, `Juge v33 – 23 nov`, `Juge v33d – 24 nov` en sont le code.

**Comment, dans l'ordre.**

1. *Fiche de l'image*, mesurée une seule fois sur l'image entière puis figée : astrométrie et catalogue Gaia profond (G < 21) hors ligne ; par couleur, largeur des étoiles, échelle des éclats, plateau de saturation, modèle d'étoile (au moins 30 étoiles isolées, rayon 28 px, aile prise sur les plus brillantes), angles des aigrettes de l'instrument.
2. *Piste 1 — soustraction.* Aux positions Gaia, seul l'éclat de chaque étoile est libre (photométrie forcée). Une étoile est retirée si son flux dépasse 1,5 fois son erreur ; la position est propre à chaque couleur si le pic dépasse 20 bruits ; deux étoiles ne sont ajustées ensemble qu'à moins de 2,5 magnitudes d'écart. **La piste 1 ne remplit rien** : elle rend aussi la liste des zones douteuses.
3. *Étape A — grosses étoiles.* Profil radial propre à chaque étoile ; forme étendue de l'image seulement pour 7 ≤ G ≤ 10,5 ; aigrettes aux angles de l'instrument (±2°), un gabarit par couleur et par filtre, suivies jusqu'à ≈ 0,45 bruit ; cœurs saturés recréés (niveau lisse du pourtour + grain du ciel).
4. *Étape B — dernière passe (« hybride »).* Reboucher ce qui reste, clair ou sombre, près des étoiles du catalogue et le long des aigrettes connues ; pixels isolés de 1 à 3 px toujours guéris ; grain de 2 à 4 px remis dans les seuls pixels changés. Les filaments fins de nébuleuse sont protégés (un reste allongé est refusé, sauf dans la bande d'une aigrette connue).
5. *Garde automatique par zone.* Rapport « lumière retirée / lumière attendue des étoiles du catalogue » : hors de 0,7 à 1,5, la zone est écartée.

**Le juge et les poids.** Une paire est `entrée`, `cible`, `poids`. Poids 1 = vrais pixels ou zone refaite et bonne ; 0,5 = réparation jugée bonne ; **0 = zone mal refaite**, que le réseau ne doit pas apprendre (anneaux, niveau faux, îlots, restes d'étoiles, halos ratés). Règles locales du juge v33 : la cible ne doit jamais être plus claire que l'entrée (tache ≥ 25 px ou écart > 5 bruits → poids 0, dilaté de 3 px) ; une étoile restée dans la cible (pic compact > 20 bruits à une position Gaia) → poids 0 sur un disque de rayon max(2,5 FWHM, reste mesuré, plafond 4 FWHM). Juge v33d (24 nov) : poids 0 sur le halo (rayon mesuré par le profil radial, plafond 40 px) et sur les étoiles faibles restées. Le juge ne fait que **mettre des poids à 0** : il ne modifie jamais la cible.

**Tuiles.** Pas de grille : 3 à 5 tuiles de 512 px **choisies** par image sur du contenu (nébuleuse, galaxie, étoiles de toutes tailles), jamais de ciel noir ; poids 0 là où la chaîne échoue. Les grosses étoiles (G ≲ 7) restent à poids 0 dans les paires réelles : elles sont apprises « à la perfection » par la synthèse. Deux versions de chaque entrée : débruitée et brute (la brute = entrée brute moins le même calque d'étoiles).

**Paramètres clés et effets mesurés (journal des essais v1 à v33).**
- Catalogue Gaia profond (G < 21) au lieu de détecter les faibles dans le résidu : étoiles faibles restées de 22 % à 2-3 %.
- Modèle d'étoile bâti sur l'image entière (au lieu de 9 étoiles par découpe) : corrige l'aile, fait disparaître les anneaux ; flux 11 fois trop fort évité par la fiche figée.
- Couples ajustés seulement à moins de 2,5 mag : l'anneau des G 12-14 disparaît ; zones « cœur résiduel » rebouchées : dipôles de 57 à 7.
- Dernière passe dans la bande des aigrettes (v21) : les pointillés d'aigrettes disparaissent, filaments et nébuleuse inchangés.
- v33 (23 nov) : disques bleus, pâtés rouges et magenta supprimés par un grain mis à l'échelle ; production 247 s par tuile (1 travailleur carte pour la piste 1 + 3 travailleurs processeur).
- Production sur les images de Nicolas : une centaine de tuiles, chaque planche revue à l'œil (15 paires lâchées) ; 109 paires débruitées et 36 brutes gardées ; en moyenne ≈ 21 % des pixels à poids 0.
- **Images web** (APOD), 24 nov : délinéarisées (étirement inverse estimé sur les étoiles, méthode publiée « courbe de réponse radiométrique »), astrométrie, Gaia profond, puis la même chaîne (variante v33b) : 65 tuiles produites, 38 gardées de 13 images.

Provenance exacte de chaque cible : `PROVENANCE.md`.

---

## Étape 3 — Cibles v33g : remettre la lumière diffuse lente (25 novembre)

**Pourquoi.** Diagnostic du 25 nov : la piste 1 soustrait le modèle de **toutes** les étoiles du catalogue profond, ailes comprises jusqu'à 150 px, même dans les zones « non touchées ». Les cibles étaient donc plus sombres que l'entrée dans le ciel diffus ; le réseau apprenait à effacer la lumière diffuse lisse (disque de M31, cœur de M65/M66).

**Comment.** `scripts/Tours N5 N6 – 25 nov/preparer_g_h.py g` : D = entrée − cible ; M = pixels de classe 0 (non touchés) hors disques des étoiles G < 8 ; L = lissage gaussien normalisé de D sur M (sigma 40 px, par canal) avec élagage itératif (4 tours, pixels écartés à plus de 2,5 écarts robustes) ; cible v33g = cible + L.

**Effet.** Médiane de (entrée − cible) sur M : +0,12 → −0,05 bruit. N5 (cibles v33g) contre N2 (cibles v33d) : étoiles G 14-17 0,313 → 0,265, fond 0,0015 → 0,0011, p99 du fond 0,12 → 0,07 ; structure 0,005 → 0,007 (légèrement plus haute).

---

## Étape 4 — Étoiles synthétiques réalistes et recette N2 (22 au 25 novembre)

**Pourquoi.** Il faut des paires réelles **et** des étoiles synthétiques posées sur de vrais fonds. Il faut des étoiles synthétiques **très réalistes**, car sous un cœur saturé la vérité n'existe dans aucune vraie image.

**Comment.**
- `scripts/Etoiles synthetiques – 22 nov/etoiles2.py` : générateur v2, 8 tours de réglage (22 nov) avec un test à l'aveugle (un juge devait distinguer vraies et fausses vignettes ; l'objectif est un taux d'erreur du juge voisin de 50 %, c'est-à-dire des étoiles indiscernables ; les notes de chaque tour sont dans `scripts/Etoiles synthetiques – 22 nov/score_aveugle.py` et ses fichiers de score, et la leçon sur ce test est dans `LECONS.md`) ; banques de formes d'étoiles réelles brillantes (médianes de 5 à 8 étoiles mesurées dans des images **avec** leurs étoiles) ; 19 ms par tuile de 256 px.
- Recette N2 (24 nov, `train_nuit.py --n2`) : poids de 0,5 portés à 1 ; sur 50 % des tuiles synthétiques, 1 ou 2 étoiles très brillantes imposées (pic de 300 à 300 000 bruits, gros halo, aigrettes) posées sur un fond de nébuleuse réel (là où la cible est au poids 1 sur un disque de 40 px) ; tirage 50 % débruitées, 25 % brutes, 25 % web ; départ E3 pas 20 000 ; lot 8, taux d'apprentissage 5e-4, montée de 1 000 pas puis cosinus ; 60 % de tuiles avec étoiles synthétiques.

**Paramètres clés.** Part de tuiles synthétiques (60 %), part d'étoiles très brillantes imposées (50 % des synthétiques), poids (0,5 → 1 : sans effet visible sur les étoiles faibles, N1 = N2 à 0,312 / 0,313).

---

## Étape 5 — Cœurs de galaxie synthétiques (N7, 25-26 novembre)

**Pourquoi.** N5 efface les cœurs de galaxie : le flux au cœur de M31 passe de 369 à 5-7 (cœurs 0,19). Hypothèse retenue : le réseau prend un cœur de galaxie ou une nébuleuse planétaire pour une étoile, faute d'exemple d'objet étendu compact **conservé** dans la cible.

**Comment.** `scripts/Tour N7 – 25 nov/paires_synthetiques_n7.py` (`coeur_synthetique`) : sur 25 % des tuiles synthétiques, 1 ou 2 cœurs (Sérsic n 1-4, rayon effectif 4-60 px, ellipticité 0-0,6, pic 50-5 000 bruits, 30 % écrêtés, galaxie jaune-blanc ou planétaire bleu-vert, largeur à mi-hauteur ≥ 2,5 fois celle des étoiles de la tuile) présents **à l'identique dans l'entrée et la cible**, poids 1.

**Effet.** Cœurs 0,19 → 0,81. Coût : étoiles G < 8 0 → 0,021 ; G 11-14 0,072 → 0,094 ; G 14-17 0,265 → 0,283 ; fond médian 0,0011 → 0,0023. Gains : p99 du fond 0,07 → 0,03, structure 0,007 → 0,004. **Effet secondaire découvert le 26 nov** : parce que les cœurs synthétiques incluent des cœurs écrêtés de 50 à 5 000 bruits, N7 garde aussi des étoiles saturées de faible rapport signal/bruit (crête plate à ≈ 100 bruits) : conflit « étoile saturée / cœur compact » (`AMELIORER.md`).

---

## Étape 6 — Affinage depuis E3

**Pourquoi.** Partir de zéro sur les paires réelles est faible (24 nov : essai depuis zéro moins bon que depuis E3) ; E3 connaît déjà la forme générale d'une étoile. **Comment.** Tous les entraînements N1 à N9 partent des poids E3 pas 20 000 (`--depart`). N5/N6 : 34 500 pas (≈ 5 h, 9 minutes pour 1 000 pas), instantané tous les 1 000 pas, **meilleur pas = plus petite erreur « sous les étoiles »** sur 6 tuiles tenues à l'écart (N5 : pas 23 000 ; N6 : 20 000 ; N7 : 22 000). La validation devient plate à partir d'environ 22 000 pas : un tour plus long ne sert à rien (N8L, 24 000 pas, n'a pas fait mieux que N7).

---

## Étape 7 — Mélange de poids entre réseaux de même départ

**Pourquoi.** Chaque entraînement a des défauts différents ; la moyenne de leurs poids est un réseau de plus, **gratuit** (aucune donnée, aucun entraînement). **Comment.** `scripts/Melanges N5 N6 N7 – 26 nov/chaine_melanges.py` : a·A + (1 − a)·B sur les 664 tenseurs flottants (même architecture, mêmes clés et formes, tampons entiers pris de A). **Règle** : mélanger seulement des réseaux partis du même point (E3) ; entre départs différents cela ne marche pas.

**Effet** (voir `resultats/tableau_melanges.md`) : N7/N5 0,75 → cœurs 0,798, étoiles G < 8 0,021, fond 0,0018, structure 0,003 (léger gain sur N7 seul : fond 0,0023 et structure 0,004) au prix de G 14-17 un peu plus hautes (0,295 contre 0,283). N7/N5 0,5 : G < 8 à 0,000 mais cœurs 0,653. Les mélanges N5/N6 gardent le cœur effacé (0,13 à 0,16).

---

## Étape 8 — Inférence à deux échelles V2 (25 novembre)

**Pourquoi.** Sur M27, la très grosse étoile (G 5,6, halo de 285 px) devenait un **disque plat gris-bleu** : la portée du réseau (≈ 180 px) est plus petite que le halo, et de grandes tuiles n'y changent rien (essayé : résidu identique). Réduire toute l'image ×2 fait disparaître le disque mais abîme le reste.

**Comment.** `scripts/Portée M27 – 25 nov/deux_echelles_v1v2.py` et `scripts/Tours N5 N6 – 25 nov/inferer_v2.py` : pour chaque étoile Gaia de G < 6,5, on mesure le rayon du halo dans l'entrée (premier rayon où le profil radial médian moins le fond local passe sous 2 bruits ; fond = médiane de la couronne 400-500 px, sinon 300-400 ; plafond 300 px). Dans ce rayon, la sortie est celle du réseau appliqué à l'image réduite ×2 puis ré-agrandie ; raccord : plein jusqu'à 70 % du rayon, cosinus jusqu'à 100 %.

**Effet.** Étoiles G < 8 restantes 0,012 → 0,000 ; fond et structure inchangés (structure 0,005 → 0,009 dans la version à formule fixe, rétablie à 0,005 avec le rayon mesuré V2) ; M27 propre. Part de l'image remplacée : 12,6 % pour M42 (V2) contre 49,5 % avec la formule de rayon en racine du flux.

---

## Ce qui n'a pas été retenu

Voir `LECONS.md` : N3/N4 (grands halos synthétiques), N6 (poids v33h), remise de la part lente à l'inférence, N8/N8L (cœurs à 10 %), N9 (cœurs compacts), mélanges N5/N6.
