# Ce qui a échoué, et pourquoi

Pour ne pas refaire les mêmes essais. Chaque ligne dit ce qui a été essayé, ce qui s'est passé, et la leçon. Dates : novembre 2026. Les essais se lisent avec la règle « un seul changement à la fois, et écrire avant ce qui a déjà été essayé et en quoi le nouvel essai diffère ».

## A. Les réseaux

| essai | date | ce qui a été fait | résultat | leçon |
|---|---|---|---|---|
| E2, E3 (synthétique seul) | 7-9 nov | entraînement sur paires 100 % générées | 3 à 4 fois pire sur de vraies étoiles ; cœurs de galaxie creusés ; le meilleur instantané du score synthétique n'était pas le meilleur sur le réel | un jeu synthétique seul ne suffit pas : il faut du réel ; le choix de l'instantané se fait sur des images réelles tenues à l'écart |
| A (E3 affiné sur paires réelles) et B (depuis zéro), 22-23 nov | 22-23 nov | 2 h sur ≈ 99 paires réelles du jeu 1, B partant de poids aléatoires | A meilleur que B (erreur sur paires tenues 0,00108 contre 0,00117) ; mais fond abîmé (0,0033 contre 0,0005) et structure brillante dégradée ; paires pleines de défauts (disques saturés, anneaux, trous) | partir de E3 ; contrôler **chaque** paire à l'œil ; les tuiles « sur grille » sans regarder le contenu donnent des champs denses presque tous à poids 0 |
| Piédestal lisse (essai C) | 23 nov | ajout d'un piédestal lisse identique à l'entrée et à la cible | arrêté avant mesure ; le défaut était la cible plus sombre que l'entrée (voir v33g) | mesurer le défaut avant de l'« enseigner » |
| N1/N2 (nuit du 24 nov) | 24 nov | paires v33d + web, poids 0,5 → 1, très brillantes imposées | étoiles brillantes bien retirées (G < 8 à 0,000), mais cœurs de M31 effacés, fond et structure plus abîmés, moitié des étoiles G 14-17 sans supervision (poids 0) | cibles plus sombres que l'entrée dans le ciel « non touché » (ailes soustraites jusqu'à 150 px) : corrigé par v33g |
| **N3** (grands halos synthétiques) | 25 nov | générateur de halos jusqu'à 285 px, 25 % des tuiles synthétiques ; juge v33e qui mesure le halo sur l'image | le disque plat de M27 reste identique ; le juge v33e prenait des nébuleuses entières pour des halos (+49 points de poids 0 sur la nébuleuse planétaire NGC 6543) | le problème n'est pas dans les données de grands halos : c'est la **portée** du réseau (voir inférence V2) |
| **N4** (juge par magnitude Gaia, 50 % de grands halos) | 25 nov | rayon du halo par loi en magnitude G, garde-fous | le juge change à peine les poids (+0,06 point) ; M27 inchangé | même conclusion ; abandon de la piste « données de halo » |
| Inférence en grandes tuiles, M27 | 25 nov | tuiles les plus grandes possibles, recouvrement 50 % | résidu identique ; le disque persiste | la portée du réseau est la limite |
| Réduction ×2 ou ×3 de toute l'image | 25 nov | image réduite puis ré-agrandie | le disque disparaît, mais le reste est abîmé (la réduction supprime aussi du signal) | n'utiliser la réduction que **dans la zone** des grosses étoiles : inférence V2 |
| **N6** (poids v33h : étoiles G 14-17 réparées → 1, restes > 10 % → 0) | 25 nov | N5 avec les poids v33h | G 14-17 plus hautes (0,351 contre 0,265), fond 0,0027 | donner plus de poids aux étoiles faibles de nos cibles les a dégradées ; N6 abandonné, N5 gardé |
| **Remise de la part lente à l'inférence** | 25 nov | N5 puis ajout de la lumière diffuse lissée (sigma 40) à la sortie | structure 0,007 → 0,004, mais fond médian 0,0011 → 0,0026 et cœurs inchangés (trou bleu-noir de M31 identique, M32 effacée) | la correction ne touche pas le défaut principal ; à ne pas retenter sans revoir l'écrêtage (ailes G 8-14, grandes structures) |
| Mélanges N5/N6 (0,5 et 0,75) | 26 nov | moyenne de poids | cœurs 0,13 à 0,16 : le cœur effacé reste effacé | un mélange ne rend pas ce qu'aucun des deux ne sait faire |
| **N8** (cœurs 10 %, tour court 10 000 pas) puis **N8L** (24 000 pas) | 26 nov | un seul changement : 10 % de cœurs au lieu de 25 % | cœurs ≈ 0,80 quelle que soit la part (25 % ou 10 %) ; étoiles plus hautes au tour court ; N8L pas mieux que N7 | la part de cœurs n'est pas le levier ; diagnostic : un cœur compact (M32, Canon 60D) est pris pour une étoile |
| **N9** (40 % de cœurs compacts) | 26 nov | cœurs de 1,3 à 1,5 fois la largeur des étoiles, Sérsic n 3-6 | M32 0,05 → 0,26, cœurs 0,81 → 0,87, MAIS étoiles gardées : G < 8 0,065 → 0,136, G 8-11 0,053 → 0,109, G 14-17 0,253 → 0,348 ; petites galaxies du jeu de contrôle toujours effacées | la forme seule ne distingue pas une étoile d'un cœur compact et coûte des étoiles ; piste : carte Gaia des étoiles connues en entrée |

## B. La chaîne classique (soustraction guidée par Gaia)

Les mêmes erreurs se sont répétées : le 22 nov, 24 essais ont refait la méthode du 20 nov, déjà jugée en échec. D'où la règle d'écrire « déjà essayé / ce qui diffère » avant chaque essai.

**Étoiles faibles.**
- Marche : le catalogue profond (G < 21) au lieu de détecter dans le résidu ; limiter les mesures annexes à G 17,5 et retirer jusqu'à G 21.
- Ne plus faire : assouplir la détection hors catalogue sur une image en filtres assemblés (des nœuds de nébuleuse sont pris pour des étoiles) ; régler la profondeur sur un « gain » mesuré quand le catalogue couvre toute l'image (artefact).

**Étoiles moyennes (anneaux, dipôles, trous).**
- Marche : modèle d'étoile sur l'image entière ; position propre à chaque couleur ; couples à moins de 2,5 mag ; PSF refaite sans les brillantes déjà retirées ; passe finale sur les « cœurs résiduels ».
- Ne plus faire : remplir seul dans un champ dense ; une largeur commune par tranche d'éclat ; un décalage de couleur global ; une PSF par couleur ; élargir les disques de remplissage contre un anneau (il repousse l'anneau à chaque passe).

**Aigrettes.**
- Marche : angles de l'instrument mesurés sur l'image entière, fixes à ±2° ; un gabarit par couleur ; suivi jusqu'à ≈ 0,45 bruit ; restes soustraits plutôt que recréés ; la passe finale qui rebouche les restes dans la bande d'une aigrette connue.
- Ne plus faire : un gabarit en boîte (rectangles colorés) ; forcer une largeur ; empiler une image d'étoile commune (**la mesure s'améliore pendant que l'image se dégrade**) ; mesurer l'aigrette étoile par étoile près du bruit ; supposer une croix à 90° et quatre aigrettes au plus (faux : M8 en a six).

**Grosses étoiles saturées (G ≲ 7).**
- Ne plus faire : soustraire le halo et les aigrettes par un modèle propre à l'étoile **et ne recréer que le cœur** : au fort étirement il reste des barres, un disque strié et un disque gris plat ; profil étendu partagé (disques magenta) ; règle « source étendue » sur une étoile écrêtée (son pic tronqué la déclenche toujours).
- Marche : le profil propre par étoile ; la saturation lue sur l'image entière ; cœurs recréés par fond harmonique jamais plus bas que leur pourtour ; plateau saturé **mesuré par étoile** (chaque étoile est écrêtée à son propre niveau) ; portée du halo **mesurée** (jusqu'à 520 px) au lieu d'une formule qui la coupait à 250-300 px.
- Limite : sous une étoile de magnitude 5, la vérité n'existe pas dans l'image. Les étoiles de la classe d'Alnitak sont presque impossibles à retirer proprement ; on marque leur disque central « à ignorer » (poids 0) et on garde halo et rayons au-delà (vrais pixels).
- Leçon générale : un essai rejeté peut redevenir bon quand un défaut en amont est corrigé : le noter « rejeté dans telles conditions », pas « jamais ».

**Nébuleuse.** Ne plus faire : recréer les grandes zones (31 % inventés, frontière nébuleuse / voie sombre perdue) ; combler les creux larges sans condition. Marche : refaire les creux larges sauf s'ils sont déjà sombres dans l'entrée ; juger un reste « réel » contre la structure locale, pas contre le bruit du ciel.

**Vitesse.** Portage de la piste 1 sur carte graphique : ≈ 5 fois plus rapide, résultat identique au bit ; deux calculs sur la carte en même temps se ralentissent de 10 à 20 fois.

## C. Juger et mesurer : ce qui a trompé

- Une **mesure qui s'améliore pendant que l'image se dégrade** (aigrettes, z3/z4). Toujours regarder l'image avant de croire la mesure.
- Une part inventée basse prise pour de la qualité ; le pixel-diff entre versions (un seul creux écarté change 65 % des pixels) ; des tolérances en bruit du ciel mesurées sur une nébuleuse (elles mesurent la nébuleuse).
- **Interprétation des colonnes.** « fond » est |sortie − entrée| en **unités de bruit** loin des étoiles : 0,001 est mille fois sous le bruit, invisible. Ce n'est pas un écart à corriger. Les vrais écarts étaient les cœurs (0,80 pour un idéal de 1) et les étoiles faibles.
- **Le juge aveugle** est un argument, pas la vérité : ses *raisons* ne sont pas des causes. Sur les étoiles synthétiques, trois corrections tirées de ses remarques (aigrettes amincies, raccord de halo, rond lissé) donnaient un meilleur taux, mais retirées une à une chacune nuisait ; il faut mesurer chaque changement seul, à juge frais.
- **Un jeu de validation incomplet cache des défauts.** Le 26 nov, une image réelle de la Tête de Cheval (étoiles saturées à faible signal sur bruit, crête plate à ≈ 100 bruits) a montré que N7 les garde : la validation ne contenait aucune étoile saturée à ce niveau (860 à 12 500 bruits seulement). Élargir la validation avant de conclure.
- **Les agents laissent des erreurs.** Les scripts de mesure N8L et N9 ont eu 3 relances à cause de noms indéfinis : exiger un contrôle de noms (py_compile ne suffit pas, il faut un contrôle des noms non définis) avant tout lancement long.
- **Le matériel compte.** Un entraînement a été arrêté par manque de mémoire vive (le garde-fou arrête proprement sous 3,5 Go libres et reprend au-dessus de 8 Go : `Tour N8L – 26 nov/train_n8l.py`).

## D. Les règles de méthode qui en sont sorties

1. Un seul changement par essai, dans une copie ; jamais de modification sur place d'un script validé.
2. Écrire avant l'essai : « déjà essayé », « ce qui diffère », ce que le changement risque de casser ailleurs.
3. Après 3 essais ciblés sans que le défaut bouge, prendre du recul au lieu d'un 4e essai.
4. Tours courts (≈ 10 000 pas) pour choisir une direction, mesure intermédiaire tous les 5 000 pas **dans** le processus d'entraînement, règle d'arrêt écrite avant le lancement, tour long seulement pour la recette gagnante (plafonné à ≈ 24 000 pas).
5. Poids 0 **local**, jamais par catégorie entière : ni « tout à zéro », ni « tout parfait ».
6. On ne mélange que des réseaux partis du même point.
