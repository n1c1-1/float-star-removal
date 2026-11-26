# Les consignes données aux agents

## Le rôle de chaque acteur

Ce travail a été conduit par **Nicolas** avec un assistant d'intelligence artificielle (Claude, Anthropic) en **session principale** : elle décide, découpe le travail, juge les résultats à l'œil et rend compte à Nicolas. Elle ne lance pas elle-même les longs calculs : elle confie chaque étape à un **agent** (une autre instance de l'assistant) à qui elle donne une **consigne écrite**. L'agent exécute (écrit un script, lance un calcul, mesure, regarde des planches), puis rend un rapport court. La session principale lit le rapport, regarde les images, et décide de la suite. Les décisions de fond (quelles données, quels critères, quand s'arrêter) viennent de Nicolas ; les agents ne décident pas.

Ces consignes sont publiées **telles qu'elles ont été données**, après nettoyage, pour que l'on puisse voir exactement ce qui a été demandé, avec quelles règles.

## Périmètre : la lignée publiée seulement

Ne sont publiées que les consignes de la **lignée des modèles publiés** : la chaîne classique de retrait guidée par Gaia (20-24 nov), les paires v33/v33g (images de Nicolas et images web), les entraînements N1 à N9 depuis E3, les mélanges, l'inférence à deux échelles, les mesures et contrôles. Les consignes d'autres travaux (autres réseaux, autres jeux de paires, interface de l'application) ne font pas partie de la méthode publiée et ne sont pas ici. Les consignes qui ont lancé les entraînements E2 et E3 (7-9 nov, données synthétiques seules) n'ont pas été retrouvées parmi les appels d'agent des transcriptions : ces étapes ont été pilotées directement dans la session principale ; leurs scripts sont dans `scripts/`.

## Contenu

`AAAA-MM-JJ/NN_HHMM_titre.md` : une consigne par fichier, groupées par jour, numérotées dans l'ordre du jour. L'en-tête donne le titre (la description de l'appel d'agent, en anglais ou en français), la date et l'heure de Paris, le type d'agent.

| jour | consignes | sujet |
|---|---|---|
| 2026-11-20 | 7 | première chaîne classique, pistes 2 et 3, catalogue Gaia profond, premiers juges |
| 2026-11-21 | 36 | versions v1 à v18 de la chaîne classique (un changement par consigne) |
| 2026-11-22 | 13 | images de test, générateur d'étoiles synthétiques, lot de validation, premier entraînement sur paires réelles |
| 2026-11-23 | 9 | correctifs de la chaîne, choix des tuiles, délinéarisation, production v33 |
| 2026-11-24 | 5 | juge v33c/v33d, essai de 2 h, images web |
| 2026-11-25 | 8 | grands halos (N3), diagnostic, N5/N6, cœur de M31, N7, remise lente |
| 2026-11-26 | 8 | mélanges, N8, N8L, N9, diagnostics, jeu de contrôle |

Total : **86 consignes**, du 20 au 26 novembre 2026.

## Ce qui a été retiré ou remplacé

- le nom du logiciel commercial de comparaison (écrit en un seul mot) : « [outil de comparaison] » ; les consignes dont le seul objet était de le faire tourner ne sont pas publiées ;
- l'expression « réseau étoiles » (écrite parfois « star net » dans les consignes d'origine) désigne **notre** réseau, jamais un outil tiers ;
- les logiciels tiers (traitement d'images, empilement) : « [logiciel tiers] » ;
- les adresses électroniques, pseudonymes de forum, comptes et dépôts personnels, noms de personnes autres que Nicolas, crédits d'auteurs des images web ;
- les chemins absolus : `<SCRIPTS>`, `<DONNEES>`, `<ESSAIS>`, `<FLOAT>`, `<DISQUE>` ;
- les lignes de co-signature de commit ;
- les lignes qui renvoyaient à d'autres jeux de données hors lignée : « [passage retiré] ».

Le fond de chaque consigne (but, règles, étapes, critères d'arrêt) est inchangé.

## Comment les lire

Les consignes donnent presque toujours : le but et la raison (« POURQUOI »), les règles strictes (où écrire, quelles ressources, quoi ne pas toucher), les étapes, et le format du rapport attendu. On y voit aussi la règle qui a structuré le travail : *un seul changement à la fois, et écrire avant « déjà essayé » et « ce qui diffère »* (`docs/LECONS.md`). Le mot « Fable » désigne un agent de « recul à froid » à qui l'on demande un avis indépendant après plusieurs essais sans progrès.
