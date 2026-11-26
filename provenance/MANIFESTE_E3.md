# Manifeste E3 : un réseau entraîné sur des données générées seulement

E3 est le point de départ de N5 et de N7 (pas 20000 : `train/E3/snap/step_20000.pt`). Ce fichier recopie ce qui
sert de preuve, sans copier les données. Les chemins sont donnés avec les racines `<ESSAIS>` (dossier des essais),
`<SCRIPTS>` (dossier des scripts) et `<DONNEES>`.

## 1. Filiation des poids

| réseau | départ | paires lues | durée | empreinte sha256 du fichier de poids repris ensuite |
|---|---|---|---|---|
| E | poids aléatoires (journal : « random weights ») | `<ESSAIS>/starRemoval/pairs_v2` (générées) | 7 nov, 17 h 20 -> | `train/E/snap/step_16000.pt` : f30192c5aacefede04085e1e5cc5106874c8076685152bf1f00a6f24e2eb9145 |
| E2 | E, pas 16000 | `<ESSAIS>/starRemoval/pairs_v3` (générées) | 7 nov, 22 h 28 -> | meilleur point `train/E2/best.pt` (sans empreinte dans ce document) |
| E3 | E2, meilleur point | `<ESSAIS>/starRemoval/pairs_v4` (générées) | 9 nov, 01 h 33 -> 08 h 03 (36000 pas) | `train/E3/snap/step_20000.pt` : 9faf2dc4be332c9d389942078a019ce86248a0dbe8b1d4a32eb57284bdc2e129 |

Dans les trois entraînements, 60 % des tuiles reçoivent en plus des étoiles tirées au vol par `train_e.synthetic_stars`
(PSF analytique, aucune image). Les journaux (`log.txt`) de E, E2 et E3 ne contiennent pas la mention « real images »
(l'option `--real` de `train_e.py` n'a pas servi) et le `config.json` de E3 n'a pas de clé `real`.
Empreintes : `config.json` de E3 fe0b5718a058525c6bb557e52c7b9ab53c24fedb313f17d35ba8fcb9fe64b5d7,
`history.json` 4d94926d0831e95182e928f7006b168efaf9d498b07d05ca964d1eef141030d9,
`log.txt` d5ae0c2094a90c6f393e0dd7a7992d2fe6260e9748b50c2f9b1377774ebee38e.

## 2. Les paires de E3 (pairs_v4) : un jeu entièrement généré

`<ESSAIS>/starRemoval/pairs_v4/manifest.json` (445091 octets, écrit le 2026-11-09 à 01:31:35, 6474 s) a pour empreinte
sha256 **5bd7602ebb189ebdb0aed607240240fb1d75e7491239ffe922f756077b924c65**. Il contient :

- `seed` 1, `n` 500 paires par jeu, `size` 2048, jeux `clean` et `defects` (1000 paires : 950 pour l'entraînement,
  50 tenues à part ; `config.json` : « held » 25 par jeu) ;
- `reads_no_image` : **true** (écrit par `make_dataset.py`, dont l'en-tête dit qu'aucune image n'est lue) ;
- `files` : 4000 empreintes sha256 (4 fichiers par paire : entrée et cible, bruitées et débruitées) ;
- `denoiser` : empreinte du réseau de débruitage de FLOAT, 2ea85a0755bfccd046e32d8f9d7110896d2a9871ab499bbb9e7cd62f7a216bdc ;
- `versions` : python 3.12.13, numpy 2.5.2, scipy 1.18.1, opencv 5.0.0, torch 2.11.0+cu128 ;
- `code` : empreintes du code qui génère les paires :

| fichier | sha256 |
|---|---|
| make_dataset.py | cd57068d1a3eee14007124984562e9421b1053ded0e36f6334cf0b9edc006654 |
| backgrounds.py | e627d48df58f6f8e64a7ecc358eee80c5b6bb1fc4d05e0c8bbccbff944fc6e1f |
| defects.py | ed5729101c4090eef2091c7f476a220392165893efd782db0e4ff225750cbfff |
| psf.py | 84a3d3459b1851578a71349efe407478ccfb5477d56820b58c7a3830037224b1 |
| instruments.py | e1de75dd1ab5d0ee6f1b2f25c5022a616a674aa2546b8d28f6ce08f7b398a9cd |
| stars.py | acfadf33ed7aa129a31e7363cd49ded14c624e37102f6d1d9a7a9f2f5eaa13e1 |
| catalogue.json | bccd5e1ab07de7a4172c0acce0d91e642d5a1ec8bb66e27121c4c8e31d9969cd |

Vérification faite le 26 nov 2026 : les empreintes des fichiers actuels de `<SCRIPTS>` (ces sept fichiers) sont
**identiques** à celles du manifeste. Le code qui a généré les paires de E3 est donc, à l'octet près, le code présent.

Ce que lit le générateur (`make_dataset.py`) : `catalogue.json` (caractéristiques publiées de télescopes, objectifs et
caméras) et le réseau de débruitage de FLOAT. Les fonds (`backgrounds.py`) et les défauts (`defects.py`) sont calculés par
code à partir de la graine ; les étoiles par `stars.py` et `psf.py` (optique de Fourier). Les images réelles n'ont servi
qu'à mesurer des plages de tirage (constantes du code, avec la mesure d'où elles viennent).

## 3. Les autres jeux générés de la filiation

| jeu | manifest.json sha256 | écrit le | backgrounds.py / stars.py / make_dataset.py dans le manifeste |
|---|---|---|---|
| pairs_v2 (E) | c48a66e416769318dd6f1afca54411fa3222c54d262fc068f8475524646ab939 | 2026-11-07 10:25 | 539ff547... / ec671b7b... / c76376b4... |
| pairs_v3 (E2) | 1ced1a48e728b821042f37588e86130e58f92984833e3eb0ace46ea358688cea | 2026-11-07 22:21 | e627d48d... / ec671b7b... / 3374b83e... |
| pairs_v4 (E3) | 5bd7602ebb189ebdb0aed607240240fb1d75e7491239ffe922f756077b924c65 | 2026-11-09 01:31 | e627d48d... / acfadf33... / cd57068d... |

Les trois manifestes portent `reads_no_image: true`. Pour pairs_v2 le code a changé depuis (empreintes différentes de
celles du présent pour backgrounds.py, stars.py et make_dataset.py) ; pour pairs_v3, stars.py et make_dataset.py ont
changé depuis : le code exact de ces deux jeux n'est pas conservé ; seul le manifeste en garde les empreintes. Le banc de
PSF `psf_bank.npz` (sha256 6075a9da6a3aee6b4ddd9ad4b2a0397fcfd58ecca66bb212ff7680288f412856) est calculé par
`synth.py build` à partir du modèle optique et d'instruments tirés (graine 2718) : aucune image.

## 4. Comment le refaire

`make_dataset.py --verify 1 --out <dossier>` régénère la première paire de chaque jeu depuis la graine et compare les
empreintes au manifeste : fichiers bruités à l'octet ; débruités à l'octet sur processeur, ou dans une tolérance si le
débruitage a tourné sur carte graphique (cas de pairs_v4 : `device` cuda:0). Deux régénérations de contrôle de pairs_v4 existent dans les
journaux de l'époque (`pairs_v4_check.log`, `pairs_v4_check2.log` : 240 fichiers chacune, ~8 min) ; elles n'affichent que les paires
régénérées, pas le verdict de la comparaison : **ce verdict ancien n'est pas conservé**.

Vérification refaite le 26 nov 2026 (`make_dataset.py --verify 1`, une paire par jeu : `clean/pair_1_0000` et `defects/pair_1_0000`,
régénérées sur le processeur pour la génération et sur la carte graphique pour le débruitage, comme le dit le manifeste) :

| fichier | résultat |
|---|---|
| `input_noisy.npy`, `target_noisy.npy` (les deux jeux) | **identiques au bit** à l'empreinte du manifeste |
| `input_denoised.npy`, `target_denoised.npy` (les deux jeux) | **différents** : plus grand écart 216 (entrée) et 88 (cible), en unités de bruit, dans une image 2048 x 2048 en float16, au-delà de la tolérance du script (0,05) |

Lecture : tout ce qui est tiré de la graine (fonds, défauts, étoiles, bruit : les fichiers « noisy ») est reproduit exactement par le code
présent. Les fichiers « denoised » sont ces mêmes images passées dans le réseau de **débruitage** de FLOAT ; les fichiers stockés ont bien
l'empreinte du manifeste, et le fichier de poids du débruiteur actuel a la même empreinte que celle du manifeste, mais le code de l'étage
de débruitage du moteur a été modifié après le 9 nov (commits des 11, 12, 13 et 15 nov, dont « graine déterministe dans le débruitage »).
**Cause probable, non établie** : ce changement de code, ou le non-déterminisme de la carte graphique. Cet écart concerne le débruitage, pas le retrait
d'étoiles : il ne change pas l'origine des paires (générées).

## 5. Ce qui reste incertain pour E3 (nommément)

- `train_e.py`, le script d'entraînement, a été modifié le 9 nov à 15 h 24, après la fin de l'entraînement de E3
  (08 h 03) : la version exacte lancée n'est pas figée par une empreinte. La modification visible est l'ajout de l'option
  `--real` (jamais utilisée pour E, E2, E3). Le dépôt git des scripts n'existe que depuis le 21 nov : aucun commit ne date
  de l'entraînement de E3.
- Le point `train/E2/best.pt` n'a pas d'empreinte dans ce document.
- `day_e4.py` (E4 : E3 + paires réelles corrigées de `real_targets.py`) existe, mais E4 n'a jamais été entraîné
  (pas de dossier `train/E4`) ; N5 et N7 partent de E3 pas 20000, pas de E4.
