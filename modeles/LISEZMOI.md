# Les modèles

Quatre réseaux NAFNet (largeur 32, ≈ 29 millions de paramètres en float32 ; architecture : `scripts/astro_gpu/stages/star_removal.py`). **Chaque fichier fait plus de 90 Mo : aucun n'est dans git.** Ils sont en pièces jointes de la version (Releases) du dépôt. Les empreintes sont dans `SHA256SUMS` :

```
sha256sum -c SHA256SUMS
```

| fichier | taille (octets) | sha256 | ce que c'est | quand l'utiliser |
|---|---|---|---|---|
| `E3_step_20000.pt` | 116 862 057 | `9faf2dc4…e2129` | E3, pas 20 000 : entraîné sur **données générées seulement** (8-9 nov) ; point de départ de N5 et N7 | essai, ou point de départ d'un nouvel entraînement ; moins bon sur le réel |
| `N5_pas_23000.pt` | 116 868 111 | `d4b0309c…f9d6` | N5, pas 23 000 (meilleur pas de validation) : E3 affiné sur paires réelles v33g (25 nov) | **champs sans cœur de galaxie ni planétaire compacte** : retire aussi les étoiles saturées de faible signal (avec l'inférence à deux échelles) ; **efface les cœurs de galaxie** |
| `N7_pas_22000.pt` | 116 868 111 | `746ce23e…5c8` | N7, pas 22 000 : N5 + cœurs de galaxie synthétiques (25 nov) | quand il faut **garder des cœurs de galaxie** ; garde certaines étoiles saturées de faible signal |
| `N7N5_075.pt` | 116 835 023 | `9c88c524…199b` | mélange de poids 0,75 × N7 + 0,25 × N5 (26 nov) | choix par défaut du lot ; se comporte comme N7 sur les étoiles saturées de faible signal |

(empreintes complètes dans `SHA256SUMS`.) Les quatre fichiers sont des dictionnaires `torch` avec les clés `net` (poids en float32 ; c'est ce que lisent `star_removal.network()` et `scripts/inferer_une_image.py`) et `step` ; E3 et N5/N7 ont aussi `model`, `config` et `val`, le mélange `melange` (ses deux sources et son coefficient).

## Comment les choisir (un compromis, pas un meilleur modèle)

| | cœurs de galaxie gardés | étoiles saturées de faible signal retirées | étoiles faibles G 14-17 restantes |
|---|---|---|---|
| N5 (deux échelles) | non (0,19) | oui | 0,265 |
| N7 | oui (0,81) | en partie non | 0,283 |
| N7/N5 0,75 | oui (0,80) | en partie non | 0,295 |

Voir `README.md` (tableau des résultats) et `docs/AMELIORER.md` (piste pour lever ce compromis).

## Origine et provenance

Tous sont issus d'entraînements publiés dans ce dépôt (`scripts/`, `prompts/`), sur des données dont la provenance est établie fichier par fichier (`docs/PROVENANCE.md`, `provenance/`). **Aucun n'a été entraîné sur la sortie d'un autre logiciel de retrait d'étoiles.**

## Licence

GPL-3.0, comme le reste du dépôt (`LICENSE`).
