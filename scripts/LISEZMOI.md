# Les scripts : quel script fait quelle étape

Ces scripts sont **ceux qui ont réellement servi** (copiés de l'espace de travail le 26 novembre 2026). Le code est gardé tel quel ; seuls ont changé : les chemins, un en-tête en français au début de chaque fichier (rôle, étape, entrées, sorties, exemple), et les noms qui contenaient un nom de tiers (voir « Nettoyage » plus bas). Les dossiers gardent leur nom d'origine, avec la date du travail, parce que les scripts s'appellent entre eux par ces noms.

## Chemins : trois variables

| variable | rôle | valeur par défaut |
|---|---|---|
| `RACINE_SCRIPTS` | le dossier `scripts/` | calculé depuis l'emplacement du fichier |
| `DONNEES` | les données et sorties (paires, entraînements, mesures) | `donnees/` |
| `ESSAIS` | les essais anciens (E2/E3, jeux d'images) | `essais/` |

Dans les chaînes de texte (docstrings, chemins en dur dans des listes), les jetons `<DONNEES>`, `<ESSAIS>`, `<FLOAT>`, `<SCRIPTS>`, `<DISQUE>` remplacent les anciens chemins absolus : ce sont des **gabarits à adapter**, pas des chemins qui fonctionnent tels quels.

## Les étapes

| étape (voir `docs/METHODE.md`) | dossier | scripts principaux |
|---|---|---|
| 1. E2 / E3, données synthétiques seules | racine de `scripts/` | `make_dataset.py` (paires générées + manifeste), `backgrounds.py`, `defects.py`, `stars.py`, `psf.py`, `instruments.py` + `catalogue.json`, `train_e.py` (entraînement), `day.py`, `day_e2.py`, `day_e3.py` (orchestration), `diagnose.py`, `models.py` (réseaux et étirement), `measure.py`, `star_measure.py`, `donut_measure.py` (mesures qui fixent les plages du générateur) |
| 2. chaîne classique v33b | `Methode v33b – 23 nov/` | `etape_piste1_gpu_2f.py` (piste 1), `etape_A.py`, `etape_B.py`, `methode_validee.py` (enchaîne le tout), dossiers `Piste1/` (modèle d'étoile et soustraction), `Passes/` (passes finales), `Piste3/` (étape A) ; `Catalogue Gaia profond – 20 nov/` (extrait Gaia) |
| 2. production et juges | `Production v33 – 23 nov/`, `Juge v33 – 23 nov/`, `Juge v33d – 24 nov/`, `Choix des tuiles – 23 nov/` | `produire_v33.py`, `produire_v33b.py`, `tuile_cpu*.py`, `construire_paire.py`, `regles_v33*.py`, `rejuger_v33d.py`, `choisir.py` |
| 3. images web | `Images web APOD – 24 nov/`, `Delinearisation – 23 nov/`, `Delinearisation APOD – 24 nov/`, `Production APOD – 24 nov/` | `moisson.py`, `telecharge.py` (collecte), `methodes.py` et `courbe_etoiles2.py` (délinéarisation « étoiles_auto »), `delineariser.py`, `resoudre.py`, `telecharger_gaia_apod.py`, `preparer_apod.py`, `produire_apod.py`, `tuile_cpu_apod.py` |
| 4. étoiles synthétiques réalistes | `Etoiles synthetiques – 22 nov/` | `etoiles2.py` (générateur v2), `paires_synthetiques.py`, `images_mesurees.py`, `banque_*.py`, `aigrettes_banque.py`, `aveugle.py` + `score_aveugle.py` (test à l'aveugle) |
| 6. entraînement sur paires réelles | `Entrainement jeu 1 – 22 nov/` (`train_jeu1.py`, base commune), `Entrainement v33c – 24 nov/`, `Entrainement nuit – 24 nov/` (N1, N2, `paires_synthetiques_n2.py`) | — |
| 7-8. N5 (cibles v33g), N6 | `Tours N5 N6 – 25 nov/` | `preparer_g_h.py` (v33g et v33h), `train_n56.py`, `enchainer_n56.py`, `mesurer_n56.py` |
| 9. N7 (cœurs synthétiques) | `Tour N7 – 25 nov/` | `paires_synthetiques_n7.py`, `train_n7.py`, `coeurs_mesure.py`, `mesurer_n7.py` |
| 10. essais écartés | `Entrainement N3 – 25 nov/`, `Entrainement N4 – 25 nov/`, `Tour N8L – 26 nov/`, `Tour N9 – 26 nov/`, `Restauration lente – 25 nov/` | `train_n3.py`, `train_n4.py`, `train_n8l.py`, `train_n9.py`, `restaurer.py` |
| 11. mélanges de poids | `Melanges N5 N6 N7 – 26 nov/` | `chaine_melanges.py` |
| 12. inférence à deux échelles V2 | `Portée M27 – 25 nov/` | `deux_echelles_v1v2.py`, `portee_m27.py` ; copie utilisée par les mesures : `inferer_v2.py` dans les dossiers `Tours N5 N6`, `Tour N7`, `Melanges` |
| 13. mesure et contrôles | `Validation – 22 nov/`, `Controle galaxies compactes – 26 nov/`, `Diagnostic *`, `Cœur M31 – 25 nov/`, `Tete de cheval – 26 nov/` | `valider_reseau.py` + `liste.py`, `mesurer_controle.py`, `chercher_galaxies.py`, `diag_coeurs.py`, `b_tests.py` |
| 14. utiliser un réseau | racine de `scripts/` | `inferer_une_image.py` (retirer les étoiles d'une image à soi), `astro_gpu/` (extrait de FLOAT : le réseau et son application par tuiles) |

`lancer_borne.ps1` lance une commande à priorité basse, sur les cœurs 4 à 15, avec des fils fixés : c'est ce qui a servi pour tous les calculs de ces journées.

## Pourquoi trois copies de `inferer_v2.py`, `valider_n7.py`, `coeurs_mesure.py` ?

Règle suivie pendant tout le travail : **on ne modifie jamais un script validé ; on le copie et on change une chose**. Les copies d'un même script sont donc des versions successives (voir leur docstring : « copie de ... »). La plus récente est dans le dossier du dernier tour.

## Chaîne v33 ou v33b ?

Les paires d'entraînement viennent des deux : v33 pour la production du 23 nov, v33b (qui corrige quatre plantages de la piste 1 et diffère par des gardes sur les bords) pour quatre images relancées et pour toutes les images web. Seule la v33b est publiée ici (le dossier `Production v33` appelle `Methode v33b`) ; les rejeux de vérification des paires v33 (`provenance/`) sont **identiques au bit** avec la chaîne v33b sur les tuiles essayées.

## Ce qui dépend de l'application FLOAT et n'est pas fourni

| composant | utilisé par |
|---|---|
| résolution astrométrique hors ligne (`astro_gpu.core.asterion`) et lecteur de catalogues locaux (importé ici sous le nom `catalogue_local` ; il porte un autre nom dans FLOAT) | `Piste1/astrometrie.py`, `resoudre.py`, `essai_astrometrie.py`, `preparer_apod.py`, `entree_brute.py`, `portee_m27.py` |
| réseau de débruitage (`astro_gpu.stages.denoise`) | `make_dataset.py` (fichiers « débruités »), `entree_brute.py` |
| extraits Gaia et astrométrie déjà calculés pour chaque image | variable `ASTRO_GPU_GAIA_EXTRACTS` ; fichiers dérivés non publiés (voir `docs/TUTORIEL.md`) |

**Fourni** : `astro_gpu/stages/star_removal.py`, extrait de FLOAT (réseau NAFNet, étirement, application par tuiles) qui suffit à l'inférence et à tous les scripts d'entraînement et de mesure.

## Nettoyage appliqué aux copies

- chemins absolus remplacés (variables ci-dessus) ;
- le nom du logiciel commercial de comparaison est remplacé par `refext` / `RefExt` dans les identifiants (variables, fichiers de sortie) ;
- les prénoms de tiers dans les noms d'images sont remplacés par `tiers1`, `tiers2` (ex. `ngc7000_tiers1`) : les noms réels de ces fichiers de données diffèrent ;
- l'opérateur de produit matriciel est écrit `operator.matmul(a, b)` et les décorateurs sont écrits comme des appels de fonction (même comportement) ;
- aucune image, aucun `.npz`, aucun cache de banque d'étoiles n'est copié. Pour les générateurs qui utilisent `cache_styles.npz`, il se refait : `python paires_synthetiques.py cache 32`.

Tous les scripts compilent (`python -m py_compile`). Beaucoup lancent de longs calculs ou lisent des données non publiées : lire l'en-tête avant de les lancer.
