# Analyse statique : ce que chaque script lit

But : montrer, en lisant le code, que ni l'entraînement de E3, de N5 et de N7, ni la fabrication des cibles de leurs
paires ne lisent la moindre sortie d'un logiciel tiers de retrait d'étoiles. Chaque ligne cite le fichier, la ligne de
code et la fonction. Racines : `<SCRIPTS>` (scripts), `<DONNEES>` (données), `<ESSAIS>` (essais), `<FLOAT>` (moteur FLOAT).
Les numéros de ligne sont ceux des fichiers au 26 novembre 2026.

## 0. Méthode et ce qui est prouvé

1. On part des scripts d'entraînement (`train_e.py` + `day_e3.py` pour E3 ; `train_n56.py` pour N5 ; `train_n7.py` pour N7)
   et de ceux qui fabriquent les cibles (`tuile_cpu*.py`, `produire_*.py`, `methode_validee.py`, `etape_A.py`, `etape_B.py`,
   piste 1, piste 3, passes, juges, `preparer_g_h.py`).
2. On calcule leur **fermeture d'imports** (analyse syntaxique Python des `import`, en suivant les dossiers de la
   variable de chemin que chaque script écrit lui-même) : 12 fichiers pour E3, 20 pour N5, 13 pour N7, 24 pour la chaîne
   de production v33, 24 pour la chaîne v33b/APOD, 8 pour les juges et l'étape g/h.
3. On cherche dans toute la fermeture : le nom du logiciel tiers (toutes graphies, avec ou sans séparateur), `starless`,
   `sans_etoiles`, `Comparaison`, `stars_dataset`, `paires_nb`, `_revue`, et toute lecture de poids de réseau
   (`torch.load`, `load_state_dict`).
4. On cherche, dans **tous** les scripts de `<SCRIPTS>`, ceux qui mentionnent à la fois `starless` et un dossier de référence
   (`stars_dataset`, `paires_nb`, `Comparaison référence`) ou un fichier `starless_fixed` / `starless_reseau` :
   30 fichiers (dont `diag_validation.py`, `part_b.py`, `coeur_m31.py` et 22 copies de `vues_mesures_m8_sho.py` rangées dans les
   dossiers de versions de la méthode, y compris v33 et v33b : elles y sont rangées mais jamais importées ni lancées par la
   chaîne). **Aucun n'appartient à une des fermetures ci-dessus** (contrôle fait par programme). Ce sont des scripts de mesure
   et de planches (les scripts `inferer_v2.py`, `valider_*.py`, `coeurs_mesure.py`, `mesurer_*.py` lisent des sorties de réseaux
   déjà entraînés pour les mesurer ; ils ne sont pas non plus dans une fermeture).
5. Résultat de la recherche (point 3) dans les fermetures : le nom du logiciel tiers n'y apparaît nulle part comme donnée lue ;
   `starless` n'y désigne que `starless.npy`, la sortie de **notre** piste 1 (dossier `sorties/<image>/mv_<tuile>/`,
   `retrait_modele.py` l. 2204, `retrait_rapide.py` l. 2851 ; lue par `piste3.py` l. 83 et `juger.py` l. 89), et des commentaires.
   Historique git (`git log -S`) sur les dossiers de la chaîne et de la production : aucun commit n'a ajouté ni retiré une lecture de
   `starless_fixed`, `cible.npy`, `Comparaison référence` ou `REF_STARLESS` ; `starless_reseau` n'apparaît que dans les copies de
   `vues_mesures_m8_sho.py` (script de planches, jamais importé ni lancé par la chaîne ; commits 17b4d20 et c24bd7a).
   E, E2 et E3 partent de poids aléatoires ; aucun poids de réseau préexistant n'est lu par l'entraînement.

Ce qui n'est PAS établi par la lecture du code seule : l'origine physique des images d'entrée (qui les a prises) et le fait
que les fichiers actuels sont ceux qui ont été lus à l'époque (voir `MANIFESTE_PAIRES.csv` : empreintes, dates, commits).

## 1. E3 (et E, E2) : entraînés sur données générées seulement

| script | ligne | fonction | ce qui est lu | sortie d'un logiciel tiers ? |
|---|---|---|---|---|
| `day_e3.py` | 17-19 | constantes | `<ESSAIS>/starRemoval/pairs_v4`, `diagnose/valset_pairs_v4.npz`, `train/E2/best.pt` | non : jeu généré, jeu de validation tiré des paires générées, poids de E2 |
| `day_e3.py` | ~33 | `main` | lance `make_dataset.py --seed 1 --n 500` (génération) si le manifeste manque | non |
| `train_e.py` | 44-46 | constantes | `pairs_v2` (défaut), `train/`, `valset_pairs_v2.npz` : remplacés par les arguments de `day_e3.py` | non |
| `train_e.py` | 134, 144 | `Tiles.__init__`, `Tiles.arrays` | `info.json`, `input_denoised.npy`, `target_denoised.npy` des dossiers de paires | non : fichiers de `make_dataset.py` (empreintes au manifeste) |
| `train_e.py` | 229 | `val_set` | le fichier de validation (`np.load(path)`) | non : coupé dans les paires tenues à part (`diagnose.py` l. 222-247) |
| `train_e.py` | 174-181 | `RealTiles.__init__` | images réelles et cibles corrigées | **non exécuté** : exige `--real` ; `config.json` de E3 n'a pas de clé `real`, `log.txt` ne contient pas « real images » |
| `train_e.py` | 60-124 | `synthetic_stars` | rien (tire des PSF analytiques) | non |
| `train_e.py` | 296-297 | `main` | `torch.load(a.init)` : poids de départ (E2 pour E3) | non : E2 a été entraîné par ce même script |
| `models.py` | 230 | `chain_nafnet` | **l'architecture** seulement (`astro_gpu.stages.star_removal.NAFNet()`, poids aléatoires), aucun fichier de poids lu | non |
| `make_dataset.py` | 58-61, 78, 144 | modules, `sha256`, `_denoiser` | `catalogue.json`, le réseau de débruitage de FLOAT | non (débruitage, pas retrait d'étoiles) |
| `backgrounds.py`, `defects.py`, `stars.py`, `psf.py`, `instruments.py` | — | — | aucune image : `np.load` absent (sauf `instruments.py` l. 45 : `catalogue.json`) | non |
| `synth.py` | 61 | `Synth.__init__` | `psf_bank.npz`, banc de PSF calculé par `synth.py build` (modèle optique, graine 2718) | non |

E, E2 et E3 partent de poids aléatoires (`log.txt` de E : « random weights ») ; aucun poids de réseau préexistant n'est lu par
l'entraînement. Seule l'architecture (la classe du réseau étoiles) est reprise de FLOAT.

## 2. N5 et N7 : d'où viennent les paires lues, les fonds synthétiques et les formes d'étoiles

### 2.1 Les scripts d'entraînement

| script | ligne | fonction | ce qui est lu |
|---|---|---|---|
| `Tours N5 N6/train_n56.py` | 41-43 | constantes | `PAIRES` = `<DONNEES>/Paires étoiles v33 – 23 nov`, `APOD` = `<DONNEES>/Paires étoiles APOD – 24 nov` |
| `train_n56.py` | 50-63 | `lister` | `PAIRES.glob("*_v33g.npz")` (réelles, débruitées et brutes) et `APOD.glob("*_v33g.npz")` ; les 6 tuiles tenues (`TENUES`) vont en validation |
| `train_n56.py` | 181 | `main` | `T1.Paires(ten, 1)` : lecture des 6 tuiles de validation |
| `train_n56.py` | 198 | `main` | `torch.load(a.depart)` : `<ESSAIS>/starRemoval/train/E3/snap/step_20000.pt` (E3, généré seulement) |
| `train_n56.py` | 116, 124-126 | `Lots.__iter__` | `paires_synthetiques_n2.paire_synthetique(fond, ...)` ; **`fond = t[i]` (la cible) ou `x[i]` (l'entrée) d'une paire réelle ou APOD tirée dans les lots**, une fois sur deux |
| `train_n56.py` | 76-90, 125 | `extras_nebuleuse` | place des étoiles très brillantes sur les pixels de la cible à poids 1 : ne lit que les tableaux de la paire |
| `Tour N7/train_n7.py` | 44-45, 53-66, 189, 206 | idem | mêmes lectures (même config : 103 + 35 + 38 paires, 6 tenues) |
| `train_n7.py` | 119, 127-135 | `Lots.__iter__` | `paires_synthetiques_n7.paire_synthetique_coeurs` : même chose + des cœurs synthétiques (Sérsic) ajoutés à la cible |
| `Entrainement jeu 1/train_jeu1.py` | 36-43 | `Paires.__init__` | `np.load(f)` des fichiers qu'on lui donne (utilisé par N5/N7 pour lire les paires, jamais pour son propre dossier) |
| `train_jeu1.py` | 31, 150 | constante, `main` | `Paires étoiles jeu 1 – 22 nov/*.npz` : **seulement dans `T1.main`, jamais appelé par N5/N7** (ils importent `Paires` et `valider`) |
| `train_jeu1.py` | 85 | `LotsSynth.__iter__` | `from paires_synthetiques import paire_synthetique` (générateur d'origine) : classe non utilisée par N5/N7 |

Les listes **exactes** des fichiers lus sont dans `config.json` de `N5` et de `N7` (clés `debruitees`, `brutes`, `apod`,
`tenues_fichiers`) : 103 débruitées + 35 brutes + 38 APOD pour l'entraînement, 6 tuiles pour la validation, identiques
pour N5 et N7. Elles sont recopiées, avec empreintes, dans `MANIFESTE_PAIRES.csv`. N5 ne reprend **pas** N2 : N5 part de
E3 pas 20000 (journal de N5 et `config.json` : `depart`). Les paires de la nuit du 24 nov (N1, N2) sont les mêmes tuiles
en version `v33d` (empreintes dans la colonne `sha256_v33d`).

### 2.2 Les « fonds » du générateur d'étoiles synthétiques

Réponse précise : les **fonds** sont les tuiles 256 px des paires réelles et APOD elles-mêmes, tirées dans le lot courant
(`train_n56.py` l. 124 : `fond = t[i] if rng.random() < 0.5 else x[i]`), donc **la cible de notre chaîne** ou l'**entrée** de la
paire. Ce ne sont ni des images d'un autre logiciel, ni un autre dossier. Le générateur ajoute des étoiles tirées
(`paire_synthetique`, 60 % des tuiles) ; la cible reste celle de la paire (poids de la paire inchangés).
La seule autre lecture de fonds est `vitesse()` (`paires_synthetiques_n2.py` l. 361-368, `paires_synthetiques_n7.py` l. 442-449,
`paires_synthetiques.py` l. 355-361) : mesure de vitesse à la ligne de commande, qui lit `sans_etoiles_tuile.npy` de deux
tuiles de **notre** chaîne (M57 et M8 SHO) ; jamais appelée par l'entraînement.

### 2.3 Les formes d'étoiles (`cache_styles.npz`, banques)

- `paires_synthetiques_n2.py` l. 45 / l. 213 et `paires_synthetiques_n7.py` l. 48 / l. 216 : `np.load(cache_styles.npz)`.
  Ce fichier (4141488 octets, 22 nov 19:52, sha256 0e270ba4b22bb15a6cb60d83b547d8e8472333f8f0f95c5ceb1230b7723fc3f8)
  est construit une fois par `construire_cache()` (l. 156 / 159, appelée seulement s'il manque) avec 32 « styles »
  tirés par `etoiles2.tirer_image` (graine 2026) : optique de Fourier (`psf.py`), instruments du catalogue
  (`instruments.py`), et 11 banques de formes réelles.
- `etoiles2.py` l. 397-410 (`banques_dispo`) : lit `Etoiles synthetiques – 22 nov/banque/banque2_*.npz`.
  Constantes `BPRP` l. 65 : centiles des couleurs Gaia de 4 champs.
- Les banques sont construites par `banque_bibliotheque.py` (l. 57 : `lire_image(Path(chemin.split(" (")[0]))`),
  `banque_individuelle.py` (l. 104 : `A.IM.load(nom)`), `banque_formes.py` (l. 85 : `A.IM.load(nom)`) : à chaque fois **l'image
  d'entrée avec ses étoiles** (`input.npy`, `input.fits`, `original.npy`) ; on y médiane des vraies étoiles brillantes isolées
  (détection `banque_detection.py`, ou positions Gaia de `astrometrie.npz`). La mention « (+ starless.npy) » dans
  l'inventaire de la bibliothèque est du texte : le code coupe le chemin avant cette parenthèse et ne lit jamais ce fichier.
  `grep` des quatre scripts de banque et de `etoiles2.py` : aucune occurrence de `starless` ni de `sans_etoiles`.
- Banques réellement utilisées par les 32 styles du cache : rosette (2), m57 (2), m8_sho (1), horsehead (1) = nos images
  de test (`Piste1/images.py`) ; E065, E062 (banque d'images), E010 (projet de l'auteur), S023, S007 (`stars_dataset`),
  W004, W008 (`stars_dataset/_revue_pn`) ; 16 styles sans banque. Les 28 fichiers `banque2_*` de `banque/` sont chargés
  par `banques_dispo`, mais seuls les styles ci-dessus ont été tirés dans le cache.
- **Doute à lever par Nicolas** : plusieurs banques viennent d'images de jeux de données ou de banques
  (`stars_dataset/set_*`, `_revue_pn`, `astro_banque`) dont l'auteur n'est pas établi par le code. Ce sont des images
  **avec** étoiles ; aucune n'est une sortie d'un logiciel de retrait d'étoiles.

## 3. Fabrication des cibles (chaîne classique) : tout ce que la chaîne lit

La cible d'une paire est `sorties/<tuile>/B/t3/sans_etoiles_tuile.npy` (`construire_paire.py` l. 149), sortie finale de :
piste 1 → étape A → étape B (3 passes) → juge (poids). Lectures, par étape :

| étape (script, ligne, fonction) | lit | nature |
|---|---|---|
| `images.py` l. 45 `load` (via `construire_paire.py` l. 142 et la piste 1) | l'image d'entrée : `input.npy`/`input.fits` de l'image (liste `IMAGES` l. 14-30 ; `selection_jeu2.csv` l. 76-82 `_jeu2` ; `entrees/apod_*` l. 89-92 `_apod`) | image **avec** étoiles (débruitée), pas une sortie de retrait |
| `Piste1/astrometrie.py` l. 18-66, 94-155 | extraits Gaia DR3 hors ligne (`gaia_extracts`) via le lecteur de catalogue du moteur (`astro_gpu.core.<lecteur>`) | catalogue d'étoiles (positions, magnitudes, spectres XP) |
| `Piste1/retrait_rapide.py` (`run`, ajustement par lots en `float64`, `ajust_lot.py`) | `astrometrie.npz`, `etalonnage_image.*`, `modele_psf_image.*`, `psf_etendue.*` de l'image, l'image d'entrée | modèle d'étoile et de halo ajusté **sur l'image elle-même**, soustraction ; écrit `starless.npy` (l. 2851), `etoiles.csv` |
| `etape_A.py` l. 30-60 + `Piste3/piste3.py` | `starless.npy` et `etoiles.csv` de la piste 1 (l. 83, 137, 203), `gaia_dr3_profond_<image>.npz`, l'image ; remplisseur « patch » (`remplir.py`, `patchfill.py`) : les patchs viennent **de l'image elle-même** | rebouchage des cœurs et zones dures |
| `etape_B.py` + `Passes/passe_finale_p.py`, `_t2_p.py`, `_t3_p.py` | `carte_remplissage_tuile.npy`, `etoiles.csv`, catalogue profond | passes sur les restes d'étoiles, points sombres, défauts |
| `Juge v33/construire_paire.py` l. 142-205, `regles_v33.py` | la cible (`sans_etoiles_tuile.npy`), `carte_remplissage_tuile.npy`, l'entrée, le catalogue profond | calcule les **poids** ; ne modifie jamais la cible |
| `Juge v33c`, `Juge v33d` (`rejuger_v33c.py`, `rejuger_v33d.py`) | paires v33 | « ne font que mettre des poids à 0 » (vérifié : cible v33 = cible v33d au bit, mode `liens` de `verifier_provenance.py`) |
| `Tours N5 N6/preparer_g_h.py` `etape_g`, `etape_h` | paires v33d, catalogue Gaia (`diag_paires.py`) | v33g = cible v33d + lumière diffuse L (lissage gaussien normalisé de σ 40 px de l'écart entrée − cible sur le ciel non touché) ; v33h = poids seulement |
| brutes : `tuile_cpu.py` l. 24-31 `brute` | `entrees_brutes/<image>_brut.npy` (fabriqué par `entree_brute.py` : demi-piles normalisées, sans réseau) | cible brute = brute − (entrée débruitée − cible débruitée), au bit |

Points précis :

- Le module `Piste3/mesure.py` (l. 12-14) place `Retrait étoiles par modèle – 20 nov` en tête du chemin d'import : pour l'étape A,
  `images` et `juger` sont lus **dans ce dossier ancien** (`juger.py` ne fait que des mesures ; `images.py` est remplacé par la
  copie de `Piste1/images.py` par `etape_A.py` l. 33-40 pour toutes les images autres que Rosette). Vérifié par import :
  `images` et `juger` viennent bien de ce dossier ; `juger.py` l. 89 relit `starless.npy` de la piste 1.
- Un réseau n'intervient qu'à un endroit du code : `Piste3/piste3.py` l. 759-766, `e3_local` (réseau E3, généré seulement) ;
  il n'est appelé que si `filler == "e3"` (l. 619, 719). **La production appelle `P3.run(..., filler="patch")`**
  (`etape_A.py` dernière ligne de code) : ce chemin n'est pas pris.
- Aucune lecture de fichier de poids dans la chaîne de production ni dans les fermetures d'entraînement autre que `E3/snap/step_20000.pt` (départ de N5 et N7) et `E2/best.pt` (départ de E3).
  Les scripts de mesure (`inferer_v2.py`, `valider_n56.py`, `valider_n7.py`, `coeurs_mesure.py`, `diag_validation.py`) lisent des réseaux déjà entraînés pour les mesurer ; ils ne sont dans aucune fermeture.
- Le débruitage des entrées (`preparer_entree.py`, `preparer_entree_cpu.py`, `stars_dataset.generate` pour les anciens jeux) utilise
  le réseau de débruitage de FLOAT (`astro_gpu.stages.denoise`), qui n'est pas un réseau de retrait d'étoiles. Le script
  `stars_dataset.py` contient aussi la fonction `generate` qui appelle un logiciel tiers ; elle écrit `input.npy` **avant**
  l'appel (l. 286 de ce fichier, `np.save(d / "input.npy", ...)` ; l'appel du logiciel : l. 288) et `starless.npy` après (l. 295) ; la chaîne importe seulement
  `valid_of` et `sky_variation_of` (`entree_brute.py` l. 17, `preparer_entree*.py` l. 21-22) et ne lit jamais `starless.npy`.

## 4. Dossiers de comparaison : lus par des scripts de mesure seulement

- `<DONNEES>/Comparaison référence – 23 nov` : lu par `Diagnostic écarts – 25 nov/diag_validation.py` l. 32 (constante `REF`), l. 152
  (`outs["reference"]`). Aucun script d'entraînement ni de production ne le mentionne.
- `starless*.npy` des dossiers `stars_dataset/*` (réseaux de retrait d'étoiles de référence, corrigés ou non) : lus uniquement par
  des scripts de mesure ou de planches : `Cœur M31/coeur_m31.py`, `Diagnostic cœurs/diag_coeurs.py`,
  `Controle galaxies compactes/mesurer_controle.py`, `Tour N7/coeurs_mesure.py`, `Tour N7/valider_n7.py`, `Melanges N5 N6 N7/*`
  (copies), `Bibliotheque images/scripts/part_b.py`, `Methode v*/vues_mesures_m8_sho.py`.
  Des fichiers `starless_fixed.npy`, `starless_reseau.npy` et `cible.npy` existent à côté des `input.npy` / `original.npy` de certaines images
  (`paires_nb/<image>`, `_revue_combi/<image>`, `stars_dataset/<set>`). **Ils ne sont pas lus** : `images.py` ouvre le fichier d'entrée par son nom,
  aucun script de la chaîne ne les nomme, et le contrôle numérique sur les 90 tuiles concernées (voir `MANIFESTE_PAIRES.csv`, colonne `preuve_cible`) montre que la cible
  n'est égale à aucun d'eux, même sur les pixels où ils diffèrent fortement de l'entrée.

## 5. Doutes, énoncés sans détour

1. **Images APOD** : ce sont des images publiées, déjà traitées par leurs auteurs avec un logiciel inconnu. Le tri visuel a écarté
   les images « étoiles réduites » et « starless », mais on ne peut pas prouver qu'aucune n'a subi un traitement d'étoiles par un
   logiciel tiers avant publication. Ce traitement est **antérieur** et fait partie de l'entrée ; la **cible** reste calculée par
   notre chaîne à partir de cette entrée (rien d'un autre logiciel de retrait n'est lu). L'entrée est une image délinéarisée
   (étirement inverse estimé sur les étoiles, `Delinearisation APOD – 24 nov`), pas un retrait d'étoiles.
2. **Propriété des images non APOD** : le code ne dit pas qui a pris chaque image (voir `RESUME.txt` du dossier hors dépôt).
3. **Versions exactes du code à l'époque** : voir les colonnes `commit_*` et `scripts_non_modifies_*` de `MANIFESTE_PAIRES.csv`.
   Pour les paires v33 de base, le dépôt git ne contient le script de production qu'après les premières paires ; l'orchestrateur
   n'a pas changé depuis ses fichiers (date de modification antérieure aux paires pour 168 lignes sur 182).
   `preparer_g_h.py` a été modifié (partie h) après l'écriture des paires v33g : le mode `g` de `verifier_provenance.py` refait
   l'étape g avec la version actuelle et la compare aux fichiers (résultat dans `LISEZMOI_PROVENANCE.md`).
4. Le lecteur de catalogue du moteur (`astro_gpu.core.<lecteur>`, utilisé par `Piste1/astrometrie.py`) porte dans son nom de module le
   nom d'un logiciel tiers : c'est un lecteur de **catalogue Gaia**, pas d'image ; à renommer avant publication si l'on veut éviter la
   question.
5. `train_e.py` a été modifié après l'entraînement de E3 (voir `MANIFESTE_E3.md`, section 5).
