# Provenance des données d'entraînement du réseau d'étoiles

Question posée : le réseau de retrait d'étoiles (NAFNet) a-t-il déjà été entraîné sur la sortie d'un autre logiciel de retrait
d'étoiles ? Ce dossier établit, fichier par fichier, d'où viennent les données lues par les modèles **E3**, **N5**, **N7** et par le
mélange **N7/N5 0,75**, et permet de refaire une cible à partir de son entrée.

## Ce que contient le dossier

| fichier | rôle |
|---|---|
| `MANIFESTE_PAIRES.csv` | une ligne par fichier de paire lu par l'entraînement de N5 et de N7 (182 fichiers : 103 débruitées + 35 brutes + 38 APOD en entraînement, 6 débruitées en validation) : empreintes sha256, tuile, image source, scripts et commits de production |
| `MANIFESTE_E3.md` | preuve que E3 (point de départ de N5 et N7) est entraîné sur des paires **générées** (empreintes du manifeste des paires, graines, code) |
| `ANALYSE_STATIQUE.md` | pour chaque script d'entraînement et de production : ce qu'il lit (fichier, ligne, fonction) et la preuve que rien n'y vient d'un logiciel tiers de retrait d'étoiles |
| `verifier_provenance.py` | refait la cible d'une paire avec la chaîne publiée et la compare au fichier (écart au bit près) |
| `refaire_paires_web.py` | refait une paire APOD depuis l'image web publiée (téléchargement facultatif, `--telecharger`), délinéarisation comprise, puis la compare au manifeste |
| `SHA256SUMS_PAIRES_NICOLAS.txt` | sha256, nom (avec alias) et taille des 54 paires de l'arborescence personnelle de Nicolas, copiées hors dépôt pour la release (`Depot public – paires/paires/`, aucun `.npz` dans le dépôt) |

Le CSV est lisible par un tableur (séparateur « ; »). Les colonnes `commit_*` donnent le commit du dépôt de scripts qui contenait le
script au moment de la production (« incertain » si le script n'était pas encore commité ; le premier commit qui suit est alors
indiqué) ; `scripts_non_modifies_depuis_prod` dit si les fichiers de la chaîne n'ont pas été touchés depuis la production.
Les noms contenant un prénom de tiers sont remplacés par l'alias `tiers1` ; la vérification retrouve la ligne par l'empreinte du fichier.

Colonnes de `MANIFESTE_PAIRES.csv` : `nom_fichier`, `modeles` (N5+N7 : lu par les deux), `role` (entraînement ou validation), `type` (debruitee,
brute, apod), `image_source`, `tx`, `ty` (coin haut gauche de la tuile de 512 px dans l'image source), `sha256_paire`, `taille_octets`,
`sha256_image_source` et `chemin_image_source` (l'entrée lue : l'image, ou la version non débruitée pour une paire brute),
`chemin_entree_debruitee` et `sha256_entree_debruitee`, `pile_origine` (dossier de la pile d'origine, d'après le `meta.json` de l'image),
`script_production`, `commit_production`, `chaine_methode` et `commit_chaine`, `juge` et `commit_juge`, `version_cible` (v33g), `filiation`,
`scripts_non_modifies_depuis_prod`, `juge_v33d_non_modifie_depuis`, `preparer_g_h_non_modifie_depuis`, `script_derivation`, `commit_derivation`,
`sha256_base_v33` et `sha256_v33d` (empreintes des versions antérieures du même fichier), `date_base_v33`, `poids_zero_pct` (part de pixels de poids 0),
`apod_info` (numéro, date, titre, adresse APOD) et `sha256_image_web_origine` (empreinte du fichier d'image téléchargé, pour les APOD).

## Qui a entraîné quoi, avec quelles données

| modèle | départ | données lues |
|---|---|---|
| E, E2, E3 (7-9 nov) | poids aléatoires, puis E, puis E2 | paires **générées** `pairs_v2/v3/v4` + étoiles tirées au vol ; aucune image (`MANIFESTE_E3.md`) |
| N5 (25 nov) | E3, pas 20000 | 182 fichiers du manifeste (version v33g) + fonds et étoiles synthétiques (voir plus bas) |
| N7 (25 nov) | E3, pas 20000 | **les mêmes** 182 fichiers (v33g) + cœurs synthétiques (Sérsic) ajoutés sur 25 % des tuiles synthétiques |
| mélange N7/N5 0,75 | — | **aucune donnée** : moyenne de poids 0,75 × N7 (pas 22000) + 0,25 × N5 (pas 23000), même architecture (journal du 26 nov) |

N5 **ne reprend pas** N2 (nuit du 24 nov) : N5 part de E3, pas 20000. N2, N1 et les tours suivants lisent les mêmes tuiles en version
v33d (empreintes dans la colonne `sha256_v33d`).

Les « fonds » du générateur d'étoiles synthétiques sont les cibles ou les entrées des paires réelles et APOD elles-mêmes (une fois
sur deux chacune), tirées dans le lot ; les formes d'étoiles viennent de banques de vraies étoiles brillantes mesurées dans des
images avec leurs étoiles (`ANALYSE_STATIQUE.md`, §2.2 et §2.3). Aucun fond, aucune forme ne vient d'une image sans étoiles
d'un autre logiciel.

## Tableau : jeu de données -> entrées -> cible -> logiciel tiers

| jeu de données | d'où viennent les entrées | comment la cible est fabriquée | sorties d'un logiciel tiers : aucune (preuve) |
|---|---|---|---|
| paires générées `pairs_v2/v3/v4` (E, E2, E3) | aucune image : fonds, défauts, étoiles et bruit tirés d'une graine | la cible est le fond sans les étoiles tirées (par construction) | manifeste `reads_no_image: true`, 4000 empreintes, code identique à celui du manifeste (`MANIFESTE_E3.md`) |
| paires réelles « débruitées » (103 + 6) | image avec ses étoiles : pile moins son gradient, fond et bruit normalisés, débruitée par le réseau de **débruitage** de FLOAT, balance des blancs (`input.npy` de l'image) | **entrée moins un calque d'étoiles** : piste 1 (modèle d'étoile et de halo ajusté sur l'image, positions et magnitudes du catalogue Gaia), étape A (rebouchage par patchs pris dans l'image), étape B (trois passes), puis juge (poids) ; v33d : poids mis à 0 seulement ; v33g : + lumière diffuse L lissée | aucune lecture (`ANALYSE_STATIQUE.md` §3 et §4) ; rejeu au bit près sur 3 tuiles (m1, m45, bubble : ci-dessous) |
| paires réelles « brutes » (35) | la même tuile, **non débruitée** (`entree_brute.py` : demi-piles normalisées, sans réseau) | brute − (entrée débruitée − cible débruitée) : on retire la même couche d'étoiles | formule vérifiée au bit sur 3 paires brutes (mode `liens` : `bubble_a7s_g1446_1310` ; rejeu : `m45_asi1600_g2650_1450`, `bubble_a7s_g2000_1367`) |
| paires APOD (38 retenues sur 65 produites, 13 images) | images web publiées par APOD, **délinéarisées** (étirement inverse estimé sur les étoiles), sans nouveau débruitage ; la liste (numéro, date, titre, adresse APOD) est dans la colonne `apod_info` | même chaîne (variante v33b) et même juge que ci-dessus ; 9 images « lâchées » à la revue à l'œil (27 tuiles) ; 7 images n'ont pas pu être produites (piste 1 plantée : 20 tuiles) | aucune lecture ; rejeu au bit près sur 1 tuile APOD (ci-dessous) |
| banques de formes d'étoiles et cache de styles | étoiles brillantes réelles isolées, médianes de 5 à 8 étoiles, mesurées dans des images **avec** étoiles (`input.npy`, `input.fits`, `original.npy`) | n'entrent pas dans une cible : servent à dessiner des étoiles ajoutées sur les fonds | `ANALYSE_STATIQUE.md` §2.3 (aucune lecture de `starless`) |

### Ce qui est établi pour les paires APOD

Le code (`Production APOD – 24 nov`) prépare chaque entrée (`preparer_apod.py` : médiane et dispersion du fond par couleur, unité = bruit moyen,
« aucun réseau, aucun téléchargement »), puis lance la piste 1 sur la carte, puis l'étape A, l'étape B et le juge sur le processeur
(`tuile_cpu_apod.py`) : les mêmes que pour les images de Nicolas, dans la variante v33b de la chaîne (correctifs de 4 plantages de la piste 1).
Les 65 fichiers de base sont exactement les 65 lignes de `paires.jsonl`. **Doute** : les images web sont des images déjà traitées par
leurs auteurs avec un logiciel inconnu (y compris, peut-être, un outil de réduction d'étoiles) ; ce traitement est dans l'**entrée** ; la
**cible** reste calculée par notre chaîne à partir de cette entrée, sans lecture d'une sortie d'un logiciel de retrait.

## Comment lancer la vérification

Variables d'environnement : `RACINE_SCRIPTS` (dossier des scripts de la chaîne) et `DONNEES` (dossier des données). Le script copie la
chaîne dans un dossier de travail ; il n'écrit rien dans `RACINE_SCRIPTS`. Quatre fils de calcul au plus.

```
# 1. rapide : empreinte dans le manifeste, relations entre versions (cible v33 = v33d, v33g = v33d + L, brute = formule)
python verifier_provenance.py <paire>.npz

# 2. refaire l'étape v33d -> v33g (quelques secondes)
python verifier_provenance.py <paire>.npz --mode g

# 3. refaire la cible depuis l'entrée (piste 1 sur processeur ; de 1 à quelques minutes sur les tuiles essayées, bien plus sur les tuiles lentes)
python verifier_provenance.py <paire>.npz --mode rejeu --device cpu --travail <dossier>

# tout
python verifier_provenance.py <paire>.npz --mode tout --rapport rapport.json
```

Le rejeu compare à la paire de base (`<image>_g<x>_<y>_debruitee.npz`) : par canal, sur les pixels de poids > 0, écart absolu maximal et
moyen (en bruits du ciel), part de pixels identiques ; poids, classes et entrée sont comparés aussi.

### Résultats des essais (26 nov 2026, processeur, 4 fils)

| paire | chaîne | mode | résultat |
|---|---|---|---|
| `m1_g896_896_debruitee` (non APOD) | v33 | rejeu (piste 1 sur processeur : 132 s ; A + B : 34 s ; 169 s au total) | cible, poids, classes, entrée : **identiques au bit** (écart max 0 sur 258458 pixels de poids > 0) |
| `m45_asi1600_g2650_1450_debruitee` et `_brute` | v33 | liens + g + rejeu (102 s) | cible débruitée, cible **brute** refaite, poids, classes : **identiques au bit** ; v33g refait à l'identique |
| `apod_070_g1536_1408_debruitee` (APOD) | v33b | liens + g + rejeu (74 s) | cible, poids, classes : **identiques au bit** ; v33g refait à l'identique |
| `bubble_a7s_g1446_1310_brute` | — | liens | cible brute = brute − (entrée débruitée − cible débruitée) : **identique au bit** ; v33 -> v33d : cible identique, poids seulement mis à 0 |
| `bubble_a7s_g2000_1367_debruitee` et `_brute` (v33b) | v33b | liens + g + rejeu (piste 1 sur processeur : 803 s ; A + B : 50 s ; 861 s au total) | cible débruitée, cible brute refaite, poids, classes : **identiques au bit** ; v33g refait à l'identique |

Même le rejeu de la piste 1 sur le **processeur** redonne, sur ces 6 tuiles, les mêmes bits que la production, qui avait fait la piste 1 sur
la carte graphique : l'ajustement par lots tourne en `float64`. Cela ne dit pas qu'il en va de même pour toutes les tuiles ni pour
une autre version de bibliothèque : seuls ces rejeux sont établis. Les étapes A, B, passes et juge n'ont donné aucun écart.

## Ce qui est établi, ce qui reste incertain

**Établi**

- N5 et N7 lisent exactement les mêmes 182 fichiers de paires (`config.json` de N5 et de N7 identiques pour les listes) ; tous sont des
  sorties de `produire_v33.py` / `produire_v33b*.py` / `produire_apod.py` (les 124 + 65 fichiers de base sont les lignes de `paires.jsonl`).
- Les cibles des paires (réelles, brutes, APOD) sont des sorties de la chaîne classique ; aucun script d'entraînement ni de production n'ouvre
  de fichier d'un logiciel tiers de retrait d'étoiles (`ANALYSE_STATIQUE.md`).
- E3 est entraîné sur des données générées (`MANIFESTE_E3.md`) : manifeste `reads_no_image: true`, code identique à celui du manifeste, et la
  régénération de contrôle (`make_dataset.py --verify 1`, 26 nov) redonne **à l'octet** les fichiers bruités des deux jeux.

**Incertain ou à décider (nommément)**

1. Les commits : le dépôt de scripts date du 21 nov ; E3 (8-9 nov) n'a pas de commit ; les paires v33 ont été écrites avant le commit
   du script de production pour 14 lignes du CSV sur 182 (colonne `commit_production`, valeur « incertain » ; les fichiers de la chaîne n'ont pourtant pas été modifiés depuis les paires pour 168 lignes, colonne `scripts_non_modifies_depuis_prod`) ; `preparer_g_h.py` a été modifié après l'écriture
   des paires v33g (le mode `g` montre que la version actuelle les reproduit au bit près pour les 3 paires essayées ; colonne `preparer_g_h_non_modifie_depuis` = non).
2. `train_e.py` a été modifié après l'entraînement de E3 (`MANIFESTE_E3.md`, §5).
3. Les images APOD ont pu être traitées par leurs auteurs avec n'importe quel logiciel avant publication (voir plus haut).
4. Plusieurs images « non APOD » viennent de banques d'images, de jeux de données ou de piles de télescopes en ligne : le code ne dit pas
   qui les a prises (voir `RESUME.txt` du dossier hors dépôt).
5. E3, fichiers « débruités » : la régénération de contrôle ne redonne **pas** les fichiers débruités du manifeste (écart max 216 et 88 en float16 ;
   cause probable : le code de l'étage de débruitage du moteur a changé depuis le 9 nov ; non établi). Les fichiers bruités, eux, sont identiques :
   l'écart concerne le réseau de débruitage, pas le retrait d'étoiles.
6. Le rejeu complet n'a été fait que sur 6 tuiles (m1, m45, 58_hoo_g3456_896, bubble_a7s_g2000_1367, bubble_a7s_g2200_1800, apod_070) ; les 182 empreintes sont vérifiables,
   les rejeux des autres tuiles n'ont pas été faits. Un essai sur `heart_a7s_g128_128` (production : 24 min de piste 1 sur la carte) a été arrêté
   après 9 minutes de piste 1 sur processeur : sans résultat.
7. Le lecteur de catalogue du moteur (`astro_gpu.core.<lecteur>`) porte dans son nom de module le nom d'un logiciel tiers : il lit des
   extraits du catalogue Gaia, pas d'images ; à renommer avant publication.

## Compléments du 26 nov (scripts publiés, paires APOD, paires de la release)

- **Scripts publiés.** `verifier_provenance.py` prend par défaut `RACINE_SCRIPTS` = `scripts/` du dépôt. Les fichiers non publiés de la chaîne
  (astrométrie et modèle d'étoile de chaque image, catalogue Gaia profond, liste des images du jeu 2, ancien dossier d'import) se donnent par
  `RACINE_ORIGINE` ; les jetons `<DONNEES>`, `<ESSAIS>`, `<FLOAT>`, `<CACHE_FLOAT>`, `<SCRIPTS>` des scripts publiés sont remplacés dans la copie de
  travail par `DONNEES`, `ESSAIS`, `RACINE_FLOAT`, `CACHE_FLOAT`, `RACINE_SCRIPTS`. Les scripts publiés ne contiennent pas le dossier de la chaîne
  « v33 production » : les paires v33 non APOD sont donc rejouées avec la chaîne v33b (elle n'en diffère que par des gardes sur les bords de l'image).
- **Essais avec les scripts publiés** (processeur, 4 fils) : `apod_070_g1536_1408` (chaîne v33b) et `m1_g896_896` (non APOD) : cible, poids, classes et entrée
  **identiques au bit** (écart max 0) ; étape v33d -> v33g refaite à l'identique pour la paire APOD.
  **Un script publié ne tourne pas tel quel** : `scripts/Methode v33b – 23 nov/Piste1/astrometrie.py` ligne 30 fait `from astro_gpu.core import asterion, catalogue_local`
  (et utilise `catalogue_local.` aux lignes 27, 66, 94-155 : `locate`, `astro_cone`, `SCALE`, `_sep_deg`, `XP_WAVELENGTHS`) ; le nettoyage a renommé le module du moteur,
  mais dans le moteur FLOAT le module porte encore son ancien nom : `ImportError: cannot import name 'catalogue_local'`. Je n'ai pas corrigé `scripts/` ;
  pour l'essai j'ai utilisé la variable `REMPLACEMENTS_COPIE` (remplacement de texte dans la copie de travail seulement). Il faut soit renommer le module dans le moteur, soit
  garder l'ancien nom dans le script publié.
- **Paires APOD : colonnes ajoutées à `MANIFESTE_PAIRES.csv`** : `url_page_apod`, `url_image`, `fichier_telecharge`, `sha256_image_telechargee` (image telle que téléchargée, 38 sur 38 sur le disque),
  `delinearisation_params` (JSON compact : méthode `etoiles_auto`, couleur choisie, `m_depart`, `m_float`, étoiles vues et utilisées, coût, normalisation fond, bruit et unité),
  `decoupe` (tx, ty, 512 px, chaîne v33b, fenêtre de la chaîne). **Attention** : pour les 38 lignes, `url_image` et `fichier_telecharge` contiennent le nom de famille du
  photographe (noms de fichiers du site) : à décider avant publication (les retirer ôte la possibilité de retélécharger sans repasser par la page APOD).
- **`refaire_paires_web.py`** : testé avec `--local` sur `apod_070_g1536_1408` : image web (sha256), délinéarisation refaite (7 s : courbe, nombre d'étoiles et couleur identiques au manifeste),
  entrée refaite **identique au bit** à l'entrée conservée, paire refaite (chaîne publiée) **identique au bit**, et le **fichier** de paire de base refait a le même sha256 que
  la colonne `sha256_base_v33`. `--telecharger` n'a pas été exécuté (aucun accès réseau).
  Ce qui exige FLOAT : l'astrométrie hors ligne et les extraits Gaia (déjà calculés pour les 38 images, fournis par `RACINE_ORIGINE`) ; le réseau de débruitage n'est pas utilisé.
  Non vérifié : les 37 autres paires APOD, ni après retéléchargement (le serveur peut servir un fichier différent : la colonne `sha256_image_telechargee` le détecterait).
- **Paires de la release** : 54 fichiers, 297,7 Mo, copiés dans `Depot public – paires/paires/` ; empreintes dans `SHA256SUMS_PAIRES_NICOLAS.txt`, revérifiées après copie.
  Les 90 fichiers (488,1 Mo) venant de banques, jeux de données, piles LCO ou « à vérifier » ne sont pas copiés : liste dans `RESUME.txt` (hors dépôt).

## Vérification décisive des cibles (26 nov, soir) : verdict pour les 182 lignes

**Verdict : établi pour les 182 lignes** (colonnes `origine_cible`, `preuve_cible`, `verdict_cible` de `MANIFESTE_PAIRES.csv`). La cible de chaque paire lue par N5 et N7 vient de
la chaîne classique (131 lignes : chaîne v33 ; 51 lignes : chaîne v33b, dont les 38 APOD), puis v33d (poids seulement) et v33g (lumière diffuse L lissée) ; pour les 35 brutes, de la formule
brute - (entrée débruitée - cible débruitée). Aucune ne vient du générateur synthétique (les tuiles synthétiques sont tirées à la volée, ce ne sont pas des fichiers de paires), d'un réseau, d'une retouche
ni d'un logiciel tiers : aucune cible n'est non établie. Preuves, ligne par ligne :

1. **Journaux de production** : chacune des 124 + 65 tuiles de base est dans `paires.jsonl` ; son journal de piste 1 (`p1_<tuile>.log`, réglages `no_fill`, dossier de travail `mv_<tuile>`) et son journal de
   fin (`cpu_<tuile>.log`, ligne `RESULTAT`) sont dans `Production v33 – 23 nov/journaux*` ou `Production APOD – 24 nov/journaux`. Seule exception : `58_hoo_g3456_896`, produite lors de l'essai initial
   (journal `journaux/58_hoo.log`, « == 58_hoo_g3456_896 rc=0 », sans `p1_`/`cpu_` séparés) ; elle est établie par le rejeu au bit près (ci-dessous).
2. **Seule l'entrée est lue.** `Piste1/images.py` (`load`) ouvre le fichier d'entrée de l'image (colonne `chemin_entree_debruitee` : `input.npy`, `original.npy` ou `input.fits`) ; aucun script de la chaîne ne nomme les fichiers
   `starless*.npy` / `cible.npy` (analyse statique) ; `git log -S` sur les dossiers de la chaîne et de la production ne trouve aucune lecture de ces fichiers ni de `Comparaison référence` à aucun commit
   (`starless_reseau` n'apparaît que dans `vues_mesures_m8_sho.py`, script de planches jamais appelé).
3. **Contrôle numérique contre les fichiers voisins.** Pour les 60 tuiles non APOD dont le dossier d'image contient un fichier `starless*.npy` ou `cible.npy` (images m1, m13, m82, m97, ngc6543, ngc6888,
   ngc7662, 58_hoo, m8_sho, horsehead, horsehead_40da, ic443_40da, m45_40da, m51_origin, m51_seestar_s30, m57, ngc869, rosette_40da, wizard_a7s), on a pris les pixels où ce fichier diffère de l'entrée de plus de
   5 bruits (au moins 142 pixels par tuile ; typiquement des milliers) : la cible n'y est égale (à 1e-4 près) au fichier voisin que sur **au plus 0.42 % des pixels** (une cible dérivée en ferait 100 %).
   Les autres images (piles de Nicolas, jeu 1, jeu 2, bancs, APOD) n'ont aucun fichier starless ni cible dans leur dossier : rien n'était à lire.
4. **Relations de versions vérifiées sur les 182 lignes** : cible v33 = cible v33d au bit ; v33g - v33d = lumière diffuse L lisse (pente max indiquée dans `preuve_cible`) ; brute = formule, au bit.
5. **Rejeu de la chaîne au bit près** sur 6 tuiles, dont 3 tuiles (5 lignes) des 14 lignes à commit « incertain » (`58_hoo_g3456_896`, `bubble_a7s_g2000_1367`, `bubble_a7s_g2200_1800`) :
   la cible refaite par la chaîne est identique à la cible du fichier. Pour les 14 lignes à commit incertain, l'historique git montre en outre qu'aucune version commitée des scripts (v32, v33, v33b) ne lit un fichier de référence.
   Non rejouées (production : 24 à 53 minutes de piste 1 par tuile) : `bubble_a7s_g1446_1310`, `heart_a7s` (3 tuiles), `ic443_40da` (4 tuiles) ; leurs dossiers d'image ne contiennent aucun fichier starless (heart, bubble) ou le contrôle du point 3 s'applique (ic443).

## Rejeu complet des paires publiées de Nicolas (dossier hors dépôt `Depot public – paires/rejeu_entrees/`)

- Contenu (469,6 Mo, 132 fichiers, empreintes dans `SHA256SUMS_REJEU_NICOLAS.txt`) : pour les 12 images de Nicolas dont des paires sont publiées (27 tuiles) : `entrees/<image>/input.npz` (l'image entière en zéros sauf les
  fenêtres de 768 px lues par la chaîne autour de chaque tuile et le maximum de chaque canal, que le juge utilise), `meta.json`, `type.json` ; `origine/…/Piste1/sorties/<image>/` (astrométrie, modèle d'étoile, étalonnage) et l'extrait Gaia profond de chaque image.
- Utilisation : `RACINE_ORIGINE=<rejeu_entrees>/origine ENTREES_REJEU=<rejeu_entrees>/entrees python verifier_provenance.py <paire>.npz --mode rejeu` (plus `RACINE_FLOAT` et `CACHE_FLOAT`). Le script publie son rejeu à partir de ces seuls fichiers ; une paire publiée au format v33g est
  comparée en refaisant la cible de la chaîne puis la lumière diffuse L (aucune paire de base n'est nécessaire).
- Essai (processeur, 4 fils, scripts publiés, seuls fichiers de `rejeu_entrees`) : `m101_a7s_g2085_883_debruitee_v33g.npz` : entrée identique à la paire, cible v33g refaite **identique au bit**, classes et entrée identiques, poids conformes (219 s).
- Taille : 469,6 Mo pour le rejeu + 297,7 Mo de paires = 767 Mo au total (moins de 1 Go : raisonnable pour une pièce jointe de version). Aucune image APOD n'est copiée.
- Nécessite encore le moteur FLOAT (astrométrie hors ligne, extraits Gaia) et la correction du nom de module du lecteur de catalogue signalée plus haut.
