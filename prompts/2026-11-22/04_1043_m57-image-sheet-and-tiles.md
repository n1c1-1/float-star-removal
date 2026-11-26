# M57 image sheet and tiles

- Date : 2026-11-22 à 10:43 (heure de Paris)
- Numéro du jour : 04
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Tu prépares la TROISIÈME image de test du retrait d'étoiles classique (sans réseau) qui doit produire les paires d'entraînement du réseau d'étoiles de FLOAT : M57 (APN Nikon D810a sur astrographe Epsilon 180, image couleur débruitée). Méthode courante gelée : <SCRIPTS>\Methode v21 SHO – 22 nov (marche sur la Rosette et M8 SHO).

LIS D'ABORD (obligatoire) :
- <SCRIPTS>\RECETTE_ET_LECONS.md, sections 1 (règle de lancement), 2 (recette, étape 1 = fiche d'image mesurée UNE fois sur l'image entière, puis figée) et 5 (règles Rosette à ne pas reprendre).
- LISEZMOI.txt de « Methode v21 SHO – 22 nov » et de « Methode v13 SHO – 21 nov » : comment l'image m8_sho a été ajoutée (images.py, astrometrie_m8_sho.py, carte_m8_sho.py, decoupes_m8_sho.json, batir_modele_image.py, batir_etalonnage_image.py, etape_piste1.py avec ses réglages par image, methode_validee.py --image).

TRAVAIL (une étape à la fois, dans cet ordre) :
1. Copie « Methode v21 SHO – 22 nov » vers « Methode v22 M57 – 22 nov » (même dossier parent) SANS son dossier sorties\ racine, ni *.log/*.err/*.png ; garde Piste1\sorties (rosette, m8_sho). Aucun fichier de v21 ne doit changer.
2. Plomberie m57 dans v22, sur le modèle de m8_sho : images.py a déjà une entrée "m57" (<DISQUE>/…\set_24_dslr_astrograph_d810a_eps180_m57\input.npy, à vérifier) ; astrométrie existante : <SCRIPTS>\Retrait étoiles par modèle – 20 nov\sorties\m57\astrometrie.npz (copie-la dans v22\Piste1\sorties\m57\ ; ne la recalcule pas) ; catalogue profond existant : <SCRIPTS>\Catalogue Gaia profond – 20 nov\gaia_dr3_profond_m57.npz (copie dans le dossier Catalogue de v22). AUCUN téléchargement (aucun accès réseau). Variable ASTRO_GPU_GAIA_EXTRACTS pointée sur l'extrait hors ligne si un script la demande.
3. FICHE DE L'IMAGE, mesurée une fois sur l'image ENTIÈRE avec les scripts existants (batir_modele_image.py m57, batir_etalonnage_image.py m57, la détection de saturation v14) : écris v22\fiche_m57.json + v22\fiche_m57.md (français, court) : taille, type (capteur couleur), bruit par couleur, largeur des étoiles (FWHM), point zéro et dispersion, parts de couleur, niveau de saturation par couleur et plateau (px à >= 92 % du max) des 10 plus brillantes, nombre et angles des aigrettes de l'instrument (mesurés sur les brillantes de l'image entière), liste des 15 étoiles les plus brillantes (G, position, saturée oui/non), position et taille apparente de M57 et de toute galaxie visible, et ce que ça change par rapport à la Rosette / M8 (réglages « calés sur decoupe6 » de la recette 5.5 à signaler, sans les changer).
4. CARTE et PROPOSITION DE 4 CARRÉS 768 px (tuile jugée = 512 central) : (a) M57 au centre (le test principal : la nébuleuse compacte et brillante ne doit pas partir avec les étoiles) ; (b) la brillante saturée à aigrettes la plus forte ; (c) le champ d'étoiles le plus dense ; (d) une zone de fond calme avec une galaxie faible si possible (IC 1296 est près de M57). Écris v22\decoupes_m57.json (même format que decoupes_m8_sho.json) et une carte PNG de l'image entière réduite, étirée (arcsinh), carrés numérotés, dans <SCRIPTS>\A_REGARDER – 21 nov soir\M57_0_carte_carres.png (cv2.imencode().tofile : chemins non ASCII). Plus, pour chaque carré, la tuile 512 d'entrée seule à x2, étirement arcsinh 60 comme dans vue_v18_v19.py de v21 : M57_<carré>_entree.png dans le même dossier.
POINT D'ARRÊT : après l'étape 4, tu t'arrêtes et tu rends compte. Tu NE lances PAS la piste 1 ni la chaîne sur les carrés : le choix des carrés revient au coordinateur.


RAPPORT FINAL (court, ≤ 15 lignes) : chiffres clés de la fiche, les 4 carrés (coordonnées et pourquoi), chemins des PNG, durées, toute surprise (ex. saturation à un autre niveau, aigrettes en croix, M57 saturée au cœur).
