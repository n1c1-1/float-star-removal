# Horsehead image sheet and tiles

- Date : 2026-11-22 à 11:02 (heure de Paris)
- Numéro du jour : 06
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Tu prépares la QUATRIÈME image de test du retrait d'étoiles classique (sans réseau) qui doit produire les paires d'entraînement du réseau d'étoiles de FLOAT : la Tête de Cheval (IC 434, image couleur débruitée « user_Horsehead », avec Alnitak et σ Orionis, étoiles géantes). Méthode courante gelée : <SCRIPTS>\Methode v21 SHO – 22 nov (marche sur la Rosette et M8 SHO).

LIS D'ABORD (obligatoire) :
- <SCRIPTS>\RECETTE_ET_LECONS.md, sections 1, 2, 3.4 (grosses étoiles, dont la question des géantes) et 5.
- LISEZMOI.txt de « Methode v21 SHO – 22 nov » et de « Methode v13 SHO – 21 nov » : comment l'image m8_sho a été ajoutée (images.py, astrometrie, carte, decoupes json, batir_modele_image.py, batir_etalonnage_image.py, etape_piste1.py réglages par image, methode_validee.py --image).
- Pour le traitement des géantes : Piste3\piste3.py de v21 (GIANT_G, try_sigma, MARGIN_GIANT) et etape_A.py (try_sigma=False). Ne change RIEN pour l'instant : tu notes seulement ce qu'il faudrait pour un essai « géantes » (un seul changement à prévoir).

TRAVAIL (une étape à la fois) :
1. Copie « Methode v21 SHO – 22 nov » vers « Methode v23 Horsehead – 22 nov » SANS son dossier sorties\ racine, ni *.log/*.err/*.png ; garde Piste1\sorties. Aucun fichier de v21 ne doit changer.
2. Plomberie « horsehead » dans v23 sur le modèle de m8_sho : images.py a une entrée "horsehead" (train_with_stars\user_Horsehead\input.npy, à vérifier) ; astrométrie : cherche <SCRIPTS>\Retrait étoiles par modèle – 20 nov\sorties\horsehead\astrometrie.npz (ou un autre nom de dossier pour cette image dans ce dossier sorties) et copie-la ; si elle n'existe pas, utilise le script d'astrométrie existant avec le catalogue HORS LIGNE seulement. Catalogue profond existant : <SCRIPTS>\Catalogue Gaia profond – 20 nov\gaia_dr3_profond_horsehead.npz. AUCUN téléchargement.
3. FICHE DE L'IMAGE, mesurée une fois sur l'image ENTIÈRE (batir_modele_image.py, batir_etalonnage_image.py, saturation v14) : v23\fiche_horsehead.json + .md (français, court) : taille, type, bruit, FWHM, point zéro, parts de couleur, saturation par couleur et plateau des 10 plus brillantes, aigrettes de l'instrument (nombre, angles), 15 plus brillantes (G, position, saturée, plateau), Alnitak et σ Ori (G, position, rayon du halo visible, rayons de reflet), position de la tête de cheval et du bord lumineux d'IC 434, de NGC 2023 ; ce qui diffère de la Rosette / M8.
4. CARTE et 4 CARRÉS 768 px (tuile jugée = 512 central) : (a) la tête de cheval et le bord d'IC 434 (test : silhouette et bord gardés) ; (b) σ Orionis (géante) ; (c) Alnitak si elle est dans l'image (sinon la plus forte brillante à aigrettes) ; (d) le champ dense le plus chargé ou NGC 2023. v23\decoupes_horsehead.json (format de decoupes_m8_sho.json), carte PNG dans <SCRIPTS>\A_REGARDER – 21 nov soir\Horsehead_0_carte_carres.png (cv2.imencode().tofile), et pour chaque carré la tuile 512 d'entrée à x2, arcsinh 60 (comme vue_v18_v19.py de v21) : Horsehead_<carré>_entree.png.
POINT D'ARRÊT : après l'étape 4, tu t'arrêtes et rends compte. Tu NE lances PAS la piste 1 ni la chaîne.


RAPPORT FINAL (≤ 15 lignes) : chiffres clés de la fiche, Alnitak / σ Ori, les 4 carrés, chemins des PNG, ce qu'il faudrait changer (un seul changement) pour l'essai « géantes », surprises.
