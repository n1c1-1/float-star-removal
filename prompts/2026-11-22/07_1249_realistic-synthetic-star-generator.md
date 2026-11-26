# Realistic synthetic star generator

- Date : 2026-11-22 à 12:49 (heure de Paris)
- Numéro du jour : 07
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Tu construis un GÉNÉRATEUR D'ÉTOILES SYNTHÉTIQUES RÉALISTES, à très forte variété, pour compléter le jeu d'entraînement du réseau d'étoiles de FLOAT (logiciel d'astrophotographie de Nicolas). Les paires principales viennent de vraies images passées au script de retrait classique ; tes étoiles seront AJOUTÉES sur de vrais fonds sans étoiles, comme le jeu « mixed » du 5 nov, la seule recette qui a battu l'outil de référence. Rien n'est entraîné ici.

CONTEXTE À LIRE (court) :
- <SCRIPTS>\RECETTE_ET_LECONS.md : sections 0, 1 (règle de lancement « Déjà essayé / Ce qui diffère »), 3.3, 3.4, 4 et 6.
- <SCRIPTS>\Lecons – 22 nov\extrait_avant_15nov.md : lignes A5, A7 à A12 (premier générateur, échecs, mesures).
- Code du premier générateur, dans <SCRIPTS>\ : README.md, psf.py (optique de Fourier : pupille, obstruction, araignée, Zernike, seeing, suivi), stars.py, instruments.py, catalogue.json, calib.py, star_measure.py. À RÉUTILISER, pas à réécrire.

CE QUI MANQUAIT AU PREMIER GÉNÉRATEUR (mesuré, à corriger) : auréole 2-3 fois trop faible ; saturation à niveau fixe, alors que chaque étoile écrêtée a SON niveau de plateau (40 à 100 % du maximum de l'image selon l'étoile : Tête de Cheval G 4,9 95-100 %, σ Ori 81-85 %, Alnitak 55-58 % ; M57 40-55 %) ; pas de décalage aléatoire entre couleurs par étoile (Rosette : 0,03-0,05 px par axe, R et B ~3 % plus larges que G) ; aigrettes idéales, alors qu'en bandes étroites elles sont en pointillés dont le pas dépend du filtre (M8 SHO, 6 aigrettes) ; halos ronds, alors qu'ils sont allongés (σ Ori, ±25 %) avec des reflets décentrés et des dizaines de rayons fins autour des géantes ; nombre d'aigrettes peu varié (on a « galéré » avec : Rosette 4, M8 6, M57 4 à 45°, Tête de Cheval croix + rayons).

DEMANDE (ses mots) : « la génération d'étoiles synthétiques réalistes avec plein de types… on a galéré avec le nombre d'aigrettes… il faut en varier la combinatoire de manière assez forte. »
Variété à couvrir par tirage combinatoire (instrument par image × paramètres par étoile) : réfracteur sans aigrette, Newton 4 aigrettes (araignée 4 branches, angle libre), 3 branches = 6 aigrettes, araignée courbe (aucune), aigrettes doubles (branches décalées), largeur et longueur qui croissent avec l'éclat ; pointillés par filtre en bandes étroites (SII/Ha/OIII) et arc-en-ciel en couleur ; halo avec aile r^-2 à r^-3 jusqu'à plusieurs centaines de px, allongement, reflets (fantômes) décentrés, anneaux de reflet ; rayons fins des géantes ; écrêtage à niveau propre par étoile (plateau plat à bord doux) ; décalage et largeur par couleur ; couleur tirée des indices Gaia (BP−RP) réalistes ; forme qui varie dans le champ (coma, astigmatisme en bord), suivi imparfait, mise au point (donut) ; densité d'un champ vide à la Voie lactée ; étoiles doubles serrées.

CALIBRATION SUR NOS VRAIES IMAGES (obligatoire) : les 4 images de test et leurs mesures existent : Rosette et M8 SHO (« Methode v21 SHO – 22 nov »), M57 (« Methode v22 M57 – 22 nov », fiche_m57.json), Tête de Cheval (« Methode v23 Horsehead – 22 nov », fiche_horsehead.json), chacune avec son modèle d'étoile de l'image entière (Piste1\sorties\<image>\, voir batir_modele_image.py), son catalogue Gaia et sa saturation. Les plages de tes tirages doivent couvrir ce qu'on y mesure (et un peu au-delà), pas des valeurs inventées.

ÉTAPE 1 (POINT D'ARRÊT) : audit + plan, sans gros calcul, environ 45 min :
(a) pour chaque image, 6 vraies étoiles (faible, moyenne, brillante, saturée, à aigrettes, la plus brillante) et l'étoile du premier générateur la plus proche en éclat et en largeur, côte à côte : vignettes x4 à l'étirement arcsinh 60 et à fort étirement, profils radiaux superposés jusqu'à 300 px, profil le long d'une aigrette ; planche dans <SCRIPTS>\A_REGARDER – 21 nov soir\Synth_0_audit_<image>.png (cv2.imencode().tofile, chemins non ASCII) ;
(b) une table des écarts mesurés (ce qui manque, de combien) ;
(c) un plan court : ce que tu changes dans le générateur, la liste des paramètres tirés et leurs plages (avec la source de chaque plage : mesure sur nos images ou référence publiée), et le test de réalisme prévu (vraies et fausses étoiles de même éclat mélangées, jugées à l'aveugle par un agent frais : on vise ~50 % d'erreur).
Écris tout dans un nouveau dossier <SCRIPTS>\Etoiles synthetiques – 22 nov\ (LISEZMOI.txt en tête : quoi, pourquoi, comment lancer) et rends compte. N'implémente pas la suite avant accord.

RAPPORT (≤ 20 lignes) : les 5 plus gros écarts mesurés, le plan, les plages, chemins des planches.
