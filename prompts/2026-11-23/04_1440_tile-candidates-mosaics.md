# Tile candidates mosaics

- Date : 2026-11-23 à 14:40 (heure de Paris)
- Numéro du jour : 04
- Type d'agent : agent général
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Tu prépares le choix des tuiles d'entraînement d'un réseau de retrait d'étoiles. C'est l'orchestrateur qui choisira à l'œil ; toi tu réduis ce qu'il doit regarder.

Matériau : <DONNEES>\Choix des tuiles – 23 nov\ (lis LISEZMOI.txt : 43 planches <image>.jpg, grille de tuiles 512 px au pas de 256, tuiles_<image>.csv avec x, y). Les images source/entrées : voir <SCRIPTS>\Paires jeu 2 – 23 nov\planches_choix.py (d'où il lit chaque image). Images écartées du jeu 2 : <SCRIPTS>\Paires jeu 2 – 23 nov\ecartees.csv — ne les propose pas (ngc4631_stellina, iris_tiers2, ngc7000_canonra, m8_asi2600, m57_lco), mais garde-les listées à part.

À faire, par image (38 images) :
1. Proposer 8 tuiles candidates de 512 px, peu chevauchantes (recouvrement ≤ 25 %), choisies par leur CONTENU : nébuleuse ou galaxie (lumière diffuse, filaments, bandes sombres) avec des étoiles de toutes tailles ; JAMAIS de ciel noir presque vide ; éviter les bords abîmés (empilement, zéro, dégradé fort) et les tuiles dont une très grosse étoile occupe plus d'un quart. Critères mesurables simples (part de pixels au-dessus du fond de plus de 5 bruits à l'échelle 8 px, nombre d'étoiles détectées, présence d'une étoile brillante) + diversité entre les 8.
2. Une mosaïque par groupe de 4 images (≈ 10 mosaïques), JPEG ≤ 2000 px de large : pour chaque image, une petite vue entière (≤ 450 px) avec les 8 tuiles numérotées 1-8, et à côté les 8 vignettes à 160 px, étirement arcsinh 60. Lisible : numéros gros, une couleur.
3. Un fichier candidats.csv (image;num;x;y;mesures) et un fichier choix_vierge.csv à remplir (image;nums choisis).
Sorties : <DONNEES>\Choix des tuiles v2 – 23 nov\ avec LISEZMOI.txt en français (quoi, d'où, date, supprimable : oui). Code dans <SCRIPTS>\Choix des tuiles – 23 nov\ (nouveau dossier, ne modifie pas Paires jeu 2).


Rapport final : 5 lignes au plus : chemins des mosaïques (liste), images sans 8 bonnes candidates et pourquoi, durée.
