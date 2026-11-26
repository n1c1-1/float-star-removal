# Validate tile choices

- Date : 2026-11-23 à 14:46 (heure de Paris)
- Numéro du jour : 06
- Type d'agent : agent général
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Tu es le valideur du choix des tuiles d'entraînement d'un réseau de retrait d'étoiles. L'orchestrateur a choisi à l'œil ; tu vérifies, tu ne choisis pas à sa place.

Entrées : <DONNEES>\Choix des tuiles v2 – 23 nov\ — choix_orchestrateur.csv (lis l'en-tête : numéros = candidates de candidats.csv ; [fx,fy] = centre de la tuile de 512 px en fraction de la vue entière de la mosaïque ; J1 = position de la tuile du jeu 1 contenant l'objet principal), candidats.csv, LISEZMOI.txt, mosaique_*.jpg. Le code qui a fait les mosaïques (et sait d'où vient chaque image, et l'échelle vue → image) : <SCRIPTS>\Choix des tuiles – 23 nov\. Tuiles du jeu 1 : <DONNEES>\Choix des tuiles – 23 nov\tuiles_<image>.csv (colonne jeu1) et les noms des paires <DONNEES>\Paires étoiles jeu 1 – 22 nov\<image>_g<x>_<y>.npz.

À faire :
1. Convertir chaque choix en coin haut gauche (x, y) de 512 px dans l'image d'ENTRÉE de la chaîne (mêmes coordonnées que tuiles_<image>.csv), bornée à l'image. Pour J1 : prendre la tuile du jeu 1 qui contient l'objet (m57 : l'anneau ; ngc7662 : la nébuleuse planétaire).
2. Vérifier chaque tuile, sur la tuile elle-même à l'étirement arcsinh 60 : (a) contient de la nébuleuse ou de la galaxie quand c'était le but (les [fx,fy] visent l'objet) ; (b) pas de ciel noir presque vide ; (c) pas de bord d'empilement, de zéro, de dégradé fort ou de bande ; (d) aucune étoile géante n'occupe plus d'un quart ; (e) recouvrement entre tuiles d'une même image ≤ 25 % ; (f) l'image d'entrée de la chaîne existe (prête, ou sinon pile brute avec dégradé : le signaler, ex. m101_a7s, m51_canon6d).
3. Pour une tuile qui échoue : proposer UN décalage ou remplacement précis (≤ 300 px) qui corrige, sans le retenir toi-même.
4. Sorties dans le même dossier : tuiles_retenues.csv (image;x;y;origine;verdict OK/A_REVOIR;motif;proposition) et UNE planche validation.jpg (≤ 2400 px de large, vignettes 200 px, étiquette image+n°, cadre rouge sur A_REVOIR).


Rapport final : 6 lignes au plus : nombre de tuiles OK / A_REVOIR, la liste courte des A_REVOIR avec motif et proposition (une ligne chacune, sinon renvoie au CSV si > 10), chemin de la planche.
