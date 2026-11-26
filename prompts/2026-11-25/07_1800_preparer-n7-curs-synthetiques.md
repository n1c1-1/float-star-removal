# Préparer N7 : cœurs synthétiques

- Date : 2026-11-25 à 18:00 (heure de Paris)
- Numéro du jour : 07
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Tu travailles pour l'orchestrateur du réseau de retrait d'étoiles (<SCRIPTS>). Rapports en 6 lignes max, en français ; aucun message d'attente.

CONTEXTE : nos réseaux (N2, N5) effacent le cœur compact brillant de M31 (trou noir au centre ; [outil de comparaison] le garde) : ils prennent un cœur de galaxie pour une étoile, faute d'exemples (leçon connue : M31, M51, M27). La chaîne en cours (PID 321448, dossier <DONNEES>\Entrainement étoiles N5 N6 – 25 nov\, code <SCRIPTS>\Tours N5 N6 – 25 nov\) entraîne N6 puis le mesure, fin vers 23:15. N5 = recette N2 avec paires *_v33g.npz ; N6 = paires *_v33h.npz. Ne touche pas à ce processus.

À FAIRE :
1. Générateur de CŒURS synthétiques, dans une copie de paires_synthetiques_n2.py (starRemoval\Entrainement nuit – 24 nov\) : sur 25 % des tuiles synthétiques, ajouter 1 ou 2 objets étendus compacts, présents à l'IDENTIQUE dans l'entrée ET la cible (on les garde) : profil de Sérsic n de 1 à 4, rayon effectif 4 à 60 px, ellipticité 0 à 0,6, éclat central de 50 à 5000 bruits, parfois écrêté au maximum comme un cœur saturé, couleur de galaxie (jaune-blanc) ou de planétaire (bleu-vert), toujours plus large qu'une étoile du même éclat (largeur à mi-hauteur ≥ 2,5 fois celle des étoiles de la tuile). Les étoiles synthétiques restent posées par-dessus et retirées dans la cible comme avant. Planche de 8 exemples entrée | cible, arcsinh 60, dans <DONNEES>\Entrainement étoiles N7 – 25 nov\planche_coeurs.jpg.
2. Mesure : ajoute à une copie du code de mesure une colonne « cœurs » = rapport moyen flux sortie / flux entrée dans 15 px autour du cœur de M31 A7S (x=2802, y=2314, voir <DONNEES>\Cœur M31 – 25 nov\coeur_resultats.json) et des autres cœurs de galaxie des images de validation que tu peux repérer (même méthode). Idéal 1. [outil de comparaison] ≈ 1.
3. Chaîne N7, qui démarre SEULE dès que le PID 321448 n'existe plus (boucle d'attente légère, 1 vérification par minute) : choisit les paires v33h si, dans <DONNEES>\Entrainement étoiles N5 N6 – 25 nov\tableau_n56.md, N6 a G 14-17 ≤ celui de N5 ET structure médiane ≤ celle de N5 + 0,001 ; sinon v33g. Écrit ce choix dans son journal. Puis N7 = cette recette exacte (même départ E3, mêmes pas que N2/N5, mêmes proportions), seul changement : les cœurs synthétiques. Puis mesure (deux échelles V2, deux_echelles_v1v2.py, + colonne cœurs), tableau avec N5, N6, N7, modèle FLOAT, [outil de comparaison] (recalcule la colonne cœurs pour N5, N6, FLOAT si des sorties existent, sinon laisse vide), planches M31 A7S cœur, Leo, M27, M42 Dobson.
Lance cette chaîne d'attente maintenant via lancer_borne et rends-moi : son PID, la planche des cœurs, l'heure de fin prévue de N7 (~5 h après 23:15).
