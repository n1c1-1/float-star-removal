# Cœur de M31 creusé : cause

- Date : 2026-11-25 à 17:52 (heure de Paris)
- Numéro du jour : 06
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Tu travailles pour l'orchestrateur du réseau de retrait d'étoiles (<SCRIPTS>). Rapport final en 5 lignes, en français ; aucun message d'attente.

PROBLÈME : sur l'image de validation M31 A7S (m31_a7s_newton), le modèle N5 creuse le cœur de la galaxie en un trou sombre bleuté (planche <DONNEES>\Entrainement étoiles N5 N6 – 25 nov\planches\m31_a7s_newton.jpg ; [outil de comparaison] garde le cœur). Les mesures utilisent l'inférence « deux échelles V2 » (<SCRIPTS>\Portée M27 – 25 nov\deux_echelles_v1v2.py) : autour des étoiles Gaia G < 6,5, la sortie vient de l'image réduite ×2, sur un rayon mesuré dans l'entrée (≤ 300 px). Hypothèse 1 : le noyau de M31 figure dans le catalogue comme source brillante et V2 le traite comme une grosse étoile. Hypothèse 2 : le réseau lui-même prend le cœur compact pour une étoile (leçon connue : M31, M51, M27).

À FAIRE, SUR PROCESSEUR SEULEMENT (la carte entraîne N6, n'y touche pas) : sur une découpe de ~800 px centrée sur le cœur, (a) liste les sources G < 8 du catalogue utilisé dans cette zone ; (b) passe N5 (meilleur pas, voir N5_meilleur_pas.txt dans le dossier ci-dessus) en simple échelle, et en V2 ; (c) idem N2 (pas 26000, <DONNEES>\Entrainement étoiles nuit – 24 nov\). Planche entrée | N2 simple | N2 V2 | N5 simple | N5 V2 | [outil de comparaison], arcsinh 60. Dis laquelle des hypothèses tient, et si le problème concerne aussi d'autres images de validation (liste les sources G < 6,5 qui tombent dans un objet étendu brillant : cœurs de galaxie, nébuleuses planétaires).
