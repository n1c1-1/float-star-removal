# Diagnostic écarts communs

- Date : 2026-11-25 à 11:30 (heure de Paris)
- Numéro du jour : 04
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Tu travailles pour l'orchestrateur du réseau de retrait d'étoiles (projet <SCRIPTS>). Rapport final en 12 lignes maximum, en français simple, chiffres à l'appui. Tu diagnostiques ; tu ne modifies ni la chaîne ni les juges ni l'entraînement.

CONTEXTE : réseau NAFNet entraîné depuis E3 (modèle sur données générées seules) par affinage sur des paires réelles (entrée avec étoiles / cible sans étoiles faite par une chaîne classique / carte de poids 1, 0,5 ou 0) plus 60 % de tuiles d'étoiles synthétiques. Meilleur essai : N2 pas 26000. Tableau de validation (13 images jamais vues) dans <DONNEES>\Entrainement étoiles nuit – 24 nov\tableau_nuit.md :
[passage retiré]
[passage retiré]
[passage retiré]
- en revanche étoiles brillantes G<8 : N2 0,012, mieux que les deux.
[passage retiré]
Code de mesure : <SCRIPTS>\Entrainement nuit – 24 nov\mesurer_nuit.py ; sorties des modèles sur la validation sous <DONNEES>\Validation étoiles – 22 nov\Resultats\. Paires : <DONNEES>\Paires étoiles v33 – 23 nov\*_v33d.npz (débruitées et brutes) et <DONNEES>\Paires étoiles APOD – 24 nov\*_v33d.npz. Registre des leçons : starRemoval\RECETTE_ET_LECONS.md (lis-le d'abord, sections 4 et 6).

QUESTIONS (une cause probable chiffrée par écart, et UNE correction proposée, un seul changement) :
1. Fond : que mesure exactement « fond médian » ? Sur la validation, l'écart de N2 est-il un décalage de niveau, une dérive de couleur, un lissage (bruit retiré), des taches, ou concentré sur quelques images ?
2. Structure : même question.
3. Étoiles faibles : dans les paires d'entraînement, pour les étoiles Gaia G14-17 (catalogues profonds utilisés par la chaîne, dans « Methode v33b – 23 nov »), quelle part garde >10 % de son flux dans la CIBLE là où le poids est > 0 ? Quelle répartition de poids (0 / 0,5 / 1) sous ces étoiles ? Même critère que la validation.
4. Fond dans les paires : dans le ciel sans étoile au poids > 0, la cible diffère-t-elle de l'entrée (décalage, rapport de bruit cible/entrée, lissage) ? Séparer versions débruitées, brutes, web.
Si une planche aide, une seule, petite, et donne son chemin.

CONTRAINTES STRICTES :
- Processeur seulement, pas de GPU (un autre agent l'utilise). Tout calcul passe par : powershell -NoProfile -File <SCRIPTS>\lancer_borne.ps1 -Fils 4 -Log <journal> -Exe python -Args "-I \"<script>\" ..." (chemins avec espaces entre \"). Au plus 2 calculs à toi en même temps, 4 fils chacun, pas de pool de processus. Mémoire vive ≤ 12 Go par processus : charge les paires une par une. Vérifie la mémoire libre avant de lancer (> 20 Go). Saturer processeur ou mémoire déconnecte toute la machine : c'est la faute à éviter absolument.
- Tu n'arrêtes que tes propres PID, jamais par nom.
- Nuance : on cherche un compromis, pas des règles extrêmes (un défaut local se traite localement).
