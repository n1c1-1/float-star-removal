# Build and launch N3 chain

- Date : 2026-11-25 à 02:42 (heure de Paris)
- Numéro du jour : 02
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Tâche : préparer puis LANCER une chaîne d'entraînement N3 (retrait d'étoiles), sans revenir me demander. Rapport final en français, 8 lignes max (ce qui a été fait, PID, heure de fin prévue, problèmes).

Contexte à lire d'abord (court) : <SCRIPTS>\Diagnostic disque – 25 nov\ (diag_disque.py, m27.json, paires.json, synth.json) et le code de la nuit <SCRIPTS>\Entrainement nuit – 24 nov\ (enchainer_nuit.py, train_nuit.py, paires_synthetiques_n2.py, mesurer_nuit.py). Juge : <SCRIPTS>\Juge v33d – 24 nov\ (rejuger_v33d.py + regles). Diagnostic : le réseau laisse un disque plat à la place des très grosses étoiles car (1) dans les paires réelles, autour des étoiles très brillantes, la cible garde le halo au poids 1 (mesure du halo plafonnée à 40 px dans le juge v33d), (2) le générateur synthétique ne fait pas de halos aussi grands que M27 (M27 : ~38 bruits à 60 px, 1 bruit à 285 px ; gabarit synthétique plafonné à 128 px de rayon, tuiles 256).

A. Générateur « grands halos » : étoiles imposées (celles de N2, sur fonds de nébuleuse réels) avec halos à l'échelle M27 : pics 3e4 à 3e5 bruits, halo à ~20-60 bruits à 60 px et descendant à 1 bruit entre 150 et 300 px, gabarit d'au moins 384 px de rayon, aigrettes ; ces tuiles synthétiques en 512 px (le reste inchangé) ; cible = fond propre, poids 1 sur tout le halo. Dosage modéré : grands halos sur ~25 % des tuiles synthétiques, pas plus. Vérifier visuellement 4 exemples (planche jpg dans le dossier) avant de lancer.
B. Juge v33e (copie de v33d) : mesure du halo par profil radial sans le plafond 40 px (jusqu'à 300 px) et poids 0 sur l'étendue où la cible garde le halo d'une étoile très brillante. Nuance : seulement les étoiles très brillantes (cœur écrêté ou quasi), pas toutes les étoiles. Rejuger les paires des deux dossiers <DONNEES>\Paires étoiles v33 – 23 nov et <DONNEES>\Paires étoiles APOD – 24 nov en écrivant des *_v33e.npz à côté (ne rien écraser) ; respecter revue_orchestrateur.csv (tuiles lâchées) ; pour APOD, ne rejuger QUE les tuiles qui ont un *_v33d.npz (38 tuiles). Donner la hausse moyenne de poids 0 (doit rester modérée ; si > +8 points en moyenne, réduire et le dire).
C. Chaîne (un seul processus GPU à la fois), chaque essai suivi de sa mesure (même mesure et même tableau que la nuit, M27 planche incluse, ajouter les lignes au tableau) :
  - N3a : levier A seul (paires v33d), départ snapshot N2 pas 26000, ~3 h.
  - N3b : leviers A + B (paires v33e), départ N2 pas 26000, ~3 h.
  - N3L (contrôle durée) : recette N3b, départ E3 <ESSAIS>\starRemoval\train\E3\snap\step_20000.pt, 8 h, programme du taux d'apprentissage adapté à 8 h.
  Sorties : <DONNEES>\Entrainement étoiles N3 – 25 nov\ (créer, avec LISEZMOI.txt en français).

Règles strictes : écrire seulement dans <SCRIPTS>\ et <DONNEES>\. Lancer TOUT calcul via powershell -NoProfile -File <SCRIPTS>\lancer_borne.ps1 -Fils <n> -Log <fichier> -Dossier <dossier> -Exe python -Args "-I \`"<script>\`" ..." (chemins avec espaces entre \`" ; total ≤ 12 fils, rejugement CPU ≤ 6 fils). GPU : uniquement la 4060 Ti (le lanceur fixe CUDA_VISIBLE_DEVICES=0), jamais la 3070. Aucun téléchargement. Ne tuer que tes propres PID (jamais par nom). Ne jamais ajouter de bruit artificiel aux images d'entraînement. Ne pas utiliser les sorties de l'outil de référence comme données. Les 13 images de validation + M27 restent hors entraînement. Lancer la chaîne dès que A et B sont prêts et vérifiés ; vérifier par nvidia-smi qu'elle tourne sur l'index 0.
