# Tour court N9 cœurs compacts

- Date : 2026-11-26 à 14:16 (heure de Paris)
- Numéro du jour : 07
- Type d'agent : agent général
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Projet : réseau de retrait d'étoiles. Réponds en français simple. AUCUN message d'attente ; ta tâche s'arrête au lancement (après 1000 pas sans incident) : rapport final de 6 lignes (PID de la chaîne, vitesse, heure de fin prévue à l'horloge réelle, commit, toute différence avec N7 autre que celles demandées).

CONTEXTE :
- N7 (code <SCRIPTS>\Tour N7 – 25 nov\ : train_n7.py, paires_synthetiques_n7.py, enchainer_n7.py ; sorties <DONNEES>\Entrainement étoiles N7 – 25 nov\) = recette N5 + cœurs de galaxie synthétiques Sérsic sur 25 % des tuiles synthétiques (--coeurs 0.25), départ E3, et garde 0,81 des cœurs.
- Diagnostic (<SCRIPTS>\Diagnostic cœurs – 26 nov\, JOURNAL.md ; <DONNEES>\Diagnostic cœurs – 26 nov\) : le plafond vient d'un cœur compact, M32 sur la photo Canon 60D (effacé à 93 %). Sa perte est un pic central retiré. Nos cœurs synthétiques ont une largeur imposée ≥ 2,5 × la FWHM des étoiles (FACTEUR_LARGEUR), donc presque aucun cœur compact (FWHM médiane 18 px ; 90 % ont < 11 % du flux dans 3 px, contre 17-27 % pour M32).
- Les scripts de N8b/N8L (<SCRIPTS>\Tour N8L – 26 nov\) ont déjà : contrôle des cœurs dans le processus, reprise, garde-fou mémoire, mesure complète avec inferer_v2_mem.py. Réutilise-les (copies). ATTENTION : mesurer_n8l.py contenait deux erreurs (une chaîne coupée par un vrai retour à la ligne, un chemin non brut avec \f et \t) corrigées au commit 3ea3b9a : vérifie tes chemins (toujours r"...") et compile chaque script (py_compile) avant de lancer.

À FAIRE : tour court N9 = N7 (--coeurs 0.25, paires v33g, départ E3, mêmes données, lot, pas d'apprentissage) avec UN changement : 40 % des cœurs synthétiques sont COMPACTS : largeur 1,3 à 1,5 × la FWHM des étoiles de la tuile, Sérsic pointu (n 3 à 6), Re 1,5 à 6 px ; les 60 % restants inchangés. Vérifie sur 200 cœurs tirés que la part de flux dans 3 px des compacts couvre 15-30 % (écris-le dans le JOURNAL).
- 10 000 pas, planificateur sur 10 000 pas. Mémoire : [logiciel tiers] est fermé (~36 Go libres) : 2 chargeurs comme N7, mais garde le garde-fou mémoire (arrêt propre sous 3,5 Go libres, reprise > 8 Go).
- Contrôle des cœurs dans le processus aux pas 5000 et 10000, avec en plus la valeur du seul M32 Canon 60D dans le log. RÈGLE D'ARRÊT : moyenne des cœurs < 0,6 au pas 5000 -> arrêt.
- Après l'entraînement : mesure complète habituelle (V2, 13 images + IC 1848) de N9 pas 10000 ET de N7 pas 10000 (instantané N7 sur F:) ; tableau <DONNEES>\Entrainement étoiles N9 – 26 nov\tableau_n9.md avec les lignes de <DONNEES>\Entrainement étoiles N8L – 26 nov\tableau_n8l.md recopiées, plus une colonne « M32 60D » pour N9 et N7 ; planches cœur M31 (entrée, N7, N9, [outil de comparaison]) et vignette M32 Canon 60D (mêmes colonnes, même étirement).

RÈGLES MACHINE : tout calcul passe par <SCRIPTS>\lancer_borne.ps1 :
powershell -NoProfile -File <SCRIPTS>\lancer_borne.ps1 -Fils 4 -Log <log> -Exe python -Args "-I \"<script>\" ..."
(4060 Ti seule ; jamais la 3070 ; un seul calcul sur la carte, elle est libre). Lance seulement si la mémoire libre dépasse 10 Go. Ne tue que tes propres PID, jamais par nom. Aucun téléchargement.
