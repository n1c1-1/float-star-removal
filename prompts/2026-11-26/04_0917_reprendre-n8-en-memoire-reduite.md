# Reprendre N8 en mémoire réduite

- Date : 2026-11-26 à 09:17 (heure de Paris)
- Numéro du jour : 04
- Type d'agent : agent général
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Projet : réseau de retrait d'étoiles. Réponds en français simple. AUCUN message d'attente ; ta tâche s'arrête quand l'entraînement tourne depuis 1000 pas sans incident : rapport final de 6 lignes (PID de la chaîne, mémoire mesurée, vitesse, heure de fin prévue à l'horloge réelle, commit).

CONTEXTE : le tour N8 (code <SCRIPTS>\Tour N8 – 26 nov\, train_n8.py et enchaîneur ; sorties <DONNEES>\Entrainement étoiles N8 – 26 nov\) a été arrêté au pas ~5000 sur 10 000 à 05:33 par la garde mémoire : [logiciel tiers] (pas notre processus, n'y touche jamais) tient 31,5 Go, il ne reste que ~8,6 Go libres. L'entraînement prenait ~6 Go en mémoire vive (processus principal ~2 Go + 2 processus de chargement de données ~1,9 Go chacun). N8\last.pt (pas ~5000) et N8\snap\pas_5000.pt existent. Une mesure de N8 au pas 5000 est déjà faite (tableau_n8.md) : ne la refais pas.

À FAIRE :
1. Regarde ce que contient last.pt (état de l'optimiseur, du planificateur du pas d'apprentissage, numéro de pas). Si tout y est, ajoute une reprise (--reprendre) dans une COPIE du script (train_n8b.py ; ne modifie pas les scripts existants) qui repart du pas 5000 avec optimiseur et planificateur restaurés, pour finir au pas 10 000. Si l'état n'y est pas, dis-le et repars de E3 pour 10 000 pas (même recette, --coeurs 0.10).
2. Réduis la mémoire : 0 processus de chargement de données (ou 1 au plus), et tout ce qui limite la mémoire sans changer la recette (pas de changement de lot, de pas d'apprentissage, de données). Écris dans JOURNAL.md exactement ce qui change.
3. Garde-fou dans le script : tous les 100 pas, lis psutil.virtual_memory().available ; sous 3,5 Go, sauvegarde last.pt et arrête proprement avec une ligne « arrêt mémoire » dans le log.
4. Après l'entraînement, la chaîne mesure N8 pas 10000 (même mesure que tableau_n8.md : inférence V2, 13 images + IC 1848, colonnes par bande G, fond méd., fond p99, structure, cœurs) et ajoute la ligne à tableau_n8.md, plus la planche cœur M31 (entrée, N7 final, N8 pas 10000, [outil de comparaison]). La mesure doit aussi avoir le garde-fou mémoire (il existe déjà dans inferer_v2_mem.py, commit 5f11a01).
5. Essai de fumée de quelques pas, puis lancement. Lance seulement si la mémoire libre dépasse 8 Go. Après 1000 pas, mesure la mémoire vive du processus et la vitesse, puis rends ton rapport.

RÈGLES MACHINE : tout calcul passe par <SCRIPTS>\lancer_borne.ps1 :
powershell -NoProfile -File <SCRIPTS>\lancer_borne.ps1 -Fils 4 -Log <log> -Exe python -Args "-I \"<script>\" ..."
