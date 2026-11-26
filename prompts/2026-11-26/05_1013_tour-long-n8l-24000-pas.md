# Tour long N8L 24000 pas

- Date : 2026-11-26 à 10:13 (heure de Paris)
- Numéro du jour : 05
- Type d'agent : agent général
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Projet : réseau de retrait d'étoiles. Réponds en français simple. AUCUN message d'attente ; ta tâche s'arrête quand l'entraînement tourne depuis 1000 pas sans incident : rapport final de 6 lignes (PID de la chaîne, mémoire, vitesse, heure de fin prévue à l'horloge réelle, commit).

CONTEXTE : le tour court N8 (code <SCRIPTS>\Tour N8 – 26 nov\ : train_n8b.py, enchainer_n8b.py, mesurer_n8_10000.py, inferer_v2_mem.py, JOURNAL.md ; sorties <DONNEES>\Entrainement étoiles N8 – 26 nov\, tableau_n8.md) = recette N7 avec --coeurs 0.10 au lieu de 0.25, 10 000 pas. Résultat : fond médian 0,0011 (N7 : 0,0023), cœurs 0,785, mais étoiles restantes plus hautes que N7 (G11-14 0,109 contre 0,094) parce que le tour est court. Décision (rythme convenu avec Nicolas : tour long seulement pour la recette gagnante, plafonné ~24 000 pas) : tour long N8L.

À FAIRE : N8L = recette N8 exacte (--coeurs 0.10, paires v33g, même données, même lot, même pas d'apprentissage), départ E3 (<ESSAIS>\starRemoval\train\E3\snap\step_20000.pt), 24 000 pas, planificateur sur 24 000 pas.
- Mémoire réduite comme N8b (1 chargeur, prefetch 2) et garde-fou mémoire (arrêt propre sous 3,5 Go libres avec last.pt, reprise automatique quand > 8 Go libres) : [logiciel tiers] (pas notre processus, n'y touche jamais) tient 31,5 Go.
- Instantanés tous les 1000 pas comme d'habitude. Mesure « cœurs » dans le processus aux pas 5000, 10 000, 15 000, 20 000, écrite dans le log. RÈGLE D'ARRÊT : cœurs < 0,6 à l'un de ces contrôles -> arrêter le tour (« arrêt : cœurs trop bas »).
- Après l'entraînement : mesure complète habituelle (inférence V2 avec garde-fou mémoire, 13 images + IC 1848, colonnes par bande G, fond méd., fond p99, structure, cœurs) au meilleur pas de validation ET au pas 24 000 ; tableau <DONNEES>\Entrainement étoiles N8L – 26 nov\tableau_n8l.md avec les lignes de tableau_n8.md recopiées ; planche cœur M31 (entrée, N7, N8L, [outil de comparaison], même étirement) et planche Leo triplet Canon 6D (mêmes colonnes).
- Essai de fumée de quelques pas, puis lancement si la mémoire libre dépasse 8 Go.

RÈGLES MACHINE : tout calcul passe par <SCRIPTS>\lancer_borne.ps1 :
powershell -NoProfile -File <SCRIPTS>\lancer_borne.ps1 -Fils 4 -Log <log> -Exe python -Args "-I \"<script>\" ..."
(4060 Ti seule ; jamais la 3070 ; un seul calcul sur la carte). Ne tue que tes propres PID, jamais par nom. Aucun téléchargement. Écris uniquement dans les deux dossiers ci-dessus.
