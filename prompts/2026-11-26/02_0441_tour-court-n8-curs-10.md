# Tour court N8 cœurs 10 %

- Date : 2026-11-26 à 04:41 (heure de Paris)
- Numéro du jour : 02
- Type d'agent : agent général
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Projet : réseau de retrait d'étoiles (astrophotographie). Réponds en français simple. AUCUN message d'attente ; ta tâche s'arrête au lancement : rapport final court (6 lignes) avec le PID de la chaîne et l'heure de fin prévue.

CONTEXTE : N7 (dossier code <SCRIPTS>\Tour N7 – 25 nov\, train_n7.py, enchainer_n7.py, JOURNAL.md ; sorties <DONNEES>\Entrainement étoiles N7 – 25 nov\) = recette N5 + cœurs de galaxie synthétiques sur 25 % des tuiles synthétiques (--coeurs 0.25), départ E3, 34 500 pas. Résultat : cœurs 0,19 -> 0,81, mais fond médian 0,0011 -> 0,0023 et étoiles restantes un peu plus hautes (G11-14 0,072 -> 0,094). Seul changement entre N5 et N7 = les cœurs, donc leur proportion est le levier à tester.

NOUVELLE RÈGLE DE RYTHME (décidée par Nicolas) : tours courts ~10 000 pas pour lire la tendance ; mesure intermédiaire ; règle d'arrêt écrite avant de lancer.

À FAIRE : tour N8 = N7 avec UN SEUL changement : --coeurs 0.10 au lieu de 0.25. Départ E3, 10 000 pas, le reste identique à N7.
2. Instantanés aux pas 5 000 et 10 000. Dans le processus d'entraînement, au pas 5 000, mesure seulement la colonne « cœurs » (rapide, même code que la chaîne N7) et écris-la dans le log. RÈGLE D'ARRÊT : si les cœurs de N8 au pas 5 000 sont inférieurs à 0,3, arrêter le tour (écrire « arrêt : cœurs trop bas » dans le log).
3. Après l'entraînement, la chaîne mesure (mesure complète habituelle de la chaîne N7, inférence V2, 13 images + IC 1848, colonnes par bande G, fond méd., fond p99, structure, cœurs) : N8 pas 5 000, N8 pas 10 000, ET N7 aux pas 5 000 et 10 000 (instantanés de N7 déjà sur F:, s'ils existent) pour comparer à pas égal. Tableau dans <DONNEES>\Entrainement étoiles N8 – 26 nov\tableau_n8.md, avec les lignes N5, N7, modèle FLOAT et [outil de comparaison] recopiées de <DONNEES>\Melanges N5 N6 N7 – 26 nov\tableau_melanges.md. Planche cœur M31 (entrée, N7, N8 10k, [outil de comparaison], même étirement) comme planche_coeur_m31.jpg de N7.
4. Un petit essai de fumée (quelques pas) avant de lancer, pour vérifier que tout tourne.

RÈGLES MACHINE (impératives, la machine a planté plusieurs fois) :
- Tout calcul passe par <SCRIPTS>\lancer_borne.ps1 :
  powershell -NoProfile -File <SCRIPTS>\lancer_borne.ps1 -Fils 4 -Log <log> -Exe python -Args "-I \"<script>\" ..."
  (4060 Ti seule, priorité basse, cœurs 4-15). Un seul calcul sur la carte à la fois (la carte est libre maintenant). Jamais la 3070.
- Ne lance que si la mémoire libre dépasse 10 Go ([logiciel tiers] en tient beaucoup) ; 12 Go au plus pour l'entraînement.
- Ne tue que tes propres PID (jamais par nom). Aucun téléchargement.
- Écris uniquement dans le dossier de code ci-dessus et dans <DONNEES>\Entrainement étoiles N8 – 26 nov\ (avec LISEZMOI.txt en français).

RAPPORT FINAL : PID de la chaîne, heure de fin prévue (horloge réelle), commit, et toute différence avec N7 autre que --coeurs et le nombre de pas.
