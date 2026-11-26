# Mesurer N8 et N7 au pas 5000

- Date : 2026-11-26 à 07:17 (heure de Paris)
- Numéro du jour : 03
- Type d'agent : agent général
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Projet : réseau de retrait d'étoiles. Réponds en français simple. AUCUN message d'attente : attends ton calcul dans un seul appel bloquant ; un seul rapport final de 6 lignes + un tableau.

CONTEXTE : la chaîne N8 (code <SCRIPTS>\Tour N8 – 26 nov\, sorties <DONNEES>\Entrainement étoiles N8 – 26 nov\) a été arrêtée à 05:33 par manque de mémoire ([logiciel tiers] tient 31 Go, ce n'est pas notre processus ; n'y touche pas). L'instantané N8\snap\pas_5000.pt existe. La chaîne prévoyait de mesurer N8 et N7 aux pas 5000 et 10000 ; fais seulement les deux mesures au pas 5000.

À FAIRE : avec exactement la mesure prévue par la chaîne N8 (lis son enchaîneur et son JOURNAL.md : inférence V2, 13 images + IC 1848, colonnes par bande G, fond méd., fond p99, structure, cœurs), mesure :
1. N8 pas 5000 (<DONNEES>\Entrainement étoiles N8 – 26 nov\N8\snap\pas_5000.pt)
2. N7 pas 5000 (instantané de N7 sous <DONNEES>\Entrainement étoiles N7 – 25 nov\, s'il existe ; sinon dis-le et ne mesure que N8).
Écris le tableau dans <DONNEES>\Entrainement étoiles N8 – 26 nov\tableau_n8.md avec les lignes N5, N7 (final), N7/N5 0,75, modèle FLOAT, [outil de comparaison] recopiées de <DONNEES>\Melanges N5 N6 N7 – 26 nov\tableau_melanges.md. Planche cœur M31 (entrée, N7 final, N8 pas 5000, [outil de comparaison], même étirement que planche_coeur_m31.jpg de N7).

MÉMOIRE (impératif : la machine n'a que ~9 Go libres) : un seul processus, les deux mesures l'une après l'autre dans un même script. Dans ce script, avant chaque image, lis la mémoire libre (psutil.virtual_memory().available) : sous 4 Go, arrête proprement en l'écrivant dans le log. Lance seulement si la mémoire libre dépasse 8 Go.

RÈGLES MACHINE : tout calcul passe par <SCRIPTS>\lancer_borne.ps1 :
powershell -NoProfile -File <SCRIPTS>\lancer_borne.ps1 -Fils 4 -Log <log> -Exe python -Args "-I \"<script>\" ..."

RAPPORT FINAL : le tableau, le chemin de la planche, en 2 lignes N8 contre N7 au même pas (cœurs, étoiles G<8 et G11-14, fond médian), le commit.
