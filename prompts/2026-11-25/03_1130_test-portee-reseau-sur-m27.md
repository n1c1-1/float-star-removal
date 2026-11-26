# Test portée réseau sur M27

- Date : 2026-11-25 à 11:30 (heure de Paris)
- Numéro du jour : 03
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Tu travailles pour l'orchestrateur du réseau de retrait d'étoiles (projet dans <SCRIPTS>). Rapport final en 6 lignes maximum, en français, sans détails de code.

BUT : savoir pourquoi notre réseau laisse un disque plat gris-bleu à la place de la très grosse étoile de M27 ([outil de comparaison] y rend un fond propre). Hypothèse à tester : la portée du réseau (~180 px) est plus petite que le halo (~285 px de rayon), ou le découpage en tuiles à l'inférence coupe le halo.

À FAIRE :
1. Trouve le meilleur instantané de l'essai N2 (pas 26000) dans <DONNEES>\Entrainement étoiles nuit – 24 nov\ et le code d'inférence utilisé par la mesure (<SCRIPTS>\Entrainement nuit – 24 nov\mesurer_nuit.py) ; trouve l'image M27 du lot de validation (<DONNEES>\Validation étoiles – 22 nov\).
2. Passe N2 sur M27 de 4 façons : (a) exactement comme la mesure ; (b) tuiles d'inférence les plus grandes que la mémoire permet, avec grand recouvrement ; (c) image réduite ×2 (moyenne par blocs), sortie ré-agrandie ; (d) réduite ×3. Pour (c)/(d), seule la zone de la grosse étoile compte.
3. Planche JPEG : découpe ~800 px centrée sur la grosse étoile, étirement arcsinh 60, colonnes entrée | (a) | (b) | (c) | (d) | sortie [outil de comparaison] et modèle FLOAT si elles existent déjà dans le dossier de validation (ne les recalcule pas).
4. Rapport : dans quelle variante le disque disparaît ou recule, plus une mesure simple (moyenne du résidu dans l'anneau 50–300 px autour de l'étoile, par variante).

CONTRAINTES STRICTES :
- Tout calcul passe par : powershell -NoProfile -File <SCRIPTS>\lancer_borne.ps1 -Fils 4 -Log <journal> -Exe python -Args "-I \"<script>\" ..." (chemins avec espaces entre \"). Il fixe priorité basse, cœurs 4-15, 4 fils, 4060 Ti seule. Jamais la 3070. Un seul calcul GPU à la fois. Mémoire vive ≤ 12 Go pour ton processus ; vérifie la mémoire libre avant de lancer (> 20 Go). Pas de pool de processus.
- Tu n'arrêtes que tes propres PID, jamais par nom.
- Aucun téléchargement.
Donne dans le rapport le chemin de la planche.
