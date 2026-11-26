# Mélanges de poids N5 N6 N7

- Date : 2026-11-26 à 04:12 (heure de Paris)
- Numéro du jour : 01
- Type d'agent : agent général
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Projet : réseau de retrait d'étoiles (astrophotographie). Réponds en français simple. AUCUN message d'attente : quand un calcul tourne, attends-le dans un seul appel bloquant ; un seul rapport final, court (8 lignes + un tableau).

BUT : tester des mélanges de poids (moyenne pondérée des paramètres, même architecture NAFNet) entre trois réseaux affinés depuis le même départ E3 : N5, N6, N7. Aucun entraînement.
- Poids : N5 dans <DONNEES>\Entrainement étoiles N5 N6 – 25 nov\N5\ (meilleur pas dans N5_meilleur_pas.txt) ; N6 dans ...\N6\ (même logique, choisis le meilleur pas comme la chaîne l'a fait, sinon le dernier) ; N7 dans <DONNEES>\Entrainement étoiles N7 – 25 nov\ (N7_meilleur_pas.txt). Le code des tours est dans <SCRIPTS>\ dossiers « Tours N5 N6 – 25 nov » et « Tour N7 – 25 nov » (lis leurs JOURNAL.md et enchainer*.py pour savoir comment la chaîne charge un instantané, applique l'inférence à deux échelles V2 et mesure).
- Moyenne : pour chaque tenseur flottant, a*A + (1-a)*B ; les tampons entiers (compteurs) pris de A. Vérifie que les clés et formes sont identiques, sinon arrête-toi et dis-le.
- Mélanges, dans cet ordre : N7/N5 a=0.75 ; N7/N5 a=0.5 ; N7/N6 a=0.5 ; N5/N6 a=0.5 ; N5/N6 a=0.75.

MESURE : exactement la mesure de la chaîne N7 (13 images de validation + IC 1848, inférence V2, colonnes : restantes par bande G, fond méd., fond p99, structure méd., cœurs). Ajoute les lignes au tableau dans un NOUVEAU fichier <DONNEES>\Melanges N5 N6 N7 – 26 nov\tableau_melanges.md en recopiant les lignes N5, N6, N7, modèle FLOAT, [outil de comparaison] de <DONNEES>\Entrainement étoiles N7 – 25 nov\tableau_n7.md.
Planches JPEG (même étirement pour toutes les vignettes, comme planche_coeur_m31.jpg de N7) : entrée, N7, les deux meilleurs mélanges, [outil de comparaison], sur M31 A7S Newton (cœur) et Leo triplet Canon 6D.

RÈGLES MACHINE (impératives, la machine a planté plusieurs fois) :
- Tout calcul passe par <SCRIPTS>\lancer_borne.ps1 :
  powershell -NoProfile -File <SCRIPTS>\lancer_borne.ps1 -Fils 4 -Log <log> -Exe python -Args "-I \"<script>\" ..."
  (il fixe la 4060 Ti seule, priorité basse, cœurs 4-15). Un seul calcul GPU à la fois : enchaîne les mélanges dans UN script qui les fait l'un après l'autre.
- Ne lance que si la mémoire libre dépasse 10 Go ([logiciel tiers] en tient beaucoup) ; 8 Go au plus pour ton processus.
- Ne tue que tes propres PID (jamais par nom). Aucun téléchargement. Jamais la carte 3070.

RAPPORT FINAL : le tableau, chemins des planches, quel mélange est le meilleur et pourquoi en 2 lignes (critère : cœurs ≥ 0,8 d'abord, puis étoiles restantes G<8 et G11-14 et fond médian), le commit.
