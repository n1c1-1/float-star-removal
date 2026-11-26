# Pourquoi les cœurs plafonnent à 0,8

- Date : 2026-11-26 à 14:12 (heure de Paris)
- Numéro du jour : 06
- Type d'agent : agent général
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Projet : réseau de retrait d'étoiles. Réponds en français simple. AUCUN message d'attente ; un seul rapport final de 10 lignes au plus + un petit tableau.

QUESTION : nos réseaux N7, N7/N5 mélangé et N8L gardent ~0,80 de la lumière des cœurs de galaxie (colonne « cœurs » : flux sortie/entrée dans 15 px autour de 6 cœurs : M31 A7S cœur et M32, M31 Canon 60D cœur et M32, M65, M66), [outil de comparaison] 0,984. Faire varier la part de cœurs synthétiques (25 % puis 10 %) ne change rien : le plafond est ailleurs. Trouve où.

DONNÉES (lecture seule) :
- Mesures par cœur : <DONNEES>\Entrainement étoiles N8L – 26 nov\coeurs\ (json par essai), idem dans <DONNEES>\Entrainement étoiles N7 – 25 nov\coeurs\ ; [outil de comparaison] aussi (cherche coeurs_*reference* ou équivalent). Code de la mesure : coeurs_mesure.py dans <SCRIPTS>\Tour N7 – 25 nov\ (ou Tour N8L).
- Sorties sans étoiles N8L pas 24000 : <DONNEES>\Entrainement étoiles N8L – 26 nov\starless_N8L_24000\ ; entrées et [outil de comparaison] : là où le code de mesure les lit.
- Code des cœurs synthétiques : paires_synthetiques_n7.py dans « Tour N7 – 25 nov » (Sérsic, Re 4-60 px, couleurs, brillance).

À FAIRE (processeur seulement, petit) :
1. Tableau par cœur : N7, N8L 24000, [outil de comparaison]. Lesquels perdent ? Tous un peu, ou un ou deux beaucoup ?
2. Pour le ou les pires : profil radial (entrée, N8L, [outil de comparaison]) sur 0-40 px. La perte est-elle un pic central retiré (noyau ponctuel pris pour une étoile), un creux annulaire, ou une baisse uniforme ?
3. Compare avec ce que fabriquent nos cœurs synthétiques : brillance de crête (relative au fond et au bruit), taille, présence ou non d'un noyau ponctuel très brillant (saturé ?). Le cas réel qui perd est-il hors de ce qu'on fabrique ?
4. Conclusion en 3 lignes : la cause la plus probable et le changement unique à essayer au prochain tour.
Une planche JPEG des profils ou vignettes du pire cœur (entrée / N8L / [outil de comparaison], même étirement) dans <DONNEES>\Diagnostic cœurs – 26 nov\ avec LISEZMOI.txt en français.
