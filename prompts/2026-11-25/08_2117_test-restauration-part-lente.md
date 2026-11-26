# Test restauration part lente

- Date : 2026-11-25 à 21:17 (heure de Paris)
- Numéro du jour : 08
- Type d'agent : agent général
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Tu travailles pour le projet de réseau de retrait d'étoiles (astrophotographie). Réponds en français simple. Aucun message d'attente : un seul rapport final, court (10 lignes maximum + un tableau).

BUT : tester, sans entraînement, une correction à l'inférence pour récupérer les fonds lisses (disques de galaxie, nébuleuses diffuses) que le réseau N5 assombrit. Idée : la couche retirée R = entrée − sortie ne doit contenir que des objets compacts (étoiles et halos). On remet dans la sortie la part lente de R : sortie_corrigée = sortie + L, où L = lissage gaussien normalisé (sigma 40 px) de R, calculé en excluant :
- les disques de halo des étoiles Gaia G < 8 (rayon comme dans la correction v33g : voir le code de fabrication des cibles v33g dans <SCRIPTS>\, dossier « Tours N5 N6 – 25 nov », commit 575a203, et son JOURNAL.md) ;
- avec écrêtage itératif (4 tours) comme pour v33g.
Dans les zones de l'inférence à deux échelles V2 (étoiles G < 6.5, code deux_echelles_v1v2.py dans le dossier « Portée M27 – 25 nov »), ne rien remettre (L = 0, transition douce sur 30 %).
Essaie aussi sigma 80 px si sigma 40 donne des auréoles autour des étoiles moyennes. Deux variantes au plus.

DONNÉES : les sorties de validation de N5 (13 images jamais vues + IC 1848), avec inférence V2, sous <DONNEES>\Entrainement étoiles N5 N6 – 25 nov\ (cherche les sorties de N5_mesure ; si elles ne sont pas sauvegardées, recalcule-les avec <DONNEES>\Entrainement étoiles N5 N6 – 25 nov\N5\ (meilleur pas dans N5_meilleur_pas.txt) MAIS sur le PROCESSEUR uniquement : la carte 4060 Ti est occupée par un entraînement, n'y touche pas).

MESURE : le code de mesure habituel (copie de mesurer_nuit.py du dossier « Tours N5 N6 – 25 nov ») avec la colonne « cœurs » (code dans le dossier « Tour N7 – 25 nov » ou « Cœur M31 – 25 nov »). Tableau : N5 seul, N5 + correction (chaque variante), avec les lignes de référence [outil de comparaison] et modèle FLOAT déjà dans <DONNEES>\Entrainement étoiles N5 N6 – 25 nov\tableau_n56.md. Colonnes : étoiles restantes par bande G, fond méd., fond p99, structure, cœurs. Le critère : fond et structure s'améliorent SANS que les étoiles restantes (surtout G8-11 et G11-14) remontent.
Planches JPEG côte à côte (entrée, N5, N5 corrigé, [outil de comparaison]) sur M31 A7S Newton et Leo triplet Canon 6D, même étirement pour toutes les vignettes.

RÈGLES MACHINE (impératives, la machine a planté plusieurs fois) :
- Tout calcul passe par <SCRIPTS>\lancer_borne.ps1 : powershell -NoProfile -File <SCRIPTS>\lancer_borne.ps1 -Fils 4 -Log <log> -Exe python -Args "-I \"<script>\" ...". 4 fils au plus. Pour forcer le processeur, mets CUDA_VISIBLE_DEVICES="" dans ton script (os.environ avant import torch) ou n'utilise pas torch.cuda.
- Ne lance que si la mémoire libre dépasse 15 Go ; 8 Go au plus pour ton processus. Un seul processus de calcul à la fois.
- Ne tue que tes propres PID (jamais par nom). Aucun téléchargement. Pas de carte 3070.
- Les sorties [outil de comparaison] ne servent qu'en référence.

RAPPORT FINAL : le tableau, chemins des planches, ton avis en 3 lignes (la correction marche-t-elle, y a-t-il des auréoles ou des étoiles qui reviennent), le commit.
