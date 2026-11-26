# Diagnose flat disc at bright stars

- Date : 2026-11-25 à 02:37 (heure de Paris)
- Numéro du jour : 01
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Diagnostic court, en français, rapport de 10 lignes maximum.

Contexte : un réseau NAFNet de retrait d'étoiles (essais N1/N2, code dans <SCRIPTS>\Entrainement nuit – 24 nov\, snapshots sous <DONNEES>\Entrainement étoiles nuit – 24 nov\) laisse, à la place des très grosses étoiles (ex. la grosse étoile à aigrettes de M27, planche <DONNEES>\Entrainement étoiles nuit – 24 nov\planches\M27_N2_pas_26000.jpg rangée 2), un DISQUE PLAT bleu-gris au lieu d'un fond propre. L'outil de référence rend un fond propre (M27_reference.jpg).

Question : pourquoi ? Vérifier par les données, sans entraîner :
1. Dans les paires d'entraînement (<DONNEES>\Paires étoiles v33 – 23 nov\*_v33d.npz, <DONNEES>\Paires étoiles APOD – 24 nov\*_v33d.npz ; clés input/target/weight), combien de pixels de poids >0 se trouvent à moins de R px d'une étoile très brillante (cœur saturé ou quasi), et que vaut la cible là (fond propre ou résidu de halo) ? Bref : le réseau voit-il jamais un exemple de « grosse étoile -> fond propre » au poids 1 ?
2. Dans le générateur synthétique (paires_synthetiques_n2.py et ce qu'il importe), les très brillantes étoiles ont-elles un halo aussi étendu que celle de M27 (rayon en px comparé), et leur cible est-elle le fond propre ?
3. Champ réceptif du NAFNet utilisé vs diamètre du halo M27 : le réseau peut-il « voir » le bord du halo depuis son centre ?
Conclure par la cause la plus probable et UN levier concret.

Règles : lecture seule sauf fichiers de travail dans <SCRIPTS>\Diagnostic disque – 25 nov\ (le créer). Python : python -I, OMP/MKL/TORCH threads = 4, pas de GPU nécessaire (si besoin, seulement CUDA_VISIBLE_DEVICES=0, jamais la 3070). Aucun téléchargement. Ne tuer aucun processus. Pas de bruit ajouté aux images.
