# Chain fixes into new version

- Date : 2026-11-23 à 14:40 (heure de Paris)
- Numéro du jour : 03
- Type d'agent : agent général
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Tu travailles pour l'orchestrateur du projet « réseau d'étoiles » (retrait d'étoiles). Tâche précise : intégrer deux correctifs déjà trouvés et testés dans une NOUVELLE version de la chaîne classique, puis rejouer des paires pour un avant/après.

Contexte à lire (seulement ces passages) :
- <SCRIPTS>\Diagnostic défauts – 23 nov\JOURNAL.md (causes et correctifs : interrupteurs DIAG_GRAIN_FENETRE dans Piste3\remplir.py add_missing_grain, et DIAG_T3_TEXTURE dans Passes\passe_finale_t3_p.py étape D ; code dans ce dossier, sous-dossier chaine).
- Chaîne de production actuelle : <SCRIPTS>\Methode v32 production – 22 nov ; production des paires : <SCRIPTS>\Paires jeu 1 – 22 nov (produire_jeu1.py, construire_paire.py, rejuger.py).
- Paires existantes : <DONNEES>\Paires étoiles jeu 1 – 22 nov\<image>_<carré>.npz.

À faire :
1. Copier v32 dans un nouveau dossier « Methode v33 production – 23 nov » (ne jamais modifier v32 ni le dossier du diagnostic), y mettre les deux correctifs ACTIFS par défaut. Ne rien changer d'autre. LISEZMOI.txt en français (quoi, d'où, date, supprimable ou non).
2. Rejouer avec v33 les 11 paires touchées (liste dans le JOURNAL du diagnostic : horsehead g128_128, ic5070 g4224_1664, m13 g1152_1152, m13 g640_128, m97 g640_128, ngc7000_gfx g7296_5248, ngc7000_tiers1 g1152_640 / g128_128 / g3200_2176, ngc869 g1152_128, rosette g1152_640) + 3 paires saines au choix (non-régression), même entrée, même juge des zones. Sorties dans <DONNEES>\Paires v33 essai – 23 nov\ (LISEZMOI.txt). Si une tuile est trop longue (piste 1 sur processeur > 1 h), utilise --piste1-gpu ; sinon passe-la et dis-le.
3. Vérifier par les valeurs : plus aucune cible hors plage (|cible| > 3 x max local de l'entrée + 50), et compter les pixels où la cible est plus lumineuse que l'entrée (> 5 bruits, lissé 1,5, taches ≥ 25 px) à poids > 0, avant/après.
4. Faire UNE planche JPEG (≤ 2400 px de large) : pour chaque paire rejouée, entrée | cible v32 | cible v33, étirement arcsinh 60, et une planche de gros plans x4 sur les 4 pires zones d'avant. Dans le même dossier F:.

Règles absolues :
- Tout calcul se lance par powershell -File <SCRIPTS>\lancer_borne.ps1 -Fils <n> -Log <fichier> -Args "<args python>" -Dossier <dossier> (priorité basse, cœurs 4-15, 4060 Ti seule). Toi : 6 fils au plus en tout, une seule tâche sur la carte à la fois, et la carte est à toi seul pour cette tâche. Garde la RAM sous 20 Go. Python avec -I pour les scripts qui lisent des fichiers téléchargés.
- Pour arrêter un calcul : seulement par le PID que tu as lancé (et ses enfants), jamais par nom.
- Écrire seulement dans <SCRIPTS>\ et <DONNEES>\. Aucun téléchargement (catalogue Gaia hors ligne : variable ASTRO_GPU_GAIA_EXTRACTS comme dans v32).
- Ne lance aucun entraînement, ne produis pas d'autres paires.

Rapport final : 8 lignes au plus, en français simple : ce qui est fait, chiffres avant/après, chemins des deux planches, durée par tuile mesurée (piste 1 carte et le reste), tout problème. Pas de détails de code.
