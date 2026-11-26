# Tours N5 N6 : fond et étoiles faibles

- Date : 2026-11-25 à 12:14 (heure de Paris)
- Numéro du jour : 05
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Tu travailles pour l'orchestrateur du réseau de retrait d'étoiles (<SCRIPTS>). Rapports en 6 lignes maximum, en français ; aucun message d'attente : attends tes calculs avec une veille qui ne se termine qu'à la fin.

CONTEXTE : le diagnostic <DONNEES>\Diagnostic écarts – 25 nov\ (lis LISEZMOI.txt, paires_resume.txt ; code dans starRemoval\Diagnostic écarts – 25 nov\) montre que les cibles des paires (*_v33d.npz dans <DONNEES>\Paires étoiles v33 – 23 nov\ et <DONNEES>\Paires étoiles APOD – 24 nov\) sont plus sombres que l'entrée dans le ciel « non touché » (ailes des étoiles du catalogue soustraites jusqu'à 150 px) : le réseau apprend à effacer la lumière diffuse (piédestal de M31, cœur de Leo). Et la moitié des étoiles G 14-17 a un poids 0 au centre, donc n'est pas enseignée.

ÉTAPE 1 — cibles v33g (processeur) : pour chaque paire, D = entrée − cible ; M = pixels de classe « non touché », moins les disques des étoiles Gaia G < 8 (rayon = halo mesuré dans l'entrée comme dans starRemoval\Portée M27 – 25 nov\deux_echelles_v1v2.py, variante V2, plafond 300 px). L = lissage gaussien normalisé de D sur M (G*(D·M) / G*M, sigma 40 px, par canal). cible_g = cible + L partout ; poids et classes inchangés. Écris *_v33g.npz à côté. Contrôles : médiane de (entrée − cible_g) sur M ≈ 0 ; part des étoiles G 14-17 gardant > 10 % du flux inchangée à 1 point près ; pas de marche visible au bord des classes. Planche de 6 paires (dont M31 A7S et la paire web au plus gros décalage) : entrée | cible v33d | cible v33g, arcsinh 60.
ÉTAPE 2 — poids v33h (processeur), sur v33g : pour chaque étoile Gaia G 14-17, disque de 2 FWHM : poids 1 si le flux restant dans la cible est entre −5 % et +5 % de l'entrée ; poids 0 s'il dépasse 10 % ; sinon inchangé. Écris *_v33h.npz. Donne la part de surface au poids > 0 avant/après.
ÉTAPE 3 — chaîne GPU, lancée par toi dès les étapes 1-2 faites, en un seul processus enchaîné : N5 = recette exacte de N2 (starRemoval\Entrainement nuit – 24 nov\, enchainer_nuit.py et paires_synthetiques_n2.py ; copie dans un nouveau dossier, ne modifie pas l'original), même départ E3, même nombre de pas que N2, mêmes proportions, seul changement : paires v33g. Puis mesure. Puis N6 = N5 avec paires v33h. Puis mesure. Les mesures utilisent l'inférence deux échelles V2 (deux_echelles_v1v2.py) et le code de mesure habituel ; tableau avec N2 (V2), N5, N6, modèle FLOAT, [outil de comparaison] ; planches M31 A7S, cœur de Leo, M42 Dobson, M27.
Dès le lancement de l'étape 3, rends-moi : PID du processus de la chaîne, heure de fin prévue de N5 et de N6. Ensuite ta tâche est finie (je surveille le PID).

CONTRAINTES STRICTES :
- Tout calcul via powershell -NoProfile -File <SCRIPTS>\lancer_borne.ps1 -Fils 4 -Log <journal> -Exe python -Args "-I \"<script>\" ..." (chemins avec espaces entre \"). 4060 Ti seule, jamais la 3070, un seul calcul GPU. Processeur : au plus 2 calculs à toi en même temps, 4 fils chacun, pas de pool. Mémoire : [logiciel tiers] occupe 17 Go ; lance seulement si mémoire libre > 15 Go ; ≤ 8 Go par processus de préparation (paires une par une), ≤ 12 Go pour l'entraînement. Saturer processeur ou mémoire déconnecte toute la machine : faute à éviter absolument.
- Tu n'arrêtes que tes propres PID, jamais par nom.
- Nuance : correction locale et dosée ; si une paire change de plus de 3 points de surface au poids > 0 à l'étape 2, signale-le sans forcer.
