# Judge v33d: halos and faint stars

- Date : 2026-11-24 à 08:35 (heure de Paris)
- Numéro du jour : 05
- Type d'agent : agent général
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Tu travailles pour l'orchestrateur du projet « retrait d'étoiles ». Écris en français, sois bref : ton rapport tient en 10 lignes au plus.

CONTEXTE :
- Un essai d'entraînement de 2 h vient de tourner sur 141 paires réelles jugées par le juge v33c. Le tableau est dans <DONNEES>\Entrainement étoiles v33c – 24 nov\tableau_validation.md.
- Ce qui progresse : les étoiles brillantes, mieux retirées que par l'outil de référence.
- Ce qui régresse, deux points, que j'ai vus à l'œil sur <DONNEES>\Validation étoiles – 22 nov\Resultats\V33C_pas_13000\m27_mono_nb_planche.jpg comparée à ...\A_E3_pas_8000\ :
  (1) HALOS : à la place de la grosse étoile à aigrettes, le nouveau modèle laisse un DISQUE PLAT gris-bleu (un halo aplati), là où l'ancien laissait un fond sombre propre. Mon hypothèse : dans les paires réelles, le poids 0 des étoiles brillantes est un disque plafonné à 4 FWHM. Le halo étendu, au-delà, reste dans la cible au poids 1, et le réseau apprend à garder ce halo.
  (2) ÉTOILES FAIBLES : sur G 14-17, la part d'étoiles qui restent passe de 0,269 à 0,307. Mon hypothèse : la règle (b) ne prend que les pics de plus de 20 bruits. Les étoiles faibles oubliées par la chaîne restent donc dans la cible au poids 1, et le réseau apprend à les garder.
- Le juge v33c est dans <SCRIPTS>\Juge v33c – 24 nov\ (rejuger_v33c.py) et le juge v33 dans ...\Juge v33 – 23 nov\regles_v33.py. Les paires sont dans <DONNEES>\Paires étoiles v33 – 23 nov\ (*_v33c.npz, ou l'ancien fichier quand v33c n'a rien changé, et les paires « lâchée » de revue_orchestrateur.csv à exclure).

1. VÉRIFIE D'ABORD les deux hypothèses sur 3 ou 4 paires, en chiffres : la lumière du halo au-delà du disque de poids 0, dans la cible et au poids 1, et les étoiles du catalogue G 14-17 encore présentes dans la cible au poids 1. Si une hypothèse est fausse, dis-le et ne fais pas sa règle.
2. RÈGLE HALO : pour chaque étoile du catalogue assez brillante, mesure dans l'ENTRÉE le profil radial (médiane par anneaux) et le rayon où il retombe au fond local + 2 bruits. C'est l'étendue du halo, sans plafond à 4 FWHM, avec un plafond raisonnable à fixer par mesure. Si, dans la CIBLE, la moyenne à l'intérieur de ce rayon dépasse le fond juste au-delà de plus de 2 bruits, mets à 0 le disque jusqu'à ce rayon.
3. RÈGLE ÉTOILES FAIBLES : à chaque position du catalogue (G 14-17 et au-delà si le catalogue va plus loin), si la cible garde un pic de plus de 5 bruits au-dessus du fond local, mets à 0 un petit disque de rayon 1,5 FWHM. Reste local et ne touche pas aux zones où l'entrée n'a pas d'étoile visible.
DOSAGE : c'est un compromis, ni « tout à 0 » ni rien. Donne le poids 0 moyen, avant et après, par paire. Si une paire gagne plus de 8 points, regarde pourquoi et resserre la règle. Fais une planche avant/après pour 4 paires représentatives (dont une avec une grosse étoile dans une nébuleuse), dans <DONNEES>\Paires étoiles v33 – 23 nov\controle_v33d\.
SORTIE : des fichiers *_v33d.npz pour TOUTES les paires gardées, y compris celles sans changement, pour qu'on ait un jeu complet et sans ambiguïté. Ajoute poids0_v33d.txt.
Le juge v33d servira aussi aux paires des images web, que la chaîne va produire aujourd'hui : rends-le appelable sur un nouveau dossier de paires.

RÈGLES MACHINE : pas de GPU, une production de paires va le prendre. Lance tout calcul via powershell -File <SCRIPTS>\lancer_borne.ps1 -Fils 2 -Log <journal> -Exe python -Args "-I \"<script>\" ..." (les \" sont nécessaires pour les chemins à espaces). Un seul processus à la fois. Écris seulement dans <SCRIPTS>\ et <DONNEES>\. Arrête seulement tes propres PID. Pas de téléchargement.

RAPPORT : les hypothèses vérifiées ou non, en chiffres, le dosage avant/après et le chemin des planches.
