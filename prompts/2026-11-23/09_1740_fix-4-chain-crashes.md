# Fix 4 chain crashes

- Date : 2026-11-23 à 17:40 (heure de Paris)
- Numéro du jour : 09
- Type d'agent : agent général
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

La piste 1 de la chaîne de retrait d'étoiles plante sur 4 images ; corrige ces bogues dans une copie, sans lancer de production.

Chaîne : <SCRIPTS>\Methode v33 production – 23 nov. Production en cours (NE PAS la toucher ni ses fichiers) : <SCRIPTS>\Production v33 – 23 nov (cherche-y les journaux d'erreur des images et la façon dont la piste 1 est appelée), sorties <DONNEES>\Paires étoiles v33 – 23 nov. Erreurs : bubble_a7s (ValueError de forme dans retrait_modele.solve), heart_a7s et ic443_40da (IndexError dans modele_psf_image.py), ic1318_stellina (AxisError). Tuiles : <DONNEES>\Choix des tuiles v2 – 23 nov\tuiles_finales.csv.

À faire : (1) reproduire chaque erreur sur le PROCESSEUR (pas la carte : elle est prise par la production), sur la plus petite donnée qui la déclenche ; (2) trouver la cause réelle (forme de tableau, liste vide d'étoiles de calibration, image à un canal, etc.) et corriger à la source dans une copie « Methode v33b – 23 nov » (seuls les fichiers corrigés diffèrent de v33 ; LISEZMOI.txt en français) ; un correctif ne doit rien changer aux images qui marchaient : vérifie sur une tuile déjà produite (même résultat qu'en v33 au bruit près, sur processeur si c'est court, sinon par lecture du code) ; (3) s'arrêter là : ne lance pas la production des tuiles manquantes (je la lancerai après celle en cours, fin prévue ~01:40).


Rapport final : 5 lignes au plus : cause de chaque erreur en une ligne simple, preuve de non-régression, commande exacte pour produire les tuiles manquantes avec v33b.
