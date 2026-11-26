# Collect amateur web images

- Date : 2026-11-23 à 15:26 (heure de Paris)
- Numéro du jour : 07
- Type d'agent : agent général
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Tu rassembles des images d'astrophotographie AMATEUR publiées sur le web (déjà étirées), qui seront délinéarisées puis servir de paires d'entraînement à un réseau de retrait d'étoiles.

Source : Wikimedia Commons d'abord (API publique, fichiers en pleine résolution, licences libres ; catégories du type « Astrophotography », « Photographs by amateur astronomers », pages des objets Messier/NGC). Autre source publique seulement si Commons ne suffit pas, sans compte ni paiement. Client générique (User-Agent sans nom ni adresse personnels). Pas de téléchargement massif : une requête à la fois.

Ce qu'on cherche (sujets qui manquent à notre bibliothèque) : objets compacts brillants (nébuleuses planétaires : M27, M57, M97, NGC 7662, NGC 6543, NGC 7293, NGC 2392 ; M42 cœur ; M1), galaxies à lumière diffuse étendue (M31, M33, M81/M82, M101, M51, NGC 891, NGC 253, M104, M63, M64, M106), quelques amas globulaires (M13, M3, M5, M92) et quelques nébuleuses à filaments fins. Visée : 40 images retenues, au plus 3 par objet, d'auteurs et d'instruments variés (couleur et filtres).

Critères pour retenir (à vérifier sur l'image et sa page de description) :
- vraie photo amateur d'une pose longue empilée, pleine résolution (côté ≥ 2000 px), échelle plausible 0,7-4"/px ;
- étoiles présentes et NON réduites (pas de « star reduction », pas de composition sans étoiles + étoiles recollées, pas d'étoiles rapetissées : profils étoilés progressifs, petites étoiles visibles) ; refuser en cas de doute ;
- fond ni écrasé au noir pur (pas de grande zone à 0) ni blanchi ; pas de mosaïque à bords, pas de forte compression visible ; pas d'image de télescope professionnel (Hubble, ESO...) ni de rendu de relevé ;
- TIFF/PNG/FITS préférés, JPEG accepté si qualité correcte.


Règles absolues : tout calcul local via powershell -File <SCRIPTS>\lancer_borne.ps1 -Fils 2 -Log <fichier> -Args "<args python>" -Dossier <dossier> ; 2 fils, pas la carte graphique, RAM < 6 Go ; Python : <FLOAT>\.venv. Arrêt seulement par ton PID. Écrire seulement dans <SCRIPTS>\ et <DONNEES>\. Ne délinéarise rien, ne fais pas d'astrométrie : collecte seulement.

Rapport final : 5 lignes au plus : nombre retenu par famille d'objets, nombre refusé et motif principal, chemin de la planche, durée.
