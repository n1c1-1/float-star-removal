# Realistic delinearisation study

- Date : 2026-11-23 à 14:40 (heure de Paris)
- Numéro du jour : 05
- Type d'agent : agent général
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Tu travailles pour l'orchestrateur du projet « réseau d'étoiles » de FLOAT (logiciel d'astrophoto de Nicolas). But de ta tâche : une DÉLINÉARISATION RÉALISTE d'images astro déjà étirées (images d'amateurs trouvées sur le web) — elle doit produire une image qui aurait raisonnablement pu être la version linéaire (niveaux du fond, profil et flux relatif des étoiles, forme du bruit, couleur). C'est un point clé : ces images serviront de paires d'entraînement, et des utilisateurs donneront aussi au modèle des images délinéarisées. Pas d'inversion exacte exigée (une courbe inconnue ne s'inverse pas exactement) : crédible, pragmatique, ni bâclée ni perfectionniste.

1. Lire la fonction existante de FLOAT qui délinéarise une image étirée (ticket #174 ; dépôt <FLOAT>, LECTURE SEULE : cherche « unstretch », « délinéar », « linearise » ; git log --grep 174). Résumer en 3 lignes ce qu'elle suppose.
2. Banc de mesure : prendre 6 de nos piles LINÉAIRES variées (couleur et filtres, nébuleuse, galaxie, amas ; voir l'inventaire dans <SCRIPTS>\Bibliotheque images – 22 nov), les étirer comme le ferait un amateur avec des traitements courants et variés (arcsinh, histogramme/courbe en S type « autostretch », MTF/STF, avec ou sans débruitage, saturation des couleurs, sortie 8 bits JPEG et 16 bits TIFF), puis les délinéariser avec la fonction FLOAT, et comparer à l'original linéaire (mis à la même échelle) : niveau et pente du fond, profil radial des étoiles par tranche d'éclat (rapport de flux), écart type du bruit et sa dépendance au signal, couleurs. Un tableau court.
3. Si c'est loin : proposer et tester des améliorations fondées sur des méthodes publiées (cite la source : estimation de la courbe de transfert depuis les étoiles via leur profil PSF connu, ou depuis la photométrie Gaia des étoiles du champ par astrométrie ; modèle de bruit poissonien+lecture ; etc.). Une amélioration à la fois, mesurée sur le même banc ; s'arrêter quand c'est crédible, ne pas tourner en rond.
4. Planche JPEG (≤ 2400 px) : pour 3 piles, original linéaire | étiré | délinéarisé FLOAT | délinéarisé amélioré, même étirement d'affichage arcsinh 60, + gros plans x4 sur une étoile brillante et une zone de nébuleuse.


Règles absolues : tout calcul via powershell -File <SCRIPTS>\lancer_borne.ps1 -Fils 3 -Log <fichier> -Args "<args python>" -Dossier <dossier> ; 3 fils au plus en tout ; PAS la carte graphique (prise par un autre agent) ; RAM < 12 Go. Arrêt seulement par tes PID. Écrire seulement dans <SCRIPTS>\ et <DONNEES>\ ; ne modifie RIEN dans le dépôt FLOAT (importer ses modules en lecture est permis). Aucun téléchargement pendant un calcul ; si un catalogue Gaia te manque pour une pile, choisis une autre pile.

Rapport final : 10 lignes au plus, en français simple, sans jargon : ce que suppose la fonction FLOAT, le tableau résumé (écarts principaux avant/après amélioration), ta recommandation, chemin de la planche, durée. Tout de suite si blocage.
