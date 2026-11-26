# Inventory of available astro images

- Date : 2026-11-22 à 15:15 (heure de Paris)
- Numéro du jour : 10
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Tu fais l'INVENTAIRE des images astro disponibles pour construire une banque de formes d'étoiles réelles (et des fonds) pour l'entraînement du réseau d'étoiles de FLOAT. LECTURE SEULE partout : tu ne déplaces, ne copies, ne modifies, ne supprimes rien hors de ton dossier de sortie. Aucun calcul lourd (au plus 2 fils ; lire des en-têtes FITS, des tailles, des fichiers .json/.csv, pas d'empilement ni d'astrométrie). Aucun téléchargement.

TROIS SOURCES (ses mots : « mes images, les jeux de test de FLOAT (quelques dizaines d'images), et les images déjà empilées voire déjà étirées que tu avais récupérées ») :
1. SES IMAGES : inventaire déjà fait le 13 nov dans <FLOAT>_inventaireG\ (inventaire_G_astro.xlsx, cibles_par_nuit.csv, astrometry.csv, calibration.csv… : lis-les d'abord) ; bruts sur <DISQUE>/astro (et <DISQUE>/astro_per_target, liens) ; ses projets FLOAT et empilements : cherche les dossiers de projets (fichiers project.json / project.astroproj, ou équivalents) sur E: (<DISQUE>/astro_float, <DISQUE>/astro_gpu…), F: (heart_nebula, NGC7822, astro-photo…), D: (astro_photo M33…). Pour chaque cible : a-t-on un EMPILEMENT linéaire (FITS/TIFF/npy) ou seulement des bruts ? capteur (A7S, APN, caméra couleur, mono + filtres), instrument/focale si connu.
[passage retiré]
[passage retiré]

POUR CHAQUE IMAGE (ou ensemble) : cible ; source (lui / test FLOAT / web) ; chemin ; état (bruts seulement / empilé linéaire / étiré non linéaire / dé-étiré) ; format ; taille en px ; capteur (couleur / mono filtres, lesquels) ; instrument et focale si connus ; échelle si une astrométrie existe ; déjà passée dans FLOAT (projet) ; déjà utilisée par le retrait classique (Rosette, M8 SHO 57_sho, M57 set_24, Tête de Cheval user_Horsehead, NGC 6543, NGC 869 set_38, IC 443 58_hoo) ; catalogue Gaia profond déjà sur disque (<SCRIPTS>\Catalogue Gaia profond – 20 nov) ; doublons (même cible/même donnée à plusieurs endroits) ; utilisable pour la banque de formes d'étoiles (oui / non et pourquoi : trop petite, étirée sans retour possible, compression JPEG forte, pas d'étoiles brillantes…), et intérêt (FWHM, aigrettes ou non, instrument différent : la VARIÉTÉ des instruments compte le plus).

SORTIE (seul endroit où tu écris) : <SCRIPTS>\Bibliotheque images – 22 nov\ : LISEZMOI.txt (quoi, date, supprimable), inventaire.csv (une ligne par image), BILAN.md en français clair : totaux par source et par état, liste des instruments/capteurs distincts, cibles de sa liste prioritaire présentes (M57, NGC 6543, M97, NGC 7662, M13, M1, M82, NGC 6888 ; M27 = juge, à signaler à part), ce qui manque, et un classement « à traiter en premier pour la banque » (variété d'instruments d'abord). Pas de git (je committerai).
RAPPORT (≤ 20 lignes) : les totaux, les instruments distincts, les 10 premières images recommandées, les trous.
