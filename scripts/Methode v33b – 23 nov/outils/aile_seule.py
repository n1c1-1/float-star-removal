# ======================================================================
# aile_seule.py
# Rôle     : Module de la chaîne (voir sa docstring ci-dessous et scripts/LISEZMOI.md).
# Étape    : 2. Chaîne classique de soustraction guidée par Gaia (v33b) (voir scripts/LISEZMOI.md)
# Entrées  : image empilée (unités de bruit), astrométrie, extrait Gaia profond hors ligne
# Sorties  : image sans étoiles, cartes de classes
# Exemple  : python aile_seule.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
# Le coeur (r 0-4 px) ne converge pas dans l'itération sur l'image entière (essais du 21 nov, JOURNAL) : le modèle de l'image
# ne corrige plus que l'aile (r >= 5 px, plein à 7 px) ; le coeur est laissé à la correction du coeur de la piste 1 sur la
# découpe (rayon 2,5 largeurs, existante). Une seule région (le terme régional servait au coeur). Lancé une fois.
from pathlib import Path
p = Path(__file__).resolve().parents[1] / "Piste1" / "modele_psf_image.py"
t = p.read_text(encoding="utf-8-sig")


def rep(a, b):
    global t
    assert t.count(a) == 1, (t.count(a), a[:80])
    t = t.replace(a, b)


rep('VERSION = "v7-2"', 'VERSION = "v7-3"')
rep('''R_CORE = 12.0                  # coeur 2D par région jusqu'ici (fondu 10 -> 12)''',
    '''R_CORE = 12.0                  # (terme régional 2D : abandonné, voir AILE)
AILE = (5.0, 7.0)              # le modèle de l'image corrige r >= 5 px (plein à 7 px) ; le coeur est laissé à la correction du
                               # coeur de la piste 1 sur la découpe (r < 2,5 largeurs ~ 7,5 px)''')
rep('''    for g in ((4, 3, 2, 1) if not __import__("os").environ.get("MPI_G1") else (1,)):''',
    '''    for g in (1,):                 # une région : le coeur (seul à varier dans le champ) est corrigé sur chaque découpe''')
rep('''                S[c, g_] += D if not __import__("os").environ.get("MPI_G1") else 0.0''', '''                pass''')
# débogage retiré
i0 = t.index('                if c == 1 and b in (int(bin_of(12.75)), int(bin_of(14.25))) and __import__("os").environ.get("MPI_DBG"):')
i1 = t.index('        accS = np.zeros((C, ng, n * n))')
t = t[:i0] + t[i1:]
rep('''                d[rbins > R] = 0.0
                drho[c, b] = d''', '''                d[rbins > R] = 0.0
                d *= np.clip((rbins - AILE[0]) / (AILE[1] - AILE[0]), 0, 1)     # coeur non touché
                drho[c, b] = d''')
rep('''            worst = max(worst, float(np.nanmax(np.abs(mn[:, :CHECK_R + 1]))))''',
    '''            worst = max(worst, float(np.nanmax(np.abs(mn[:, int(AILE[0]):CHECK_R + 1]))))''')
rep('''pire |reste| tranches 12-16.5, r <= {CHECK_R} :''', '''pire |reste| tranches 12-16.5, r {int(AILE[0])}-{CHECK_R} :''')
rep('''Forme :  modèle(étoile) = S_région (coeur 2D, interpolé entre régions) + rho_tranche(r) (profil radial par tranche d'éclat G).
  S_région : forme 2D du coeur (r <= 12 px), propre à chaque région de l'image (optique), commune à tous les éclats ;
  rho_tranche : correction radiale propre à chaque tranche de G (image débruitée : la forme dépend de l'éclat), r <= 28 px,
                au-delà de 12 px tirée des étoiles de la tranche ET des tranches plus brillantes (aile étendue).''',
    '''Forme :  modèle(étoile) = S (forme 2D de départ, empilée sur les étoiles G < 14,5 de l'image entière)
                          + rho(r ; G) (correction radiale par demi-magnitude G, interpolée en G pour chaque étoile ; image
                            débruitée : la forme change avec l'éclat, même à l'intérieur d'une tranche de 1-1,5 mag),
  rho corrige l'AILE seulement (r >= 5 px, plein à 7 px) ; au-delà de 12 px tirée des étoiles de la tranche ET des tranches
  plus brillantes (aile étendue). Le COEUR (r < 5 px) ne converge pas dans cette itération (essais du 21 nov : dépassement
  puis blocage à -10 / +6 bruits, voir JOURNAL) : il est laissé à la correction du coeur de la piste 1 sur chaque découpe.''')
rep('''L'ajustement des étoiles est CELUI de la piste 1 (fit_star, mêmes rayons, même anneau de fond, position par couleur v4) :
la table des écarts est donc mesurée pour la façon dont le modèle est ensuite ajusté (principe de DAOPHOT).
Arrêt : profils empilés plats à +-0.1 bruit de ciel dans chaque tranche G 12-13.5 / 13.5-14.5 / 14.5-15.5 / 15.5-16.5 et
chaque couleur (r 0-24 px), ou 4 mises à jour.''',
    '''Positions : l'ajustement de la piste 1 (fit_star, position par couleur v4) ; flux de chaque couleur réajusté contre le ciel
loin de l'étoile (plan sur l'anneau 29-35 px, comme EPSFBuilder) ; un fond pris dans l'aile faisait dériver le modèle.
Arrêt : profils empilés plats à +-0.1 bruit de ciel dans chaque tranche G 12-13.5 / 13.5-14.5 / 14.5-15.5 / 15.5-16.5 et
chaque couleur (r 5-24 px), ou 4 mises à jour.''')
p.write_text(t, encoding="utf-8")
print("ok")
