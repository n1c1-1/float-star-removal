# ======================================================================
# generer_exemples.py
# Rôle     : Refait EXACTEMENT les images d'exemple du dépôt (docs/images) : des paires synthétiques « entrée | cible |
#            différence », avec le générateur publié dans scripts/. Aucune image n'est lue : tout sort du code et des graines.
# Entrées  : aucune image. Le code de scripts/ (fonds : backgrounds.py, defects.py ; étoiles : etoiles2.py,
#            paires_synthetiques_n7.py, paires_synthetiques_n3.py ; catalogue.json des instruments).
#            Les graines sont écrites ci-dessous (TUILES, GRAINE_CACHE).
# Sorties  : <sortie>/exemple_01_champ_lisse.jpg ... exemple_06_fond_defauts.jpg, <sortie>/planche_synthetique.jpg,
#            <empreintes> (sha256 des tableaux entrée et cible de chaque tuile, puis des images).
# Exemple  : python generer_exemples.py --sortie ../docs/images --empreintes EMPREINTES.txt
#            (options : --scripts <dossier des scripts>, --cache <fichier>)
# Ce que fait chaque tuile : on tire un fond SYNTHÉTIQUE (le ciel, sans étoile, avec son bruit) ; on pose dessus des étoiles
#            synthétiques ; la CIBLE est ce fond, copié tel quel (on le connaît avant d'y poser les étoiles) ; l'ENTRÉE est le
#            fond plus les étoiles (et leur propre bruit de photons). Le bruit du fond est le même dans l'entrée et la cible.
# Étirement : le même pour les trois panneaux et pour toutes les images : asinh(v / 3) / 10, v en unités de bruit du ciel
#            (celui du réseau), multiplié par un gain d'affichage fixe (GAIN) et décalé d'un noir fixe (NOIR).
# Modèle d'étoiles : chemin SANS banque de formes mesurées (il n'y en a aucune dans le dépôt) : le modèle paramétrique seul
#            (instrument tiré du catalogue, optique de Fourier, auréole, aigrettes, plateau d'écrêtage, couleurs, bruit de photons).
# Calcul   : processeur seulement, 4 fils.
# ======================================================================
"""Refait les images d'exemple du dépôt. Voir l'en-tête ci-dessus et docs/images/LEGENDES.md."""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True      # ne laisse aucun __pycache__ dans le dépôt

# Calcul sur processeur seulement, au plus 4 fils (avant tout import de bibliothèque de calcul)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
_FILS = "4"
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _FILS

import argparse
import hashlib
import importlib.util
import math
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------------------------------------------- GRAINES
#: graine du cache des « styles d'image » (instrument, PSF, auréole, aigrettes), et nombre de styles construits
GRAINE_CACHE = 2026
N_STYLES = 16

#: une tuile = un fond (graine_fond, taille, défauts ou non) + des étoiles (graine_etoiles) + ce qui est imposé
#:   extras : étoiles très brillantes imposées (x, y, pic en bruits) ; avec 3 colonnes de plus (A60, gamma, r1) : grand halo
#:   coeurs : True : un ou deux cœurs de galaxie synthétiques (gardés dans l'entrée ET la cible)
TUILES = [
    dict(nom="exemple_01_champ_lisse", titre="1. Champ d'étoiles sur fond lisse", taille=256, graine_fond=1114,
         defauts=False, graine_etoiles=1114, module="n7", extras=None, coeurs=False),
    dict(nom="exemple_02_tres_brillantes", titre="2. Étoiles très brillantes sur nébuleuse", taille=384, graine_fond=1203,
         defauts=False, graine_etoiles=1203, module="n7", extras=[(110, 120, 30000.0), (290, 90, 9000.0)],
         coeurs=False),
    dict(nom="exemple_03_halo_aigrettes", titre="3. Étoile avec grand halo et aigrettes", taille=384, graine_fond=1304,
         defauts=False, graine_etoiles=1304, module="n3", extras=[(192, 192, 37000.0, 38.0, 2.33, 285.0)], coeurs=False),
    dict(nom="exemple_04_coeur_galaxie", titre="4. Cœur de galaxie gardé", taille=384, graine_fond=1407,
         defauts=False, graine_etoiles=1407, module="n7", extras=None, coeurs=True),
    dict(nom="exemple_05_champ_dense", titre="5. Champ dense, étoiles faibles", taille=256, graine_fond=1501,
         defauts=False, graine_etoiles=1501, module="n7", extras=None, coeurs=False),
    dict(nom="exemple_06_fond_defauts", titre="6. Fond avec défauts d'empilement", taille=256, graine_fond=1611,
         defauts=True, graine_etoiles=1611, module="n7", extras=None, coeurs=False),
]

#: affichage : v en unités de bruit du ciel ; image = GAIN * asinh(v / 3) / 10 + NOIR, bornée à [0, 1]
GAIN = 4.0
NOIR = 0.04

QUALITE_JPEG = 82


# ---------------------------------------------------------------------------------------------------------------- code publié
def _charger_modules(scripts: Path, cache: Path):
    """Charge le générateur publié (sans le modifier) ; le cache de styles est celui de la sortie temporaire."""
    os.environ["RACINE_SCRIPTS"] = str(scripts)
    for sous in ("", "Etoiles synthetiques – 22 nov"):
        p = str(scripts / sous) if sous else str(scripts)
        if p not in sys.path:
            sys.path.insert(0, p)

    def charger(nom, rel):
        spec = importlib.util.spec_from_file_location(nom, str(scripts / rel))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[nom] = mod
        spec.loader.exec_module(mod)
        return mod

    n7 = charger("paires_n7", "Tour N7 – 25 nov/paires_synthetiques_n7.py")
    n3 = charger("paires_n3", "Entrainement N3 – 25 nov/paires_synthetiques_n3.py")
    n7.HERE = scripts / "Etoiles synthetiques – 22 nov"     # construire_cache y cherche etoiles2.py
    n7.CACHE = cache
    n3.CACHE = cache
    return n7, n3


# ---------------------------------------------------------------------------------------------------------------- le fond
def fabriquer_fond(graine: int, n: int, defauts: bool):
    """Le fond synthétique (3, n, n) float32, en unités de bruit du ciel, AVEC son bruit : le ciel sans étoile
    (backgrounds.py), ses défauts d'empilement si demandé (defects.py), puis le bruit du ciel (corrélé entre pixels voisins,
    comme celui de nos empilements) et le bruit de photons de la lumière du fond. Aucune image lue."""
    import numpy as np
    from backgrounds import background
    from defects import defects as tirer_defauts
    from make_dataset import correlated_noise
    from psf import loguniform
    bg, info = background(np.random.default_rng([graine, 0]), n)
    if defauts:
        d, dinfo = tirer_defauts(np.random.default_rng([graine, 1]), n)
        bg = bg + d
        info["defauts"] = dinfo
    rng_n = np.random.default_rng([graine, 3])
    corr = float(rng_n.uniform(0.09, 0.42))
    k = loguniform(rng_n, 0.01, 0.3)
    bruit_ciel = correlated_noise(rng_n, n, corr) / math.sqrt(2)
    lumiere = np.clip(bg, 0, None)
    bruit_photon = np.sqrt(lumiere * k) * correlated_noise(rng_n, n, corr)
    fond = (bg + bruit_ciel + bruit_photon).astype(np.float32)
    info.update(correlation_bruit=round(corr, 3), photon_k=round(k, 4))
    return fond, info


def fabriquer_paire(t: dict, n7, n3):
    """(entrée, cible, infos) d'une tuile. La cible est le fond (plus les cœurs de galaxie s'il y en a), copié tel quel."""
    import numpy as np
    fond, info = fabriquer_fond(t["graine_fond"], t["taille"], t["defauts"])
    g = t["graine_etoiles"]
    if t["coeurs"]:
        entree, cible, coeur, infos_c = n7.paire_synthetique_coeurs(fond, g, extras=t["extras"], p_coeurs=1.0)
        info["coeurs"] = [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in i.items()} for i in infos_c]
        assert np.array_equal(cible, (fond + coeur).astype(np.float32))
    else:
        mod = n3 if t["module"] == "n3" else n7
        entree, cible = mod.paire_synthetique(fond, g, extras=t["extras"])
        assert np.array_equal(cible, fond)                   # la cible est le fond, copié tel quel
    S = (n3 if t["module"] == "n3" else n7)._charger()
    k = int(np.random.default_rng(g).integers(len(S["meta"])))
    m = S["meta"][k]
    info["style"] = dict(indice=k, fwhm_px=round(m["fwhm"], 2), aigrettes=m["aigrettes"], mode=m["mode"],
                         photon_k_etoiles=round(m["photon_k"], 4))
    return entree.astype(np.float32), cible.astype(np.float32), info


# ---------------------------------------------------------------------------------------------------------------- images
def etirer(v):
    """Le même étirement asinh pour tous les panneaux : (3, H, W) en unités de bruit -> (H, W, 3) uint8 RVB."""
    import numpy as np
    x = GAIN * np.arcsinh(v / 3.0) / 10.0 + NOIR
    return (np.clip(x, 0, 1).transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8)


def trois_panneaux(entree, cible, titre, cv2, np):
    """Entrée | cible | différence (= les étoiles retirées), même étirement ; titres simples en français."""
    pan = [etirer(entree), etirer(cible), etirer(entree - cible)]
    noms = ["Entrée (avec étoiles)", "Cible (sans étoiles)", "Différence = étoiles retirées"]
    bandes = []
    for p, nom in zip(pan, noms):
        bande = np.full((26, p.shape[1], 3), 24, np.uint8)
        cv2.putText(bande, nom, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        bandes.append(np.concatenate([bande, p], 0))
    sep = np.full((bandes[0].shape[0], 4, 3), 255, np.uint8)
    ligne = np.concatenate([bandes[0], sep, bandes[1], sep, bandes[2]], 1)
    haut = np.zeros((30, ligne.shape[1], 3), np.uint8)
    cv2.putText(haut, titre, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([haut, ligne], 0)


def encoder_jpeg(rvb, cv2, qualite=QUALITE_JPEG):
    ok, buf = cv2.imencode(".jpg", rvb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, qualite, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
    assert ok
    return bytes(buf)


def planche(lignes, cv2, np, w=250):
    """Une ligne par exemple, trois colonnes (entrée, cible, différence), tuiles ramenées à w pixels de côté."""
    rangs = []
    for entree, cible, titre in lignes:
        pan = [cv2.resize(etirer(v), (w, w), interpolation=cv2.INTER_AREA) for v in (entree, cible, entree - cible)]
        sep = np.full((w, 3, 3), 255, np.uint8)
        ligne = np.concatenate([pan[0], sep, pan[1], sep, pan[2]], 1)
        bande = np.full((22, ligne.shape[1], 3), 24, np.uint8)
        cv2.putText(bande, titre, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
        rangs.append(np.concatenate([bande, ligne], 0))
    tete = np.zeros((24, rangs[0].shape[1], 3), np.uint8)
    for i, nom in enumerate(("Entrée", "Cible", "Différence (étoiles retirées)")):
        cv2.putText(tete, nom, (i * (w + 3) + 6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([tete] + rangs, 0)


def sha(octets: bytes) -> str:
    return hashlib.sha256(octets).hexdigest()


# ---------------------------------------------------------------------------------------------------------------- principal
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Refait les images d'exemple (paires synthétiques entrée | cible | différence).")
    ap.add_argument("--sortie", default=str(Path(__file__).resolve().parents[1] / "docs" / "images"),
                    help="dossier où écrire les images JPEG")
    ap.add_argument("--empreintes", default=str(Path(__file__).resolve().parent / "EMPREINTES.txt"),
                    help="fichier des empreintes sha256")
    ap.add_argument("--scripts", default=str(Path(__file__).resolve().parents[1] / "scripts"),
                    help="dossier des scripts publiés (le générateur)")
    ap.add_argument("--cache", default="", help="fichier du cache de styles (par défaut : un dossier temporaire)")
    a = ap.parse_args(argv)
    import cv2
    import numpy as np
    cv2.setNumThreads(int(_FILS))
    t0 = time.time()
    sortie = Path(a.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="styles_") as td:
        cache = Path(a.cache) if a.cache else Path(td) / "cache_styles.npz"
        n7, n3 = _charger_modules(Path(a.scripts).resolve(), cache)
        if not cache.exists():
            print(f"cache de {N_STYLES} styles (modèle paramétrique, sans banque) ...", flush=True)
            n7.construire_cache(N_STYLES, GRAINE_CACHE)
        out = []
        lignes = []
        for t in TUILES:
            tt = time.time()
            entree, cible, info = fabriquer_paire(t, n7, n3)
            h_e = sha(entree.tobytes())
            h_c = sha(cible.tobytes())
            jpg = encoder_jpeg(trois_panneaux(entree, cible, t["titre"], cv2, np), cv2)
            (sortie / (t["nom"] + ".jpg")).write_bytes(jpg)
            lignes.append((entree, cible, t["titre"]))
            n_diff = int((entree != cible).any(0).sum())
            out.append((t["nom"], h_e, h_c, sha(jpg), len(jpg), n_diff, info))
            print(f"{t['nom']}: {len(jpg) / 1024:.0f} Ko, {time.time() - tt:.1f} s", flush=True)
        jpg = encoder_jpeg(planche(lignes, cv2, np), cv2, 80)
        (sortie / "planche_synthetique.jpg").write_bytes(jpg)
        out.append(("planche_synthetique", "", "", sha(jpg), len(jpg), 0, {}))
        print(f"planche_synthetique: {len(jpg) / 1024:.0f} Ko", flush=True)
    with open(a.empreintes, "w", encoding="utf-8", newline="\n") as f:
        f.write("# sha256 des tableaux float32 (entree, cible) et des images JPEG de chaque exemple\n")
        f.write(f"# numpy {np.__version__}, opencv {cv2.__version__}, python {sys.version.split()[0]}, 4 fils\n")
        for nom, he, hc, hj, ko, npx, _ in out:
            f.write(f"{nom}\tentree={he or '-'}\tcible={hc or '-'}\tjpeg={hj}\toctets={ko}\n")
    infos = Path(a.empreintes).with_name("PARAMETRES_TIRES.txt")
    infos.write_text("\n".join(f"{o[0]}\tpixels_differents={o[5]}\t{o[6]}" for o in out) + "\n", encoding="utf-8")
    print(f"fini en {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
