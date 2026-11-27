# ======================================================================
# piste3.py
# Role     : Track 3: bright stars removed by measured mask + content-aware filling (multi-scale PatchMatch, Wexler 2007 / Barnes 2009, grain copy at the last level, Poisson blending Pérez 2003).
# Step     : 2. Classic Gaia-guided subtraction chain (v33b) (see scripts/LISEZMOI.md)
# Inputs   : stacked image (noise units), astrometry, offline deep Gaia extract
# Outputs  : star-free image, class maps
# Example  : python piste3.py <tour> [cas,cas] [--sigma]      (--sigma: also tries sigma Orionis, G < 4.5)
# ======================================================================
"""Track 3: bright stars removed by measured mask + content-aware filling (multi-scale PatchMatch,
Wexler 2007 / Barnes 2009, grain copy at the last level, Poisson blending Pérez 2003). Faint and medium stars:
the star-free image of track 1 (last test >= 10), read, never modified.

    python piste3.py <tour> [cas,cas] [--sigma]      (--sigma: also tries sigma Orionis, G < 4.5)
"""
from __future__ import annotations

import json
import sys
import time
sys.dont_write_bytecode = True
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mesure as M  # noqa: E402
import remplir as F  # noqa: E402
import soustraire as S4  # noqa: E402
sys.path.insert(1, str(M.P1))
from juger import save_img, to_rgb8, hole_metrics  # noqa: E402

MARGIN_GIANT = 520    # essai sur une géante (--sigma)
MARGIN = 200           # contexte autour de la tuile (sources et halos d'étoiles voisines)
BRIGHT = 0.5           # étoile brillante : coeur >= 50 % du niveau de saturation de sa couleur
OCC_MAX = 0.15       # au-delà, trop d'étoiles dans le halo pour lire la nébuleuse dessous
SIGMA_FROM_INPUT = True   # --sigma : la zone de la géante part de l'entrée (option montrée au propriétaire)
DESTRIPE = True      # géante : rayons en éventail retirés du détail en coordonnées polaires
KEEP_NEBULA = True   # rendre la structure de la nébuleuse sous les halos
DEBUG_XY = tuple(int(v) for v in __import__('os').environ['P3_SUIVI'].split(',')) if __import__('os').environ.get('P3_SUIVI') else None
DEEP = {"rosette": (Path(__import__("os").environ.get("RACINE_SCRIPTS") or Path(__file__).resolve().parents[2]) / "Catalogue Gaia profond – 20 nov" / "gaia_dr3_profond_rosette.npz")}
DEEP_R = 2.0         # rayon d'exclusion des sources, en largeurs à mi-hauteur
DEEP_GMAX = 99.0     # toutes les étoiles du catalogue profond
MID_SPIKES = True    # hybride : aigrettes des étoiles moyennes soustraites aussi
MID_G = 11.0
DIPS = True          # creux larges de la base : basse fréquence refaite
ARBITRATE = True     # hybride : pour chaque étoile, la piste 1 seule ou la piste 3, la meilleure
MEDIUM = True        # reprise des étoiles moyennes où la piste 1 a laissé un défaut
V26_ECHELLE_ETOILE = True  # v26 (22 nov) : le reste de v25 est jugé à l'échelle de l'étoile, aux positions du catalogue
V31_HORS_GEANTES = True  # v31 (22 nov) : la règle v26 ne s'applique pas dans la zone d'une géante (base = entrée)
V25_RESTE_REEL = True  # v25 (22 nov) : règle « vrai reste » de v18 étendue aux moyennes corrigées et défauts recréés
GIANT_G = 4.5          # géantes (sigma Ori) laissées entières, sauf --sigma
GEANTE_DISQUE_PLATEAU = False  # h3 REJETÉ (22 nov) : disque recréé de la géante limité à 3 x rayon du coeur saturé
GEANTE_PORTEE_MESUREE = True  # h1 (22 nov) : portée du halo de la géante mesurée jusqu'à 520 px
GEANTE_MODE4 = True    # g2 (22 nov) : en mode 4, la géante reçoit aussi halo soustrait + coeur recréé (avant : sautée)
GEANTE_SANS_GABARIT_AIGRETTES = False  # g6 REJETÉ (22 nov) : la géante n'a pas de gabarit d'aigrettes (le passage polaire les retire déjà)
GEANTE_QUADRUPOLE = True  # h2 : repris avec h1 (g5 rejeté quand le halo était coupé à 222 px) (22 nov) : halo de la géante allongé (2 lobes) : terme en 2θ retiré par anneau
GEANTE_SECTEURS = False  # g4 REJETÉ (22 nov) : halo de la géante par 8 secteurs (reflets plus forts d'un côté)
V18_RESTE_REEL = True  # v18 (22 nov) : zone dure d'une brillante (disque du coeur, bandes d'aigrettes) recréée seulement où la
                       # piste 1 a laissé un vrai reste (reste_reel.py) ; ailleurs la valeur de la piste 1 est gardée

SIX = [(1706, 1009, 0), (2043, 968, 0), (2038, 1036, 0), (1762, 1159, 0), (2078, 1220, 0), (1999, 1327, 0)]  # les 6 étoiles entourées
REF_ROSETTE = [  # piste 1, coordonnées image entière : (x, y, G)
    (2043.6, 1203.9, 11.88), (2002.4, 1066.7, 12.01), (2151.1, 986.8, 11.92), (2009.0, 1273.0, 12.08),
    (1941.0, 1127.0, 5.60)]


def run(tour, case, try_sigma=False, log=print, mode4=False, filler="patch"):
    t0 = time.time()
    name, crop, tile = M.CASES[case]
    out = HERE / "sorties" / tour / case
    out.mkdir(parents=True, exist_ok=True)
    full = M.images.load(name)
    img = full[:, crop[1]:crop[3], crop[0]:crop[2]]
    Hc, Wc = img.shape[1:]
    tx0, ty0, tx1, ty1 = tile
    margin = MARGIN_GIANT if try_sigma else MARGIN       # une géante : son halo dépasse 400 px
    wx0, wy0 = max(0, tx0 - margin), max(0, ty0 - margin)
    wx1, wy1 = min(Wc, tx1 + margin), min(Hc, ty1 + margin)
    # base de la piste 1 : sa dernière sortie complète ; la fenêtre utilisée est copiée ici (sorties/base_piste1),
    # pour que le tour reste rejouable si la piste 1 efface ses sorties ; à défaut, la dernière copie
    d1 = M.p1_dir(name, case)
    cache_root = HERE / "sorties" / "base_piste1" / case
    if d1 is not None:
        cdir = cache_root / f"{d1.name}_m{margin}"
        if not (cdir / "fenetre_sans_etoiles.npy").exists():
            cdir.mkdir(parents=True, exist_ok=True)
            unpack0 = lambda f: np.unpackbits(np.load(f))[:Hc * Wc].reshape(Hc, Wc).astype(bool) if f.exists() else np.zeros((Hc, Wc), bool)
            np.save(cdir / "fenetre_sans_etoiles.npy", np.load(d1 / "starless.npy", mmap_mode="r")[:, wy0:wy1, wx0:wx1])
            np.save(cdir / "fenetre_remplissage.npy", unpack0(d1 / "remplissage.npy")[wy0:wy1, wx0:wx1])
            np.save(cdir / "fenetre_exclu.npy", unpack0(d1 / "exclu.npy")[wy0:wy1, wx0:wx1])
            (cdir / "etoiles.csv").write_bytes((d1 / "etoiles.csv").read_bytes())
            (cdir / "resume.json").write_bytes((d1 / "resume.json").read_bytes())
            if (d1 / "zones_a_remplir.npz").exists():
                zz = np.load(d1 / "zones_a_remplir.npz")
                fz = tuple(int(v) for v in zz["forme"])
                np.savez_compressed(cdir / "fenetre_zones.npz", **{k: np.unpackbits(zz[k])[:fz[0] * fz[1]].reshape(fz)[wy0:wy1, wx0:wx1]
                                                                for k in zz.files if k != "forme"})
    else:
        cands_ = sorted(cache_root.glob(f"essai*_m{margin}"), key=lambda d: int(d.name[5:].split("_")[0])) if cache_root.exists() else []
        if not cands_ and __import__("os").environ.get("P3_BASE_ENTREE") == "1":
            # essai seulement : l'entrée sert de base (étoiles faibles non retirées) ; jamais pour un tour rendu
            cdir = HERE / "sorties" / "base_entree_essai" / case
            cdir.mkdir(parents=True, exist_ok=True)
            np.save(cdir / "fenetre_sans_etoiles.npy", img[:, wy0:wy1, wx0:wx1])
            np.save(cdir / "fenetre_remplissage.npy", np.zeros((wy1 - wy0, wx1 - wx0), bool))
            np.save(cdir / "fenetre_exclu.npy", np.zeros((wy1 - wy0, wx1 - wx0), bool))
            (cdir / "etoiles.csv").write_text("x,y,G" + chr(10), encoding="utf-8")
        elif not cands_:
            raise SystemExit(f"{case} : aucune base de la piste 1 (ni sortie, ni copie)")
        else:
            cdir = cands_[-1]
    d1 = cdir
    p1 = img.copy(); p1[:, wy0:wy1, wx0:wx1] = np.load(cdir / "fenetre_sans_etoiles.npy")
    p1fill = np.zeros((Hc, Wc), bool); p1fill[wy0:wy1, wx0:wx1] = np.load(cdir / "fenetre_remplissage.npy")
    p1ex = np.zeros((Hc, Wc), bool); p1ex[wy0:wy1, wx0:wx1] = np.load(cdir / "fenetre_exclu.npy")
    p1zones = None
    if (cdir / "fenetre_zones.npz").exists():
        zz = np.load(cdir / "fenetre_zones.npz")
        p1zones = {k: zz[k].astype(bool) for k in zz.files}
        uni = np.zeros(p1fill.shape, bool)
        uni[wy0:wy1, wx0:wx1] = np.any([v for v in p1zones.values()], 0)
        p1fill |= uni                      # sources interdites : toute zone que la piste 1 aurait remplie
    log(f"[{case}] base piste 1 : {d1.name} (copie de la fenêtre : {cdir})")
    I = img[:, wy0:wy1, wx0:wx1].copy(); B = p1[:, wy0:wy1, wx0:wx1].copy()
    L = I.mean(0); Ls = B.mean(0)
    H, W = L.shape
    sigc = [M.noise_sigma(c) for c in I]
    sig = M.noise_sigma(L)
    sat = img.reshape(img.shape[0], -1).max(1)
    log(f"bruit lum {sig:.3f}, par couleur {np.round(sigc, 3).tolist()}, saturation {np.round(sat, 0).tolist()}")
    # catalogue
    z = np.load(M.P1 / "sorties" / name / "astrometrie.npz", allow_pickle=True)["cat"].item()
    gx = z["x"] - crop[0]; gy = z["y"] - crop[1]; gG = z["g"]
    # étoiles brillantes de la fenêtre (règle : coeur >= 50 % de la saturation)
    inw = (gx > wx0) & (gx < wx1) & (gy > wy0) & (gy < wy1) & (gG < 12)
    stars = []; left_alone = []
    # v19 : éclat prévu par la piste 1 (etoiles.csv, flux_predit, mêmes coordonnées que gx/gy) et largeur de l'image
    import csv as _csv, json as _json
    _fw = float(_json.load(open(d1 / "resume.json", encoding="utf-8"))["fwhm"])
    _pf = 4 * np.log(2) / (np.pi * _fw * _fw)
    _pred = []
    with open(d1 / "etoiles.csv", encoding="utf-8") as _f:
        for _r in _csv.DictReader(_f):
            try:
                _pred.append((float(_r["x"]), float(_r["y"]), float(_r["flux_predit"])))
            except ValueError:
                pass
    _pred = np.array(_pred) if _pred else np.zeros((0, 3))
    sat_image = full.reshape(full.shape[0], -1).max(1)          # v20 : maximum de l'image entière, par filtre

    def pic_attendu(x_, y_):
        """Pic attendu (pixel central) de l'étoile en (x_, y_) ; 0 si elle n'est pas dans la liste de la piste 1."""
        if not len(_pred):
            return 0.0
        dd = np.hypot(_pred[:, 0] - x_, _pred[:, 1] - y_)
        k = int(np.argmin(dd))
        return float(_pred[k, 2] * _pf) if dd[k] < 1.5 else 0.0
    for x, y, g in zip(gx[inw], gy[inw], gG[inw]):
        X, Y = int(round(x)), int(round(y))
        pk = img[:, max(0, Y - 2):Y + 3, max(0, X - 2):X + 3].reshape(img.shape[0], -1).max(1)
        ratio = float(np.max(pk / sat))
        if ratio < BRIGHT:
            continue
        # étoile sur un coeur de nébuleuse saturé (étoile centrale de NGC 6543) : la nébuleuse passe avant
        Lr = img.mean(0)
        ya_, yb_, xa_, xb_ = max(0, Y - 16), min(Hc, Y + 17), max(0, X - 16), min(Wc, X + 17)
        yy, xx = np.mgrid[ya_:yb_, xa_:xb_]
        rr = np.hypot(xx - x, yy - y)
        ring = Lr[ya_:yb_, xa_:xb_][(rr >= 8) & (rr <= 15)]
        pkl = float(Lr[max(0, Y - 2):Y + 3, max(0, X - 2):X + 3].max())
        # v19 : une étoile dont le pic ATTENDU (éclat prévu par la piste 1 x pic d'une gaussienne de la largeur de
        # l'image) atteint BRIGHT x saturation est une vraie brillante écrêtée : son pic est tronqué, donc son anneau
        # dépasse toujours 4 % du pic mesuré. La règle « source étendue » (écrite pour l'étoile centrale de
        # NGC 6543, sur un coeur de nébuleuse) ne vaut que pour une étoile qui ne devrait PAS saturer.
        # v20 : le pic attendu du catalogue trompe (G 9,96 de M8 a : attendu 8 087, réel 55-70 % du max, aucun
        # plateau ; traitée en brillante, elle a rouvert le globule). Critère MESURÉ sur l'image (saturation v14) :
        # au moins 9 px à >= 92 % du maximum de l'IMAGE ENTIÈRE dans un filtre, à moins de 8 px de l'étoile.
        _blk = full[:, max(0, Y + crop[1] - 8):Y + crop[1] + 9, max(0, X + crop[0] - 8):X + crop[0] + 9]
        n_plateau = int(max(int((_blk[c] >= 0.92 * sat_image[c]).sum()) for c in range(full.shape[0])))
        attendu_sature = n_plateau >= 9
        pa = n_plateau
        if np.median(ring) > 0.04 * pkl and attendu_sature:
            log(f"  G {g:.2f} ({x - tx0:.0f}, {y - ty0:.0f}) tuile : anneau à {100 * np.median(ring) / pkl:.0f} % du pic mais plateau saturé de {pa} px : brillante écrêtée, traitée (v20)")
        if np.median(ring) > 0.04 * pkl and not attendu_sature and not (g < GIANT_G and try_sigma):   # source étendue : anneau 8-15 px > 4 % du pic
            left_alone.append((x - wx0, y - wy0))
            log(f"  G {g:.2f} ({x - tx0:.0f}, {y - ty0:.0f}) tuile : anneau 8-15 px à {100 * np.median(ring) / pkl:.0f} % du pic : source étendue (coeur de nébuleuse), laissée")
            continue
        giant = g < GIANT_G
        if giant and not try_sigma:
            log(f"  G {g:.2f} ({x:.0f}, {y:.0f}) : géante, laissée entière (règle du propriétaire)")
            continue
        stars.append(dict(x=float(x - wx0), y=float(y - wy0), g=float(g), ratio=ratio, giant=bool(giant),
                          rmax=(MARGIN_GIANT if (giant and GEANTE_PORTEE_MESUREE)
                                else int(np.clip(30 * (12 - g), 40, 220 if not giant else 600)))))
        # h1 (22 nov) : géante : portée de recherche du halo = marge de la fenêtre (520 px), le rayon du halo est ensuite
        # MESURÉ ; la formule en G (246 px σ Ori, 306 Alnitak) coupait des halos visibles jusqu'à 490-530 px
    # dans l'exclusion de la piste 1 (sigma Ori), la base est l'entrée : on garde la règle sauf --sigma
    exw = p1ex[wy0:wy1, wx0:wx1]
    if not try_sigma:
        stars = [s for s in stars if not exw[int(np.clip(s["y"], 0, H - 1)), int(np.clip(s["x"], 0, W - 1))]]
    # autres étoiles masquées dans les profils
    other_all = np.zeros((H, W), np.uint8)
    sel = (gx > wx0 - 20) & (gx < wx1 + 20) & (gy > wy0 - 20) & (gy < wy1 + 20) & (gG < 15)
    for x, y, g in zip(gx[sel] - wx0, gy[sel] - wy0, gG[sel]):
        cv2.circle(other_all, (int(round(x)), int(round(y))), int(max(3, (15 - g) * 2.5)), 1, -1)
    # + étoiles hors catalogue retirées par la piste 1 (sa liste etoiles.csv)
    import csv
    with open(d1 / "etoiles.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            x, y = float(row["x"]) - wx0, float(row["y"]) - wy0
            if -20 < x < W + 20 and -20 < y < H + 20:
                try:
                    g = float(row["G"])
                except ValueError:
                    g = float("nan")
                r = int(max(3, (15 - g) * 2.5)) if np.isfinite(g) else 4
                cv2.circle(other_all, (int(round(x)), int(round(y))), r, 1, -1)
    # angles des aigrettes : mesurés sur les étoiles brillantes de toute l'image (une petite découpe en a trop peu)
    Lc = full.mean(0)
    Hf_, Wf_ = Lc.shape
    allb = []
    for x, y, g in zip(gx + crop[0], gy + crop[1], gG):
        if 260 < x < Wf_ - 260 and 260 < y < Hf_ - 260 and GIANT_G <= g < 9.5:
            X, Y = int(round(x)), int(round(y))
            pk = full[:, Y - 2:Y + 3, X - 2:X + 3].reshape(full.shape[0], -1).max(1)
            if np.max(pk / sat) >= BRIGHT:
                core = np.nonzero(M.radial_profile(Lc, x, y, 40) > 0.3 * float(np.min(sat)))[0]
                allb.append(dict(x=x, y=y, g=g, r_core=int(core.max()) + 1 if len(core) else 3))
    allb = sorted(allb, key=lambda s: s["g"])[:30]
    angles = M.spike_angles(Lc, allb, log) if len(allb) >= 3 else []
    # mesure et masque par étoile
    D = Ls - cv2.GaussianBlur(Ls, (0, 0), 6)
    sigD = M.noise_sigma(Ls)
    mask = np.zeros((H, W), bool); hard = np.zeros((H, W), bool); soft = np.zeros((H, W), bool)
    hard_b = np.zeros((H, W), bool); hard_m = np.zeros((H, W), bool); hard_t = np.zeros((H, W), bool)
    for s in stars:
        oth = other_all.astype(bool).copy()
        X, Y = int(round(s["x"])), int(round(s["y"]))
        cv2.circle(oth.view(np.uint8), (X, Y), int(max(3, (15 - s["g"]) * 2.5)), 0, -1)
        M.measure_star(L, Ls, s, sig, float(np.min(sat)), oth, angles, D, sigD)
        if s["giant"]:
            # géante : ses rayons couvrent tout le champ et noient la mesure ; aigrettes de l'instrument (angles mesurés
            # sur les autres étoiles, affinés sur elle) sur toute la longueur du halo, largeur mesurée ou 5 px
            have = {round(sp["angle"]) for sp in s["spikes"]}
            for a in angles + [(a + 180) % 360 for a in angles]:
                if all(min(abs(a - q), 360 - abs(a - q)) > 8 for q in have):
                    s["spikes"].append(dict(angle=float(a), length=int(s["r_halo"]), hw=5))
            for sp in s["spikes"]:
                sp["length"] = max(sp["length"], int(s["r_halo"])); sp["hw"] = max(sp["hw"], 5)
        hm, sm = M.star_mask((H, W), s)
        s["px"] = int((hm | sm).sum())
        hard |= hm; soft |= sm; mask |= hm | sm; hard_b |= hm
        log(f"  G {s['g']:.2f} ({s['x'] + wx0 - tx0:.0f}, {s['y'] + wy0 - ty0:.0f}) tuile : coeur {s['r_core']} px, "
            f"dur {s['r_hard']} px, halo {s['r_halo']} px, résidu p1 {s['r_res']} px, masque R {s['R']} ; aigrettes "
            + ", ".join(f"{sp['angle']:.0f}°:{sp['length']}px/±{sp['hw']}" for sp in s["spikes"]) + f" (propres {s['angles_propres']})")
    fixed_m = np.zeros((H, W), bool); sub_zone = np.zeros((H, W), bool)
    zone_tpl = np.zeros((H, W), bool); core_tpl = np.zeros((H, W), bool)   # v18 : gabarit des zones dures des brillantes
    B0 = B.copy()                                   # base de la piste 1, pour l'arbitrage étoile par étoile
    H_tot = np.zeros_like(B)                        # halos des brillantes soustraits (gardés si la piste 1 l'emporte)
    if try_sigma and SIGMA_FROM_INPUT:
        # géante : sa zone part de l'entrée, pas de la base de la piste 1 (qui l'a laissée entière, bord compris) ;
        # passage progressif sur 30 px au bord du disque de 1.6 R
        for s_ in stars:
            if s_["giant"]:
                yy_, xx_ = np.mgrid[0:H, 0:W]
                rr_ = np.hypot(xx_ - s_["x"], yy_ - s_["y"])
                w_ = np.clip((1.6 * s_["R"] - rr_) / 30.0, 0, 1).astype(np.float32)
                B = w_[None] * I + (1 - w_[None]) * B
                Ls = B.mean(0); D = Ls - cv2.GaussianBlur(Ls, (0, 0), 6)
                log(f"  géante G {s_['g']:.2f} : sa zone (1.6 R = {1.6 * s_['R']:.0f} px) part de l'entrée")
    if mode4:
        # tour 4 : halos et aigrettes soustraits ; seul le coeur (et ce qui reste au-delà du seuil) est recréé
        hard[:] = False; soft[:] = False; hard_b[:] = False; mask[:] = False
        core_disc = np.zeros((H, W), np.uint8)
        for s in stars:
            cv2.circle(core_disc, (int(round(s["x"])), int(round(s["y"]))), int(s["r_core"] + 4), 1, -1)
        core_disc = core_disc.astype(bool)
        for s in sorted(stars, key=lambda t: t["g"]):
            if s["giant"] and not GEANTE_MODE4:
                continue
            oth = other_all.astype(bool).copy()
            cv2.circle(oth.view(np.uint8), (int(round(s["x"])), int(round(s["y"]))), int(max(3, (15 - s["g"]) * 2.5)), 0, -1)
            own = np.zeros((H, W), np.uint8); cv2.circle(own, (int(round(s["x"])), int(round(s["y"]))), int(s["r_core"] + 4), 1, -1)
            badm = oth | (core_disc & ~own.astype(bool))
            Bb_ = B
            if s["giant"] and GEANTE_SECTEURS:          # g4 : halo de la géante mesuré par secteurs (non rond)
                B, _, _ = S4.subtract_halo_secteurs(B, s, badm, s["r_core"] + 3, s["R"])
            else:
                B, _, _ = S4.subtract_halo(B, s, badm, s["r_core"] + 3, s["R"])
            if s["giant"] and GEANTE_QUADRUPOLE:        # g5 : allongement du halo (2 lobes) retiré anneau par anneau
                B = S4.subtract_quadrupole(B, s, badm, s["r_core"] + 3, s["R"])
            H_tot += Bb_ - B
            if s["giant"] and GEANTE_SANS_GABARIT_AIGRETTES:   # g6 : aigrettes de la géante laissées au retrait des rayons
                left = np.zeros((H, W), bool)
            else:
                B, left = S4.subtract_spikes(B, s, sig, int(s["r_core"] + 4))
            # coeur recréé : jusqu'où le reste après soustraction s'écarte encore du fond de plus d'un bruit (quart du halo
            # au plus) ; une étoile double ou un coeur mal soustrait garde ainsi un disque assez grand
            pa = M.radial_profile(B.mean(0), s["x"], s["y"], int(s["R"]) + 10, badm & ~own.astype(bool))
            bga = np.nanmedian(pa[int(s["R"]):])
            ra = int(s["r_core"] + 4)
            dev_ = np.abs(np.nan_to_num(pa - bga))
            cap_ = 0.25 * s["R"]
            if s["giant"] and GEANTE_DISQUE_PLATEAU:      # h3 : disque recréé <= 3 x rayon du plateau saturé
                cap_ = min(cap_, 3 * s["r_core"])
            for k_ in range(ra, int(max(ra + 1, cap_))):
                if np.all(dev_[k_:k_ + 4] < sig):
                    break
                ra = k_ + 1
            own[:] = 0; cv2.circle(own, (int(round(s["x"])), int(round(s["y"]))), int(ra), 1, -1)
            s["r_m4"] = int(ra)                       # g3 : rayon du disque recréé en mode 4
            hd = own.astype(bool) | left
            hard |= hd; hard_b |= hd; mask |= hd
            zone_tpl |= hd; core_tpl |= own.astype(bool)
            disc = np.zeros((H, W), np.uint8); cv2.circle(disc, (int(round(s["x"])), int(round(s["y"]))), int(s["R"]), 1, -1)
            sub_zone |= disc.astype(bool)
            log(f"  G {s['g']:.2f} : halo soustrait jusqu'à {s['R']} px, aigrettes soustraites ; reste recréé {int(hd.sum())} px")
        soft = sub_zone & ~hard
        Ls = B.mean(0); D = Ls - cv2.GaussianBlur(Ls, (0, 0), 6)
    mid_stars = []
    if mode4 and MID_SPIKES and angles:
        # hybride : les étoiles moyennes (G <= 11) gardent souvent leurs aigrettes dans la base de la piste 1 ; même
        # gabarit d'aigrette que pour les brillantes, là où une aigrette est mesurée au-delà du coeur
        bx = [(t["x"], t["y"]) for t in stars]
        selm = (gx > wx0 + 20) & (gx < wx1 - 20) & (gy > wy0 + 20) & (gy < wy1 - 20) & (gG <= MID_G)
        for x, y, g in zip(gx[selm] - wx0, gy[selm] - wy0, gG[selm]):
            if any(np.hypot(x - a, y - b) < 4 for a, b in bx) or hard[int(round(y)), int(round(x))]:
                continue
            s_ = dict(x=float(x), y=float(y), g=float(g), ratio=0.0, giant=False, rmax=40, medium=True)
            oth = other_all.astype(bool).copy()
            cv2.circle(oth.view(np.uint8), (int(round(x)), int(round(y))), int(max(3, (15 - g) * 2.5)), 0, -1)
            M.measure_star(L, Ls, s_, sig, float(np.min(sat)), oth, angles, D, sigD)
            sp_ = [t for t in s_["spikes"] if t["length"] > s_["r_core"] + 8 and
                   any(min(abs(t["angle"] - q) % 360, 360 - abs(t["angle"] - q) % 360) < 8 for q in [(a + k) % 360 for a in angles for k in (0, 90, 180, 270)])]
            if not sp_:
                continue
            for t in sp_:
                t["hw"] = min(t["hw"], 4)
            s_["spikes"] = sp_
            B, left = S4.subtract_spikes(B, s_, sig, int(s_["r_core"] + 4))
            hard |= left; hard_b |= left; mask |= left; zone_tpl |= left
            mid_stars.append(s_)
        Ls = B.mean(0); D = Ls - cv2.GaussianBlur(Ls, (0, 0), 6)
        log(f"  étoiles moyennes (G <= {MID_G}) dont les aigrettes sont soustraites : {len(mid_stars)}")
    v18 = None
    if mode4 and V18_RESTE_REEL and zone_tpl.any():
        # v18 : le gabarit (disques des coeurs, bandes d'aigrettes des brillantes et des moyennes) n'est recréé que là où la
        # piste 1 a laissé un vrai reste ; P = piste 1 moins les halos radiaux de l'étape A (nuls hors des brillantes) ;
        # coeurs saturés (zones « coeur_sature » de la piste 1, saturation v14) toujours recréés
        import reste_reel as RRe
        fw18 = float(json.loads((d1 / "resume.json").read_text(encoding="utf-8"))["fwhm"]) if (d1 / "resume.json").exists() else 3.0
        P18 = B0 - H_tot
        excl18 = other_all.astype(bool) | core_tpl
        real18, z18, nz18 = RRe.restes_reels(P18, zone_tpl, excl18, fw18, sigc)
        satz = np.zeros((H, W), bool)
        if p1zones is not None and "coeur_sature" in p1zones:
            satz = p1zones["coeur_sature"].astype(bool) & zone_tpl
        keep18 = zone_tpl & ~real18 & ~satz
        B[:, keep18] = P18[:, keep18]
        hard &= ~keep18; hard_b &= ~keep18; mask &= ~keep18
        Ls = B.mean(0); D = Ls - cv2.GaussianBlur(Ls, (0, 0), 6)
        tT = (slice(ty0 - wy0, ty1 - wy0), slice(tx0 - wx0, tx1 - wx0))
        v18 = dict(k=RRe.K, rayon_fond_px=int(round(RRe.SCALE_FWHM * fw18)), fwhm=round(fw18, 3), dilatation_px=RRe.DIL,
                   bruit_local_median=[round(float(v), 3) for v in nz18],
                   gabarit_px=int(zone_tpl.sum()), recree_px=int((zone_tpl & ~keep18).sum()), garde_piste1_px=int(keep18.sum()),
                   sature_force_px=int(satz.sum()),
                   gabarit_tuile_px=int(zone_tpl[tT].sum()), recree_tuile_px=int((zone_tpl & ~keep18)[tT].sum()))
        log(f"  v18 reste réel (k {RRe.K}, fond médiane r {v18['rayon_fond_px']} px = {RRe.SCALE_FWHM} FWHM, dilatation {RRe.DIL} px, "
            f"bruit local médian {v18['bruit_local_median']}) : gabarit {v18['gabarit_px']} px -> recréé {v18['recree_px']} px "
            f"(tuile {v18['gabarit_tuile_px']} -> {v18['recree_tuile_px']}), coeurs saturés forcés {v18['sature_force_px']} px")
    # étoiles moyennes : la soustraction de la piste 1 reste la base ; on ne reprend que celles où elle a laissé un
    # défaut visible (coeur moins anneau 6-12 px hors +-5 bruits, ou creux de détail < -4 bruits dans les 8 px)
    if MEDIUM:
        dl = cv2.GaussianBlur(D, (0, 0), 1.0)
        sdl = float(np.median(np.abs(dl - np.median(dl))) * 1.4826)
        cands = [(x, y, g) for x, y, g in zip(gx[sel] - wx0, gy[sel] - wy0, gG[sel])]
        with open(d1 / "etoiles.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["G"] in ("", "nan"):
                    cands.append((float(row["x"]) - wx0, float(row["y"]) - wy0, 14.5))
        nmed = 0
        for x, y, g in cands:
            X, Y = int(round(x)), int(round(y))
            if X < 16 or Y < 16 or X >= W - 16 or Y >= H - 16 or hard[Y, X] or (mask[Y, X] and not try_sigma) or exw[Y, X] and not try_sigma:
                continue
            if any(np.hypot(x - lx, y - ly) < 40 for lx, ly in left_alone):
                continue
            yy, xx = np.mgrid[Y - 12:Y + 13, X - 12:X + 13]
            rr = np.hypot(xx - x, yy - y)
            cut = Ls[Y - 12:Y + 13, X - 12:X + 13]
            c = cut[rr <= 3].mean() - np.median(cut[(rr >= 6) & (rr <= 12)])
            dk = dl[Y - 12:Y + 13, X - 12:X + 13][rr <= 8].min()
            ring_d = np.median(cut[(rr >= 4) & (rr <= 9)]) - np.median(cut[(rr >= 12) & (rr <= 12.5)]) if False else                 np.median(cut[(rr >= 4) & (rr <= 8)]) - np.median(cut[(rr >= 10) & (rr <= 12)])
            # seuils rapportés à la texture locale (sur la nébuleuse, le détail varie plus que sur le ciel)
            ya_, xa_ = max(0, Y - 24), max(0, X - 24)
            blk = dl[ya_:Y + 25, xa_:X + 25]
            yb_, xb_ = np.mgrid[ya_:ya_ + blk.shape[0], xa_:xa_ + blk.shape[1]]
            far_ = np.hypot(xb_ - x, yb_ - y) > 12
            sl_ = max(sdl, float(np.median(np.abs(blk[far_] - np.median(blk[far_]))) * 1.4826))
            sg_ = sig * sl_ / sdl
            # couleur : point coloré au coeur (une couleur s'écarte des deux autres), invisible en luminance
            cc_ = [B[k_, Y - 12:Y + 13, X - 12:X + 13] for k_ in range(B.shape[0])]
            chroma = max(abs((cc_[k_][rr <= 2].mean() - np.median(cc_[k_][(rr >= 6) & (rr <= 12)])) -
                             (cut[rr <= 2].mean() - np.median(cut[(rr >= 6) & (rr <= 12)]))) / sigc[k_] for k_ in range(B.shape[0]))
            if DEBUG_XY and np.hypot(x - DEBUG_XY[0] - tx0 + wx0, y - DEBUG_XY[1] - ty0 + wy0) < 6:
                log(f"    [suivi] candidate G {g:.2f} ({x - tx0 + wx0:.0f},{y - ty0 + wy0:.0f}) : c {c / sg_:.1f}, creux {dk / sl_:.1f}, "
                    f"anneau {ring_d / sl_:.1f}, couleur {chroma:.1f}, fond {np.median(cut[(rr >= 10) & (rr <= 12)]) / sig:.1f} bruits")
            if abs(c) <= 5 * sg_ and dk >= -4 * sl_ and abs(ring_d) <= 2.5 * sl_ and chroma <= 4:   # coeur, creux, anneau ou couleur laissés
                continue
            if np.median(cut[(rr >= 10) & (rr <= 12)]) > 15 * sig:   # sur nébuleuse brillante : la nébuleuse d'abord
                continue
            sm_ = dict(x=float(x), y=float(y), g=float(g), ratio=0.0, giant=False, rmax=30, medium=True)
            oth = other_all.astype(bool).copy()
            cv2.circle(oth.view(np.uint8), (X, Y), int(max(3, (15 - g) * 2.5)), 0, -1)
            M.measure_star(L, Ls, sm_, sig, float(np.min(sat)), oth, [], D, sigD)
            sm_["spikes"] = []
            if mode4:
                bad_ = oth | hard
                if fixed_m[Y, X]:
                    continue        # dans le disque d'une correction voisine : pas de double correction
                if DEBUG_XY and np.hypot(x - DEBUG_XY[0] - tx0 + wx0, y - DEBUG_XY[1] - ty0 + wy0) < 20:
                    log(f"    [suivi] moyenne G {g:.2f} corrigée à ({x - tx0 + wx0:.0f},{y - ty0 + wy0:.0f}), coeur-anneau avant {c / sig:.1f} bruits")
                B = S4.medium_fix(B, x, y, bad_, None)
                fd = np.zeros((H, W), np.uint8); cv2.circle(fd, (X, Y), 12, 1, -1)
                fixed_m |= fd.astype(bool)
                nmed += 1
                # défaut encore là après correction (creux profond, reste non radial) : recréé, petit disque
                Lb_ = B.mean(0); cut2 = Lb_[Y - 12:Y + 13, X - 12:X + 13]
                ya2, xa2 = max(0, Y - 30), max(0, X - 30)
                blk2 = Lb_[ya2:Y + 31, xa2:X + 31].astype(np.float32)
                Db_ = cv2.GaussianBlur(blk2 - cv2.GaussianBlur(blk2, (0, 0), 6), (0, 0), 1.0)
                dl2 = Db_[Y - 12 - ya2:Y + 13 - ya2, X - 12 - xa2:X + 13 - xa2]
                c2 = cut2[rr <= 3].mean() - np.median(cut2[(rr >= 6) & (rr <= 12)])
                dip2 = cut2[rr <= 6].mean() < np.median(cut2[(rr >= 8) & (rr <= 12)]) - 1.5 * sig   # vrai creux, pas un flanc
                # petit puits (1-3 px) à côté d'un coeur encore clair : la moyenne du disque ne le voit pas ; on cherche le
                # minimum lissé à 1 px, nettement sous l'anneau
                pit = float(cv2.GaussianBlur(cut2.astype(np.float32), (0, 0), 1.0)[rr <= 5].min()) < np.median(cut2[(rr >= 8) & (rr <= 12)]) - 3 * sig
                if abs(c2) > 5 * sg_ or (dl2[rr <= 8].min() < -6 * sl_ and (dip2 or pit)):
                    hd_ = np.zeros((H, W), np.uint8)
                    rh_ = int(max(sm_["r_core"] + 3, min(sm_["r_res"] + 2, 20))) if not (pit and not dip2 and abs(c2) <= 5 * sg_) else 6
                    cv2.circle(hd_, (X, Y), rh_, 1, -1)
                    hard |= hd_.astype(bool); mask |= hd_.astype(bool); hard_t |= hd_.astype(bool)
                continue
            hm, smk = M.star_mask((H, W), sm_, pad=2)
            hard |= hm; mask |= hm; hard_m |= hm          # pas de zone de halo : la soustraction l'a déjà retiré, la nébuleuse reste
            nmed += 1
        log(f"  étoiles moyennes reprises (défaut laissé par la soustraction) : {nmed}")
    # sources interdites : masque, remplissages de la piste 1 (zones synthétiques), exclusion (étoiles restées)
    forbid = ndi.binary_dilation(p1fill[wy0:wy1, wx0:wx1], iterations=2)
    if not try_sigma:
        forbid |= exw
    # sources sans étoile : pas de patch pris autour d'une étoile du catalogue (restes de la soustraction)
    forbid_src = forbid | other_all.astype(bool)
    # ni les creux sombres de la base (défauts de soustraction) : détail < -3.5 bruits de détail, élargi de 3 px
    dl_ = cv2.GaussianBlur(D, (0, 0), 1.0); sdl_ = float(np.median(np.abs(dl_ - np.median(dl_))) * 1.4826)
    forbid_src |= ndi.binary_dilation(dl_ < -3.5 * sdl_, iterations=3)
    log(f"  sources interdites : {100 * forbid_src.mean():.1f} % de la fenêtre")
    # catalogue Gaia profond : aucun centre de patch à moins de DEEP_R largeurs à mi-hauteur d'une étoile du catalogue
    deep_forbid = None
    deep_f = DEEP.get(name)
    if mode4 and deep_f is not None and deep_f.exists():
        zd = np.load(deep_f)
        fw_ = float(json.loads((d1 / "resume.json").read_text(encoding="utf-8"))["fwhm"]) if (d1 / "resume.json").exists() else 3.0
        dx_ = zd["x_pix"] - crop[0] - wx0; dy_ = zd["y_pix"] - crop[1] - wy0
        md_ = (dx_ > -20) & (dx_ < W + 20) & (dy_ > -20) & (dy_ < H + 20) & (zd["Gmag"] < DEEP_GMAX)
        deep_forbid = np.zeros((H, W), np.uint8)
        for a_, b_ in zip(dx_[md_], dy_[md_]):
            cv2.circle(deep_forbid, (int(round(a_)), int(round(b_))), int(round(DEEP_R * fw_)), 1, -1)
        deep_forbid = deep_forbid.astype(bool)
        # zones que la piste 1 aurait remplies (non convergées, coeurs saturés, aigrettes) : pas de source non plus
        deep_forbid |= forbid
        log(f"  catalogue profond : {int(md_.sum())} étoiles, rayon {DEEP_R} x {fw_:.2f} px ; centres de patch interdits "
            f"{100 * deep_forbid.mean():.1f} % de la fenêtre")
    soft &= ~hard
    # dans le halo, les défauts de la base (creux sombres de la soustraction, restes d'étoiles) passent en dur :
    # taches de détail < -3 bruits de détail (>= 5 px), ou > +3 sur une position d'étoile connue
    dl = cv2.GaussianBlur(D, (0, 0), 1.0); sdl = M.noise_sigma(dl) if False else float(np.median(np.abs(dl - np.median(dl))) * 1.4826)
    if mode4:      # la base a changé (halos, aigrettes, moyennes corrigés) : le détail est relu
        Ls = B.mean(0); D = Ls - cv2.GaussianBlur(Ls, (0, 0), 6)
        dl = cv2.GaussianBlur(D, (0, 0), 1.0); sdl = float(np.median(np.abs(dl - np.median(dl))) * 1.4826)
        soft = sub_zone & ~hard          # les taches ne sont cherchées que dans les halos soustraits
    gsoft = np.zeros((H, W), bool)          # halo d'une géante : ses rayons sont traités en polaire, pas en taches
    for s_ in stars:
        if s_["giant"]:
            yy_, xx_ = np.mgrid[0:H, 0:W]
            gsoft |= np.hypot(xx_ - s_["x"], yy_ - s_["y"]) <= 1.6 * s_["R"]
    blobs = soft & ~gsoft & ((dl < -3 * sdl) | ((dl > 3 * sdl) & other_all.astype(bool)))
    lab, nb = ndi.label(blobs)
    if nb:
        sizes = ndi.sum(blobs, lab, np.arange(1, nb + 1))
        keep = np.isin(lab, 1 + np.nonzero(sizes >= 5)[0])
        add = ndi.binary_dilation(keep, iterations=3) & (soft | hard)
        if mode4:
            # tour 4 : une tache (reste d'une étoile faible sous le halo) est corrigée comme une étoile moyenne
            # (partie radiale retranchée, bruit ramené au ciel) ; seules les grandes taches (> 150 px) sont recréées
            big_t = np.zeros((H, W), bool)
            for k_, sl_ in enumerate(ndi.find_objects(lab), 1):
                if sizes[k_ - 1] < 5:
                    continue
                comp_ = lab == k_
                # un vrai creux est plus bas que le fond autour ; le flanc d'une volute de nébuleuse est sombre dans la
                # couche de détail mais pas sous le fond : celui-là n'est pas un défaut, on n'y touche pas
                ring_ = ndi.binary_dilation(comp_, iterations=12) & ~ndi.binary_dilation(comp_, iterations=4) & ~other_all.astype(bool) & ~hard
                if ring_.sum() > 20:
                    Lb__ = B.mean(0)
                    if Lb__[comp_].mean() > np.median(Lb__[ring_]) - 1.5 * sig:
                        continue
                if sizes[k_ - 1] > 150:
                    big_t |= comp_
                    continue
                cy_, cx_ = ndi.center_of_mass(comp_)
                rf = int(max(5, np.sqrt(sizes[k_ - 1] / np.pi) + 4))
                if fixed_m[int(round(cy_)), int(round(cx_))]:
                    continue        # déjà corrigé (étoile moyenne ou tache voisine) : pas de double correction
                if DEBUG_XY and np.hypot(cx_ - DEBUG_XY[0] - tx0 + wx0, cy_ - DEBUG_XY[1] - ty0 + wy0) < 20:
                    log(f"    [suivi] tache corrigée à ({cx_ - tx0 + wx0:.0f},{cy_ - ty0 + wy0:.0f}), {int(sizes[k_ - 1])} px, rayon {rf}")
                B = S4.medium_fix(B, cx_, cy_, hard | other_all.astype(bool) & ~comp_, None, r_fix=rf)
                fd = np.zeros((H, W), np.uint8); cv2.circle(fd, (int(round(cx_)), int(round(cy_))), rf, 1, -1)
                fixed_m |= fd.astype(bool)
            add = ndi.binary_dilation(big_t, iterations=3) & (soft | hard)
            log(f"  défauts de la base dans les halos : {int((sizes >= 5).sum())} taches, corrigées sur place sauf "
                f"{int(add.sum())} px recréés")
        else:
            log(f"  défauts de la base dans les halos : {int((sizes >= 5).sum())} taches, {int(add.sum())} px passés en dur")
        hard_t |= add & ~hard; hard |= add; soft &= ~hard
    # g3 (22 nov) : en mode 4 le disque recréé de la géante a le rayon « ra » (r_m4), pas r_hard : les rayons partent de son bord
    giants = [(s_["x"], s_["y"], s_.get("r_m4", s_["r_hard"]) + 2, s_["R"]) for s_ in stars if s_["giant"]] if DESTRIPE else []
    if mode4 and p1zones is not None:
        # base sans remplissage (c9) : coeurs saturés et aigrettes de la piste 1 remplis d'office ; zones « boucle non
        # convergée » et « écart du coeur » seulement si l'étoile s'y voit encore ou y laisse un creux sous l'anneau
        must = (p1zones.get("coeur_sature", 0) | p1zones.get("aigrette", 0)) & ~hard
        cand_ = (p1zones.get("boucle_non_convergee", 0) | p1zones.get("coeur_ecart", 0)) & ~hard & ~must
        labz, nz = ndi.label(cand_)
        Lz = B.mean(0); take = np.zeros((H, W), bool); nt = 0
        for k_, sl_ in enumerate(ndi.find_objects(labz), 1):
            comp_ = labz == k_
            ring_ = ndi.binary_dilation(comp_, iterations=6) & ~ndi.binary_dilation(comp_, iterations=1) & ~(must | cand_ | other_all.astype(bool))
            if ring_.sum() < 10:
                ring_ = ndi.binary_dilation(comp_, iterations=6) & ~ndi.binary_dilation(comp_, iterations=1) & ~(must | cand_)
            if ring_.sum() < 10:
                continue
            rm_ = np.median(Lz[ring_]); vals = Lz[comp_]
            sm1 = cv2.GaussianBlur(Lz.astype(np.float32), (0, 0), 1.0)[comp_]
            star_left = sm1.max() > rm_ + 5 * sig          # l'étoile se voit encore
            hole_ = sm1.min() < rm_ - 3 * sig or vals.mean() < rm_ - 1.5 * sig   # creux sous l'anneau
            if star_left or hole_:
                take |= comp_; nt += 1
        zp1 = must | take
        hard |= zp1; mask |= zp1; hard_t |= zp1 & ~must; hard_b |= must
        log(f"  zones de la piste 1 : coeurs saturés + aigrettes {int(must.sum())} px remplis ; "
            f"non convergées / écart du coeur : {nt} sur {nz} reprises ({int(take.sum())} px)")
    if mode4 and __import__("os").environ.get("P3_REMPLIR_ZONES_P1") == "1":
        # base de la piste 1 sans aucun remplissage : ses zones « à remplir » (étoiles non convergées, coeurs saturés,
        # aigrettes) sont remplies ici, comme les zones dures
        zp1 = p1fill[wy0:wy1, wx0:wx1] & ~hard
        hard |= zp1; mask |= zp1; hard_t |= zp1
        log(f"  zones à remplir laissées par la piste 1 : {int(zp1.sum())} px ajoutés aux zones recréées")
    if mode4 and V25_RESTE_REEL:
        # v25 (22 nov) : même règle que v18, étendue aux deux autres reprises de l'étape A : étoiles moyennes corrigées sur
        # place (fixed_m, classe 3) et défauts / zones de la piste 1 recréés (hard_t, classe 4). Leurs tests comparent le
        # reste au BRUIT DU CIEL ; sur une nébuleuse brillante (M57) la structure les déclenche toujours : l'anneau de M57
        # était détruit alors que la piste 1 l'avait laissé intact. Un pixel n'est repris que là où la piste 1 a laissé un
        # vrai reste (> 3 bruits de la structure locale) ; ailleurs on remet la piste 1 (moins les halos de l'étape A).
        # Les coeurs saturés de la piste 1 restent toujours recréés.
        import reste_reel as RRe
        fw25 = float(json.loads((d1 / "resume.json").read_text(encoding="utf-8"))["fwhm"]) if (d1 / "resume.json").exists() else 3.0
        P25 = B0 - H_tot
        z25 = (fixed_m | hard_t) & ~hard_b
        if V31_HORS_GEANTES and try_sigma:
            # v31 : dans la zone d'une géante (1,6 R), la base de travail est l'ENTRÉE (la piste 1 avait laissé la géante
            # entière) : rendre un pixel « à la piste 1 » y remet la géante. La règle v26 ne s'y applique pas.
            for s_ in stars:
                if s_["giant"]:
                    yy_, xx_ = np.mgrid[0:H, 0:W]
                    z25 &= np.hypot(xx_ - s_["x"], yy_ - s_["y"]) > 1.6 * s_["R"]
        if z25.any():
            if V26_ECHELLE_ETOILE:
                # v26 : un reste d'étoile a la taille d'une étoile : filtre à l'échelle de la PSF (gaussienne 0,5 FWHM moins
                # gaussienne 2 FWHM, comme le test des passes v9) sur la luminance de la piste 1 ; par morceau de z25, gardé
                # seulement si, à moins de 1,5 FWHM d'une étoile du catalogue du morceau, |filtre| > 5 x son écart robuste
                # dans l'anneau 4-12 px autour du morceau. La structure d'une nébuleuse (M57 : échelle 10-20 px) y est faible.
                Lp = P25.mean(0).astype(np.float32)
                Dp = cv2.GaussianBlur(Lp, (0, 0), 0.5 * fw25) - cv2.GaussianBlur(Lp, (0, 0), 2.0 * fw25)
                real25 = np.zeros((H, W), bool)
                rr25 = int(np.ceil(1.5 * fw25)); rd25 = int(np.ceil(3 * fw25))
                nkeep25 = 0
                for x_, y_, g_ in zip(gx - wx0, gy - wy0, gG):
                    X_, Y_ = int(round(x_)), int(round(y_))
                    if not (12 <= X_ < W - 12 and 12 <= Y_ < H - 12) or not z25[Y_, X_]:
                        continue
                    yy_, xx_ = np.mgrid[Y_ - 12:Y_ + 13, X_ - 12:X_ + 13]
                    r_ = np.hypot(xx_ - x_, yy_ - y_)
                    blk_ = Dp[Y_ - 12:Y_ + 13, X_ - 12:X_ + 13]
                    ring_ = blk_[(r_ >= 4) & (r_ <= 12)]
                    nr_ = 1.4826 * float(np.median(np.abs(ring_ - np.median(ring_)))) + 1e-9
                    if float(np.abs(blk_[r_ <= rr25]).max()) > 5 * nr_:
                        cv2.circle(real25.view(np.uint8), (X_, Y_), rd25, 1, -1)
                        nkeep25 += 1
                real25 &= z25
                log(f"  v26 : {nkeep25} étoiles du catalogue avec un vrai reste à l'échelle de l'étoile (disques de {rd25} px)")
            else:
                real25, _, _ = RRe.restes_reels(P25, z25, other_all.astype(bool) & ~z25, fw25, sigc)
            sat25 = p1zones["coeur_sature"].astype(bool) if (p1zones is not None and "coeur_sature" in p1zones) else np.zeros((H, W), bool)
            back25 = z25 & ~real25 & ~sat25
            B[:, back25] = P25[:, back25]
            hard &= ~back25; hard_t &= ~back25; mask &= ~back25; fixed_m &= ~back25
            log(f"  v25 reste réel (moyennes corrigées + défauts recréés) : {int(z25.sum())} px candidats, "
                f"{int(back25.sum())} rendus à la piste 1, {int((z25 & ~back25).sum())} gardés")
    if mode4:
        # trous durs fermés : pas de petits îlots de la base (creux de la soustraction) laissés au milieu d'un coeur
        # recréé, sinon la surface harmonique s'y accroche et creuse le coeur
        closed = ndi.binary_fill_holes(ndi.binary_closing(hard, structure=np.ones((5, 5), bool)))
        add_c = closed & ~hard
        hard |= add_c; mask |= add_c; hard_b |= add_c & ndi.binary_dilation(hard_b, iterations=3)
        hard_t |= add_c & ~hard_b
    # référence des grandes échelles : l'entrée moins les modèles d'étoiles (étoiles de la piste 1, halos, aigrettes et
    # corrections soustraits), avant tout remplissage
    np.save(out / "reference_entree_moins_modeles_tuile.npy", B[:, ty0 - wy0:ty1 - wy0, tx0 - wx0:tx1 - wx0])
    np.save(out / "zones_dures_tuile.npy", np.packbits(hard[ty0 - wy0:ty1 - wy0, tx0 - wx0:tx1 - wx0]))
    soft_fill = np.zeros_like(soft) if mode4 else soft     # tour 4 : plus de membrane sur le halo (il est soustrait)
    res = F.structure_texture2(B, hard, soft_fill, forbid_src & ~mask, sigc, log=log, giants=giants,
                               filler=("gauss" if filler == "gauss" else "patch"), ring_excl=other_all.astype(bool) | mask,
                               center_forbid=deep_forbid)
    if filler == "e3":
        res = e3_local(I, B, res, hard, sigc, log)
    # la membrane du halo efface aussi la structure de la nébuleuse sous le halo ; on la rend : écart de l'entrée
    # (flou de 6 px, sans les zones dures ni les autres étoiles) à sa médiane par anneau autour de l'étoile (le halo
    # est rond, la nébuleuse non), nul au bord du halo (la membrane y porte déjà l'extérieur) et dans le dur
    if KEEP_NEBULA and not mode4:
        excl = hard | ndi.binary_dilation(other_all.astype(bool), iterations=2)
        lowI = F.lowpass_masked(I, excl, 6.0)
        own = np.zeros((H, W), np.int32)
        for k, s in enumerate(stars):
            if s["giant"]:
                continue
            hm, sm = M.star_mask((H, W), dict(s))
            own[sm & ~hard] += 1
        # poids doux : seulement là où un seul halo agit et où le flou s'appuie sur assez de pixels sûrs
        cov = np.clip((cv2.GaussianBlur((~excl).astype(np.float32), (0, 0), 6.0) - 0.4) / 0.4, 0, 1)
        one = cv2.GaussianBlur(((own == 1) & soft).astype(np.float32), (0, 0), 4.0)
        wk = cov * np.clip((one - 0.5) * 2, 0, 1)
        for s in stars:
            if s["giant"]:
                continue
            Rr = s["R"]
            X, Y = int(round(s["x"])), int(round(s["y"]))
            ya, yb, xa, xb = max(0, Y - Rr), min(H, Y + Rr + 1), max(0, X - Rr), min(W, X + Rr + 1)
            yy, xx = np.mgrid[ya:yb, xa:xb]
            rr = np.hypot(xx - s["x"], yy - s["y"])
            ri = rr.astype(int)
            zone = (rr <= Rr) & soft[ya:yb, xa:xb] & (own[ya:yb, xa:xb] == 1)
            # seulement pour une étoile isolée : halos croisés (amas) = la médiane par anneau ne sépare plus rien
            disc = rr <= Rr
            occ = float(excl[ya:yb, xa:xb][disc & (rr > s["r_hard"] + 3)].mean())
            if not zone.any() or zone.sum() < 0.8 * (disc & ~hard[ya:yb, xa:xb]).sum() or occ > OCC_MAX:
                continue
            s["nebuleuse_rendue"] = True
            log(f"  structure de nébuleuse rendue sous le halo de G {s['g']:.2f} ({s['x'] + wx0 - tx0:.0f}, {s['y'] + wy0 - ty0:.0f}), occupation {occ:.2f}")
            w = np.clip((Rr - rr) / (0.3 * Rr), 0, 1)
            ok = (rr <= Rr) & ~excl[ya:yb, xa:xb]
            for c in range(I.shape[0]):
                v = lowI[c, ya:yb, xa:xb]
                med = np.full(Rr + 2, np.nan)
                for k in range(Rr + 1):
                    sel_ = ok & (ri == k)
                    if sel_.sum() >= 5:
                        med[k] = np.median(v[sel_])
                kk = np.arange(Rr + 2); good = np.isfinite(med)
                if good.sum() < 3:
                    continue
                med = np.interp(kk, kk[good], med[good])
                dev = v - med[np.clip(ri, 0, Rr + 1)]
                res[c, ya:yb, xa:xb][zone] += (w * wk[ya:yb, xa:xb] * dev)[zone]
    # contrôle : faux trous de même forme sur des zones sans étoile brillante, base piste 1
    ctrl = control(B, mask, stars, forbid_src, sigc, sig, other_all, log)
    # résultat dans la découpe
    dip_zone = np.zeros((H, W), bool)
    if mode4 and DIPS:
        # creux larges (plus de ~6 px) de la base, sous le fond local : la couche de détail ne les voit pas ; on ne refait
        # que leur basse fréquence (rien d'inventé dans le détail)
        Lr_ = res.mean(0)
        sm3 = cv2.GaussianBlur(Lr_.astype(np.float32), (0, 0), 3.0)
        bgl = ndi.median_filter(sm3, size=41)
        sdev = float(np.median(np.abs(sm3 - bgl)) * 1.4826)
        dips = (sm3 - bgl < -max(3 * sdev, 1.0 * sig)) & ~ndi.binary_dilation(hard, iterations=2)
        labd, nd = ndi.label(dips)
        if nd:
            szd = ndi.sum(dips, labd, np.arange(1, nd + 1))
            keepd = np.isin(labd, 1 + np.nonzero(szd >= 40)[0])
            # v3 : un creux déjà sombre dans l'ENTRÉE est une structure réelle (couloir, globule) : on ne le refait pas
            # entrée = luminance, flou de 3 px, sans le dur (dilaté 2) ni les autres étoiles (dilatées 2)
            excl_d = ndi.binary_dilation(hard, iterations=2) | ndi.binary_dilation(other_all.astype(bool), iterations=2)
            Lin_ = F.lowpass_masked(I, excl_d, 3.0).mean(0)
            n_found = int((szd >= 40).sum()); n_drop = 0
            for k in (1 + np.nonzero(szd >= 40)[0]):
                comp = labd == k
                ring = ndi.binary_dilation(comp, iterations=8) & ~ndi.binary_dilation(comp, iterations=2)
                rv = ring & ~excl_d; iv = comp & ~excl_d
                if rv.sum() < 10 or iv.sum() < 10:
                    continue
                d_res = np.median(sm3[ring]) - np.median(sm3[comp])
                d_in = np.median(Lin_[rv]) - np.median(Lin_[iv])
                if not d_in < 0.5 * d_res:
                    keepd[comp] = False; n_drop += 1
            log(f"  creux larges v3 : trouvés {n_found}, gardés {n_found - n_drop}, écartés (sombres dans l'entrée) {n_drop}")
            dip_zone = ndi.binary_dilation(keepd, iterations=4) & ~hard
            if dip_zone.any():
                res = F.relevel_lowfreq(res, dip_zone)
            log(f"  creux larges de la base : {int((szd >= 40).sum())}, {int(dip_zone.sum())} px, basse fréquence refaite")
    arb = None
    if mode4 and ARBITRATE:
        res, arb, won = arbitrate(B0, H_tot, res, hard, gx - wx0, gy - wy0, gG, angles, sig, other_all.astype(bool),
                                  (tx0 - wx0, ty0 - wy0, tx1 - wx0, ty1 - wy0), log, sigc)
        hard &= ~won; hard_b &= ~won; hard_t &= ~won; mask &= ~won
    R_crop = p1.copy()
    R_crop[:, wy0:wy1, wx0:wx1] = res
    np.save(out / "sans_etoiles_tuile.npy", R_crop[:, ty0:ty1, tx0:tx1])
    np.save(out / "masque_tuile.npy", np.packbits(mask[ty0 - wy0:ty1 - wy0, tx0 - wx0:tx1 - wx0]))
    # carte des pixels inventés, pour la base d'entraînement : 0 intact, 1 halo (basse fréquence refaite, détail
    # gardé), 2 recréé étoile brillante, 3 recréé étoile moyenne, 4 recréé défaut de la base dans un halo
    cls = np.zeros((H, W), np.uint8)
    if mode4:      # 1 halo soustrait, 3 moyenne corrigée : pas inventés ; 2, 4 recréés ; 5 recréé par E3
        cls[(sub_zone | dip_zone) & ~hard] = 1; cls[fixed_m & ~hard] = 3; cls[hard_t] = 4; cls[hard_b] = 2
        if filler == "e3":
            cls[hard] = 5
    else:
        cls[soft & ~hard] = 1; cls[hard_t] = 4; cls[hard_m & ~hard_b] = 3; cls[hard_b] = 2
    ct = cls[ty0 - wy0:ty1 - wy0, tx0 - wx0:tx1 - wx0]
    np.save(out / "carte_remplissage_tuile.npy", ct)
    names = ((1, "halo_soustrait"), (2, "recree_brillantes"), (3, "moyennes_corrigees"), (4, "recree_defauts_base"),
             (5, "recree_E3")) if mode4 else ((1, "halo_basse_frequence"), (2, "recree_brillantes"), (3, "recree_moyennes"),
                                               (4, "recree_defauts_base"))
    part = {nm: round(float((ct == k).mean() * 100), 2) for k, nm in names}
    part["total_invente"] = round(float(np.isin(ct, [2, 4, 5] if mode4 else [2, 3, 4]).mean() * 100), 2)
    log(f"  part de la tuile (%) : {part}")
    # référence : le grain de la base (piste 1) sur le ciel intact sans étoile, interpolé sous les masques ;
    # (contre l'entrée, les étoiles faibles non cataloguées gonflent le « grain » de l'entrée)
    gm = grain_map(B, res, other_all.astype(bool) | mask, hard, cls)

    save_img(out / "6_grain_rapport_local.png", gm["image"][ty0 - wy0:ty1 - wy0, tx0 - wx0:tx1 - wx0])
    log(f"  grain local (sans étoiles / entrée) par classe : {gm['stats']}")
    meas = dict(base_piste1=d1.name, angles=angles, controle=ctrl, secondes=None, part_tuile_pct=part, grain=gm['stats'], arbitrage=arb, v18=v18)
    meas["etoiles"] = [dict(G=round(s["g"], 2), x_tuile=round(s["x"] + wx0 - tx0, 1), y_tuile=round(s["y"] + wy0 - ty0, 1),
                            coeur=s["r_core"], halo=s["r_halo"], residu_p1=s["r_res"], R=s["R"], aigrettes=s["spikes"],
                            r_dur=s["r_hard"], **border_stats(res, B, np.logical_or(*M.star_mask((H, W), dict(s))), sigc),
                            grain_zone_dure=border_stats(res, B, M.star_mask((H, W), dict(s))[0], sigc)["grain_dedans_sur_dehors"])
                       for s in stars]
    meas.update(metrics(case, name, crop, tile, full, img, p1, R_crop, p1ex, gx, gy, gG, log))
    render(case, name, tile, full, img, p1, R_crop, mask, (wx0, wy0), stars, out, meas, hard)
    meas["secondes"] = round(time.time() - t0)
    (out / "mesures.json").write_text(json.dumps(meas, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    log(f"[{case}] fini en {meas['secondes']} s -> {out}")
    return meas


_E3 = {}


def e3_local(I, B, res, hard, sigc, log):
    """Variante : réseau E3 de FLOAT (entraîné sur étoiles synthétiques) appliqué à la fenêtre d'entrée, gardé seulement
    dans les zones dures ; par zone : niveau recalé sur l'anneau 2-8 px de la base (E3 creuse sous les vraies étoiles),
    grain fin ramené à celui de l'anneau, bord fondu (Poisson)."""
    import torch
    sys.path.insert(0, r"<FLOAT>")
    from astro_gpu.stages import star_removal as sr
    torch.set_num_threads(4)
    key = (I.shape, float(I[:, ::97, ::97].sum()))
    if key not in _E3:
        dev = torch.device("cpu")
        net = sr.NAFNet().to(dev)
        st = torch.load(r"<ESSAIS>/starRemoval/train/E3/snap/step_20000.pt", map_location=dev, weights_only=False)["net"]
        net.load_state_dict({k: v.float() for k, v in st.items()}); net.eval()
        t = time.time()
        _E3[key] = sr.apply(net, I.astype(np.float32), dev)
        log(f"  E3 sur la fenêtre : {time.time() - t:.0f} s")
    E = _E3[key]
    out = res.copy()
    lab, n = ndi.label(hard)
    mad = lambda v: float(np.median(np.abs(v - np.median(v))) * 1.4826)
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        y0, y1 = max(0, sl[0].start - 12), min(hard.shape[0], sl[0].stop + 12)
        x0, x1 = max(0, sl[1].start - 12), min(hard.shape[1], sl[1].stop + 12)
        comp = lab[y0:y1, x0:x1] == k
        ring = ndi.binary_dilation(comp, iterations=8) & ~ndi.binary_dilation(comp, iterations=2) & ~hard[y0:y1, x0:x1]
        if ring.sum() < 10:
            continue
        cand = E[:, y0:y1, x0:x1].copy()
        for c in range(cand.shape[0]):
            cand[c] += np.median(B[c, y0:y1, x0:x1][ring]) - np.median(cand[c][ring])
        Lc_ = cand.mean(0); hpE = Lc_ - cv2.GaussianBlur(Lc_.astype(np.float32), (0, 0), 1.5)
        Lb_ = B[:, y0:y1, x0:x1].mean(0); hpB = Lb_ - cv2.GaussianBlur(Lb_.astype(np.float32), (0, 0), 1.5)
        inner = ndi.binary_erosion(comp, iterations=1)
        if inner.sum() > 10:
            f = float(np.clip(mad(hpB[ring]) / max(mad(hpE[inner]), 1e-9), 0.5, 2.2))
            for c in range(cand.shape[0]):
                lo = cv2.GaussianBlur(cand[c].astype(np.float32), (0, 0), 1.5)
                cand[c] = lo + f * (cand[c] - lo)
        blended = F.poisson_blend(B[:, y0:y1, x0:x1], cand, comp)
        out[:, y0:y1, x0:x1][:, comp] = blended[:, comp]
    return out


def _spike_left(Lx, x, y, angles, r0, r1):
    """Reste d'aigrette : |bande centrale - bandes de côté| moyen sur r0..r1 le long des angles de l'instrument."""
    if not angles:
        return 0.0
    ds = np.arange(r0, r1, 1.0)
    vals = []
    for a in set([float(a) % 360 for a in angles] + [(float(a) + 180) % 360 for a in angles]):
        t = np.deg2rad(a); ux, uy = np.cos(t), np.sin(t); nx, ny = -uy, ux
        st = lambda o: ndi.map_coordinates(Lx, [y + ds * uy + o * ny, x + ds * ux + o * nx], order=1, mode="nearest")
        cen = np.mean([st(o) for o in (-1, 0, 1)], 0); side = np.median([st(o) for o in (-7, -5, 5, 7)], 0)
        vals.append(abs(float(np.mean(cen - side))))
    return float(np.max(vals))


def arbitrate(B0, H_tot, res, hard, xs, ys, gs, angles, sig, starz, tile_w, log, sigc=None):
    """Pour chaque étoile Gaia G < 13 : sa zone (rayon 3 (14 - G), 8 à 40 px) jugée sur la piste 1 seule (halos des
    brillantes retirés, pour comparer à niveau égal) et sur la piste 3 : note = |coeur - anneau| + creux sous l'anneau
    (compté double) + reste d'aigrette (en bruits) + 2 x part de pixels inventés ; la plus basse l'emporte. Si la
    piste 1 l'emporte, sa zone est reprise (bord fondu sur 3 px)."""
    C, H, W = res.shape
    P1h = B0 - H_tot
    L1 = P1h.mean(0); L3 = res.mean(0)
    out = res.copy(); won = np.zeros((H, W), bool)
    n1 = n3 = 0
    detail = []
    order = np.argsort(gs)
    for i in order:
        x, y, g = float(xs[i]), float(ys[i]), float(gs[i])
        if g >= 13 or not (20 <= x < W - 20 and 20 <= y < H - 20):
            continue
        rz = int(np.clip(3 * (14 - g), 8, 40))
        x0, x1 = max(0, int(x) - rz - 8), min(W, int(x) + rz + 9)
        y0, y1 = max(0, int(y) - rz - 8), min(H, int(y) + rz + 9)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        rr = np.hypot(xx - x, yy - y)
        core = rr <= 3; ring = (rr >= rz) & (rr <= rz + 6) & ~starz[y0:y1, x0:x1]
        if ring.sum() < 10:
            ring = (rr >= rz) & (rr <= rz + 6)
        def score(Lx, inv, X3):
            c = (Lx[y0:y1, x0:x1][core].mean() - np.median(Lx[y0:y1, x0:x1][ring])) / sig
            sp = _spike_left(Lx, x, y, angles, 4, rz) / sig
            # point coloré : écart de la couleur la plus éloignée à la luminance, coeur moins anneau, en bruits de couleur
            ch = 0.0
            if sigc is not None:
                for k in range(X3.shape[0]):
                    ck = X3[k, y0:y1, x0:x1][core].mean() - np.median(X3[k, y0:y1, x0:x1][ring])
                    ch = max(ch, abs(ck - c * sig) / sigc[k])
            return abs(c) + max(0.0, -c) + sp + 0.5 * ch + 2 * inv
        zone = rr <= rz
        inv3 = float(hard[y0:y1, x0:x1][zone].mean())
        s1, s3 = score(L1, 0.0, P1h), score(L3, inv3, res)
        inT = tile_w[0] <= x < tile_w[2] and tile_w[1] <= y < tile_w[3]
        if s1 < s3:
            w = np.clip((rz - rr) / 3.0, 0, 1).astype(np.float32)
            out[:, y0:y1, x0:x1] = w * P1h[:, y0:y1, x0:x1] + (1 - w) * out[:, y0:y1, x0:x1]
            won[y0:y1, x0:x1] |= zone
            n1 += inT
        else:
            n3 += inT
        if inT:
            detail.append(dict(x=round(x - tile_w[0]), y=round(y - tile_w[1]), G=round(g, 2), note_p1=round(s1, 2), note_p3=round(s3, 2),
                               choix="piste1" if s1 < s3 else "piste3"))
    if log:
        log(f"  arbitrage étoile par étoile (zone jugée, G < 13) : piste 1 seule {n1}, piste 3 {n3}")
    return out, dict(piste1=int(n1), piste3=int(n3), etoiles=detail), won


def grain_map(Bse, R, starz, hard, cls):
    """Grain par zone recréée : écart type robuste (MAD) du détail à 2 px dans la zone (rognée de 2 px, bord exclu)
    contre celui de la base (piste 1) dans l'anneau 4-16 px autour, hors étoiles et hors masques. Rapport par zone,
    image des zones colorées (bleu < 0.9 trop lisse, gris 0.9-1.1, rouge > 1.1), parts en pixels par classe."""
    hpR = R.mean(0) - cv2.GaussianBlur(R.mean(0), (0, 0), 2.0)
    hpB = Bse.mean(0) - cv2.GaussianBlur(Bse.mean(0), (0, 0), 2.0)
    mad = lambda v: float(np.median(np.abs(v - np.median(v))) * 1.4826)
    lab, n = ndi.label(hard)
    ratio = np.full(hard.shape, np.nan, np.float32)
    okring = ~(starz | hard | np.isin(cls, [2, 4, 5]))   # anneau : pixels vrais (intacts, halo soustrait, moyenne corrigée)
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        y0, y1 = max(0, sl[0].start - 16), min(hard.shape[0], sl[0].stop + 16)
        x0, x1 = max(0, sl[1].start - 16), min(hard.shape[1], sl[1].stop + 16)
        comp = lab[y0:y1, x0:x1] == k
        inner = ndi.binary_erosion(comp, iterations=2)
        if inner.sum() < 20:
            inner = comp
        ring = F.rim(comp, ~okring[y0:y1, x0:x1], hard[y0:y1, x0:x1])      # même anneau que la correction
        if ring.sum() < 40:
            continue
        r = mad(hpR[y0:y1, x0:x1][inner]) / max(mad(hpB[y0:y1, x0:x1][ring]), 1e-9)
        ratio[y0:y1, x0:x1][comp] = r
    stats = {}
    for k, nm in ((2, "brillantes"), (3, "moyennes"), (4, "defauts")):
        z = (cls == k) & np.isfinite(ratio)
        if z.sum() > 50:
            r = ratio[z]
            stats[nm] = dict(mediane=round(float(np.median(r)), 2), px_hors_0_9_1_1_pct=round(float(100 * ((r < 0.9) | (r > 1.1)).mean()), 1),
                             px_sous_0_9_pct=round(float(100 * (r < 0.9).mean()), 1))
    rgb = np.full(hard.shape + (3,), 40, np.uint8)
    g = np.isfinite(ratio)
    rgb[g] = (150, 150, 150)
    rgb[g & (ratio < 0.9)] = (60, 60, 230)
    rgb[g & (ratio > 1.1)] = (230, 60, 60)
    return dict(image=rgb, stats=stats, ratio=ratio)


def border_stats(res, base, m, sigc):
    """Marche au bord (anneau 1-3 px dedans moins 1-3 px dehors, en bruits) et rapport de grain dedans / dehors."""
    inner = m & ~ndi.binary_erosion(m, iterations=3)
    outer = ndi.binary_dilation(m, iterations=3) & ~m
    L = res.mean(0); s = float(np.mean(sigc))
    hp = L - cv2.GaussianBlur(L, (0, 0), 2)
    core = ndi.binary_erosion(m, iterations=4)
    ring = ndi.binary_dilation(m, iterations=25) & ~ndi.binary_dilation(m, iterations=5)
    mad = lambda v: float(np.median(np.abs(v - np.median(v))) * 1.4826) if v.size else float("nan")
    return dict(marche_bord_bruits=round(float((np.median(L[inner]) - np.median(L[outer])) / s), 2),
                grain_dedans_sur_dehors=round(mad(hp[core]) / max(mad(hp[ring]), 1e-9), 2) if core.any() else None)


def control(B, mask, stars, forbid, sigc, sig, other, log):
    """Faux trous : la forme de chaque masque d'étoile posée sur une zone sans étoile brillante de la base piste 1,
    remplie, comparée au vrai : écart de niveau (bruits et % du niveau), rapport de grain, marche au bord."""
    H, W = mask.shape
    rng = np.random.default_rng(1)
    rows = []
    mb = np.zeros((H, W), bool)
    for s_ in stars:
        a_, b_ = M.star_mask((H, W), dict(s_)); mb |= a_ | b_
    busy = ndi.binary_dilation(mb, iterations=10)     # loin des étoiles brillantes (les reprises moyennes, permises)
    for s in stars[:6]:
        hm, sm = M.star_mask((H, W), dict(s))
        ys, xs = np.nonzero(hm | sm)
        cy, cx = int(round(s["y"])), int(round(s["x"]))
        for _ in range(400):
            dy = int(rng.integers(-H, H)); dx = int(rng.integers(-W, W))
            ny, nx = ys + dy, xs + dx
            if ny.min() < 4 or ny.max() >= H - 4 or nx.min() < 4 or nx.max() >= W - 4:
                continue
            if busy[ny, nx].any():
                continue
            fm = np.zeros((H, W), bool); fm[ny, nx] = True
            fh = np.zeros((H, W), bool); hy, hx = np.nonzero(hm); fh[hy + dy, hx + dx] = True
            filled = F.structure_texture2(B, fh, fm & ~fh, (mask | forbid) & ~fm, sigc)
            L0 = B.mean(0); L1 = filled.mean(0)
            lvl = float(np.median(L0[fm]))
            hp0 = L0 - cv2.GaussianBlur(L0, (0, 0), 2); hp1 = L1 - cv2.GaussianBlur(L1, (0, 0), 2)
            mad = lambda v: float(np.median(np.abs(v - np.median(v))) * 1.4826)
            core = ndi.binary_erosion(fm, iterations=4)
            r = dict(G=round(s["g"], 1), px=int(fm.sum()), niveau=round(lvl, 2),
                     ecart_bruits=round(float(np.mean(L1[fm] - L0[fm]) / sig), 2),
                     ecart_pct=round(float(100 * np.mean(L1[fm] - L0[fm]) / max(abs(lvl), 1e-6)), 2),
                     grain_rempli_sur_vrai=round(mad(hp1[core]) / max(mad(hp0[core]), 1e-9), 2) if core.any() else None,
                     **{"marche_bord_bruits": border_stats(filled, B, fm, sigc)["marche_bord_bruits"]})
            rows.append(r)
            log(f"  contrôle faux trou {r}")
            break
    return rows


def ref_core_annulus(full_sl, crop, refs, fwhm, sig):
    """Mesure de la piste 1 : coeur (r <= fwhm) moyen moins médiane de l'anneau 2-4 fwhm, en bruits."""
    L = full_sl.mean(0)
    out = []
    for x, y, g in refs:
        x -= crop[0]; y -= crop[1]
        rr = int(4 * fwhm) + 1
        xi, yi = int(round(x)), int(round(y))
        gy, gx = np.mgrid[-rr:rr + 1, -rr:rr + 1]
        r = np.hypot(gx + xi - x, gy + yi - y)
        c = L[yi - rr:yi + rr + 1, xi - rr:xi + rr + 1]
        v = c[r <= max(1.0, fwhm)].mean() - np.median(c[(r >= 2 * fwhm) & (r <= 4 * fwhm)])
        out.append(dict(G=g, coeur_moins_anneau_bruits=round(float(v / sig), 2)))
    return out


def metrics(case, name, crop, tile, full, img, p1, R_crop, p1ex, gx, gy, gG, log):
    x0, y0, x1, y1 = tile
    H, W = img.shape[1:]
    inT = (gx > x0) & (gx < x1) & (gy > y0) & (gy < y1) & (gG < 13)
    inT &= ~p1ex[np.clip(gy.astype(int), 0, H - 1), np.clip(gx.astype(int), 0, W - 1)]
    m = dict(creux_piste3=hole_metrics(R_crop, gx, gy, None, 3, inT), creux_piste1=hole_metrics(p1, gx, gy, None, 3, inT))
    # bruit global comme juger.py (luminance de l'image entière)
    Lf = full.mean(0)
    sigf = M.noise_sigma(Lf)
    m["bruit_juger"] = round(sigf, 4)
    if name == "rosette":
        sigf = 0.093   # bruit donné par la piste 1 pour ses mesures de référence
    if case == "rosette_decoupe6":
        sig6 = 0.093
        m["six_etoiles_piste3"] = ref_core_annulus(R_crop, crop, SIX, 3.19, sig6)
        m["six_etoiles_piste1"] = ref_core_annulus(p1, crop, SIX, 3.19, sig6)
    if name == "rosette" and case != "rosette_decoupe6":
        fw = 2.985   # largeur à mi-hauteur de la piste 1 (resume.json, Rosette)
        m["references_piste3"] = ref_core_annulus(R_crop, crop, REF_ROSETTE, fw, sigf)
        m["references_piste1"] = ref_core_annulus(p1, crop, REF_ROSETTE, fw, sigf)
        m["references_entree"] = ref_core_annulus(img, crop, REF_ROSETTE, fw, sigf)
    log(f"  mesures : {json.dumps({k: v for k, v in m.items()}, ensure_ascii=False)}")
    return m


def render(case, name, tile, full, img, p1, R_crop, mask, woff, stars, out, meas, hard=None):
    x0, y0, x1, y1 = tile
    sl = lambda a: a[:, y0:y1, x0:x1]
    I, A, Bm = sl(img), sl(p1), sl(R_crop)
    meta = M.images.meta(name)
    top = float(meta.get("top") or full.max()); off = float(meta.get("offset", 3.0)); soft = float(meta.get("soft", 3.0))
    L = I.mean(0); sig = M.noise_sigma(L)
    g = full[1] if full.shape[0] > 1 else full[0]
    bk = float(np.median(g[::7, ::7])); t998 = float(np.percentile(g[::5, ::5], 99.8))

    def revue(a):
        v = np.arcsinh(np.clip(a + off, 0, None) / soft) / np.arcsinh((top + off) / soft)
        return (np.transpose(np.clip(v, 0, 1), (1, 2, 0)) * 255 + 0.5).astype(np.uint8)

    def post(a):
        x = np.clip((a - bk) / max(t998 - bk, 1e-9), 0, None)
        v = np.arcsinh(40 * x) / np.arcsinh(40)
        return (np.transpose(np.clip(v, 0, 1), (1, 2, 0)) * 255 + 0.5).astype(np.uint8)

    def pousse(a):
        return to_rgb8(a, -2 * sig, float(np.percentile(L, 60)) + 15 * sig)
    sep = np.full((I.shape[1], 6, 3), 255, np.uint8)
    for label, f in (("1_revue", revue), ("2_post", post), ("3_extreme", pousse)):
        save_img(out / f"{label}_entree_piste1_piste3_couche.jpg",
                 np.concatenate([f(I), sep, f(A), sep, f(Bm), sep, f(I - Bm)], 1))
    ov = post(I).copy()
    mt = mask[y0 - woff[1]:y1 - woff[1], x0 - woff[0]:x1 - woff[0]]
    edge = mt & ~ndi.binary_erosion(mt)
    ov[edge] = (0, 255, 255)
    if hard is not None:
        ht = hard[y0 - woff[1]:y1 - woff[1], x0 - woff[0]:x1 - woff[0]]
        ov[ht & ~ndi.binary_erosion(ht)] = (255, 255, 0)
    save_img(out / "4_masques_post.jpg", ov)
    # plus grande zone recréée (détail copié) de la tuile, à 100 % et agrandie x3 : contrôle des répétitions de patches
    if hard is not None:
        ht = hard[y0 - woff[1]:y1 - woff[1], x0 - woff[0]:x1 - woff[0]]
        lab, nl = ndi.label(ht)
        if nl:
            k = int(np.argmax(ndi.sum(ht, lab, np.arange(1, nl + 1)))) + 1
            sy, sx = ndi.find_objects(lab)[k - 1]
            ya, yb = max(0, sy.start - 20), min(y1 - y0, sy.stop + 20); xa, xb = max(0, sx.start - 20), min(x1 - x0, sx.stop + 20)
            meas["plus_grande_zone_recreee"] = dict(px=int((lab == k).sum()), x=[xa, xb], y=[ya, yb])
            for label, f in (("post", post), ("extreme", pousse)):
                c = lambda a: f(a[:, ya:yb, xa:xb])
                sp_ = np.full((yb - ya, 4, 3), 255, np.uint8)
                row = np.concatenate([c(I), sp_, c(A), sp_, c(Bm)], 1)
                save_img(out / f"5_plus_grande_zone_{label}_100pc_entree_p1_p3.jpg", row)
                if label == "post":
                    z = c(Bm)
                    save_img(out / "5_plus_grande_zone_post_x3_p3.png", cv2.resize(z, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
    # zone recréée sur la nébuleuse la plus structurée (gradient de l'entrée sans étoiles lissée à 3 px autour) :
    # contrôle de la continuité des filaments, à 100 % et x3
    if hard is not None:
        ht = hard[y0 - woff[1]:y1 - woff[1], x0 - woff[0]:x1 - woff[0]]
        lab, nl = ndi.label(ht)
        Ab = cv2.GaussianBlur(A.mean(0), (0, 0), 3.0)
        gmag = np.hypot(cv2.Sobel(Ab, cv2.CV_32F, 1, 0), cv2.Sobel(Ab, cv2.CV_32F, 0, 1))
        best = None
        for k, sl_ in enumerate(ndi.find_objects(lab), 1):
            comp = lab == k
            if comp.sum() < 400:
                continue
            ring = ndi.binary_dilation(comp, iterations=12) & ~ndi.binary_dilation(comp, iterations=3)
            sc_ = float(np.percentile(gmag[ring], 90))       # le bord le plus structuré (filament ou bord de nébuleuse)
            if best is None or sc_ > best[0]:
                best = (sc_, sl_)
        if best is not None:
            sy, sx = best[1]
            cy_, cx_ = (sy.start + sy.stop) // 2, (sx.start + sx.stop) // 2
            hh = max(48, (sy.stop - sy.start) // 2 + 24, (sx.stop - sx.start) // 2 + 24)
            ya, yb = max(0, cy_ - hh), min(y1 - y0, cy_ + hh); xa, xb = max(0, cx_ - hh), min(x1 - x0, cx_ + hh)
            meas["crop_filament"] = dict(x=[xa, xb], y=[ya, yb])
            for label, f in (("post", post), ("extreme", pousse)):
                c = lambda a: f(a[:, ya:yb, xa:xb])
                sp_ = np.full((yb - ya, 4, 3), 255, np.uint8)
                row = np.concatenate([c(I), sp_, c(A), sp_, c(Bm)], 1)
                save_img(out / f"7_filament_{label}_100pc_entree_p1_p3.jpg", row)
                save_img(out / f"7_filament_{label}_x3_entree_p1_p3.png", cv2.resize(row, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
    if case == "rosette_decoupe6":
        for (fx, fy, _) in SIX:
            cx, cy = int(fx - M.CASES[case][1][0]), int(fy - M.CASES[case][1][1])
            ya, yb, xa, xb = max(0, cy - 40), min(img.shape[1], cy + 40), max(0, cx - 40), min(img.shape[2], cx + 40)
            c = lambda a: post(a[:, ya:yb, xa:xb])
            sp_ = np.full((yb - ya, 4, 3), 255, np.uint8)
            row = np.concatenate([c(img), sp_, c(p1), sp_, c(R_crop)], 1)
            save_img(out / f"9_six_{fx}_{fy}_100pc_entree_p1_p3.png", row)
            save_img(out / f"9_six_{fx}_{fy}_x3_entree_p1_p3.png", cv2.resize(row, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
    # crops à 100 % de chaque étoile traitée (post) : entrée | piste 1 | piste 3 ; 128 px et étendue du masque
    for s in stars:
        cx, cy = int(round(s["x"] + woff[0])), int(round(s["y"] + woff[1]))
        if not (x0 <= cx < x1 and y0 <= cy < y1):
            continue
        for tag, hw in (("128", 64), ("masque", max(64, max([s["R"]] + [sp["length"] for sp in s["spikes"]]) + 20))):
            ya, yb = max(0, cy - hw), min(img.shape[1], cy + hw); xa, xb = max(0, cx - hw), min(img.shape[2], cx + hw)
            c = lambda a: a[:, ya:yb, xa:xb]
            sp_ = np.full((yb - ya, 4, 3), 255, np.uint8)
            save_img(out / f"crop_G{s['g']:.2f}_x{cx - x0}_y{cy - y0}_{tag}_entree_p1_p3.jpg",
                     np.concatenate([post(c(img)), sp_, post(c(p1)), sp_, post(c(R_crop))], 1))
