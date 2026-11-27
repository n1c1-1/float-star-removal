# ======================================================================
# mesure.py
# Role     : Track 3: per-bright-star measurements (halo radius, spike angles and lengths) and mask.
# Step     : 2. Classic Gaia-guided subtraction chain (v33b) (see scripts/LISEZMOI.md)
# Inputs   : stacked image (noise units), astrometry, offline deep Gaia extract
# Outputs  : star-free image, class maps
# Example  : python mesure.py [arguments]  (see the docstring; paths through the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Track 3: per-bright-star measurements (halo radius, spike angles and lengths) and mask."""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

P1 = (Path(__import__("os").environ.get("RACINE_SCRIPTS") or Path(__file__).resolve().parents[2]) / "Retrait étoiles par modèle – 20 nov")
sys.path.insert(0, str(P1))
import images  # noqa: E402

CASES = {
    # nom : (image, découpe x0, y0, x1, y1, tuile dans la découpe x0, y0, x1, y1) : mêmes que essai_tuiles.py (piste 1)
    "rosette_etoile_sur_nebuleuse": ("rosette", (917, 103, 2965, 2151), (768, 768, 1280, 1280)),
    "tete_de_cheval_sigma_ori": ("horsehead", (3800, 0, 5848, 2048), (768, 326, 1280, 838)),
    "ngc6543_nebuleuse_et_etoile_centrale": ("ngc6543", (0, 0, 2400, 2400), (989, 927, 1501, 1439)),
    # boucle de découpe de la piste 1 : les 6 étoiles entourées par le propriétaire, zone jugée = 512 px centraux
    "rosette_decoupe6": ("rosette", (1530, 790, 2298, 1558), (128, 128, 640, 640)),
}
# entrée linéaire non débruitée (21 nov) : même découpe des 6 étoiles, dans le repère de cet empilement (tourné ~178°)
LIN_DIR = (Path(__import__("os").environ.get("RACINE_SCRIPTS") or Path(__file__).resolve().parents[2]) / "Entrée non débruitée – 21 nov")
CASES["rosette_lin_decoupe6"] = ("rosette_lin", (1628, 1357, 2428, 2157), (144, 144, 656, 656))
_LOADED = {}


def load_image(name):
    """Image en unités de bruit (C, H, W) ; « rosette_lin » = l'empilement linéaire non débruité (lu en lecture seule),
    sinon le chargeur de la piste 1."""
    if name == "rosette_lin":
        if name not in _LOADED:
            a = np.load(LIN_DIR / "rosette_non_debruite.npy", mmap_mode="r")
            a = np.ascontiguousarray(np.array(a, np.float32)); a[~np.isfinite(a)] = 0
            _LOADED[name] = a
        return _LOADED[name]
    return images.load(name)


def meta_image(name):
    if name == "rosette_lin":
        import json as _j
        return _j.loads((LIN_DIR / "meta.json").read_text(encoding="utf-8"))
    return images.meta(name)


# sortie de la piste 1 imposée pour une tuile (sinon : dernier essai complet)
P1_OVERRIDE = {"rosette_decoupe6": __import__("os").environ.get("P3_BASE_DECOUPE", "c1_decoupe6"),
               "rosette_lin_decoupe6": __import__("os").environ.get("P3_BASE_LIN", "L3_decoupe6")}


def p1_dir(name, case):
    """Dernier essai de la piste 1 (essai10 ou plus) pour la tuile."""
    if case in P1_OVERRIDE:
        d = P1 / "sorties" / name / P1_OVERRIDE[case]
        return d if (d / "starless.npy").exists() and (d / "resume.json").exists() else None
    best = None
    for d in (P1 / "sorties" / name).glob(f"essai*_{case}"):
        try:
            k = int(d.name[5:].split("_")[0])
        except ValueError:
            continue
        if k >= 10 and (d / "starless.npy").exists() and (d / "resume.json").exists() and (best is None or k > best[0]):
            best = (k, d)
    return best[1] if best else None


def noise_sigma(a):
    hp = a - cv2.GaussianBlur(a, (0, 0), 6)
    return float(np.median(np.abs(hp - np.median(hp))) * 1.4826)


def polar(L, x, y, rs, th):
    R, T = np.meshgrid(rs, th, indexing="ij")
    return ndi.map_coordinates(L, [y + R * np.sin(T), x + R * np.cos(T)], order=1, mode="nearest")


def radial_profile(L, x, y, rmax, other=None):
    """Médiane par anneau de 1 px (autres étoiles exclues), r = 0..rmax."""
    H, W = L.shape
    x0, x1 = max(0, int(x - rmax)), min(W, int(x + rmax) + 1)
    y0, y1 = max(0, int(y - rmax)), min(H, int(y + rmax) + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    r = np.hypot(xx - x, yy - y).astype(int)
    v = L[y0:y1, x0:x1]
    ok = r <= rmax
    if other is not None:
        ok &= ~other[y0:y1, x0:x1]
    prof = np.full(rmax + 1, np.nan)
    order = np.argsort(r[ok], kind="stable")
    rr = r[ok][order]; vv = v[ok][order]
    edges = np.searchsorted(rr, np.arange(rmax + 2))
    for k in range(rmax + 1):
        s = vv[edges[k]:edges[k + 1]]
        if len(s) >= 3:
            prof[k] = np.median(s)
    return prof


def sym_angular(L, x, y, r_core, th):
    """Profil angulaire en z robuste (par anneau, r_core+4 .. r_core+44), rendu symétrique : min(z(t), z(t + 180))."""
    P = polar(L, x, y, np.arange(r_core + 4, r_core + 44, 1.0), th)
    med = np.median(P, 1, keepdims=True)
    mad = np.median(np.abs(P - med), 1, keepdims=True) * 1.4826 + 1e-6
    a = ndi.uniform_filter1d(np.clip((P - med) / mad, -5, 20).mean(0), 3, mode="wrap")
    return np.minimum(a, np.roll(a, len(th) // 2))


def peaks(S, zmin, step=0.5, sep=8):
    b = np.median(S); s = np.median(np.abs(S - b)) * 1.4826 + 1e-6
    out = []
    for i in np.argsort(-S):
        z = (S[i] - b) / s
        if z < zmin:
            break
        d = i * step
        if all(min(abs(d - q), 360 - abs(d - q)) > sep for q, _ in out):
            out.append((float(d), round(float(z), 1)))
    return out


def spike_angles(L, stars, log):
    """Angles des aigrettes de l'instrument : médiane, sur les étoiles brillantes de la découpe, du profil angulaire
    symétrique (une aigrette va dans les deux sens) ; les voisins d'une étoile ne pèsent pas sur la médiane."""
    th = np.deg2rad(np.arange(0, 360, 0.5))
    prof = [sym_angular(L, s["x"], s["y"], s["r_core"], th) for s in stars]
    S = np.median(prof, 0)
    # v15 (SHO 21 nov) : plus de plafond de 4 pics (un profil symétrique donne 2 pics par aigrette, [:4] = une croix) ;
    # tous les pics significatifs (z >= 2,5) dont l'opposé à 180 deg (+-8) est aussi un pic : M8 SHO a 6 aigrettes à 60 deg
    pk_all = peaks(S, 2.5)
    pk = [(a, z) for a, z in pk_all if any(abs(((a - q) % 360) - 180) <= 8 for q, _ in pk_all)]
    log(f"angles des aigrettes (deg, 0 = +x, sens image ; médiane de {len(prof)} étoiles, force en z) : {pk}")
    return sorted(a for a, _ in pk)


def measure_star(L, Ls, s, sig, sat, other, angles, D, sigD):
    """Rayon du coeur (>= 30 % de la saturation), rayon du halo (excès < 0.7 bruit sur l'entrée et sur le résidu de la
    piste 1), longueur et largeur de chaque aigrette."""
    x, y = s["x"], s["y"]
    rmax = int(s["rmax"])
    p = radial_profile(L, x, y, rmax, other)
    pr = radial_profile(Ls, x, y, rmax, other)
    tail = slice(int(rmax * 0.8), rmax + 1)
    b = np.nanmedian(p[tail]); br = np.nanmedian(pr[tail])
    ex = ndi.uniform_filter1d(np.nan_to_num(p - b), 3)
    exr = ndi.uniform_filter1d(np.nan_to_num(pr - br), 3)
    thr = 0.7 * sig
    core = np.nonzero(np.nan_to_num(p) > 0.3 * sat)[0]
    r_core = int(core.max()) + 1 if len(core) else 2
    above = np.nonzero(ex > thr)[0]
    r_halo = int(above.max()) + 1 if len(above) else r_core
    # premier passage durable sous le seuil (évite les voisins loin dans l'anneau)
    below = np.nonzero(ex[r_core:] <= thr)[0]
    for k in below:
        rr = r_core + k
        if np.all(ex[rr:rr + 6] <= thr):
            r_halo = rr
            break
    abv = np.nonzero(np.abs(exr) > thr)[0]
    r_res = r_core
    belr = np.nonzero(np.abs(exr[r_core:]) <= thr)[0]
    for k in belr:
        rr = r_core + k
        if np.all(np.abs(exr[rr:rr + 6]) <= thr):
            r_res = rr
            break
    # zone dure : là où le DÉTAIL de la base (piste 1) est faux autour de l'étoile : médiane par anneau du détail
    # (base moins son flou de 6 px) hors 0.5 bruit de détail, ou dispersion par anneau > 1.4 fois celle du ciel
    md = radial_profile(D, x, y, rmax, other)
    sd = radial_profile(np.abs(D - np.median(D)), x, y, rmax, other) * 1.4826
    badr = (np.abs(np.nan_to_num(md)) > 0.5 * sigD) | (np.nan_to_num(sd) > 1.4 * sigD)
    r_hard = r_core + 3
    for k in range(r_core, len(badr)):
        if not badr[k:k + 5].any():
            r_hard = max(r_core + 3, k + 1)
            break
    # le résidu de la piste 1 jusqu'à r_res n'entre pas d'office dans le dur : sa partie lisse est refaite par la membrane
    # du halo, ses taches (creux, rayons) passent en dur une à une ; une étoile moyenne garde un dur court
    # dans un amas, les anneaux ne redeviennent jamais propres (voisins) : le dur reste dans le quart intérieur du halo,
    # les défauts plus loin passent en dur tache par tache
    r_hard = int(min(r_hard, max(r_core + 6, 0.25 * r_halo)))
    if s.get("giant"):
        r_hard = int(max(r_hard, 0.3 * r_halo))   # géante : le coeur du reflet (rayons serrés, couleurs) est recréé
    if s.get("medium"):
        r_hard = int(max(r_core + 3, min(max(r_hard, r_res + 2), 20)))   # tout le défaut laissé, 20 px au plus
    s.update(r_core=r_core, r_halo=max(r_halo, r_hard), r_res=r_res, r_hard=r_hard, bg=float(b))
    # aigrettes mesurées sur la base (piste 1, qui ne les modélise pas) moins son modèle radial (médiane par anneau)
    H, W = Ls.shape
    R2 = int(min(rmax * 1.5, 300))
    x0, x1 = max(0, int(x) - R2), min(W, int(x) + R2 + 1); y0, y1 = max(0, int(y) - R2), min(H, int(y) + R2 + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.hypot(xx - x, yy - y)
    pf = np.nan_to_num(pr, nan=br)
    Rm = Ls[y0:y1, x0:x1] - np.interp(rr, np.arange(len(pf)), pf, right=br)
    lx, ly = x - x0, y - y0
    th = np.deg2rad(np.arange(0, 360, 0.5))
    own = [a for a, z in peaks(sym_angular(L, x, y, r_core, th), 6.0, sep=12)][:4]
    own += [a for a, z in peaks(sym_angular(Ls, x, y, r_core, th), 6.0, sep=12)][:4]
    s["angles_propres"] = sorted(set(round(a) for a in own))
    cand = own + list(angles) + [(a + 90) % 360 for a in list(angles) + own]
    cand = cand + [(a + 180) % 360 for a in cand]
    ds = np.arange(0, R2, 1.0)
    def strip(a, off, dd=ds):
        t = np.deg2rad(a); ux, uy = np.cos(t), np.sin(t); nx, ny = -uy, ux
        return ndi.map_coordinates(Rm, [ly + dd * uy + off * ny, lx + dd * ux + off * nx], order=1, mode="nearest")
    def excess(a):
        cen = np.mean([strip(a, o) for o in (-1, 0, 1)], 0)
        side = np.median([strip(a, o) for o in (-12, -9, -6, 6, 9, 12)], 0)
        return ndi.uniform_filter1d(np.abs(cen - side), 7)   # reste d'aigrette clair ou sombre (sur-soustraction)
    # chaque angle candidat affiné à +-6 deg sur l'étoile (excès moyen sur les 40 px après le coeur)
    st0 = max(r_core, 3, int(r_hard) - 2)
    angs = []
    for a in cand:
        best = max(np.arange(a - 2, a + 2.01, 0.5), key=lambda b: float(np.mean(excess(b)[st0:st0 + 60])))   # la direction de l'instrument est fixe : affinage de ±2°
        best = float(best % 360)
        if all(min(abs(best - q), 360 - abs(best - q)) > 8 for q in angs):
            angs.append(best)
    # seuil par distance : angles témoins loin de toute aigrette candidate (médiane + 4 MAD)
    ctrl = [a for a in np.arange(7.5, 360, 15) if all(min(abs(a - q), 360 - abs(a - q)) > 10 for q in angs)]
    E = np.array([excess(a) for a in ctrl])
    mE = np.median(E, 0); sE = np.median(np.abs(E - mE), 0) * 1.4826
    thr_d = mE + np.maximum(4 * ndi.uniform_filter1d(sE, 15), 0.3 * sig)
    spikes = []; spikes_raw = []
    for a in angs:
        e = excess(a)
        start = max(r_core, 3, int(r_hard) - 2)   # dans le disque dur, la base est déjà refaite
        length = start; gap = 0; seen = False
        for d in range(start, len(ds)):
            if e[d] > thr_d[d]:
                length = d; gap = 0; seen = True
            else:
                gap += 1
                # près du coeur, le reste de la soustraction peut masquer le début de l'aigrette : on la cherche
                # jusqu'à 40 px, puis on s'arrête après 10 px sans elle
                if (seen and gap > 10) or (not seen and d > start + 40):
                    break
        spikes_raw.append(dict(angle=float(a), length=int(length), start=start, e=e))
    # une aigrette de diffraction est symétrique : l'opposée (à 3 deg près) reçoit au moins 80 % de la longueur
    for sp in spikes_raw:
        for o in spikes_raw:
            if abs(((o["angle"] - sp["angle"]) % 360) - 180) <= 3:
                sp["length"] = max(sp["length"], int(0.8 * o["length"]))
    for sp in spikes_raw:
        a, length, start = sp["angle"], sp["length"], sp["start"]
        if length - start < 6:
            continue
        dd = np.arange(start + 2, max(start + 6, start + (length - start) // 3), 1.0)
        offs = np.arange(-15, 16)
        prof = np.array([strip(a, o, dd).mean() for o in offs])
        pb = np.median(np.r_[prof[:5], prof[-5:]])
        pk = prof[13:18].max() - pb
        over = np.abs(offs[prof - pb > max(0.2 * pk, 0.3 * sig)])
        over = over[over <= 12]
        hw = int(over.max()) if len(over) else 1
        spikes.append(dict(angle=float(a), length=int(length), hw=int(max(hw, 1))))
    s["spikes"] = spikes
    return s


def star_mask(shape, s, pad=3):
    """(dur, doux) : dur = coeur + halo fort (excès > 2 bruits) + aigrettes (le détail y est faux, à recréer) ;
    doux = halo faible jusqu'au rayon mesuré (seule la basse fréquence y est fausse)."""
    hard = np.zeros(shape, np.uint8); soft = np.zeros(shape, np.uint8)
    Rh = int(s["r_hard"] + pad); Rs = int(max(s["r_halo"], s["r_hard"]) + pad)
    s["R"] = Rs
    cx, cy = int(round(s["x"])), int(round(s["y"]))
    cv2.circle(hard, (cx, cy), Rh, 1, -1)
    cv2.circle(soft, (cx, cy), Rs, 1, -1)
    for sp in s["spikes"]:
        if sp["length"] <= Rh - 2 or sp["length"] - sp["hw"] < 6:
            continue
        t = np.deg2rad(sp["angle"])
        ux, uy = np.cos(t), np.sin(t); nx, ny = -uy, ux
        L = sp["length"] + pad + 4
        w0 = sp["hw"] + pad; w1 = max(2, sp["hw"] * 0.6) + pad
        pts = np.array([[s["x"] + nx * w0, s["y"] + ny * w0], [s["x"] - nx * w0, s["y"] - ny * w0],
                        [s["x"] + ux * L - nx * w1, s["y"] + uy * L - ny * w1],
                        [s["x"] + ux * L + nx * w1, s["y"] + uy * L + ny * w1]])
        cv2.fillConvexPoly(hard, np.round(pts).astype(np.int32), 1)
    hard = hard.astype(bool)
    return hard, soft.astype(bool) & ~hard
