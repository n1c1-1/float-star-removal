# ======================================================================
# aveugle.py
# Rôle     : Test de réalisme à l'aveugle du générateur v2 : par image, jusqu'à 10 vraies étoiles par classe d'éclat (moyenne G 12-13,5 ; brillante G 9-11 non écrêtée ; saturée = plateau >= 9 px ; très brillante G < 6,5) et autant de fausses d
# Étape    : 4. Étoiles synthétiques réalistes (22 nov) (voir scripts/LISEZMOI.md)
# Entrées  : fonds sans étoiles (cibles de paires), banques de formes d'étoiles
# Sorties  : paires (entrée, cible), planches, test à l'aveugle
# Exemple  : Usage : python aveugle.py [image ...]  -> aveugle\<image>_<classe>_<k>.png ; la clé va dans le dossier passé par --cle.
# ======================================================================
"""Test de réalisme à l'aveugle du générateur v2 : par image, jusqu'à 10 vraies étoiles par classe d'éclat (moyenne
G 12-13,5 ; brillante G 9-11 non écrêtée ; saturée = plateau >= 9 px ; très brillante G < 6,5) et autant de fausses de
même éclat, rendues par etoiles2 (paramètres d'image mesurés, images_mesurees ; couleur par BP-RP Gaia et la loi de
couleur de l'image ajustée sur ses étoiles G 11-14) et AJOUTÉES sur l'image réelle à un endroit sans étoile du catalogue
de cet éclat (le voisinage reste réel, comme pour les vraies).
Vignettes x4, arcsinh 60 | étirement fort, numérotées, mélangées ; 10 par planche, une classe par planche.
Usage : python aveugle.py [image ...]  -> aveugle\\<image>_<classe>_<k>.png ; la clé va dans le dossier passé par --cle.
"""
from __future__ import annotations

import os as _os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    _os.environ[_k] = "2"                                  # au plus 2 fils (consigne du 22 nov après la saturation)

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_etoiles as A                                            # noqa: E402
import etoiles2 as E                                                 # noqa: E402
from images_mesurees import IMP                                      # noqa: E402

CLASSES = {"moyenne": 24, "brillante": 32, "saturee": 48, "tres_brillante": 64}     # demi-côté de la vignette
DEEP = {"rosette": A.V21, "m8_sho": A.V21, "m57": A.V22, "horsehead": A.V23}
OUTD = HERE / "aveugle"
TOUR = [1]
MAXI = {}                                                # par classe, au plus (tour 3)
OPT = dict(sans=[], classes=None, suffixe="")            # tour 8 : essais en retirant un changement du tour 7


def couleur_image(a, x, y, g, bp, fw, sig, imax):
    """Loi de couleur de l'image : parts R/V/B d'étoiles G 11-14 isolées (ouverture 3 FWHM), ajustées en BP-RP (degré 2)."""
    tree = cKDTree(np.c_[x, y])
    h = int(math.ceil(3 * fw)) + 3
    rows = []
    for i in np.flatnonzero((g >= 11) & (g <= 14) & np.isfinite(bp)):
        if not (h < x[i] < a.shape[2] - h and h < y[i] < a.shape[1] - h):
            continue
        if any(j != i and g[j] < g[i] + 4 for j in tree.query_ball_point([x[i], y[i]], 6 * fw)):
            continue
        s = A.stamp(a, x[i], y[i], h)
        pr = A.ring_median(s)
        bg = np.nanmedian(pr[:, int(2.5 * fw):int(3 * fw) + 2], axis=1)
        r, *_ = A.rings(h)
        f = np.array([np.sum(np.where(r <= 3 * fw, s[c] - bg[c], 0)) for c in range(3)])
        if np.any(f <= 0) or np.any(np.nanmax(s, axis=(1, 2)) > 0.3 * imax):
            continue
        rows.append([bp[i], *(f / f.sum())])
        if len(rows) >= 600:
            break
    R = np.array(rows)
    co = [np.polyfit(R[:, 0], R[:, 1 + c], 2) for c in range(3)]

    def fn(b):
        b = np.clip(np.atleast_1d(b), np.percentile(R[:, 0], 1), np.percentile(R[:, 0], 99))
        p = np.clip(np.stack([np.polyval(co[c], b) for c in range(3)]), 0.02, None)
        return (p / p.sum(0, keepdims=True)).astype(np.float32)
    return fn, len(R)


def run(name, cle_dir, rng):
    cfg = A.IMAGES[name]
    d = A.SR / cfg["dir"] / "Piste1" / "sorties" / name
    a = A.IM.load(name)
    aud = json.loads((HERE / "audit_mesures.json").read_text(encoding="utf-8"))[name]
    fw, zp, kap = aud["fwhm"], aud["zp"], aud["stats_image"]["k_ouverture_sur_catalogue"]
    sig, imax = np.array(aud["bruit"]), np.array(aud["max_image"])
    bk = float(np.median(a[1, ::7, ::7])); wt = float(np.percentile(a[1, ::5, ::5], 99.8))
    cat = np.load(d / "astrometrie.npz", allow_pickle=True)["cat"].item()
    x, y, g = (np.asarray(cat[k], float) for k in ("x", "y", "g"))
    for (ex, ey, eg, en) in cfg.get("extra", []):
        x, y, g = np.r_[x, ex], np.r_[y, ey], np.r_[g, eg]
    z = np.load(A.SR / DEEP[name] / "Catalogue Gaia profond – 20 nov" / f"gaia_dr3_profond_{name}.npz")
    dt = cKDTree(np.c_[z["x_pix"], z["y_pix"]])
    dd, ii = dt.query(np.c_[x, y])
    bp = np.where(dd < 2.0, (z["BPmag"] - z["RPmag"])[ii], np.nan)
    Hh, W = a.shape[1:]
    fn, ncol = couleur_image(a, x, y, g, bp, fw, sig, imax)
    imp = dict(IMP[name], sat=imax)
    P = E.tirer_image(np.random.default_rng([77, len(name)]), 1201, imp, E.ETAPES)
    P["couleur_fn"] = fn
    for f_ in OPT["sans"]:
        P["t7_" + f_] = False
    tree = cKDTree(np.c_[x, y])
    valid = (x > 0) & (y > 0) & (x < W) & (y < Hh)

    def plateau_px(i):
        s = A.stamp(a, x[i], y[i], 60)
        return A.plateau(s, 40 if g[i] < 4.5 else 15, 60)[1][1]

    sigL = float(np.sqrt((sig ** 2).sum()))

    def present(i):
        """Tour 5 : une vraie doit être VUE là où le catalogue la met (tour 4 : une vignette de M8 sans étoile, une
        G 13,2 du catalogue absente de l'image) : pic net au centre et flux d'ouverture cohérent avec son G."""
        h = int(3 * fw) + 4
        st_ = A.stamp(a, x[i], y[i], h)
        if np.isnan(st_).any():
            return False
        L = st_.sum(0)
        r_, *_ = A.rings(h)
        bgl = float(np.median(L[r_ > h - 2]))
        if float(L[r_ <= 2].max()) - bgl < 20 * sigL:
            return False
        ap = float(np.sum(np.where(r_ <= 3 * fw, L - bgl, 0)))
        return 0.3 < ap / (kap * 10 ** (-0.4 * (g[i] - zp))) < 3.0 or g[i] < 9.5

    def dominant(i, R):
        return all(j == i or g[j] > g[i] + 1.5 for j in tree.query_ball_point([x[i], y[i]], 0.6 * R))

    # tour 3 : les vraies de toutes les classes d'abord (la banque de l'image est refaite SANS elles), au plus
    # MAXI[classe] par classe ; très brillantes G < 7, saturées G 7-10 à plateau, prises sur toute l'image
    allpicks = {}
    CL = ({"brillante": CLASSES["brillante"], "tres_brillante": CLASSES["tres_brillante"]} if TOUR[0] >= 7 else
          {"tres_brillante": CLASSES["tres_brillante"]} if TOUR[0] >= 6 else CLASSES)
    if OPT["classes"]:
        CL = {c: CLASSES[c] for c in OPT["classes"]}
    for cl, hc in CL.items():
        m = 2 * hc + 20
        inside = valid & (x > m) & (y > m) & (x < W - m) & (y < Hh - m)
        if cl == "moyenne":
            cand = np.flatnonzero(inside & (g >= 12) & (g <= 13.5))
        elif cl == "brillante":
            cand = np.flatnonzero(inside & (g >= 9) & (g <= 11))
        elif cl == "saturee":
            cand = np.flatnonzero(inside & (g >= (7 if TOUR[0] >= 3 else 6)) & (g < 10))
        else:
            cand = np.flatnonzero(inside & (g < (7 if TOUR[0] >= 3 else 6.5)))
        rng.shuffle(cand)
        picks = []
        for i in cand:
            if not dominant(i, hc):
                continue
            if TOUR[0] >= 5 and not present(i):
                continue
            pl = plateau_px(i) if cl in ("brillante", "saturee", "tres_brillante") else 0
            if cl == "brillante" and pl >= 9:
                continue
            if cl == "saturee" and pl < 9:
                continue
            picks.append(i)
            if len(picks) >= MAXI.get(cl, 10):
                break
        allpicks[cl] = picks
    if TOUR[0] >= 3:
        import banque_formes as BF
        exc = [(x[i], y[i]) for v in allpicks.values() for i in v]
        if TOUR[0] >= 4:                                     # tour 4 : étoiles individuelles nettoyées
            import banque_individuelle as BI
            bq = BI.construire(name, a=a, exclure=exc, verbose=False)
            if bq is not None and len(bq["angles"]) < len(IMP[name]["angles"]):
                bq["angles"] = list(IMP[name]["angles"])      # aigrettes trop faibles pour la détection (Rosette)
        else:
            bq = BF.construire(name, exclure=exc, a=a, verbose=False)
        if bq is not None and TOUR[0] >= 6:
            # tour 6 : d'autres images de la banque de même configuration (même nombre d'aigrettes), ramenées à la
            # largeur de l'image cible et tournées vers ses angles (mesurés) ; l'image elle-même pèse 0,3
            import banque_individuelle as BI
            ang_t = list(IMP[name]["angles"])
            nsp = len(ang_t)
            autres = []
            for f in sorted(BI.OUTB.glob("banque2_*.npz")):
                nm = f.stem.replace("banque2_", "")
                if nm == name:
                    continue
                o = BI.charger(nm)
                if o is None or len(o.get("angles", [])) != nsp or len(o["G"]) < 5:
                    continue
                if nsp:
                    per = 360.0 / nsp
                    d = (min(a_ % per for a_ in ang_t) - min(a_ % per for a_ in o["angles"])) % per
                    rot = d - per if d > per / 2 else d
                else:
                    rot = 0.0
                ech = float(bq["fwhm"] / o["fwhm"])
                if TOUR[0] >= 8 and not (0.7 <= ech <= 1.4):
                    continue                                  # tour 8 : formes trop réduites ou grossies : non
                autres.append(dict(b=o, rot=float(rot), echelle=float(np.clip(ech, 0.4, 2.5)),
                                   poids=float(len(o["G"]))))
            tot = sum(e["poids"] for e in autres)
            for e in autres:
                e["poids"] = 0.7 * e["poids"] / tot
            P["banque"] = [dict(b=bq, rot=0.0, echelle=1.0, poids=1.0 if not autres else 0.3)] + autres
            P["banque_rotation"], P["banque_echelle"], P["banque_aureole"] = 0.0, 1.0, (0.8, 1.25)
            print(name, "banques :", [(e["b"]["image"], round(e["rot"], 1), round(e["echelle"], 2), round(e["poids"], 2))
                                      for e in P["banque"]], flush=True)
        elif bq is not None:
            P["banque"] = bq
            P["banque_rotation"], P["banque_echelle"], P["banque_aureole"] = 0.0, 1.0, (0.8, 1.25)
            print(name, "banque sans les vraies du test :", len(bq["G"]), "formes", flush=True) if TOUR[0] < 6 else None
    key = []
    for cl, hc in CL.items():
        m = 2 * hc + 20
        picks = allpicks[cl]
        items = []
        for i in picks:
            items.append(dict(vraie=True, G=float(g[i]), xy=[float(x[i]), float(y[i])],
                              crop=A.stamp(a, x[i], y[i], hc)))
            # la fausse : même éclat ; BP-RP d'une étoile du champ d'éclat voisin ; endroit sans étoile de cet éclat
            Fk = kap * 10 ** (-0.4 * (g[i] - zp))
            near = np.flatnonzero(np.isfinite(bp) & (np.abs(g - g[i]) < 1.0))
            bpi = float(bp[rng.choice(near)]) if near.size else 1.2
            for _ in range(4000):
                fx, fy = rng.uniform(m, W - m), rng.uniform(m, Hh - m)
                nb = tree.query_ball_point([fx, fy], 0.6 * hc)
                # tour 2 : plus d'exigence « aucune étoile à 3 FWHM » (elle isolait les fausses plus que les vraies,
                # « coeur isolé, sans voisines » selon le juge) ; même règle que pour les vraies
                if any(g[j] < g[i] + 1.5 for j in nb):
                    continue
                loc = A.stamp(a, fx, fy, int(3 * fw))
                if np.isnan(loc).any() or np.any(loc == 0):
                    continue
                L = np.nansum(loc, 0)
                if L.max() - np.median(L) > 15 * np.sqrt((sig ** 2).sum()):
                    continue
                break
            R = 600 if g[i] < 7 else 200
            xi, yi = int(round(fx)), int(round(fy))
            reg = A.stamp(a, xi, yi, R)
            reg = np.nan_to_num(reg)
            champ = dict(x=[R + fx - xi], y=[R + fy - yi], flux=[Fk], bp_rp=[bpi])
            o = E.ajouter_etoiles(reg, [int(rng.integers(1 << 30))], champ=champ, P=P)
            out = o["entree"]
            items.append(dict(vraie=False, G=float(g[i]), xy=[float(fx), float(fy)], bp_rp=bpi,
                              crop=out[:, R - hc:R + hc + 1, R - hc:R + hc + 1]))
        order = rng.permutation(len(items))
        items = [items[j] for j in order]
        for sh in range(0, len(items), 10):
            part = items[sh:sh + 10]
            nom = f"{name}_{cl}_{sh // 10 + 1}"
            panels = []
            for k, it in enumerate(part):
                c = it["crop"]
                p1 = (A.stretch60(c, bk, wt) * 255).astype(np.uint8)
                p2 = (A.stretch_strong(c, sig, imax) * 255).astype(np.uint8)
                up = lambda q: cv2.resize(q, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
                pan = np.concatenate([up(p1), np.full((p1.shape[0] * 4, 6, 3), 255, np.uint8), up(p2)], 1)
                lab = np.full((28, pan.shape[1], 3), 255, np.uint8)
                cv2.putText(lab, f"{k + 1}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                panels.append(np.concatenate([lab, pan], 0))
                key.append(dict(planche=nom, numero=k + 1, vraie=it["vraie"], G=round(it["G"], 2),
                                xy=[round(v, 1) for v in it["xy"]], classe=cl))
            while len(panels) % 2:
                panels.append(np.full_like(panels[0], 255))
            gap = np.full((panels[0].shape[0], 16, 3), 255, np.uint8)
            rows = [np.concatenate([panels[r], gap, panels[r + 1]], 1) for r in range(0, len(panels), 2)]
            hgap = np.full((16, rows[0].shape[1], 3), 255, np.uint8)
            sheet = np.concatenate(sum([[r_, hgap] for r_ in rows], [])[:-1], 0)
            OUTD.mkdir(exist_ok=True, parents=True)
            ok, buf = cv2.imencode(".png", cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)); assert ok
            buf.tofile(str(OUTD / f"{nom}.png"))
        print(name, cl, len(picks), "vraies", flush=True)
    return key, ncol


def main():
    args = [s for s in sys.argv[1:] if not s.startswith("--")]
    global OUTD
    cle_dir = Path(next(s.split("=", 1)[1] for s in sys.argv[1:] if s.startswith("--cle=")))
    TOUR[0] = int(next((s.split("=", 1)[1] for s in sys.argv[1:] if s.startswith("--tour=")), 1))
    for s_ in sys.argv[1:]:
        if s_.startswith("--sans="):
            OPT["sans"] = [v for v in s_.split("=", 1)[1].split(",") if v]
        if s_.startswith("--classes="):
            OPT["classes"] = s_.split("=", 1)[1].split(",")
        if s_.startswith("--suffixe="):
            OPT["suffixe"] = s_.split("=", 1)[1]
    if TOUR[0] > 1:
        OUTD = HERE / f"aveugle_tour{TOUR[0]}{OPT['suffixe']}"
    if TOUR[0] >= 3:
        MAXI.update(moyenne=6, brillante=6, saturee=12, tres_brillante=12)
    if TOUR[0] >= 4:
        MAXI.update(saturee=15, tres_brillante=15)
    names = args or list(A.IMAGES)
    allk = {}
    for n in names:
        rng = np.random.default_rng([2026, 11, 22, TOUR[0], len(n)])
        k, ncol = run(n, cle_dir, rng)
        allk[n] = dict(cle=k, etoiles_loi_couleur=ncol)
    cle_dir.mkdir(parents=True, exist_ok=True)
    (cle_dir / (f"cle_aveugle_tour{TOUR[0]}{OPT['suffixe']}.json" if TOUR[0] > 1 else "cle_aveugle.json")).write_text(json.dumps(allk, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
