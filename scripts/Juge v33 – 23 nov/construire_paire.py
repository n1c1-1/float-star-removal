# ======================================================================
# construire_paire.py
# Rôle     : Une paire d'entraînement (jeu 1) à partir d'une tuile traitée par la chaîne classique.
# Étape    : 2. Juge des zones : poids 0 où la chaîne échoue (v33) (voir scripts/LISEZMOI.md)
# Entrées  : entrée, cible de la chaîne, zones
# Sorties  : carte de poids
# Exemple  : usage : construire_paire.py <dossier méthode> <image> <carré> [<x0 découpe> <y0 découpe>]
# ======================================================================
"""Une paire d'entraînement (jeu 1) à partir d'une tuile traitée par la chaîne classique.

usage : construire_paire.py <dossier méthode> <image> <carré> [<x0 découpe> <y0 découpe>]
  - entrée  : la tuile 512 de l'image débruitée (découpe 768, tuile = découpe + 128) ;
  - cible   : sorties\\<carré>\\B\\t3\\sans_etoiles_tuile.npy ;
  - classes : sorties\\<carré>\\B\\t3\\carte_remplissage_tuile.npy (0 intact, 1 halo soustrait, 3 moyenne corrigée sur place :
              vrais pixels ; 2 recréé autour des brillantes ; 4-8 défauts et restes rebouchés par l'étape A et les passes) ;
  - poids   : RECETTE_ET_LECONS.md section 8 : vrais pixels 1 ; « inévitable » (classe 2 sur le cœur écrêté d'une étoile,
              dilaté de 6 px) 1 si la zone est bonne, 0 sinon ; « réparation » (le reste des classes 2 et 4-8) 0,5 si bonne, 0 sinon.
  - zone bonne : grain (écart robuste du détail fin) entre 0,85 et 1,15 de celui de l'anneau 4-16 px autour (zones de plus de
              30 px), marche au bord < 1,5 bruit local, moins de 2 % de pixels au-delà de 5 écarts du filtre à l'échelle de
              l'étoile (anneaux, rayons, taches).
Sortie : <DONNEES>\\Paires étoiles jeu 1 – 22 nov\\<image>_<carré>.npz (input, target, weight, classes ; float32 / uint8)
         + <image>_<carré>.jpg (entrée | cible | poids, étirement arcsinh 60) + ligne dans paires.jsonl.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Paires étoiles v33 – 23 nov")
ROSETTE = {"decoupe6": (1530, 790), "coeur_est": (2066, 1266), "filaments": (1466, 1591), "lisse": (2016, 1666)}


def mad(v):
    v = np.asarray(v, np.float64)
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) + 1e-12


def juger_zone(L, comp, autres, fw):
    """Rend (bonne, mesures) pour une zone recréée ``comp`` de la luminance L de la cible."""
    ring = ndi.binary_dilation(comp, iterations=16) & ~ndi.binary_dilation(comp, iterations=4) & ~autres
    if ring.sum() < 40:
        ring = ndi.binary_dilation(comp, iterations=16) & ~ndi.binary_dilation(comp, iterations=4)
    det = L - cv2.GaussianBlur(L, (0, 0), 2.0)
    m = {}
    if comp.sum() > 30:
        m["grain"] = mad(det[comp]) / mad(det[ring])
    inner = comp & ~ndi.binary_erosion(comp, iterations=2)
    outer = ndi.binary_dilation(comp, iterations=2) & ~comp & ~autres
    sm = cv2.GaussianBlur(L, (0, 0), 1.0)
    bruit = mad((L - cv2.GaussianBlur(L, (0, 0), 4.0))[ring])
    if inner.sum() > 5 and outer.sum() > 5:
        m["marche"] = abs(float(np.median(sm[inner]) - np.median(sm[outer]))) / bruit
    D = cv2.GaussianBlur(L, (0, 0), 0.5 * fw) - cv2.GaussianBlur(L, (0, 0), 2.0 * fw)
    m["motifs"] = float((np.abs(D[comp]) > 5 * mad(D[ring])).mean())
    bonne = (0.85 <= m.get("grain", 1.0) <= 1.15) and m.get("marche", 0.0) < 1.5 and m["motifs"] < 0.02
    if comp.sum() >= 200:
        # 22 nov soir (M1, NGC 6543, IC 443) : anneaux concentriques, disques sombres et cœurs restés passaient les trois
        # tests ci-dessus ; deux tests de plus sur les grandes zones
        m["anneau"] = score_anneaux(L, comp)
        m["niveau"] = score_niveau(L, comp, ring)
        bonne = bonne and m["anneau"] <= 0.6 and abs(m["niveau"]) <= 3.0
    return bonne, {k: round(v, 3) for k, v in m.items()}


def restes_etoiles(T, cl, W, fw):
    """22 nov soir (NGC 6888) : une étoile pas retirée (cœur troué dans l'entrée, hors catalogue…) laisse une tache dans
    la cible, et le bord de la tache (classe 1 ou 3, vrais pixels) gardait le poids 1. Tache = détail de la cible à
    l'échelle de l'étoile (flou 0,5 FWHM moins médiane 4 FWHM) au-dessus de 5 dispersions, d'au moins 9 px, posée sur
    un endroit que la chaîne a touché (classe != 0 sous la tache) : un nœud de nébuleuse intact (classe 0) est gardé.
    La tache dilatée de 1 FWHM passe à 0. Rend le nombre de taches."""
    L = T.mean(0)
    k = max(3, int(round(4 * fw)) | 1)
    det = cv2.GaussianBlur(L, (0, 0), max(0.5 * fw, 0.7)) - ndi.median_filter(L, size=k)
    s = mad(det)
    lab, n = ndi.label(det > 5 * s)
    if not n:
        return 0
    idx = np.arange(1, n + 1)
    taille = ndi.sum(np.ones_like(L), lab, idx)
    touche = ndi.mean((cl != 0).astype(np.float32), lab, idx)
    garde = idx[(taille >= 9) & (touche > 0.5)]
    if not len(garde):
        return 0
    m = ndi.binary_dilation(np.isin(lab, garde), iterations=max(1, int(round(fw))))
    W[m] = 0.0
    return int(len(garde))


def ilots_dans_rejet(W, inv):
    """22 nov soir (M1) : une zone refaite gardée dont l'entourage (4-16 px) est à plus de moitié à poids 0 (halo mal
    soustrait autour) n'a plus de référence juste : son niveau suit un halo rejeté. Elle passe à 0. Rend le nombre."""
    lab, _ = ndi.label(inv)
    n = 0
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        comp = lab == k
        if W[comp].max() == 0:
            continue
        ring = ndi.binary_dilation(comp, iterations=16) & ~ndi.binary_dilation(comp, iterations=4)
        if float((W[ring] == 0).mean()) > 0.5:
            W[comp] = 0.0
            n += 1
    return n


def score_anneaux(L, comp):
    """Anneau = détail fin (0,5 px moins 1,5 px) du même signe tout le tour : |moyenne des signes| par anneau de 1 px
    (pas 0,5 px, à partir de 3 px du centre, au moins 30 px et 3/4 du tour) ; rend le plus fort. Bruit : ~0,1-0,4 ;
    anneaux et disques sombres : 0,6-0,95 (planche du 22 nov soir)."""
    cy, cx = ndi.center_of_mass(comp)
    yy, xx = np.nonzero(comp)
    r = np.hypot(xx - cx, yy - cy)
    d = (cv2.GaussianBlur(L, (0, 0), 0.5) - cv2.GaussianBlur(L, (0, 0), 1.5))[yy, xx]
    best = 0.0
    for a in np.arange(3, r.max(), 0.5):
        m = (r >= a) & (r < a + 1.0)
        if m.sum() < 30:
            continue
        ang = np.arctan2(yy[m] - cy, xx[m] - cx)
        if len(np.unique((ang * 8 / np.pi).astype(int))) < 12:
            continue
        best = max(best, abs(float(np.mean(np.sign(d[m])))))
    return best


def score_niveau(L, comp, ring):
    """Écart de niveau de la zone au plan de son entourage (lissé 2 px), en dispersion de l'entourage : cœur resté
    (positif) ou disque sombre (négatif) ; une nébuleuse lisse sous la zone suit le plan."""
    sm = cv2.GaussianBlur(L, (0, 0), 2.0)
    ry, rx = np.nonzero(ring)
    A = np.c_[np.ones(len(rx)), rx, ry]
    coef, *_ = np.linalg.lstsq(A, sm[ry, rx], rcond=None)
    res_r = sm[ry, rx] - __import__('operator').matmul(A, coef)
    zy, zx = np.nonzero(comp)
    res_z = sm[zy, zx] - __import__('operator').matmul(np.c_[np.ones(len(zx)), zx, zy], coef)
    return float(np.median(res_z) - np.median(res_r)) / mad(res_r)


def main():
    meth, image, carre = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    sys.path.insert(0, str(meth / "Piste1"))
    import images
    if len(sys.argv) > 5:
        cx0, cy0 = int(sys.argv[4]), int(sys.argv[5])
    elif image == "rosette":
        cx0, cy0 = ROSETTE[carre]
    else:
        cx0, cy0 = json.loads((meth / f"decoupes_{image}.json").read_text(encoding="utf-8"))[carre][:2]
    x0, y0 = cx0 + 128, cy0 + 128
    full = images.load(image)
    I = np.array(full[:, y0:y0 + 512, x0:x0 + 512], np.float32)
    S = meth / "sorties" / carre / "B" / "t3"
    T = np.load(S / "sans_etoiles_tuile.npy").astype(np.float32)
    cl = np.load(S / "carte_remplissage_tuile.npy").astype(np.uint8)
    try:
        fw = float(json.loads(next((meth / "Piste1" / "sorties" / image).glob(f"*{carre}*/resume.json")).read_text(encoding="utf-8"))["fwhm"])
    except Exception:
        fw = 3.0
    # cœurs écrêtés : plateau à >= 98 % du pic local (9x9) et au-dessus de 30 % du max de l'image, dilaté de 6 px
    sat = np.zeros(cl.shape, bool)
    for c in range(I.shape[0]):
        mx = ndi.maximum_filter(I[c], 9)
        sat |= (I[c] >= 0.98 * mx) & (I[c] >= 0.3 * float(full[c].max()))
    lab_s, _ = ndi.label(sat)
    sizes = ndi.sum(sat, lab_s, np.arange(1, lab_s.max() + 1)) if lab_s.max() else np.zeros(0)
    sat = np.isin(lab_s, 1 + np.nonzero(sizes >= 9)[0])
    coeur = ndi.binary_dilation(sat, iterations=6)
    W = np.ones(cl.shape, np.float32)
    inv = (cl == 2) | (cl >= 4)
    L = T.mean(0)
    lab, n = ndi.label(inv)
    zones = []
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        comp = lab == k
        inevitable = bool((comp & coeur).sum() > 0.5 * comp.sum())
        bonne, m = juger_zone(L, comp, inv & ~comp, fw)
        W[comp] = (1.0 if inevitable else 0.5) if bonne else 0.0
        zones.append(dict(px=int(comp.sum()), inevitable=inevitable, bonne=bonne, **m))
    # zones de halo soustrait (classe 1, vrais pixels) : un halo trop ou pas assez soustrait laisse un anneau sombre ou
    # clair ; il serait appris avec un poids 1. Test de niveau : par anneau de 4 px autour de chaque brillante (centre
    # = centre de la composante de classe 1), médiane lissée comparée à l'anneau extérieur (hors zones) ; écart > 2 bruits
    # locaux dans un anneau -> toute la zone de halo à poids 0.
    halo = cl == 1
    lab1, n1 = ndi.label(halo)
    Ls = cv2.GaussianBlur(L, (0, 0), 2.0)
    yy, xx = np.mgrid[0:L.shape[0], 0:L.shape[1]]
    for k, sl in enumerate(ndi.find_objects(lab1), 1):
        comp = lab1 == k
        if comp.sum() < 200:
            continue
        cy, cx = ndi.center_of_mass(comp)
        r = np.hypot(xx - cx, yy - cy)
        rmax = float(r[comp].max())
        out_ring = (r > rmax) & (r < rmax + 16) & ~inv & ~halo
        if out_ring.sum() < 50:
            continue
        ref, bruit = float(np.median(Ls[out_ring])), mad((L - cv2.GaussianBlur(L, (0, 0), 4.0))[out_ring])
        pires = 0.0
        for a in np.arange(0, rmax, 4):
            an = comp & (r >= a) & (r < a + 4) & ~inv
            if an.sum() > 30:
                pires = max(pires, abs(float(np.median(Ls[an])) - ref) / bruit)
        if pires > 2.0:
            W[comp] = 0.0
        zones.append(dict(px=int(comp.sum()), inevitable=False, halo=True, bonne=bool(pires <= 2.0), ecart_anneau=round(pires, 2)))
    n_restes = restes_etoiles(T, cl, W, fw)
    if n_restes:
        zones.append(dict(px=0, inevitable=False, bonne=False, restes_etoiles=n_restes))
    n_ilots = ilots_dans_rejet(W, inv)
    if n_ilots:
        zones.append(dict(px=0, inevitable=False, bonne=False, ilots_mis_a_0=n_ilots))
    # juge v33 (23 nov) : deux regles locales ajoutees, rien d'autre ne change (regles_v33.py)
    import regles_v33
    r33 = regles_v33.appliquer(I, T, W, cl, image, x0, y0, fw)
    if r33["taches_plus_claires"] or r33["etoiles_restees"]:
        zones.append(dict(px=0, inevitable=False, bonne=False, **r33))
    OUT.mkdir(parents=True, exist_ok=True)
    nom = f"{image}_{carre}"
    np.savez_compressed(OUT / f"{nom}.npz", input=I, target=T, weight=W, classes=cl)
    bk = [float(np.median(full[c, ::9, ::9])) for c in range(3)]
    top = [float(np.percentile(full[c, ::9, ::9], 99.8)) for c in range(3)]

    def st(a):
        o = [np.arcsinh(60 * np.clip((a[c] - bk[c]) / (top[c] - bk[c]), 0, None)) / np.arcsinh(60) for c in range(3)]
        return (np.clip(np.stack(o, -1), 0, 1) * 255 + .5).astype(np.uint8)
    wv = np.stack([255 * (W == 0), 255 * (W == 0.5), 255 * (W == 1.0)], -1).astype(np.uint8)
    wv[(W == 1.0) & (cl == 0)] = 40
    sep = np.full((512, 6, 3), 255, np.uint8)
    cv2.imencode(".jpg", np.concatenate([st(I), sep, st(T), sep, wv], 1)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(OUT / f"{nom}.jpg"))
    resume = dict(paire=nom, methode=meth.name, image=image, carre=carre, tuile=[x0, y0],
                  poids_1=round(float((W == 1).mean()), 4), poids_05=round(float((W == 0.5).mean()), 4),
                  poids_0=round(float((W == 0).mean()), 4), zones=len(zones),
                  zones_mauvaises=sum(1 for z in zones if not z["bonne"]),
                  inevitables=sum(1 for z in zones if z["inevitable"]))
    with open(OUT / "paires.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(resume, ensure_ascii=False) + "\n")
    (OUT / f"{nom}_zones.json").write_text(json.dumps(zones, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(resume, ensure_ascii=False))


if __name__ == "__main__":
    main()
