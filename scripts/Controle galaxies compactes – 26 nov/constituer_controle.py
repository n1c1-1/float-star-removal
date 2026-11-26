# ======================================================================
# constituer_controle.py
# Rôle     : Constitue le jeu de controle : galaxies retenues A L'OEIL (revue\<image>.jpg, numero = rang dans revue\<image>.json), etoile Gaia de meme brillance pour chacune, controle.json et planche_controle.jpg (vignettes 64x64, etirement as
# Étape    : 13. Jeu de contrôle des petites galaxies (26 nov) (voir scripts/LISEZMOI.md)
# Entrées  : images web et extraits Gaia (non publiés)
# Sorties  : controle.json, mesures par réseau
# Exemple  : python constituer_controle.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
r"""Constitue le jeu de controle : galaxies retenues A L'OEIL (revue\<image>.jpg, numero = rang dans revue\<image>.json), etoile Gaia de meme brillance
pour chacune, controle.json et planche_controle.jpg (vignettes 64x64, etirement asinh commun, galaxie a gauche, etoile a droite).

    constituer_controle.py
"""
import os, sys, json, glob, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np, cv2
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).resolve().parent))
import chercher_galaxies as CG

APOD, GAIA, OUT = CG.APOD, CG.GAIA, CG.OUT
#: choix a l'oeil sur revue\<image>.jpg (rang dans revue\<image>.json)
CHOIX = {"apod_017": [0, 3, 5, 9], "apod_018": [3, 6, 9, 15], "apod_023": [0, 12, 15, 22], "apod_028": [8, 9, 15], "apod_071": [2, 4, 7]}
AUTEURS = {"apod_017": "Galaxies in the Furnace (Fornax A) - Simone Curzi et equipe ShaRA, APOD 2025-12-04",
           "apod_018": "Globular Cluster M15 Deep Field - [auteur], APOD 2025-11-26",
           "apod_023": "NGC 4565: Galaxy on Edge - Jose Rodrigues, APOD 2025-09-04",
           "apod_028": "Curly Spiral Galaxy M63 - [auteur], APOD 2025-05-22",
           "apod_071": "The Hydra Cluster of Galaxies - Marco Lorenzi, Angus Lau, Tommy Tse, APOD 2023-02-16"}


def vignette(img, x, y, h=32, k=3):
    C, H, W = img.shape
    x0, y0 = int(round(x)) - h, int(round(y)) - h
    v = np.zeros((C, 2 * h, 2 * h), np.float32)
    s = img[:, max(y0, 0):y0 + 2 * h, max(x0, 0):x0 + 2 * h]
    v[:, :s.shape[1], :s.shape[2]] = s
    g = np.clip(np.arcsinh(np.clip(v, 0, None) / 6) / np.arcsinh(500 / 6), 0, 1)
    im = (g.transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)
    return cv2.resize(im, (2 * h * k, 2 * h * k), interpolation=cv2.INTER_CUBIC)


def main():
    gal, tiles = [], CG.tuiles_apod()
    planche = []
    for nom, rangs in CHOIX.items():
        sel = json.loads((OUT / "revue" / f"{nom}.json").read_text(encoding="utf-8"))
        img = np.array(np.load(APOD / "entrees" / nom / "input.npy", mmap_mode="r"), np.float32)
        C, H, W = img.shape
        L = img.mean(0)
        z = np.load(GAIA / f"gaia_dr3_profond_{nom}.npz")
        gx, gy, gg = z["x_pix"].astype(float), z["y_pix"].astype(float), z["Gmag"].astype(float)
        ok = np.isfinite(gg)
        gx, gy, gg = gx[ok], gy[ok], gg[ok]
        gt = cKDTree(np.c_[gx, gy])
        sig = CG.bruit_local(L)
        sat = float(np.percentile(L, 99.99))
        TUI = tiles.get(nom, [])
        # etoiles Gaia candidates (isolees, non saturees, loin des bords et des tuiles) avec leur flux dans 3 px
        rng = np.random.default_rng(1)
        idx = np.flatnonzero((gg > 11) & (gg < 20.5) & (gx > CG.MARGE) & (gx < W - CG.MARGE) & (gy > CG.MARGE) & (gy < H - CG.MARGE))
        idx = idx[rng.permutation(len(idx))[:6000]]
        etoiles = []
        for i in idx:
            d, _ = gt.query([gx[i], gy[i]], k=2)
            if d[1] < 14:
                continue
            dt = CG.dist_tuile(TUI, gx[i], gy[i])
            if dt is not None and dt < CG.MARGE_TUILE:
                continue
            p = CG.profil(L, gx[i], gy[i])
            if p is None or p["crete"] < 15 * sig[int(gy[i]), int(gx[i])] or p["crete"] > 0.9 * sat or np.hypot(p["cx"] - gx[i], p["cy"] - gy[i]) > 2.5:
                continue
            if p["f3"] > 0 and p["fwhm"] < 2.2 * np.median([r["fwhm_etoiles"] for r in sel]):
                etoiles.append((p["f3"], p["cx"], p["cy"], gg[i], p["fwhm"], p["part"] if "part" in p else p["f3"] / max(p["f12"], 1e-9)))
        etoiles = np.array(etoiles)
        for k in rangs:
            r = sel[k]
            f3 = r["flux3"]
            j = int(np.argmin(np.abs(np.log(etoiles[:, 0] / f3))))
            e = etoiles[j]
            dt = r.get("dist_tuile_px")
            gal.append({
                "id": len(gal) + 1, "image": nom, "source_image": AUTEURS[nom],
                "x": r["x"], "y": r["y"], "taille_image": [W, H],
                "fwhm_px": r["fwhm"], "fwhm_etoiles_px": r["fwhm_etoiles"], "rapport_fwhm": r["rapport_fwhm"],
                "crete_bruits": r["crete_bruits"], "part_flux_3px": r["part_3px"], "ellipticite": r["ellipticite"],
                "distance_gaia_px": r["dist_gaia"],
                "independance": {
                    "image_dans_validation_ou_ic1848": False,
                    "tuiles_d_entrainement_dans_l_image": len(TUI),
                    "distance_tuile_la_plus_proche_px": dt,
                    "preuve": ("image sans aucune tuile d'entrainement (aucun fichier apod_%s_g*.npz dans les paires APOD ni v33)" % nom[5:]) if not TUI
                              else f"{len(TUI)} tuiles de 512 px dans cette image, la plus proche a {dt} px (marge exigee 64 px)"},
                "etoile_comparaison": {"x": round(float(e[1]), 2), "y": round(float(e[2]), 2), "G": round(float(e[3]), 2), "flux_3px": round(float(e[0]), 1),
                                       "rapport_flux_3px_a_la_galaxie": round(float(e[0] / f3), 2), "fwhm_px": round(float(e[4]), 2)},
                "flux_3px": r["flux3"], "flux_12px": r["flux12"]})
            g = gal[-1]
            a = vignette(img, g["x"], g["y"]); b = vignette(img, g["etoile_comparaison"]["x"], g["etoile_comparaison"]["y"])
            for im, txt in ((a, f"{g['id']}  {nom[5:]}  x{g['rapport_fwhm']:.1f}  p{g['crete_bruits']:.0f}"), (b, f"etoile G{g['etoile_comparaison']['G']:.1f}")):
                cv2.putText(im, txt, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                cv2.rectangle(im, (0, 0), (im.shape[1] - 1, im.shape[0] - 1), (60, 60, 60), 1)
            planche.append(np.hstack([a, b]))
    while len(planche) % 3:
        planche.append(np.zeros_like(planche[0]))
    cv2.imencode(".jpg", np.vstack([np.hstack(planche[i:i + 3]) for i in range(0, len(planche), 3)]), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(OUT / "planche_controle.jpg"))
    cams = {"apod_017": "non documente", "apod_018": "non documente", "apod_023": "non documente", "apod_028": "non documente", "apod_071": "non documente"}
    info = {"cree": time.strftime("%Y-%m-%d %H:%M"), "nombre": len(gal), "images": sorted({g["image"] for g in gal}),
            "regle": "ne sert qu'a juger un reseau : jamais a regler ni a entrainer",
            "entrees": "<DONNEES>\\Paires etoiles APOD - 24 nov\\entrees\\<image>\\input.npy (unites de bruit, fond 0, entree sans retrait)",
            "mesures": "FWHM par profil radial moyen (tailles en pixels), rapport a la mediane des 15 etoiles Gaia de reference les plus proches ; crete = maximum lisse / bruit local (MAD) ; part 3 px = flux (r<=3 px) / flux (r<=12 px), fond = mediane de la couronne 16-24 px"}
    (OUT / "controle.json").write_text(json.dumps({"info": info, "galaxies": gal}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(len(gal), "galaxies", info["images"])


if __name__ == "__main__":
    main()
