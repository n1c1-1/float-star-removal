# ======================================================================
# telecharger_gaia_apod.py
# Role     : Deep Gaia DR3 extracts (G < 21) of the solved web images, PREPARATION only.
# Step     : 3. De-linearisation and astrometry of the APOD web images (24 Nov) (see scripts/LISEZMOI.md)
# Inputs   : downloaded APOD web images
# Outputs  : linear images, astrometry, tiles
# Example  : python telecharger_gaia_apod.py [arguments]  (see the docstring; paths via the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
r"""Deep Gaia DR3 extracts (G < 21) of the solved web images, PREPARATION only. Adapted copy of
Catalogue Gaia profond – 20 nov\telecharger_gaia_profond.py (same query, same columns, same output format), which read
astrometrie.npz: here the WCS comes from lineaire\NNN_astrometrie.npz and the image size from lineaire\NNN.npy.
GAVO TAP service (gaia.dr3lite), sequential queries (sync: async no longer answered on 24 Nov), 3 s between two, generic User-Agent; an error is
retried with doubled wait, then the image is recorded as failed (no stubbornness).
Output: <DONNEES>\Catalogue Gaia profond APOD – 24 nov\gaia_dr3_profond_apod_NNN.npz. Usage: telecharger_gaia_apod.py [--compte] n..."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))
import numpy as np  # noqa: E402
import requests  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.table import unique, vstack  # noqa: E402
from astropy.wcs import WCS  # noqa: E402

from outils_apod import LIN  # noqa: E402

OUT = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Catalogue Gaia profond APOD – 24 nov")
OUT.mkdir(parents=True, exist_ok=True)
TAP = "https://dc.zah.uni-heidelberg.de/tap"
GMAX = 21.0
PAUSE = 3.0


def tuiles(n):
    z = np.load(LIN / f"{n:03d}_astrometrie.npz", allow_pickle=True)
    w = WCS(fits.Header.fromstring(str(z["wcs"])))
    H, W = np.load(LIN / f"{n:03d}.npy", mmap_mode="r").shape[1:]
    mx, my = 0.05 * W + 60, 0.05 * H + 60
    x0, x1, y0, y1 = -mx, W + mx, -my, H + my
    step = int(0.7 / (float(z["scale"]) / 3600))
    xs = list(range(int(x0), int(x1), step)) + [int(x1)]
    ys = list(range(int(y0), int(y1), step)) + [int(y1)]
    T = [(xs[i], xs[i + 1], ys[j], ys[j + 1]) for i in range(len(xs) - 1) for j in range(len(ys) - 1)]
    return z, w, W, H, (x0, x1, y0, y1), T


def main():
    args = sys.argv[1:]
    compte = "--compte" in args
    nums = [int(a) for a in args if a != "--compte"]
    if compte:
        for n in nums:
            _, _, W, H, _, T = tuiles(n)
            print(n, f"{W}x{H}", len(T), "requetes", flush=True)
        return
    sess = requests.Session()
    sess.headers["User-Agent"] = "astro-catalogue-extract/1.0"
    MAXREC = 2_000_000
    # 24 nov : le service asynchrone (TAP /async) ne repond plus (coupure apres 120 s, meme pour une requete de 8 000 lignes) ;
    # le service synchrone repond en ~1 s. Meme requete ADQL, via /sync ; une tuile qui echoue ou atteint MAXREC est coupee en 4.

    def q(adql, tries=3):
        from astropy.io import ascii
        for i in range(tries):
            try:
                r = sess.post(TAP + "/sync", data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "MAXREC": MAXREC, "QUERY": adql}, timeout=180)
                if r.status_code == 200 and r.text.startswith("source_id"):
                    return ascii.read(r.text, format="csv")
                print("  reponse", r.status_code, " ".join(r.text[:150].split()), flush=True)
                if r.status_code in (400, 403, 429, 503):
                    time.sleep(30 * 2 ** i)
            except Exception as e:  # noqa: BLE001
                print("  nouvel essai", i, str(e)[:150], flush=True)
                time.sleep(20 * 2 ** i)
        return None

    def tuile_ad(w, a0, a1, b0, b1, prof=0):
        px, py = [a0, a1, a1, a0], [b0, b0, b1, b1]
        ra, de = w.all_pix2world(px, py, 0)
        pts = ", ".join(f"{a:.7f}, {b:.7f}" for a, b in zip(ra, de))
        sql = ("SELECT source_id, ra, dec, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, pmra, pmdec FROM gaia.dr3lite "
               f"WHERE 1=CONTAINS(POINT('ICRS', ra, dec), POLYGON('ICRS', {pts})) AND phot_g_mean_mag < {GMAX}")
        t = q(sql)
        time.sleep(PAUSE)
        if (t is None or len(t) >= MAXREC) and prof < 3 and (a1 - a0) > 64:
            am, bm = (a0 + a1) // 2, (b0 + b1) // 2
            print("  tuile coupee en 4", flush=True)
            out = []
            for (u0, u1, v0, v1) in ((a0, am, b0, bm), (am, a1, b0, bm), (a0, am, bm, b1), (am, a1, bm, b1)):
                out += tuile_ad(w, u0, u1, v0, v1, prof + 1)
            return out
        if t is None:
            raise RuntimeError("serveur : echec")
        return [t]

    for n in nums:
        f = OUT / f"gaia_dr3_profond_apod_{n:03d}.npz"
        if f.exists():
            print(n, "deja fait", flush=True)
            continue
        t0 = time.time()
        z, w, W, H, (x0, x1, y0, y1), T = tuiles(n)
        parts = []
        for k, (a0, a1, b0, b1) in enumerate(T):
            ps = tuile_ad(w, a0, a1, b0, b1)
            print(n, k + 1, "/", len(T), sum(len(p) for p in ps), flush=True)
            parts += ps
        t = unique(vstack([p for p in parts if len(p)]), keys="source_id")
        ra, de = np.asarray(t["ra"], float), np.asarray(t["dec"], float)
        x, y = w.all_world2pix(ra, de, 0)
        m = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        g_ = lambda k: np.ma.filled(np.ma.asarray(t[k]).astype(float), np.nan)[m]  # noqa: E731
        g = g_("phot_g_mean_mag")
        np.savez_compressed(f, Source=np.ma.filled(t["source_id"], -1).astype(np.int64)[m], RA_ICRS=ra[m], DE_ICRS=de[m], Gmag=g,
                            BPmag=g_("phot_bp_mean_mag"), RPmag=g_("phot_rp_mean_mag"), pmRA=g_("pmra"), pmDE=g_("pmdec"),
                            x_pix=x[m], y_pix=y[m], footprint=np.array([x0, x1, y0, y1]))
        print("RESULT", n, "n=", int(m.sum()), "G max", float(np.nanmax(g)), f"{f.stat().st_size / 1e6:.1f} Mo", f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
