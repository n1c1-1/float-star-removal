# ======================================================================
# preparer.py
# Role     : Prepares the Horsehead Nebula (Nicolas's linear stack, read in place) like the Stars stage of FLOAT: gradient removed (degree-2 polynomial, default settings of the Gradient block), background = median per channel, unit = sky spread (sk
# Stage    : 13. Trial on the Horsehead Nebula (26 Nov) (see scripts/LISEZMOI.md)
# Inputs   : linear stack of the Horsehead Nebula (not published)
# Outputs  : starless images (not published)
# Example  : python preparer.py [arguments]  (see the docstring; paths via variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
r"""Prepares the Horsehead Nebula (Nicolas's linear stack, read in place) like the Stars stage of FLOAT: gradient removed
(degree-2 polynomial, default settings of the Gradient block), background = median per channel, unit = sky spread (sky_spread), image in
noise units. No denoising (no half-stacks). Gaia stars G<6.5 from the offline extract, placed by the header's
WCS. The image is flipped (the FITS of [third-party software] is bottom-up) so that it is seen normally."""
import os, sys, time
os.environ["ASTRO_GPU_GAIA_EXTRACTS"] = r"<CACHE_FLOAT>\cache\gaia_extracts"; os.environ["ASTRO_GPU_OFFLINE"] = "1"
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"<FLOAT>")
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astro_gpu.stages.gradient import fit_poly_sky, poly_sky, GradientParams
from astro_gpu.stages.star_removal import sky_unit
from astro_gpu.core import catalogue_local
SRC = r"<DISQUE>/astro\2024-12_Astro-Mix\tosend\tosend\horse_head.fit"
OUT = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Tete de cheval – 26 nov" / "travail")
h = fits.open(SRC); hdr = h[0].header; d = np.asarray(h[0].data, np.float32)
C, H, W = d.shape
bp = GradientParams()
valid = np.ones((H, W), bool)
fit = fit_poly_sky(d, valid, bp.degree, bp.samples, bp.object_clip)
sky = poly_sky(fit["coefs"], bp.degree, H, W)
lin = d - sky + np.median(sky, axis=(1, 2), keepdims=True)    # gradient retire, niveau moyen du ciel conserve
shown = np.ones((H, W), bool)
unit = sky_unit(lin, shown)
bk = np.array([np.median(lin[c][::7, ::7]) for c in range(3)], np.float32)[:, None, None]
print("gradient: amplitude", [float(np.ptp(sky[c])) for c in range(3)], "unite", unit, "fond", bk.ravel())
wcs = WCS(hdr, naxis=2)
ra0, de0 = hdr["CRVAL1"], hdr["CRVAL2"]
cone = catalogue_local.astro_cone(ra0, de0, 1.65, 6.5)
x, y = wcs.all_world2pix(cone["RA_ICRS"], cone["DE_ICRS"], 0)
yt = H - 1 - y
ok = (x > -300) & (x < W + 300) & (yt > -300) & (yt < H + 300)
print("etoiles G<6,5 dans le champ:", int(ok.sum()), "completude", cone["complete_to"])
for i in np.argsort(cone["Gmag"][ok])[:8]:
    j = np.flatnonzero(ok)[i]; print("  G", cone["Gmag"][j], "x,y", round(x[j]), round(yt[j]))
flip = lambda a: np.ascontiguousarray(a[:, ::-1, :])
lin_f = flip(lin)
np.savez(OUT / "entree.npz", lin=lin_f.astype(np.float32), brut=flip(d), bk=bk, unit=np.float32(unit),
         gx=x[ok], gy=yt[ok], G=cone["Gmag"][ok])
print("ok", lin_f.shape, "max", float(lin_f.max()))
