# ======================================================================
# instruments.py
# Rôle     : Instruments réels (télescopes, objectifs, caméras) tirés de catalogue.json (fiches publiques).
# Étape    : 1. Entraînement sur données synthétiques seules (E2 / E3, 7-9 nov) (voir scripts/LISEZMOI.md)
# Entrées  : générateur : graines + catalogue.json (fiches d'instruments publiques) ; aucune image
# Sorties  : paires synthétiques (input/target/info.json + manifest.json), réseau E3
# Exemple  : python instruments.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Real instruments: a telescope or lens and a camera, from their published specifications (catalogue.json, each row
with its source), turned into what psf.py computes.

- the image scale: 206.265 x pixel (um) / focal length (mm) arcsec per pixel; lambda / D at 530 nm: 109.3 / D (mm)
  arcsec; the seeing (arcsec, drawn: median 2.5, Racine 1996) and the guiding, turned into pixels;
- the obstruction and the spider of the instrument (its secondary mirror, or the camera of a RASA, over the aperture);
- the image (2048 px) is a window of the sensor at a random place: the field position of its pixels runs over the
  sensor's half diagonal, so a corner window shows the corners' aberrations;
- a Newtonian's coma, from third-order optics: the tangential coma of a paraboloid is 3 theta f / (16 N^2) long
  (theta the field angle, N the focal ratio; e.g. Rutten & van Venrooij, Telescope Optics, 1988), the corrector taking
  most of it off when there is one; a Zernike coma coefficient c (waves) throws rays up to about 40 c lambda / D;
- a refractor's field curvature without its flattener: a defocus at the edge of h^2 / (2 R), R the Petzval radius
  (n f for a thin lens, about 1.4 f), its blur h^2 / (2 R N) (an 80 mm f/7.5 on APS-C: about 5 px in the corners);
  camera lenses are designed flat: their astigmatism and coma instead;
- the stack's own blur (demosaicing, the registration's interpolation and errors): 0.9 - 1.5 px times the drizzle;
- the star density per megapixel follows the scale: a sky density (stars per square arcminute, drawn from a sparse high
  latitude to the Milky Way in Cygnus) times the pixel's area;
- every stack's stars a little long (the guiding's, the drift's and the registration's residuals): a segment of 0.55 -
  2.3 FWHM.

The seeing, the stack's blur, the drizzle's shares, the elongation and the density are calibrated on what our 29 real
images measure (calib.py against diagnose.py units: median ellipticity 0.13, FWHM 3.2 - 3.7 px, 570 stars per megapixel
above 3 sigmas): the drawn fields measure 0.11, 3.6 px and 460.

The night is drawn too: well tuned (45 %: seeing 1.0 - 1.8", guiding 0.2 - 0.6", in focus, collimated, the
corrector in place), average (35 %) or bad (20 %: seeing 2.5 - 4.5", poor guiding or trails, missed focus rings, a
miscollimated mirror, a bent vane, a tilted sensor, pinched optics, the corrector missing). Every defect seen on our
images is extrapolated to every instrument of the catalogue: the optics give it that instrument's shape (a filled
ring for a refractor, a ring with the secondary's shadow and the spider's spikes for a Newtonian).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from psf import Instrument, loguniform

CATALOGUE = Path(__file__).resolve().parent / "catalogue.json"


def load() -> dict:
    return json.loads(CATALOGUE.read_text(encoding="utf-8"))


#: the sky's star richness by kind of field: emission and dark nebulae lie in the Milky Way's plane, galaxies far from
#: it (1: Cygnus as in our North America)
RICHNESS = {"emission": (0.3, 2.0), "dark": (0.3, 2.0), "reflection": (0.1, 1.0), "compact": (0.05, 1.0),
            "galaxy": (0.03, 0.4), "empty": (0.03, 1.5)}


def draw(rng, cat: dict | None = None, n: int = 2048, kind: str | None = None) -> tuple[Instrument, dict]:
    """An instrument from the catalogue (a telescope and a camera that go together), with its defects drawn; and what
    was drawn (names, scale, the window on the sensor, the sky's star density)."""
    cat = cat or load()
    scopes = cat["telescopes"]
    w = np.array([1 / math.sqrt(float(s["focal_mm"])) for s in scopes])    # short focal lengths image more often
    sc = scopes[int(rng.choice(len(scopes), p=w / w.sum()))]
    if sc.get("camera"):                                          # a smart telescope: its own sensor
        cam = {"name": sc["name"] + " sensor", "pixel_um": sc["camera"]["pixel_um"],
               "width_px": sc["camera"]["width_px"], "height_px": sc["camera"]["height_px"]}
    else:
        cams = cat["cameras"]
        cam = cams[int(rng.integers(len(cams)))]
    D, f = float(sc["aperture_mm"]), float(sc["focal_mm"])
    stopped = sc["design"] == "lens" and rng.random() < 0.5       # a lens closed one stop: smaller, polygonal
    if stopped:
        D /= math.sqrt(2)
    N = f / D
    pix = float(cam["pixel_um"])
    binning = 2 if (pix * 206.265 / f < 0.6 and rng.random() < 0.7) else 1   # oversampled: binned 2 x 2
    # the stack's output grid: drizzled 1.5 x for most of our stacks, sometimes 1 or 2 (stars wider in pixels)
    # (shares calibrated on the FWHM our stacks measure: calib.py)
    drizzle = float(rng.choice([1.0, 1.5, 2.0], p=[0.65, 0.28, 0.07])) if binning == 1 else 1.0
    scale = 206.265 * pix * binning / f / drizzle                 # arcsec / px of the stack
    inst = Instrument(kind=sc["design"])
    inst.airy_px = (109.3 / D) / scale
    # the night: well tuned (good seeing and guiding, in focus, collimated, corrected), average, or bad (every defect
    # seen on our images may come, on any instrument: the optics give it that instrument's own shape)
    night = ("good", "average", "bad")[int(rng.choice(3, p=[0.45, 0.35, 0.20]))]
    # the seeing in the stack (arcsec), the ranges calibrated on the FWHM our stacks measure (calib.py)
    seeing = {"good": rng.uniform(1.0, 1.8), "average": rng.uniform(1.6, 2.8), "bad": rng.uniform(2.5, 4.5)}[night]
    guide = {"good": rng.uniform(0.2, 0.6), "average": rng.uniform(0.4, 1.0), "bad": rng.uniform(0.8, 2.0)}[night]
    seeing, guide = float(seeing), float(guide)
    optics = float(sc.get("spot_arcsec", 0.0)) * float(rng.uniform(0.7, 1.5))   # a lens's own spot, wide open
    # the stack's own blur: demosaicing, the interpolation of the registration, its small errors (our stacks never
    # go under 2.3 px FWHM, even at 8"/px: star_measure.py)
    stack_px = float(rng.uniform(0.9, 1.5)) * drizzle             # calibrated on the measured FWHM (calib.py)
    inst.seeing_fwhm = math.sqrt((seeing ** 2 + optics ** 2) / scale ** 2 + stack_px ** 2)
    inst.jitter_px = guide / scale * float(rng.uniform(0.3, 1.0))
    inst.seeing_beta = float(rng.uniform(3.0, 8.0))               # our denoised stacks: a gaussian-like core
    inst.obstruction = float(sc.get("obstruction", 0.0))
    inst.vanes = int(sc.get("vanes", 0))
    if inst.vanes:
        lo, hi = sc.get("vane_mm_range", [1.0, 1.0])
        inst.vane_width = float(rng.uniform(lo, hi)) / D
        inst.vane_angle = float(rng.uniform(0, 90))
    elif sc.get("cables") and rng.random() < 0.5:                 # a RASA's camera cables across the aperture
        inst.vanes = int(rng.choice([1, 2]))
        inst.vane_width = float(rng.uniform(3.0, 6.0)) / D
        inst.vane_angle = float(rng.uniform(0, 180))
    if stopped:
        inst.blades = int(sc.get("blades", 0))
        inst.blade_angle = float(rng.uniform(0, 360))
    # the window: where the 2048 px sit on the sensor, its field position running over the sensor's half diagonal
    W, H = cam["width_px"] / binning * drizzle, cam["height_px"] / binning * drizzle
    pix = pix / drizzle                                           # the stack's pixel, in sensor microns
    half_diag_px = 0.5 * math.hypot(W, H)
    wx = float(rng.uniform(-max(W - n, 0) / 2, max(W - n, 0) / 2))
    wy = float(rng.uniform(-max(H - n, 0) / 2, max(H - n, 0) / 2))
    # psf.py's field position: (u, v) in -1..1 across the image, h = |(u, v)| / sqrt 2 = 1 at the image's corner.
    # The optical axis sits at -(wx, wy) from the window's centre; the window's corner is at this fraction of the
    # sensor's half diagonal:
    reach = math.hypot(n / 2 + abs(wx), n / 2 + abs(wy)) / half_diag_px
    inst.centre = (-wx / (n / 2), -wy / (n / 2))
    # psf.py's aberrations are given at the image's corner distance from the axis (n / sqrt 2 px): from the sensor's
    # corner, coma scales as the distance, curvature and astigmatism as its square
    k1 = (n / math.sqrt(2)) / half_diag_px
    theta_corner = half_diag_px * pix * binning * 1e-3 / f        # the sensor corner's field angle (rad)
    # the corrector or flattener: in place on a good night, sometimes missing or misspaced otherwise
    corr = sc.get("corrected", True) and not (night == "bad" and rng.random() < 0.5) and \
        not (night == "average" and rng.random() < 0.15)
    if sc["design"] == "newton":
        coma_mm = 3 * theta_corner * f / (16 * N * N)             # tangential coma at the sensor's corner
        coma_px = coma_mm / (pix * binning * 1e-3)
        keep = 1.0 if not corr else float(rng.uniform(0.03, 0.15))
        inst.coma_edge = coma_px * keep * k1 / (40 * inst.airy_px)
    if sc["design"] == "refractor" and not corr:                  # (camera lenses are designed flat)
        R = 1.4 * f                                               # the Petzval radius of a thin lens: n f
        h = half_diag_px * pix * binning * 1e-3
        blur_mm = h * h / (2 * R * N)
        inst.curvature_px = blur_mm / (pix * binning * 1e-3) * k1 ** 2
    if sc["design"] == "lens":
        inst.astig_edge = float(rng.uniform(0.0, 0.6)) * k1 ** 2
        inst.chroma_px = (float(rng.uniform(0, 1.5)), 0.0, float(rng.uniform(0, 3.0)))
    if sc.get("achromat"):
        inst.chroma_px = (float(rng.uniform(0, 1)), 0.0, float(rng.uniform(1, 4)))
    # the defects of the night, over the specifications: (probability on an average night, on a bad one); none on a
    # good night but a hint of focus error
    P = {"good": 0.0, "average": 1.0, "bad": 1.0}[night]
    bad = night == "bad"
    r = rng.random()
    if bad and r < 0.35:                                          # the rings of a missed focus (a real M31)
        inst.defocus_ring_px = loguniform(rng, 8, 40)
        inst.vignetting = float(rng.uniform(0.0, 0.5)) if inst.obstruction else 0.0
    elif r < (0.6 if bad else 0.3 if P else 0.1):
        inst.defocus_ring_px = loguniform(rng, 1.0, 4.0) if P else float(rng.uniform(0.5, 1.5))
    if inst.vanes and P and rng.random() < (0.35 if bad else 0.1):
        inst.vane_twist = float(rng.uniform(0.5, 4.0))            # a bent or turned vane: split spikes
    if inst.obstruction and P and rng.random() < (0.45 if bad else 0.15):
        inst.coma_edge += float(rng.uniform(0.1, 0.6)) if bad else float(rng.uniform(0.05, 0.2))   # miscollimated
    if P and rng.random() < (0.4 if bad else 0.15):
        s = 5 if bad else 2
        inst.tilt = (float(rng.uniform(-s, s)), float(rng.uniform(-s, s)))
    if bad and rng.random() < 0.15:
        inst.trefoil = float(rng.uniform(0.1, 0.4))               # pinched optics
        inst.trefoil_angle = float(rng.uniform(0, 120))
    r = rng.random()
    if bad and r < 0.15:
        inst.track_px = loguniform(rng, 6, 30)                    # trailed (untracked, a slipping mount)
    else:
        # every stack's stars are a little long, whatever the night (the guiding's and the drift's residuals, the
        # registration's): our 29 images measure a median ellipticity of 0.13 (0.06 - 0.21 from image to image,
        # diagnose.py units); a segment of about one FWHM gives it (calib.py)
        inst.track_px = inst.seeing_fwhm * loguniform(rng, 0.55, 2.3)
    inst.track_angle = float(rng.uniform(0, 180))
    # the sky's star density, through the pixel's area (5000 / Mpx above 1 sigma at 8"/px for a Cygnus-rich field)
    richness = loguniform(rng, *RICHNESS.get(kind or "", (0.03, 1.3)))
    depth = (D / 80.0) ** 0.8                                     # a larger aperture goes deeper
    # the scale is set so that the draws span our images' densities (54 to 5450 per Mpx, median about 575: they lean
    # to the Milky Way) with the sparse fields of long focal lengths below them
    density = 60000.0 * richness * depth * (scale / 8.0) ** 2     # 2.5 x: calibrated on the measured densities
    ml = cam.get("microlens_rings")
    inst.extra = {"night": night, "corrected": bool(corr), "stopped_down": stopped, "drizzle": drizzle,
                  "star_eater": bool(cam.get("star_eater")) and rng.random() < 0.6,
                  "microlens": bool(ml) and rng.random() < (0.1 if ml == "rare" else 0.6),
                  "telescope": sc["name"], "camera": cam["name"], "binning": binning, "scale_arcsec": round(scale, 3),
                  "seeing_arcsec": round(seeing, 2), "window": [round(wx), round(wy)], "reach": round(reach, 3),
                  "density": round(density, 1)}
    return inst, inst.extra
