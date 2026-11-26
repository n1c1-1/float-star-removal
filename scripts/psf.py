# ======================================================================
# psf.py
# Rôle     : Image d'une étoile calculée par l'optique de Fourier (pupille, aberrations de Zernike, turbulence, pixels).
# Étape    : 1. Entraînement sur données synthétiques seules (E2 / E3, 7-9 nov) (voir scripts/LISEZMOI.md)
# Entrées  : générateur : graines + catalogue.json (fiches d'instruments publiques) ; aucune image
# Sorties  : paires synthétiques (input/target/info.json + manifest.json), réseau E3
# Exemple  : python psf.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Star images computed from the optics: the point spread function (PSF) of an instrument at a place of the field.

Fraunhofer diffraction, as GalSim's OpticalPSF (Rowe et al. 2015, Astronomy & Computing 10, 121): the PSF is the
squared modulus of the Fourier transform of the pupil, P(x, y) exp(2 pi i W(x, y) / lambda).

- the pupil: a disc, or the polygon of a stopped-down lens's blades (N blades give N spikes, 2N when N is odd), its
  central obstruction (the secondary mirror of a Newtonian, a Schmidt-Cassegrain, a RASA; none for a refractor or a
  lens), the spider's vanes (or a RASA's camera cables: one or two thick "vanes"; one vane turned or offset gives split
  or doubled spikes, as a bent spider does), cut at the edge of the field by an offset disc when the beam is vignetted
  (the half moons of defocused stars at the edges);
- the wavefront W in Zernike polynomials (Noll 1976, JOSA 66, 207: Z4 defocus, Z5-6 astigmatism, Z7-8 coma, Z9-10
  trefoil, Z11 spherical), varying over the field as the third-order (Seidel) aberrations do: coma as the field height
  h along the field's radius, astigmatism and field curvature as h^2, a tilted sensor as a defocus slope across the
  field;
- three colours, each at its wavelength (the diffraction scale and the wavefront in waves scale with it), a
  refractor's chromatism as an extra defocus per colour (the blue / violet halos);
- then the atmosphere (a Moffat profile: Moffat 1969, A&A 3, 455; Racine 1996, PASP 108, 699), the tracking (a line,
  trailed stars when untracked) and its jitter, and the pixels (the computation is oversampled, then binned).

The defocus is set by the ring's diameter: a wavefront W_pv rho^2 (waves) throws the rays at the pupil's edge to
4 W_pv lambda / D from the centre: the ring's diameter is 8 W_pv in units of lambda / D; Z4 = a sqrt(3) (2 rho^2 - 1)
has W_pv = 2 sqrt(3) a.

Units: the PSF is returned in image pixels, (3, S, S), each colour summing to 1.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

#: the colours' wavelengths (nm), the reference (green)
WAVE = (640.0, 530.0, 460.0)
WAVE0 = 530.0
#: the pupil grid (the FFT size) and the largest field of view of the optical PSF (image px)
N_PUPIL = 1024
FOV_MAX = 256


class Instrument:
    """What makes the stars' shape in one image (drawn by ``draw_instrument``)."""
    kind: str = "refractor"
    airy_px: float = 0.4                 # lambda / D at 530 nm, in image pixels
    obstruction: float = 0.0             # diameter ratio of the central obstruction
    vanes: int = 0                       # spider vanes (0, 3, 4)
    vane_width: float = 0.01             # in pupil diameters
    vane_angle: float = 0.0              # degrees
    vane_twist: float = 0.0              # one vane turned by this (degrees): split / doubled spikes
    blades: int = 0                      # a lens stopped down: a polygonal aperture of this many blades (0: round)
    blade_angle: float = 0.0
    seeing_fwhm: float = 2.5             # px
    seeing_beta: float = 3.5
    defocus_ring_px: float = 0.0         # the defocused ring's diameter at the centre (px), 0: in focus
    tilt: tuple = (0.0, 0.0)             # extra ring diameter (px) at the right / bottom edge: a tilted sensor
    curvature_px: float = 0.0            # extra ring diameter at the corners (field curvature)
    coma_edge: float = 0.0               # waves (Z7/8) at the field's corner
    astig_edge: float = 0.0              # waves (Z5/6) at the corner
    spherical: float = 0.0               # waves (Z11)
    trefoil: float = 0.0                 # waves (Z9/10), pinched optics
    trefoil_angle: float = 0.0
    chroma_px: tuple = (0.0, 0.0, 0.0)   # extra ring diameter per colour (refractor chromatism)
    vignetting: float = 0.0              # offset of the cutting disc at the corner, in pupil radii
    track_px: float = 0.0                # trailing length (px)
    track_angle: float = 0.0
    jitter_px: float = 0.0               # guiding jitter (gaussian sigma, px)
    centre: tuple = (0.0, 0.0)           # the optical axis off the image centre (fraction of the half-size)
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}
Instrument = dataclass(Instrument)


# ---------------------------------------------------------------------------------------------- the pupil
def zernike(rho: np.ndarray, th: np.ndarray, j: int) -> np.ndarray:
    """Noll-normalised Zernike polynomial j (4 to 11)."""
    r2 = rho * rho
    return {
        4: math.sqrt(3) * (2 * r2 - 1),
        5: math.sqrt(6) * r2 * np.sin(2 * th),
        6: math.sqrt(6) * r2 * np.cos(2 * th),
        7: math.sqrt(8) * (3 * r2 * rho - 2 * rho) * np.sin(th),
        8: math.sqrt(8) * (3 * r2 * rho - 2 * rho) * np.cos(th),
        9: math.sqrt(8) * r2 * rho * np.sin(3 * th),
        10: math.sqrt(8) * r2 * rho * np.cos(3 * th),
        11: math.sqrt(5) * (6 * r2 * r2 - 6 * r2 + 1),
    }[j]


def pupil(inst: Instrument, rp: float, u: float, v: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The pupil's transmission on the N_PUPIL grid (radius ``rp`` grid cells) at field position (u, v) in -1..1,
    with its polar coordinates."""
    n = N_PUPIL
    yy, xx = (np.mgrid[0:n, 0:n] - n / 2).astype(np.float32)
    rho = np.hypot(xx, yy) / rp
    th = np.arctan2(yy, xx)
    P = (rho <= 1.0).astype(np.float32)
    if inst.blades >= 5:                              # a stopped-down lens: the regular polygon of its blades
        seg = 2 * math.pi / inst.blades
        t = np.mod(th - math.radians(inst.blade_angle), seg) - seg / 2
        P *= (rho * np.cos(t) <= math.cos(seg / 2)).astype(np.float32)
    if inst.obstruction > 0:
        P *= rho >= inst.obstruction
    for k in range(inst.vanes):                       # each vane a half-line from the centre, width vane_width x D
        a = math.radians(inst.vane_angle + k * 360.0 / inst.vanes + (inst.vane_twist if k == 0 else 0.0))
        along = xx * math.cos(a) + yy * math.sin(a)
        across = -xx * math.sin(a) + yy * math.cos(a)
        P *= ~((np.abs(across) < inst.vane_width * rp) & (along > 0))
    if inst.vignetting > 0 and (u or v):                                  # the beam cut at the field's edge
        h = math.hypot(u, v) / math.sqrt(2)
        d = inst.vignetting * h * rp
        a = math.atan2(v, u)
        P *= np.hypot(xx - d * math.cos(a), yy - d * math.sin(a)) <= rp
    return P, rho, th


def wavefront(inst: Instrument, rho, th, u: float, v: float, colour: int) -> np.ndarray:
    """W in waves at the reference wavelength, at field position (u, v)."""
    h = math.hypot(u, v) / math.sqrt(2)                                  # 1 at the corner
    phi = math.atan2(v, u)
    ring = inst.defocus_ring_px + inst.tilt[0] * (u + 1) / 2 + inst.tilt[1] * (v + 1) / 2 \
        + inst.curvature_px * h * h + inst.chroma_px[colour]
    a4 = max(ring, 0.0) / (16 * math.sqrt(3) * inst.airy_px)             # ring diameter -> Z4 (see the module)
    W = a4 * zernike(rho, th, 4)
    if inst.coma_edge:
        c = inst.coma_edge * h
        W += c * (math.sin(phi) * zernike(rho, th, 7) + math.cos(phi) * zernike(rho, th, 8))
    if inst.astig_edge:
        s = inst.astig_edge * h * h
        W += s * (math.sin(2 * phi) * zernike(rho, th, 5) + math.cos(2 * phi) * zernike(rho, th, 6))
    if inst.spherical:
        W += inst.spherical * zernike(rho, th, 11)
    if inst.trefoil:
        t = math.radians(inst.trefoil_angle)
        W += inst.trefoil * (math.sin(3 * t) * zernike(rho, th, 9) + math.cos(3 * t) * zernike(rho, th, 10))
    return W


# ---------------------------------------------------------------------------------------------- the PSF
def moffat_kernel(fwhm: float, beta: float, size: int) -> np.ndarray:
    """A Moffat profile on size x size, faded to 0 over the outer quarter of its radius (no edge in the wings)."""
    yy, xx = np.mgrid[0:size, 0:size] - (size - 1) / 2
    alpha = fwhm / (2 * math.sqrt(2 ** (1 / beta) - 1))
    r = np.hypot(xx, yy)
    h = (size - 1) / 2
    fade = np.clip((h - r) / (0.25 * h), 0, 1)
    k = (1 + r ** 2 / alpha ** 2) ** (-beta) * fade * fade * (3 - 2 * fade)
    return (k / k.sum()).astype(np.float32)


def blur(img: np.ndarray, ker: np.ndarray) -> np.ndarray:
    return cv2.filter2D(img, -1, ker, borderType=cv2.BORDER_CONSTANT)


def sampling(inst: Instrument) -> tuple[int, float]:
    """(oversampling, pupil radius in grid cells): the FFT's cell is 1 / os image px; the pupil fits the grid with a
    margin (the FFT's cell is (2 rp / N) lambda / D)."""
    os_ = int(min(8, max(1, math.ceil(2.4 / inst.airy_px))))
    while N_PUPIL / os_ < 96 and os_ > 1:                                # keep a usable field of view
        os_ -= 1
    rp = N_PUPIL / (2 * os_ * inst.airy_px)
    return os_, min(rp, 0.45 * N_PUPIL)


def psf_at(inst: Instrument, u: float, v: float, size: int = 129) -> np.ndarray:
    """The PSF (3, size, size) in image px at field position (u, v) in -1..1 (0, 0: the optical axis), each colour
    summing to 1."""
    os_, rp0 = sampling(inst)
    # the atmosphere, the tracking, the jitter: one kernel on the oversampled grid (a star 1 - 2 px wide is binned
    # after its blur, as the light falls on the pixels)
    fw = inst.seeing_fwhm * os_
    ks = int(min(N_PUPIL // 2, 2 * math.ceil(12 * fw) + 1)) | 1
    ker = moffat_kernel(fw, inst.seeing_beta, ks)
    if inst.track_px > 0.5:
        tl = inst.track_px * os_
        L = int(math.ceil(tl)) | 1
        line = np.zeros((L + 2, L + 2), np.float32)
        a = math.radians(inst.track_angle)
        for t in np.linspace(-tl / 2, tl / 2, 4 * L):
            line[int(round((L + 1) / 2 + t * math.sin(a))), int(round((L + 1) / 2 + t * math.cos(a)))] += 1
        ker = cv2.filter2D(np.pad(ker, L), -1, line / line.sum(), borderType=cv2.BORDER_CONSTANT)
    if inst.jitter_px > 0.05:
        ker = cv2.GaussianBlur(ker, (0, 0), inst.jitter_px * os_)
    out = np.zeros((3, size, size), np.float32)
    n = N_PUPIL // os_ * os_
    c0 = N_PUPIL // 2
    half = min(size // 2, n // (2 * os_) - 1)
    k = 2 * half + 1
    for c, lam in enumerate(WAVE):
        rp = rp0 * WAVE0 / lam                                           # the diffraction scale follows lambda
        P, rho, th = pupil(inst, rp, u, v)
        W = wavefront(inst, rho, th, u, v, c) * WAVE0 / lam              # the same path, in this colour's waves
        E = P * np.exp(2j * np.pi * W)
        I = (np.abs(np.fft.fftshift(np.fft.fft2(E))) ** 2).astype(np.float32)
        I = blur(I, ker)
        # bin to image px (centred), keep size x size
        sub = I[c0 - half * os_ - os_ // 2: c0 + (half + 1) * os_ - os_ // 2,
                c0 - half * os_ - os_ // 2: c0 + (half + 1) * os_ - os_ // 2]
        binned = sub[:k * os_, :k * os_].reshape(k, os_, k, os_).sum(axis=(1, 3))
        o = (size - k) // 2
        out[c, o:o + k, o:o + k] = binned
    # the wings fade out before the stamp's edge (a square edge would show around bright stars; the far kernel of
    # stars.py carries the light beyond)
    h = size // 2
    rr = np.hypot(*(np.mgrid[0:size, 0:size] - h))
    taper = np.clip((h - rr) / (0.3 * h), 0, 1) ** 2
    taper = (taper * taper * (3 - 2 * taper)).astype(np.float32)
    for c in range(3):
        out[c] *= taper
        out[c] /= max(float(out[c].sum()), 1e-12)
    return out


# ---------------------------------------------------------------------------------------------- the instruments
def loguniform(rng, lo: float, hi: float) -> float:
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


#: the kinds of instrument and their shares in the draws
KINDS = {"refractor": 0.30, "newton": 0.22, "sct": 0.10, "rasa": 0.06, "lens": 0.20, "smart": 0.12}


def draw_instrument(rng, fwhm_px: float | None = None) -> Instrument:
    """An instrument at random, its total star FWHM about ``fwhm_px`` (drawn as measured on our images when None:
    median 3.7 px, 10 % under 2.5, 10 % over 6.5 - star_measure.py)."""
    kinds = list(KINDS)
    kind = kinds[rng.choice(len(kinds), p=list(KINDS.values()))]
    fw = fwhm_px or float(np.clip(math.exp(rng.normal(math.log(3.7), 0.35)), 1.6, 12.0))
    inst = Instrument(kind=kind)
    inst.airy_px = loguniform(rng, 0.25, 1.2)
    inst.seeing_fwhm = math.sqrt(max(fw ** 2 - (1.03 * inst.airy_px) ** 2, 0.8))
    inst.seeing_beta = float(rng.uniform(2.5, 5.0))
    if kind == "newton":
        inst.obstruction = float(rng.uniform(0.2, 0.35))
        inst.vanes = 4
        inst.vane_width = loguniform(rng, 0.006, 0.025)
        inst.coma_edge = float(rng.uniform(0.0, 0.9)) if rng.random() < 0.6 else 0.0   # no / partial corrector
    elif kind == "sct":
        inst.obstruction = float(rng.uniform(0.3, 0.42))
        inst.vanes = 0
    elif kind == "rasa":
        inst.obstruction = float(rng.uniform(0.35, 0.45))
        inst.vanes = 4 if rng.random() < 0.3 else 0                        # the camera's cable, sometimes a spider
        inst.vane_width = loguniform(rng, 0.005, 0.02)
    elif kind == "lens":
        inst.astig_edge = float(rng.uniform(0.0, 0.8))
        inst.coma_edge = float(rng.uniform(0.0, 0.6))
        inst.chroma_px = (float(rng.uniform(0, 2)), 0.0, float(rng.uniform(0, 4)))
    elif kind == "refractor":
        inst.chroma_px = ((float(rng.uniform(0, 1)), 0.0, float(rng.uniform(0, 3)))
                          if rng.random() < 0.5 else (0.0, 0.0, 0.0))     # an achromat's blue halo, or an apo
        inst.astig_edge = float(rng.uniform(0.0, 0.3))
    else:                                                                  # a smart telescope: small, apochromat
        inst.airy_px = loguniform(rng, 0.6, 1.5)
        inst.seeing_fwhm = math.sqrt(max(fw ** 2 - (1.03 * inst.airy_px) ** 2, 0.8))
    if inst.vanes:
        inst.vane_angle = float(rng.uniform(0, 90))
        inst.vane_twist = float(rng.uniform(0.5, 4.0)) if rng.random() < 0.15 else 0.0   # a bent vane
    # focus: mostly sharp, sometimes soft, rarely the rings of a missed focus (a real M31: 20 px rings)
    r = rng.random()
    if r < 0.08:
        inst.defocus_ring_px = loguniform(rng, 8, 40)
        inst.vignetting = float(rng.uniform(0.0, 0.5)) if inst.obstruction else 0.0
    elif r < 0.30:
        inst.defocus_ring_px = loguniform(rng, 1.0, 5.0)
    if rng.random() < 0.15:                                                # a tilted sensor
        inst.tilt = (float(rng.uniform(-6, 6)), float(rng.uniform(-6, 6)))
    if rng.random() < 0.2:
        inst.curvature_px = float(rng.uniform(0, 6))
    if rng.random() < 0.1:
        inst.trefoil = float(rng.uniform(0.1, 0.5))                        # pinched optics
        inst.trefoil_angle = float(rng.uniform(0, 120))
    if rng.random() < 0.15:
        inst.spherical = float(rng.uniform(-0.4, 0.4))
    # tracking: mostly fine (a slight elongation), sometimes trailed, rarely untracked
    r = rng.random()
    if r < 0.05:
        inst.track_px = loguniform(rng, 6, 30)
    elif r < 0.35:
        inst.track_px = loguniform(rng, 0.5, 3.0)
    inst.track_angle = float(rng.uniform(0, 180))
    inst.jitter_px = float(rng.uniform(0, 0.8))
    inst.centre = (float(rng.uniform(-0.1, 0.1)), float(rng.uniform(-0.1, 0.1)))
    return inst
