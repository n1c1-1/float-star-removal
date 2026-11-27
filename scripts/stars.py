# ======================================================================
# stars.py
# Role     : Field of stars drawn and rendered through an instrument (psf.py).
# Step     : 1. Training on synthetic data only (E2 / E3, 7-9 Nov) (see scripts/README.md)
# Inputs   : generator: seeds + catalogue.json (public instrument sheets); no image
# Outputs  : synthetic pairs (input/target/info.json + manifest.json), network E3
# Example  : python stars.py [arguments]  (see the docstring; paths through the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""A field of stars drawn and rendered through an instrument (psf.py): the stars of one synthetic image.

The field (in the noise-sigma units of the backgrounds, a star's "peak" being its central value):

- the counts: N(> peak) proportional to peak^-alpha, alpha drawn 0.55 - 0.75, the density above 1 sigma drawn 50 to
  6000 per megapixel (measured on our images, star_measure.py: 54 for the Celestron Origin's M51 to 5450 for North
  America and NGC 7822; the ratio N(>1) / N(>100) of 13 to 40 gives the slope); down to 0.05 sigma (the unresolved
  stars of the Milky Way), up to far past the saturation; steeper above 100 sigmas, alpha drawn 0.7 - 1.0 (on our
  images N(> 300) / N(> 100) is 0.37 - 0.46 and N(> 1000) / N(> 300) 0.31 - 0.42);
- sometimes an open cluster (a few hundred stars around a centre) or a globular cluster (thousands, unresolved in its
  core: King 1962, AJ 67, 471);
- rarely one or two stars of Alnitak's class: thousands of times the saturation, their aureole still a few sigma 800
  px away (Alnitak in our Horsehead: 420 sigma at 40 px, 45 at 150 px, 3 at 800 px, about r^-1.7);
- the colours: B - V spread as the field stars, turned into the white-balanced camera's multipliers.

The rendering, on a canvas wider than the image (the stars outside throw their halos and spikes in):

- the core: the instrument's PSF at 3 x 3 places of the field, each star's light shared between the four nearest
  (bilinear weights), each place's map convolved with its PSF: a PSF varying smoothly over the field;
- the far light, one kernel for the whole field: the aureole I(r) ~ (1 + (r / r0)^2)^(-gamma / 2), gamma 1.5 - 2.6
  (Racine 1996, PASP 108, 699: r^-2; measured on our bright stars: r^-1 to r^-2.6), holding 0.3 to 5 % of the light
  (more makes a dense field's sum of halos a veil no real field shows);
  the diffraction spikes beyond the optical PSF's field (I ~ r^-2 along the spike, longer in red), split when a vane
  is turned;
- ghosts of the brightest stars: a defocused disc or ring, at a place mirrored through the optical axis (a reflection
  between the filter or the sensor's window and the optics);
- the part of each star the target keeps: a weight w(peak) rising from 0 at 0.3 sigma to 1 at 1.5 sigma (a star seen
  as a star); below, its light stays in the target, smoothed (8 px): the unresolved stars' light is the sky's;
- the giants stay: what lies under a star of Alnitak's class cannot be known, not even by hand, and a tile cannot take
  off a halo whose star it does not see. A star's reach is the diameter inside which its own light is over 50 noise
  units; up to 40 px the star is taken off, from 64 px it stays whole in the target (core, halo, spikes, ghosts, rings),
  a growing share of it in between; and these 40 px are at least 8 times the star's width (a defocused or smeared
  instrument's stars are wide without being giants) (measured on the 23 images: the common bright stars reach 16 - 40 px; Alnitak, the
  Flaming Star's AE Aurigae, the brightest of the NGC 7000 field reach 50 - 110 px).
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.signal import fftconvolve

from psf import Instrument, draw_instrument, loguniform, psf_at

#: the PSF's size (px), the far kernel's half size (px)
PSF_SIZE, FAR = 129, 1024
#: the peaks a star is seen from (w = 0 below VIS0, 1 above VIS1), the smoothing of the light the target keeps
VIS0, VIS1, KEEP_SIGMA = 0.3, 1.5, 8.0
#: the light a star's reach is measured at, the reaches (px) from and up to which the target keeps it
REACH_LEVEL, REACH0, REACH1 = 50.0, 40.0, 64.0
#: the widest stars drawn (px, their extent at half their peak): the widest of our 29 images measures 6.9 px
MAX_WIDTH = 10


def bv_colour(bv: np.ndarray) -> np.ndarray:
    """(3, n) multipliers of a star of colour B - V in the white-balanced camera (the white at the Sun's 0.65)."""
    col = np.stack([10 ** (0.2 * (bv - 0.65)), np.ones_like(bv), 10 ** (-0.36 * (bv - 0.65))])
    return (col / col.mean(0, keepdims=True)).astype(np.float32)


def draw_field(rng, n: int, margin: int, psf_peak: float, info: dict, density: float | None = None) -> dict:
    """Star positions on the canvas (n + 2 margin), peaks (sigma), fluxes, colours. ``density``: stars per megapixel
    above 1 sigma (the instrument's, from its scale), else drawn."""
    C = n + 2 * margin
    mp = C * C / 1e6
    dens = float(np.clip(density, 5, 8000)) if density else loguniform(rng, 50, 6000)
    alpha = float(rng.uniform(0.55, 0.75))
    pmin = 0.05
    nstar = int(dens * mp * pmin ** -alpha)
    nstar = min(nstar, 1_500_000)
    peak = pmin * rng.random(nstar) ** (-1 / alpha)
    bright = float(rng.uniform(0.7, 1.0))
    peak = np.where(peak > 100, 100 * (peak / 100) ** (alpha / bright), peak)
    x, y = rng.uniform(0, C, nstar), rng.uniform(0, C, nstar)
    info.update(density=round(dens, 1), alpha=round(alpha, 3), alpha_bright=round(bright, 3), stars=nstar)
    extra = []
    if rng.random() < 0.15:                                       # an open cluster
        k = int(loguniform(rng, 50, 600))
        cx, cy, s = rng.uniform(margin, margin + n, 2).tolist() + [loguniform(rng, 40, 300)]
        extra.append((cx + s * rng.standard_normal(k), cy + s * rng.standard_normal(k),
                      0.3 * rng.random(k) ** (-1 / 0.6)))
        info["open_cluster"] = k
    if rng.random() < 0.05:                                       # a globular cluster: King profile, unresolved core
        k = int(loguniform(rng, 3000, 30000))
        cx, cy = rng.uniform(margin + 0.2 * n, margin + 0.8 * n, 2)
        rc = loguniform(rng, 5, 40)
        r = rc * np.sqrt(np.clip(1 / rng.random(k) ** 1.0 - 1, 0, 400))     # surface density ~ 1 / (1 + (r/rc)^2)
        t = rng.uniform(0, 2 * np.pi, k)
        extra.append((cx + r * np.cos(t), cy + r * np.sin(t), 0.05 * rng.random(k) ** (-1 / 0.5)))
        info["globular"] = k
    for ex, ey, ep in extra:
        x, y, peak = np.r_[x, ex], np.r_[y, ey], np.r_[peak, ep]
    peak = np.minimum(peak, 3e5)
    if rng.random() < 0.15:                                       # Alnitak's class
        k = int(rng.integers(1, 3))
        xs, ys = rng.uniform(margin - 200, margin + n + 200, k), rng.uniform(margin - 200, margin + n + 200, k)
        ps = np.exp(rng.uniform(np.log(5e5), np.log(5e7), k))
        x, y, peak = np.r_[x, xs], np.r_[y, ys], np.r_[peak, ps]
        info["giants"] = [[round(float(a) - margin, 1), round(float(b) - margin, 1), float(f"{p:.3g}")]
                          for a, b, p in zip(xs, ys, ps)]
    bv = np.clip(0.7 + 0.35 * rng.standard_normal(len(peak)), -0.3, 2.0)
    return {"x": x, "y": y, "peak": peak, "flux": peak / psf_peak, "col": bv_colour(bv)}


def eat_faint(f: dict, rng, info: dict) -> dict:
    """A Sony camera's 'star eater' (its in-camera filter on long exposures takes faint stars for hot pixels): the
    stars under a level are gone, from the input and the target alike."""
    t = float(rng.uniform(0.5, 3.0))
    keep = f["peak"] > t
    info["star_eater_below"] = round(t, 2)
    return {"x": f["x"][keep], "y": f["y"][keep], "peak": f["peak"][keep], "flux": f["flux"][keep],
            "col": f["col"][:, keep]}


def splat(C: int, x, y, values) -> np.ndarray:
    """Point sources on a C x C map (bilinear: sub-pixel positions)."""
    m = np.zeros(C * C, np.float64)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x - x0, y - y0
    for dy, dx, w in ((0, 0, (1 - fx) * (1 - fy)), (0, 1, fx * (1 - fy)), (1, 0, (1 - fx) * fy), (1, 1, fx * fy)):
        xx, yy = x0 + dx, y0 + dy
        ok = (xx >= 0) & (xx < C) & (yy >= 0) & (yy < C)
        m += np.bincount(yy[ok] * C + xx[ok], (values * w)[ok], minlength=C * C)
    return m.reshape(C, C).astype(np.float32)


def far_kernel(rng, inst: Instrument, core_psf: np.ndarray, info: dict) -> tuple[np.ndarray, float]:
    """The far light's kernel (3, 2 FAR + 1, 2 FAR + 1): the aureole and the spikes beyond the optical PSF; and the
    aureole's share of the light (the core takes the rest)."""
    K = 2 * FAR + 1
    yy, xx = (np.mgrid[0:K, 0:K] - FAR).astype(np.float32)
    r = np.hypot(xx, yy)
    gamma = float(rng.uniform(1.5, 2.6))
    fwhm = math.hypot(inst.seeing_fwhm, 1.03 * inst.airy_px)
    r0 = fwhm * float(rng.uniform(1.0, 3.0))
    share = loguniform(rng, 0.003, 0.05)
    a = (1 + (r / r0) ** 2) ** (-gamma / 2) * (r <= FAR)
    a = a / a.sum() * share
    out = np.repeat(a[None], 3, 0)
    info.update(aureole_gamma=round(gamma, 2), aureole_share=round(share, 4), aureole_r0=round(r0, 2))
    if inst.vanes or inst.blades >= 5:
        # the spikes beyond the optical PSF: matched to its level at the edge of its field, I ~ r^-2, longer in red;
        # a vane's spikes run across it, a blade's across its edge
        h = PSF_SIZE // 2
        g = core_psf[1]
        dirs = []
        for k in range(inst.vanes):
            base = inst.vane_angle + k * 360.0 / inst.vanes + (inst.vane_twist if k == 0 else 0.0)
            dirs += [base + 90.0, base - 90.0]
        for k in range(inst.blades if inst.blades >= 5 else 0):
            normal = inst.blade_angle + (k + 0.5) * 360.0 / inst.blades
            dirs += [normal, normal + 180.0]
        lvl = []
        for d in dirs:
            t = math.radians(d)
            px, py = int(round(h + 0.8 * h * math.cos(t))), int(round(h + 0.8 * h * math.sin(t)))
            lvl.append(float(g[py, px]) - float(np.median(g[np.hypot(*np.mgrid[-h:h + 1, -h:h + 1]) > 0.75 * h])))
        level = max(np.median(lvl), 0.0)
        r_match = 0.8 * h
        width = max(0.6 * fwhm, 0.8)
        spikes = np.zeros((K, K), np.float32)
        for d in dirs:
            t = math.radians(d)
            along = xx * math.cos(t) + yy * math.sin(t)
            across = -xx * math.sin(t) + yy * math.cos(t)
            spikes += np.exp(-across ** 2 / (2 * width ** 2)) * (along > r_match) * \
                (r_match / np.maximum(along, 1)) ** 2
        for c, lam in enumerate((640.0, 530.0, 460.0)):
            out[c] += level * spikes * (lam / 530.0) ** 2
        info["spike_level"] = float(f"{level:.3g}")
    return out, share


def reach_share(peak: np.ndarray, core_psf: np.ndarray, far: np.ndarray, share: float, psf_peak: float) -> np.ndarray:
    """The share of each star the target keeps (0 - 1), from its reach: the diameter inside which its own light (the
    median of each ring, so without the spikes) is over REACH_LEVEL."""
    def rings(k: np.ndarray) -> np.ndarray:
        h = k.shape[0] // 2
        a, r = np.arange(32)[:, None] * 2 * math.pi / 32, np.arange(h + 1)
        return np.median(k[np.rint(h + r * np.sin(a)).astype(int), np.rint(h + r * np.cos(a)).astype(int)], axis=0)

    prof = rings(far[1])
    c = rings(core_psf[1])
    prof[:len(c)] += (1 - share) * c
    prof = np.minimum.accumulate(prof / psf_peak)
    reach = 2 * np.searchsorted(-prof, -REACH_LEVEL / peak, side="right")
    start = max(REACH0, 8.0 * 2 * np.searchsorted(-prof, -0.5, side="right"))
    return np.clip((reach - start) / (start * (REACH1 / REACH0 - 1)), 0, 1).astype(np.float32)


def star_width(inst: Instrument) -> int:
    """The extent (px) of a star at half its peak, at the centre of the field, along its longest side."""
    g = psf_at(inst, -inst.centre[0], -inst.centre[1], PSF_SIZE)[1]
    y, x = np.nonzero(g >= 0.5 * g.max())
    return int(max(np.ptp(y), np.ptp(x))) + 1


def ghost_stamp(rng, radius: float, ring: bool) -> np.ndarray:
    s = int(2 * radius + 9) | 1
    yy, xx = np.mgrid[0:s, 0:s] - s // 2
    r = np.hypot(xx, yy) / radius
    g = np.exp(-((r - 0.85) / 0.12) ** 2) if ring else (r < 1).astype(np.float32)
    g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), max(radius * 0.05, 0.7))
    return g / g.sum()


def render(rng, n: int, info: dict, inst: Instrument | None = None,
           kind: str | None = None) -> tuple[np.ndarray, np.ndarray, Instrument]:
    """(stars in the input, the stars' light the target keeps: the unseen ones and the giants), both (3, n, n) in
    sigma units; the instrument (drawn
    from the catalogue, the sky's richness by the background's ``kind``)."""
    if inst is None:
        from instruments import CATALOGUE, draw
        for _ in range(50):                                       # no instrument wider than our images ever are
            inst = draw(rng, n=n, kind=kind)[0] if CATALOGUE.exists() else draw_instrument(rng)
            if star_width(inst) <= MAX_WIDTH:
                break
    info["instrument"] = inst.as_dict()
    grid = (-1.0, 0.0, 1.0)
    psfs = {}
    for j, v in enumerate(grid):
        for i, u in enumerate(grid):
            psfs[(i, j)] = psf_at(inst, u - inst.centre[0], v - inst.centre[1], PSF_SIZE)
    centre_psf = psfs[(1, 1)]
    far, share = far_kernel(rng, inst, centre_psf, info)
    psf_peak = float(centre_psf[1].max()) * (1 - share)
    margin = FAR
    C = n + 2 * margin
    f = draw_field(rng, n, margin, psf_peak, info, inst.extra.get("density"))
    if inst.extra.get("star_eater"):
        f = eat_faint(f, rng, info)
    # the field coordinate of each star: -1 .. 1 across the image (beyond it outside)
    u = np.clip((f["x"] - margin) / n * 2 - 1, -1, 1)
    v = np.clip((f["y"] - margin) / n * 2 - 1, -1, 1)
    # the core: 3 x 3 places, bilinear weights, on the image and a margin of the PSF's half size
    m2 = PSF_SIZE
    C2 = n + 2 * m2
    near = (f["x"] > margin - m2) & (f["x"] < margin + n + m2) & (f["y"] > margin - m2) & (f["y"] < margin + n + m2)

    def light(flux: np.ndarray) -> np.ndarray:
        core = np.zeros((3, C2, C2), np.float32)
        for (i, j), p in psfs.items():
            wu = np.clip(1 - np.abs(u[near] - grid[i]), 0, 1)
            wv = np.clip(1 - np.abs(v[near] - grid[j]), 0, 1)
            w = wu * wv * (flux[near] > 0)
            if not w.any():
                continue
            for c in range(3):
                pts = splat(C2, f["x"][near] - (margin - m2), f["y"][near] - (margin - m2),
                            flux[near] * f["col"][c, near] * w * (1 - share))
                core[c] += fftconvolve(pts, p[c], mode="same").astype(np.float32)
        out = core[:, m2:m2 + n, m2:m2 + n]
        # the far light: every star on the wide canvas, one kernel
        for c in range(3):
            pts = splat(C, f["x"], f["y"], flux * f["col"][c])
            out[c] += fftconvolve(pts, far[c], mode="same")[margin:margin + n, margin:margin + n].astype(np.float32)
        return out

    stars = light(f["flux"])
    # the giants: what the target keeps of them
    stay = reach_share(f["peak"], centre_psf, far, share, psf_peak)
    big = light(f["flux"] * stay) if stay.any() else np.zeros_like(stars)
    info["giants_kept"] = int((stay > 0).sum())
    # ghosts of the brightest stars
    top_idx = np.argsort(-f["flux"])[:3]
    gh = []
    if rng.random() < 0.3:
        k = float(rng.uniform(-0.15, 0.15))
        for idx in top_idx:
            if f["peak"][idx] < 3e3:
                continue
            cx = margin + n / 2 * (1 + inst.centre[0])
            cy = margin + n / 2 * (1 + inst.centre[1])
            gx, gy = cx + (f["x"][idx] - cx) * (1 + k), cy + (f["y"][idx] - cy) * (1 + k)
            rad = loguniform(rng, 10, 150)
            st = ghost_stamp(rng, rad, rng.random() < 0.6)
            frac = loguniform(rng, 1e-4, 3e-3)
            tint = np.array([rng.uniform(0.6, 1.2), 1.0, rng.uniform(0.8, 1.6)], np.float32)
            s2 = st.shape[0] // 2
            x0, y0 = int(gx - margin) - s2, int(gy - margin) - s2
            ys, xs = slice(max(0, y0), min(n, y0 + st.shape[0])), slice(max(0, x0), min(n, x0 + st.shape[1]))
            if ys.start < ys.stop and xs.start < xs.stop:
                part = st[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]
                for c in range(3):
                    stars[c, ys, xs] += frac * f["flux"][idx] * f["col"][c, idx] * tint[c] * part
                    big[c, ys, xs] += stay[idx] * frac * f["flux"][idx] * f["col"][c, idx] * tint[c] * part
                gh.append([round(float(gx - margin), 1), round(float(gy - margin), 1), round(rad, 1)])
    if gh:
        info["ghosts"] = gh
    if inst.extra.get("microlens"):
        # the rings some sensors draw around bright stars (light bounced between the cover glass and the microlenses:
        # forum reports only), concentric, coloured, regularly spaced
        s = float(rng.uniform(8, 20))
        amp = loguniform(rng, 1e-5, 1e-4)
        R = int(7 * s)
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        rr = np.hypot(xx, yy)
        rings = []
        for c in range(3):
            ph = c * 2 * np.pi / 3
            rg = (0.5 + 0.5 * np.cos(2 * np.pi * rr / s + ph)) * np.exp(-rr / (3 * s)) * (rr > s)
            rings.append(rg / max(rg.sum(), 1e-9))
        for idx in top_idx:
            if f["peak"][idx] < 1e3:
                continue
            x0, y0 = int(f["x"][idx] - margin) - R, int(f["y"][idx] - margin) - R
            ys, xs = slice(max(0, y0), min(n, y0 + 2 * R + 1)), slice(max(0, x0), min(n, x0 + 2 * R + 1))
            if ys.start < ys.stop and xs.start < xs.stop:
                for c in range(3):
                    ring = amp * f["flux"][idx] * rings[c][ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]
                    stars[c, ys, xs] += ring
                    big[c, ys, xs] += stay[idx] * ring
        info["microlens_px"] = round(s, 1)
    # the light the target keeps: the unseen stars, smoothed
    w = np.clip((f["peak"] - VIS0) / (VIS1 - VIS0), 0, 1)
    keep = np.zeros((3, n, n), np.float32)
    inside = (f["x"] >= margin) & (f["x"] < margin + n) & (f["y"] >= margin) & (f["y"] < margin + n)
    for c in range(3):
        pts = splat(n, f["x"][inside] - margin, f["y"][inside] - margin,
                    f["flux"][inside] * f["col"][c, inside] * (1 - w[inside]))
        keep[c] = cv2.GaussianBlur(pts, (0, 0), KEEP_SIGMA)
    return np.clip(stars, 0, None), keep + big, inst
