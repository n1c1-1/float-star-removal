# ======================================================================
# backgrounds.py
# Rôle     : Fonds de ciel sans étoiles, générés (nébuleuses, galaxies, poussière) sur lesquels on dessine les étoiles synthétiques.
# Étape    : 1. Entraînement sur données synthétiques seules (E2 / E3, 7-9 nov) (voir scripts/LISEZMOI.md)
# Entrées  : générateur : graines + catalogue.json (fiches d'instruments publiques) ; aucune image
# Sorties  : paires synthétiques (input/target/info.json + manifest.json), réseau E3
# Exemple  : python backgrounds.py --n 24 [--size 2048] [--seed 0] [--out <ESSAIS>/starRemoval/backgrounds]
# ======================================================================
"""Star-free sky backgrounds, generated: the backgrounds the synthetic stars are drawn on to train the star removal.

Each image is linear, in the units the star removal network works in (astro_gpu/stages/star_removal.py): each colour
minus its sky, over one noise sigma of the sky; 0 is the sky. No noise and no star here: they come after, on top.

Most fields are a dark sky with an object; some are filled with nebula. The kinds, drawn at random:

- **emission nebula**: a Gaussian random field with a power-law spectrum P(k) ~ k^-beta (the interstellar medium is
  self-similar: beta about 2.7 to 3 in the far infrared, Miville-Deschenes et al. 2010, A&A 518, L104; steeper on our
  images, off their stars, smoothed by the denoising: measure.py real_input), made lognormal (the column density of
  turbulent clouds is lognormal:
  Kainulainen et al. 2009, A&A 508, L35), under a soft envelope that fades into the sky; the fine detail only where the
  nebula is bright; sometimes a bright core (M42, M8), an ionisation front (an edge sharp on one side), filaments
  (everywhere in the interstellar medium: Andre et al. 2010, A&A 518, L102; drawn as ridges of a field, the "ridged"
  noise of procedural textures: Ebert, Musgrave et al., Texturing & Modeling, 2003), an OIII region (teal, compact);
- **galaxy**: a Sersic bulge and an exponential disc (Sersic 1963; b_n from Ciotti & Bertin 1999, A&A 352, 447),
  inclined, logarithmic spiral arms with H II knots, a dust lane when seen edge-on; and small background galaxies, a few
  pixels wide, which look like stars and must stay;
- **reflection nebula**: blue, soft, around a few centres (the Pleiades);
- **dark nebula**: the faint light of the Milky Way crossed by dust lanes;
- **compact**: a small bright nebula or a planetary nebula (a ring, teal inside, red at the rim) in an empty sky;
- **empty**: the sky, faint cirrus at most.

Dust in absorption crosses some fields (the light behind times exp(-tau), redder where thicker). The sky: a level near 0
and a faint residual gradient (the Gradient block never takes it all). The colours are those measured on our images
off their stars (measure.py real_input, R/G and B/G where there is signal): emission R/G 1.4 to 3.4 (7.3 with a
dual-band filter), B/G 0.64 to 1.12; reflection R/G about 0.6, B/G 1.4 to 1.6; galaxies R/G 1.0 to 1.2, B/G 0.75 to
0.85. The brightness: the 99.9th percentile of the luminance drawn around that of our images off their stars (10 %
under 1.3 sigma, median 6.7, 10 % over 50).

    python backgrounds.py --n 24 [--size 2048] [--seed 0] [--out <ESSAIS>/starRemoval/backgrounds]

Writes <out>/bg_<seed>_<i>.fits (3, H, W) float32, bg_<seed>_<i>.json (what was drawn), previews/bg_<...>.jpg.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

OUT = (Path(__import__("os").environ.get("ESSAIS", "essais")) / "starRemoval" / "backgrounds")
LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)
#: the kinds of field and their shares
KINDS = {"emission": 0.30, "galaxy": 0.25, "reflection": 0.08, "dark": 0.10, "compact": 0.10, "empty": 0.17}
#: the 99.9th percentile of the luminance (sigma): log10 drawn normal around our images' (median 4, 90 % under ~70),
#: per kind (a galaxy's core and a compact nebula stand out more than a wide nebula)
P999_LOG10 = {"emission": 0.6, "galaxy": 0.9, "reflection": 0.6, "dark": 0.6, "compact": 1.0, "empty": -0.2}
P999_SD, P999_RANGE = 0.8, (0.5, 500.0)


# ---------------------------------------------------------------------------------------------- colours
def colour(r_g: float, b_g: float) -> np.ndarray:
    """An RGB colour of ratios R/G and B/G, of luminance 1."""
    c = np.array([r_g, 1.0, b_g], np.float32)
    return c / float(__import__('operator').matmul(LUM, c))


def emission_colour(rng) -> np.ndarray:
    r = float(np.exp(rng.uniform(np.log(1.4), np.log(4.0)))) if rng.random() < 0.9 else float(rng.uniform(4, 8))
    return colour(r, float(rng.uniform(0.6, 1.2)))


def oiii_colour(rng) -> np.ndarray:
    return colour(float(rng.uniform(0.35, 0.7)), float(rng.uniform(0.9, 1.3)))


def reflection_colour(rng) -> np.ndarray:
    return colour(float(rng.uniform(0.5, 0.75)), float(rng.uniform(1.35, 1.7)))


def galaxy_colours(rng) -> tuple[np.ndarray, np.ndarray]:
    """(bulge, disc): the bulge warmer, the disc and its arms bluer."""
    return (colour(float(rng.uniform(1.1, 1.25)), float(rng.uniform(0.72, 0.85))),
            colour(float(rng.uniform(0.95, 1.1)), float(rng.uniform(0.95, 1.2))))


# ---------------------------------------------------------------------------------------------- fields
def power_field(rng, n: int, beta: float, kmin: float = 1.0) -> np.ndarray:
    """A Gaussian random field n x n, P(k) ~ k^-beta (k in cycles per image, below ``kmin`` flat), mean 0, std 1."""
    ky = np.fft.fftfreq(n)[:, None] * n
    kx = np.fft.rfftfreq(n)[None, :] * n
    k = np.maximum(np.hypot(ky, kx), kmin)
    amp = k ** (-beta / 2)
    amp[0, 0] = 0
    z = (rng.standard_normal(amp.shape) + 1j * rng.standard_normal(amp.shape)) * amp
    f = np.fft.irfft2(z, s=(n, n)).astype(np.float32)
    return (f - f.mean()) / (f.std() + 1e-12)


def lognormal(g: np.ndarray, s: float) -> np.ndarray:
    e = np.exp(s * g)
    return e / np.median(e)


def ridges(g: np.ndarray, sharp: float) -> np.ndarray:
    """Ridged noise: 1 - |g| in 0..1, raised to ``sharp`` (higher: thinner filaments)."""
    return (1 - np.clip(np.abs(g) / 2.5, 0, 1)) ** sharp


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def grid(n: int):
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    return xx, yy


def blob(rng, n: int, cx: float, cy: float, size: float, warp: float) -> np.ndarray:
    """A soft region around (cx, cy) of about ``size`` px, elongated and turned at random, its edge warped by a smooth
    field and fading slowly into the sky (no hard edge)."""
    xx, yy = grid(n)
    q = float(rng.uniform(0.4, 1.0))
    a = rng.uniform(0, np.pi)
    dx, dy = xx - cx, yy - cy
    u = (dx * np.cos(a) + dy * np.sin(a)) / size
    v = (-dx * np.sin(a) + dy * np.cos(a)) / (size * q)
    r = np.hypot(u, v) * np.exp(warp * power_field(rng, n, 3.6, kmin=1.5) * 0.5)
    return np.exp(-r ** float(rng.uniform(1.2, 2.5)))


def sersic(r: np.ndarray, re: float, nn: float) -> np.ndarray:
    """Sersic profile of index ``nn``, 1 at the effective radius ``re`` (b_n: Ciotti & Bertin 1999)."""
    bn = 2 * nn - 1 / 3 + 4 / (405 * nn) + 46 / (25515 * nn ** 2)
    return np.exp(-bn * ((np.maximum(r, 1e-3) / re) ** (1 / nn) - 1)).astype(np.float32)


# ---------------------------------------------------------------------------------------------- the objects
def emission(rng, n: int, rgb: np.ndarray, info: dict, filled: bool):
    beta = float(rng.uniform(3.0, 4.2))
    s_ln = float(rng.uniform(0.4, 1.0))
    base = power_field(rng, n, beta)
    if filled:
        env = sigmoid((power_field(rng, n, 4.5, kmin=1.5) + float(rng.uniform(-0.5, 1.0))) /
                      float(rng.uniform(0.4, 1.2)))
    else:
        c = rng.uniform(0.25, 0.75, 2) * n
        env = blob(rng, n, float(c[0]), float(c[1]), float(rng.uniform(0.12, 0.45)) * n, float(rng.uniform(0.3, 1.0)))
    neb = lognormal(base, s_ln) * env
    # the fine detail where the nebula is bright only
    fine = power_field(rng, n, float(rng.uniform(2.4, 3.0)))
    neb *= np.exp(float(rng.uniform(0.1, 0.35)) * fine * np.clip(neb / np.percentile(neb, 99), 0, 1))
    info.update(beta=round(beta, 2), lognormal=round(s_ln, 2), filled=filled)
    if rng.random() < 0.5:                                       # filaments, following the nebula's own structure
        g2 = 0.7 * base + 0.7 * power_field(rng, n, float(rng.uniform(2.6, 3.4)))
        k = float(rng.uniform(0.3, 1.5))
        neb *= 1 + k * ridges(g2 / g2.std(), float(rng.uniform(3.0, 10.0)))
        info["filaments"] = round(k, 2)
    if rng.random() < 0.3:                                       # an ionisation front: a bright rim, sharp on one side
        xx, yy = grid(n)
        ang = rng.uniform(0, 2 * np.pi)
        d = ((xx - n / 2) * np.cos(ang) + (yy - n / 2) * np.sin(ang)) + power_field(rng, n, 3.8) * n * 0.05
        d -= float(rng.uniform(-0.25, 0.25)) * n
        w = float(rng.uniform(0.02, 0.08)) * n
        rim = np.where(d > 0, 0.15 + 0.85 * np.exp(-d / w) + 0.6 * np.exp(-d / (w / 4)), 1.0).astype(np.float32)
        import cv2
        neb *= cv2.GaussianBlur(rim, (0, 0), float(rng.uniform(2.0, 6.0)))         # an edge, not a drawn line
        info["front_px"] = round(w, 1)
    rgb += emission_colour(rng)[:, None, None] * neb[None]
    if rng.random() < 0.3:                                       # a bright core (M42, M8)
        c = np.unravel_index(np.argmax(env * lognormal(base, 1.0)), env.shape)
        core = blob(rng, n, float(c[1]), float(c[0]), float(rng.uniform(0.02, 0.08)) * n, 0.5)
        k = float(rng.uniform(5, 40))
        rgb += emission_colour(rng)[:, None, None] * (k * core * np.percentile(neb, 99) * lognormal(fine, 0.2))[None]
        info["core"] = round(k, 1)
    if rng.random() < 0.4:                                       # an OIII region: teal, where the nebula is densest
        o = (neb / np.percentile(neb, 99.5)) ** float(rng.uniform(1.5, 3.0))
        w = float(rng.uniform(0.1, 0.6))
        rgb += oiii_colour(rng)[:, None, None] * (w * o * np.percentile(neb, 99.5))[None]
        info["oiii"] = round(w, 2)


def galaxy(rng, n: int, rgb: np.ndarray, info: dict, cx: float, cy: float, size: float, detail: bool):
    """A Sersic bulge and an exponential disc of half-light radius ``size`` px, inclined, with arms, H II knots and a
    dust lane when ``detail``."""
    xx, yy = grid(n)
    q = float(rng.uniform(0.15, 1.0))                            # axis ratio: edge-on (0.15) to face-on (1)
    pa = rng.uniform(0, np.pi)
    dx, dy = xx - cx, yy - cy
    u = dx * np.cos(pa) + dy * np.sin(pa)
    v = (-dx * np.sin(pa) + dy * np.cos(pa)) / q
    r = np.hypot(u, v)
    bulge_c, disc_c = galaxy_colours(rng)
    disc = sersic(r, size, 1.0)
    bulge = sersic(r, size * float(rng.uniform(0.08, 0.3)), float(rng.uniform(2.0, 4.0)))
    bt = float(rng.uniform(0.2, 2.0))                            # bulge to disc, at the centre's scale
    if detail and q > 0.3:                                       # logarithmic spiral arms
        m = int(rng.choice([2, 2, 2, 3, 4]))
        pitch = np.radians(rng.uniform(10, 30))
        th = np.arctan2(v, u)
        # the arms' phase wanders (no two arms alike, no drawn rings), their contrast stays moderate
        wander = power_field(rng, n, 4.0, kmin=2.0) * float(rng.uniform(0.3, 0.8))
        arm = 0.5 + 0.5 * np.cos(m * (th - np.log(np.maximum(r, 1) / size) / np.tan(pitch)) + wander)
        arm = arm ** float(rng.uniform(1.0, 2.5)) * sigmoid((r / size - 0.4) * 4)      # no arms in the bulge
        disc = disc * (0.7 + float(rng.uniform(0.3, 0.9)) * arm * lognormal(power_field(rng, n, 2.8), 0.3))
        knots = np.clip(power_field(rng, n, 2.2), 2.2, None) - 2.2              # H II regions along the arms
        rgb += emission_colour(rng)[:, None, None] * (knots * arm * sersic(r, size, 1.0) * 1.5)[None]
        info.setdefault("arms", m)
    light = disc_c[:, None, None] * disc[None] + bulge_c[:, None, None] * (bt * bulge / max(bulge.max(), 1e-9) *
                                                                        disc.max())[None]
    if detail and q < 0.45 and rng.random() < 0.8:               # the dust lane of an edge-on disc
        lane = np.exp(-(((-dx * np.sin(pa) + dy * np.cos(pa)) - float(rng.uniform(-0.3, 0.3)) * size * q) /
                        (size * q * float(rng.uniform(0.15, 0.35)))) ** 2) * (np.abs(u) < 3 * size)
        tau = lane * float(rng.uniform(0.8, 2.5)) * lognormal(power_field(rng, n, 3.0), 0.4)
        light = light * np.exp(-tau[None] * np.array([0.75, 1.0, 1.3], np.float32)[:, None, None])
        info["dust_lane"] = True
    if rng.random() < 0.6:
        # a point-like nucleus (a nuclear star cluster or an active nucleus: Böker et al. 2002, AJ 123, 1389), as
        # wide as a star, kept in the starless image: the network must learn to leave it. Its concentration (peak over
        # the 4 - 6 px ring) drawn 8 - 40 around the 27 measured on M51's nucleus in our Seestar image (diagnose)
        L = light.sum(0)
        iy, ix = int(round(cy)), int(round(cx))
        if 8 <= iy < n - 8 and 8 <= ix < n - 8:
            rr = np.hypot(xx - cx, yy - cy)
            ring = float(L[(rr >= 4) & (rr <= 6)].mean())
            conc = float(np.exp(rng.uniform(np.log(8), np.log(40))))
            amp = conc * ring - float(L[iy, ix])
            if amp > 0:
                w = float(rng.uniform(2.0, 5.0))
                core = np.exp(-4 * np.log(2) * rr ** 2 / w ** 2)
                light = light + (amp / bulge_c.sum()) * bulge_c[:, None, None] * core[None]
                info["nucleus"] = {"concentration": round(conc, 1), "fwhm": round(w, 2)}
    rgb += light


def reflection(rng, n: int, rgb: np.ndarray, info: dict):
    k = int(rng.integers(1, 6))
    field = np.zeros((n, n), np.float32)
    for _ in range(k):
        c = rng.uniform(0.15, 0.85, 2) * n
        field += float(rng.uniform(0.3, 1.0)) * blob(rng, n, float(c[0]), float(c[1]), float(rng.uniform(0.03, 0.15)) * n,
                                                     float(rng.uniform(0.3, 1.0)))
    field *= lognormal(power_field(rng, n, float(rng.uniform(3.0, 3.8))), float(rng.uniform(0.3, 0.8)))
    rgb += reflection_colour(rng)[:, None, None] * field[None]
    info["centres"] = k


def compact(rng, n: int, rgb: np.ndarray, info: dict):
    xx, yy = grid(n)
    cx, cy = rng.uniform(0.2, 0.8, 2) * n
    size = float(np.exp(rng.uniform(np.log(6), np.log(120))))
    if rng.random() < 0.5:                                       # a planetary nebula: a ring, teal inside, red rim
        q = float(rng.uniform(0.6, 1.0))
        a = rng.uniform(0, np.pi)
        dx, dy = xx - cx, yy - cy
        r = np.hypot(dx * np.cos(a) + dy * np.sin(a), (-dx * np.sin(a) + dy * np.cos(a)) / q) / size
        ring = np.exp(-((r - 1) / 0.25) ** 2)
        inside = np.exp(-(r / 0.9) ** 4)
        rgb += emission_colour(rng)[:, None, None] * ring[None] + oiii_colour(rng)[:, None, None] * (0.8 * inside)[None]
        rgb += emission_colour(rng)[:, None, None] * (0.03 * np.exp(-r / 2.5))[None]       # a faint halo
        info["planetary_px"] = round(size, 1)
    else:
        b = blob(rng, n, float(cx), float(cy), size, 0.6) * lognormal(power_field(rng, n, 3.2), 0.5)
        rgb += emission_colour(rng)[:, None, None] * b[None]
        info["nebula_px"] = round(size, 1)


def milky_way(rng, n: int) -> np.ndarray:
    mw = sigmoid(power_field(rng, n, 4.5, kmin=1.0) * 1.2) * lognormal(power_field(rng, n, 3.4), 0.35)
    return colour(1.1, 0.85)[:, None, None] * mw[None]


def dust(rng, n: int, info: dict, strength: float, lanes: bool = False) -> np.ndarray:
    """exp(-tau) per colour: globules, and with ``lanes`` mostly thin sharp lanes (NGC 7000's), redder where thicker."""
    t = lognormal(power_field(rng, n, float(rng.uniform(3.0, 3.8))), float(rng.uniform(0.8, 1.5)))
    if lanes:
        t = 0.3 * t + float(rng.uniform(2.0, 5.0)) * ridges(power_field(rng, n, float(rng.uniform(3.2, 3.8))),
                                                            float(rng.uniform(6, 16))) * t
    else:
        t *= 1 + float(rng.uniform(0.5, 3.0)) * ridges(power_field(rng, n, 3.0), float(rng.uniform(3, 8)))
    t *= sigmoid((power_field(rng, n, 4.5, kmin=1.5) + float(rng.uniform(-1.0, 0.5))) / 0.5)
    tau = t / np.percentile(t, 99) * strength
    info["dust_tau99"] = round(float(np.percentile(tau, 99)), 2)
    return np.exp(-tau[None] * np.array([0.75, 1.0, 1.3], np.float32)[:, None, None])


# ---------------------------------------------------------------------------------------------- one image
def background(rng, n: int) -> tuple[np.ndarray, dict]:
    kinds = list(KINDS)
    kind = kinds[rng.choice(len(kinds), p=list(KINDS.values()))]
    info: dict = {"kind": kind}
    rgb = np.zeros((3, n, n), np.float32)
    if kind == "emission":
        emission(rng, n, rgb, info, filled=rng.random() < 0.35)
    elif kind == "galaxy":
        main_size = float(np.exp(rng.uniform(np.log(0.02), np.log(0.25)))) * n
        c = rng.uniform(0.3, 0.7, 2) * n
        galaxy(rng, n, rgb, info, float(c[0]), float(c[1]), main_size, True)
        info["size_px"] = round(main_size, 1)
        for _ in range(int(rng.integers(0, 3))):                 # companions (Leo triplet, M51b)
            cc = rng.uniform(0.1, 0.9, 2) * n
            sub = np.zeros_like(rgb)
            galaxy(rng, n, sub, info, float(cc[0]), float(cc[1]), main_size * float(rng.uniform(0.2, 0.7)), True)
            rgb += sub * float(rng.uniform(0.2, 0.8)) * rgb.max() / max(float(sub.max()), 1e-9)
    elif kind == "reflection":
        reflection(rng, n, rgb, info)
    elif kind == "dark":                                         # a glow (the Milky Way, or a nebula) behind dust lanes
        if rng.random() < 0.5:
            rgb += milky_way(rng, n)
        else:
            emission(rng, n, rgb, info, filled=True)
        rgb *= dust(rng, n, info, float(rng.uniform(1.5, 4.0)), lanes=True)
    elif kind == "compact":
        compact(rng, n, rgb, info)
    else:
        cir = lognormal(power_field(rng, n, float(rng.uniform(3.0, 3.8))), 0.8)
        rgb += colour(1.0, 1.0)[:, None, None] * (cir * sigmoid(power_field(rng, n, 4.5, kmin=1.5)))[None]

    # the brightness: the 99.9th percentile of the luminance drawn around our images'
    top = float(np.clip(10 ** rng.normal(P999_LOG10[kind], P999_SD), *P999_RANGE))
    if kind == "empty":
        top = float(np.clip(top, 0.2, 1.5))
    L = np.tensordot(LUM, rgb, 1)
    rgb *= top / max(float(np.percentile(L, 99.9)), 1e-9)
    info["p999"] = round(top, 2)

    # background galaxies, a few px wide: they look like stars, and must stay
    if kind != "dark" and rng.random() < 0.5:
        m = int(rng.integers(1, 15))
        for _ in range(m):
            cc = rng.uniform(0, 1, 2) * n
            r0 = float(np.exp(rng.uniform(np.log(1.5), np.log(8))))
            x0, y0 = int(cc[0]), int(cc[1])
            h = int(10 * r0) + 2
            ys, xs = slice(max(0, y0 - h), min(n, y0 + h)), slice(max(0, x0 - h), min(n, x0 + h))
            sub = np.zeros((3, ys.stop - ys.start, xs.stop - xs.start), np.float32)
            loc = np.random.default_rng(int(rng.integers(1 << 30)))
            xx, yy = np.meshgrid(np.arange(xs.start, xs.stop), np.arange(ys.start, ys.stop))
            q, pa = float(loc.uniform(0.3, 1.0)), loc.uniform(0, np.pi)
            dx, dy = xx - cc[0], yy - cc[1]
            r = np.hypot(dx * np.cos(pa) + dy * np.sin(pa), (-dx * np.sin(pa) + dy * np.cos(pa)) / q)
            prof = sersic(r, r0, float(loc.uniform(0.8, 4.0)))
            sub += galaxy_colours(loc)[int(loc.integers(2))][:, None, None] * prof[None]
            rgb[:, ys, xs] += sub * float(np.exp(loc.uniform(np.log(0.5), np.log(8)))) / max(float(prof.max()), 1e-9)
        info["small_galaxies"] = m

    # dust in absorption across some of the other fields
    if kind in ("emission", "reflection") and rng.random() < 0.3:
        rgb *= dust(rng, n, info, float(rng.uniform(0.5, 2.0)))

    # the sky: a level near 0, a faint residual gradient
    xx, yy = grid(n)
    ga = float(rng.uniform(0, 0.3))
    ang = rng.uniform(0, 2 * np.pi)
    rgb += float(rng.uniform(-0.5, 0.0)) + ga * ((xx / n - 0.5) * np.cos(ang) + (yy / n - 0.5) * np.sin(ang))[None]
    info["gradient"] = round(ga, 2)
    return rgb.astype(np.float32), info


# ---------------------------------------------------------------------------------------------- previews
def preview(rgb: np.ndarray, black: float = -1.0, soft: float = 0.5, white: float = 300.0) -> np.ndarray:
    """One stretch for every image, generated or real (asinh, sigma units): the levels compare between images."""
    x = np.arcsinh(np.clip(rgb - black, 0, None) / soft) / np.arcsinh((white - black) / soft)
    return (np.clip(x, 0, 1).transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    import cv2
    from astropy.io import fits
    out = Path(a.out)
    (out / "previews").mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("bg_*.fits")) + list(out.glob("bg_*.json")) + list((out / "previews").glob("bg_*.jpg")):
        old.unlink()                                              # one version at a time in the folder
    for i in range(a.n):
        t = time.time()
        rng = np.random.default_rng([a.seed, i])
        rgb, info = background(rng, a.size)
        name = f"bg_{a.seed}_{i:04d}"
        fits.writeto(out / f"{name}.fits", rgb, overwrite=True)
        (out / f"{name}.json").write_text(json.dumps(info))
        cv2.imwrite(str(out / "previews" / f"{name}.jpg"), preview(rgb), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"{name} {time.time() - t:4.1f} s {info}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
