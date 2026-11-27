# ======================================================================
# real_targets.py
# Role     : Automatic targets for 23 real images from E3's output (a trial of the E3 period, unrelated to the v33 pairs).
# Step     : 1. Training on synthetic data only (E2 / E3, 7-9 Nov) (see scripts/README.md)
# Inputs   : generator: seeds + catalogue.json (public instrument sheets); no image
# Outputs  : synthetic pairs (input/target/info.json + manifest.json), network E3
# Example  : python real_targets.py [--names a,b] [--device cuda:0]
# ======================================================================
"""Targets for our 23 real training images: E3's output (its step 20000) corrected automatically, then checked by eye on the sheets written here.

    python real_targets.py [--names a,b] [--device cuda:0]

Each change is logged in report.json, with its place, so it can be checked:
1. dimples: in the star zones (the star layer above its surroundings), what the output lacks below the level around the zone is added, per colour, smoothed so that the grain stays;
2. stars left: a compact point of the output where the original has a star and the surroundings are flat takes the level around it (points standing on a glow, the nucleus of a galaxy, are left and logged);
   in the star zones the colour is that of the surroundings, the brightness kept;
3. giants: the same rule as stars.py (reach, the diameter inside which the star's own light exceeds 50 noise units): up to max(40, 8 x star width) taken off, 1.6 x that kept whole, progressive between;
4. the diffuse light taken off the sky takes the colour of the stars taken off, one value per colour for the image;
5. the extended objects that may have been taken for stars (a small galaxy, a knot of nebula) are only logged; those the check by eye confirms are listed in OBJECTS and come back from the original without their stars (6), or become a zone left out of the training.

Writes <ESSAIS>/starRemoval/real_targets/<image>/target.npy, report.json, sheet.jpg (and the crops of what was changed).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage as ndi

sys.path.insert(0, "<FLOAT>")
from models import load  # noqa: E402
from stars import REACH0, REACH1, REACH_LEVEL  # noqa: E402
from astro_gpu.stages import star_removal as SR  # noqa: E402

TRAIN = Path(__file__).resolve().parent / "data" / "train_with_stars"
NET = "<ESSAIS>/starRemoval/train/E3/snap/step_20000.pt"
OUT = (Path(__import__("os").environ.get("ESSAIS", "essais")) / "starRemoval" / "real_targets")
WIDTHS = "<ESSAIS>/A_REGARDER/geantes/controle_jeu.json"
LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)
DIMPLE_MAX = 0.5                 # noise units: a dimple is shallow; more than this beside a bright object is its glow, not a dimple
GRAIN = 0.01                     # noise units: what is not a dimple, the grain of the denoised sky
LEFT_MIN, ORIG_MIN = 0.4, 6.0    # noise units above the surroundings: a point left in the output, a star in the original
#: small objects the network took for stars, seen by eye: (y, x, radius)
OBJECTS = {"user_M31": [(3386, 3154, 130)], "set_16_dslr_lens_canon60d_m31_untracked": [(1704, 3093, 55)], "set_24_dslr_astrograph_d810a_eps180_m57": [(2403, 3770, 35)]}
#: the check by eye, place by place (one JSON per line: image, remedy FILL, RESTORE or EXCLUDE, y, x, r, why), and the images it drops whole
REMEDIES = (Path(__import__("os").environ.get("ESSAIS", "essais")) / "starRemoval" / "real_targets" / "remedies.jsonl")
DROPPED = ("set_38_smart_dwarfmini_ngc869", "set_55_dslr_refractor_canon40da_jellyfish")
ZONE = 0.12         # noise units above the surroundings: a star zone
MIN_AREA, WIDE_SHARE = 200, 0.03        # in the star layer, a part this large that keeps this share of its peak at 1.5-2.5 half-peak widths is no star (a star keeps 0.2 %)


def lum(a):
    return np.tensordot(LUM, a, 1)


def blur(a, s):
    return cv2.GaussianBlur(a, (0, 0), s)


def level_around(chan, out_of):
    """The level of a colour around the zones: a masked average, at the finest scale that still has surroundings to look at."""
    around = None
    for sigma in (32, 16, 8):
        w = blur(out_of, sigma)
        e = blur(chan * out_of, sigma) / np.maximum(w, 1e-6)
        around = e if around is None else np.where(w > 0.25, e, around)
    return around


def blown_level(L):
    return float(np.percentile(L[L > np.percentile(L, 99.9)], 99.9))


def ring_share(s, y, x, w50, p):
    """The share of a peak still there at 1.5 to 2.5 widths at half peak."""
    H, W = s.shape
    R = int(2.5 * w50) + 2
    if y < R or x < R or y >= H - R or x >= W - R:
        return 0.0
    yy, xx = np.ogrid[-R:R + 1, -R:R + 1]
    d = np.hypot(yy, xx)
    return float(np.median(s[y - R:y + R + 1, x - R:x + R + 1][(d >= 1.5 * w50) & (d < 2.5 * w50)])) / p


def giants(L, top, fwhm):
    """The stars whose reach passes the line: [(y, x, reach, share kept)]. The profile is taken around the blown objects; it has to be round (a nebula's bright part is not)."""
    H, W = L.shape
    lab, n = ndi.label(L > 0.8 * top)
    area = np.bincount(lab.ravel())
    area[0] = 0
    objs = ndi.find_objects(lab)
    start = max(REACH0, 8.0 * fwhm)
    full = start * REACH1 / REACH0
    found = []
    R, ST = 700, 2
    for l in [int(i) for i in np.argsort(-area)[:40] if area[i] >= 12]:
        s = objs[l - 1]
        cy, cx = (s[0].start + s[0].stop) // 2, (s[1].start + s[1].stop) // 2
        y0, y1, x0, x1 = max(0, cy - R), min(H, cy + R), max(0, cx - R), min(W, cx + R)
        c = L[y0:y1:ST, x0:x1:ST]
        yy, xx = np.mgrid[y0:y1:ST, x0:x1:ST]
        dist = np.hypot(yy - cy, xx - cx)
        k = (dist / 4).astype(int)
        prof = np.array([np.median(c[k == j]) if (k == j).sum() > 3 else np.nan for j in range(R // 4)])
        p = prof - np.nanmin(prof[:75])                            # the level around it (within 300 px): a bright nebula is not its light
        below = np.where(~(p > REACH_LEVEL))[0]
        reach = int(8 * below[0]) if len(below) else 2 * R
        share = float(np.clip((reach - start) / (full - start), 0, 1))
        if share <= 0:
            continue
        # round: the eight sectors of the ring between a quarter and a half of the reach
        ring = (dist >= reach / 4) & (dist < reach / 2)
        ang = ((np.arctan2(yy - cy, xx - cx) + np.pi) / (2 * np.pi) * 8).astype(int) % 8
        sky = float(np.nanmin(prof[:75]))
        sect =[np.median(c[ring & (ang == a)]) - sky for a in range(8) if (ring & (ang == a)).sum() > 3]
        if len(sect) < 8 or min(sect) < 0.35 * max(sect):
            continue
        found.append((cy, cx, reach, round(share, 2)))
    return found


def correct(img, o, fwhm, name=""):
    rep: dict = {}
    L0, Lo = lum(img), lum(o)
    s = lum(img - o)
    zone = ndi.binary_dilation(s - blur(s, 16) > ZONE, iterations=3)
    out_of = (~zone).astype(np.float32)
    soft = blur(zone.astype(np.float32), 1.0)
    # 1. dimples
    fine = max(0.7, 0.35 * fwhm)                                   # the width of the smoothing follows the stars': a wide one misses the pits of fine stars
    fixed = o.copy()
    for c in range(3):
        lack = np.maximum(blur(level_around(o[c], out_of) - o[c], fine) - GRAIN, 0)
        fixed[c] = o[c] + soft * lack * np.clip(2 - lack / DIMPLE_MAX, 0, 1)                 # deeper than a dimple can be: the glow of a bright object beside it, nothing to add
    rep["dimples"] = {"zones_percent": round(100 * float(zone.mean()), 1), "raised_mean": round(float(lum(fixed - o)[zone].mean()), 3)}
    # the diffuse light taken off the sky is unresolved stars: it has the colour of the stars taken off
    light = img - o
    per_col = np.array([float(light[c][zone].sum()) for c in range(3)], np.float32)
    ratio = per_col / max(per_col[1], 1e-6)
    veil = np.array([float(np.median(light[c][~zone])) for c in range(3)], np.float32)
    shift = veil - veil[1] * ratio                             # one value per colour for the image: a local one leaves stains
    fixed = fixed + shift[:, None, None]
    rep["veil"] = {"star_colour": np.round(ratio, 3).tolist(), "taken_off_median": np.round(veil, 3).tolist(), "shift": np.round(shift, 3).tolist()}
    # 2. stars left: a point where the original has a star, on a flat surrounding (the nucleus of a galaxy stands on its own glow, a knot on its nebula); twice, what one pass leaves next to a filled star shows in the second
    left, kept = [], []
    H, W = L0.shape
    orig = L0 - blur(L0, 6)
    r = int(max(4, round(3 * fwhm)))
    R = int(round(16 * fwhm))
    q = int(round(8 * fwhm))
    for _pass in range(2):
        Lf = lum(fixed)
        local = Lf - blur(Lf, 6)
        mx = ndi.maximum_filter(local, 7)
        ys, xs = np.nonzero((local == mx) & (local > LEFT_MIN) & (orig > ORIG_MIN))
        spots = np.zeros(Lf.shape, bool)
        for y, x in zip(ys.tolist(), xs.tolist()):
            y0, y1, x0, x1 = max(0, y - R), min(H, y + R + 1), max(0, x - R), min(W, x + R + 1)
            cut = Lf[y0:y1, x0:x1]
            yy, xx = np.ogrid[y0 - y:y1 - y, x0 - x:x1 - x]
            d = np.hypot(yy, xx)
            near, far = (d >= 2 * fwhm) & (d < 4 * fwhm), (d >= 10 * fwhm) & (d < 16 * fwhm)
            if float(np.median(cut[near]) - np.median(cut[far])) > 0.15 * float(local[y, x]):
                kept.append([y, x, round(float(local[y, x] / orig[y, x]), 2)])
                continue
            left.append([y, x, round(float(local[y, x] / orig[y, x]), 2)])
            # the shape of what is left (elongated, bitten), or a disc if it runs into its neighbours
            b0, b1, c0, c1 = max(0, y - q), min(H, y + q + 1), max(0, x - q), min(W, x + q + 1)
            lab, _ = ndi.label(local[b0:b1, c0:c1] > max(LEFT_MIN / 2, 0.04 * float(local[y, x])))
            shape = lab == lab[y - b0, x - c0]
            if shape[0].any() or shape[-1].any() or shape[:, 0].any() or shape[:, -1].any():
                yy, xx = np.ogrid[b0 - y:b1 - y, c0 - x:c1 - x]
                shape = np.hypot(yy, xx) <= r
            else:
                shape = ndi.binary_dilation(shape, iterations=max(3, int(round(fwhm))))
            spots[b0:b1, c0:c1] |= shape
        if not spots.any():
            break
        spot_soft = blur(spots.astype(np.float32), 1.0)
        away = (~ndi.binary_dilation(spots, iterations=2)).astype(np.float32)
        for c in range(3):
            fixed[c] = fixed[c] + spot_soft * np.minimum(level_around(fixed[c], away) - fixed[c], 0)      # only takes away: beside a bright object the level around is its glow
    # the coloured rings at the stars: in the zones, the colour of what surrounds them, the brightness kept
    round_ = np.stack([level_around(fixed[c], out_of) for c in range(3)])
    Lr, Lf = lum(round_), lum(fixed)
    calm = np.clip(1 - np.abs(Lr - Lf), 0, 1)                         # not beside a bright object, whose glow is no colour to copy
    for c in range(3):
        fixed[c] = fixed[c] + soft * calm * np.clip(blur((round_[c] - Lr) - (fixed[c] - Lf), 2.0), -DIMPLE_MAX, DIMPLE_MAX)
    rep["stars_left_filled"], rep["points_kept_unsure"] = left, kept
    # 3. giants
    top = blown_level(L0)
    gs = giants(L0, top, fwhm)
    rep["giants"] = []
    hp = s - blur(s, 16) > ZONE
    lab, _n = ndi.label(hp)
    for cy, cx, reach, share in gs:
        H, W = Lo.shape
        yy, xx = np.ogrid[:H, :W]
        d = np.hypot(yy - cy, xx - cx)
        w = np.clip((reach - d) / (reach / 2), 0, 1)                   # whole up to half the reach, nothing past it
        w = 0.5 - 0.5 * np.cos(np.pi * w)
        mine = lab[cy, cx]
        other = hp & (lab != mine) if mine else hp
        w = w.astype(np.float32) * (1 - blur(ndi.binary_dilation(other, iterations=2).astype(np.float32), 1.0))    # the other stars around it stay off
        fixed = fixed + share * w * (img - fixed)
        rep["giants"].append({"y": cy, "x": cx, "reach": reach, "share": share})
    # 5. extended objects taken for stars
    rep["extended"] = []
    smooth = blur(s, 1.0)
    lab2, n = ndi.label(smooth - blur(s, 16) > ZONE)
    for l, sl in enumerate(ndi.find_objects(lab2), 1):
        cut = lab2[sl] == l
        if cut.sum() < 30:
            continue
        v = smooth[sl][cut]
        p = float(v.max())
        if p < 10 or cut.sum() < MIN_AREA or float(L0[sl][cut].max()) >= 0.8 * top:
            continue
        box = smooth[sl]
        if int(((box == ndi.maximum_filter(box, 5)) & cut & (box > 0.2 * p)).sum()) > 2:
            continue                                                         # several stars run together, not one object
        w50 =2 * float(np.sqrt(int((v > 0.5 * p).sum()) / np.pi))
        i = int(np.argmax(v))
        cy, cx = sl[0].start + int(np.nonzero(cut)[0][i]), sl[1].start + int(np.nonzero(cut)[1][i])
        share = ring_share(s, cy, cx, w50, p)
        if share < WIDE_SHARE:
            continue
        if any(np.hypot(cy - g["y"], cx - g["x"]) < g["reach"] for g in rep["giants"]):
            continue
        rep["extended"].append({"y": cy, "x": cx, "width_at_half": round(w50, 1), "peak": round(p, 1), "share": round(share, 3)})
    # 6. the objects listed by the check by eye: back from the original, without the stars in front of them
    rep["objects"] = []
    H, W = L0.shape
    todo = [json.loads(line) for line in REMEDIES.read_text().splitlines() if line.strip()] if REMEDIES.exists() else []
    todo = [t for t in todo if t["image"] == name]
    for y, x, r in OBJECTS.get(name, []) + [(t["y"], t["x"], t["r"]) for t in todo if t["remedy"] == "RESTORE"]:
        y0, y1, x0, x1 = max(0, y - r), min(H, y + r + 1), max(0, x - r), min(W, x + r + 1)
        size = 2 * int(round(3 * fwhm)) + 1
        free = np.stack([ndi.median_filter(img[c, y0:y1, x0:x1], size=size) for c in range(3)])
        yy, xx = np.ogrid[y0 - y:y1 - y, x0 - x:x1 - x]
        d = np.hypot(yy, xx).astype(np.float32)
        core = np.clip((3 * fwhm - d) / fwhm, 0, 1)                    # the nucleus: the original itself, a median would flatten it
        free = free + core * (img[:, y0:y1, x0:x1] - free)
        edge = np.clip((r - d) / (0.3 * r), 0, 1)
        fixed[:, y0:y1, x0:x1] = fixed[:, y0:y1, x0:x1] + edge * (free - fixed[:, y0:y1, x0:x1])
        rep["objects"].append({"y": y, "x": x, "r": r})
    # 7. the places listed by the check by eye: the level around them
    fills = [t for t in todo if t["remedy"] == "FILL"]
    if fills:
        mask = np.zeros(L0.shape, bool)
        for t in fills:
            y0, y1, x0, x1 = max(0, t["y"] - t["r"]), min(H, t["y"] + t["r"] + 1), max(0, t["x"] - t["r"]), min(W, t["x"] + t["r"] + 1)
            yy, xx = np.ogrid[y0 - t["y"]:y1 - t["y"], x0 - t["x"]:x1 - t["x"]]
            mask[y0:y1, x0:x1] |= np.hypot(yy, xx) <= t["r"]
        fill_soft = blur(mask.astype(np.float32), 1.0)
        away = (~ndi.binary_dilation(mask, iterations=2)).astype(np.float32)
        for c in range(3):
            fixed[c] = fixed[c] + fill_soft * (level_around(fixed[c], away) - fixed[c])
    rep["remedies"] = {"fill": len(fills), "excluded_discs": [[t["y"], t["x"], t["r"]] for t in todo if t["remedy"] == "EXCLUDE"]}
    return fixed, rep


def view(a, st):
    return st(a)


def sheet(img, o, fixed, rep, path):
    from apply_real import Stretch, label
    st = Stretch(img)
    H, W = img.shape[1:]
    scale = max(1, max(H, W) // 900)

    def small(a, title):
        return label(st(a[:, ::scale, ::scale]), title)

    diff = lum(fixed - o)
    d = np.clip(np.abs(diff[::scale, ::scale]) / 2.0, 0, 1)
    d = (np.stack([d, d, d], -1) * 255).astype(np.uint8).copy()
    top = np.vstack([np.hstack([small(img, "original"), small(o, "E3 step 20000")]), np.hstack([small(fixed, "target (corrected)"), label(d, "what the correction changed (x0.5/unit)")])])
    S = 256
    places = [("giant", g["y"], g["x"]) for g in rep["giants"]] + [("extended", e["y"], e["x"]) for e in rep["extended"]]
    places += [("star left", p[0], p[1]) for p in sorted(rep["stars_left_filled"], key=lambda q: -q[2])[:6]]
    places += [("unsure, kept", p[0], p[1]) for p in sorted(rep["points_kept_unsure"], key=lambda q: -q[2])[:6]]
    rows = []
    for kind, y, x in places[:20]:
        y0, x0 = int(np.clip(y - S // 2, 0, H - S)), int(np.clip(x - S // 2, 0, W - S))
        cut = lambda a: st(a[:, y0:y0 + S, x0:x0 + S])
        rows.append(np.hstack([label(cut(img), f"{kind} {y},{x}"), cut(o), cut(fixed)]))
    W0 = top.shape[1]
    if rows:
        grid = [np.hstack(rows[i:i + 2]) if len(rows[i:i + 2]) == 2 else np.hstack([rows[i], np.zeros_like(rows[i])]) for i in range(0, len(rows), 2)]
        crops = np.vstack(grid)
        crops = cv2.resize(crops, (W0, int(crops.shape[0] * W0 / crops.shape[1])), interpolation=cv2.INTER_AREA)
        top = np.vstack([top, crops])
    cv2.imwrite(str(path), top, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--names", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--outputs-only", action="store_true", help="only run the network and keep its outputs")
    a = ap.parse_args(argv)
    dev = torch.device(a.device)
    net = load(NET, a.device)[0]
    meas = {r["name"]: r for r in json.load(open(WIDTHS)) if r["group"] == "train"}
    only = set(a.names.split(",")) if a.names else None
    for d in sorted(p for p in TRAIN.iterdir() if (p / "input.npy").exists()):
        if only and d.name not in only:
            continue
        img = np.asarray(np.load(d / "input.npy"), np.float32)
        out = OUT / d.name
        out.mkdir(parents=True, exist_ok=True)
        if (out / "output.npy").exists():                         # the network's output is kept: only the corrections change between rounds
            o = np.load(out / "output.npy")
        else:
            o = SR.apply(net, img, dev, None, SR.TILE, SR.OVERLAP)
            np.save(out / "output.npy", o.astype(np.float32))
        if a.outputs_only:
            print(d.name, "output kept", flush=True)
            continue
        fwhm = float(meas[d.name]["fwhm"])
        fixed, rep = correct(img, o, fwhm, d.name)
        rep["fwhm"] = fwhm
        np.save(out / "target.npy", fixed.astype(np.float32))
        (out / "report.json").write_text(json.dumps(rep))
        sheet(img, o, fixed, rep, out / "sheet.jpg")
        print(d.name, f"giants {len(rep['giants'])}, extended {len(rep['extended'])}, stars left filled {len(rep['stars_left_filled'])}, unsure kept {len(rep['points_kept_unsure'])}, dimple zones {rep['dimples']['zones_percent']} %", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
