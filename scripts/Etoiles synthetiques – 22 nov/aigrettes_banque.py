# ======================================================================
# aigrettes_banque.py
# Role     : Round 6 (1): spike angles of a bank, new detection, and diagnostic on the Newtons.
# Step     : 4. Realistic synthetic stars (22 Nov) (see scripts/LISEZMOI.md)
# Inputs   : starless backgrounds (pair targets), star shape banks
# Outputs  : pairs (input, target), sheets, blind test
# Example  : python aigrettes_banque.py diag      -> before / after table on the Newton banks, sheet Synth_19
# ======================================================================
"""Round 6 (1): spike angles of a bank, new detection, and diagnostic on the Newtons.

angles_v2(formes, fw): for EACH bright star of the bank, luminance divided by its round profile (per ring) in
the ring [max(3 FWHM, 2 x plateau radius, 8 px), 140 px]; MEAN angular profile (1°) over all the bright stars;
smoothed (1.5°); peaks more than 4 noise units above the median, and a twin at 180° (±5°) more than 2 noise units.
The old detection (banque_individuelle.angles_aigrettes): median of the shapes, 3 FWHM-60 px, 6 noise units, twin
at the same threshold.

    python aigrettes_banque.py diag      -> before / after table on the Newton banks, sheet Synth_19
    python aigrettes_banque.py ecrire    -> rewrites the angles (key “angles”) of all the banque2_*.npz banks,
                                            old set kept under “angles_v1”
"""
from __future__ import annotations

import os as _os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    _os.environ[_k] = "2"                                  # au plus 2 fils

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import banque_individuelle as BI                                     # noqa: E402

#: banques de Newton (instrument lu dans les notes de la bibliothèque et les fiches des 4 images), nombre attendu
#: d'aigrettes : lu à l'œil sur les vraies étoiles (planches d'audit, fiches) ; « ? » = non vérifié
NEWTON = {"rosette": 4, "m8_sho": 6, "m57": 4, "horsehead": 4, "E006": 4, "E011": "?", "S002": "?", "T002": "?",
          "E066": "?", "E063": "?"}


def profil_angulaire(formes, fw, par_etoile=False, r_min=None):
    """(360,) profil angulaire relatif moyen des brillantes, et le nombre d'étoiles utilisées (par_etoile : aussi la
    liste des profils de chaque étoile)."""
    n, _, S, _ = formes.shape
    h = S // 2
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1].astype(np.float32)
    rr = np.hypot(xx, yy)
    ri = np.rint(rr).astype(int).ravel()
    th = (np.degrees(np.arctan2(yy, xx)) % 360).astype(int).ravel()
    acc = np.zeros(360); cnt = 0
    chacun = []
    for k in range(n):
        L = formes[k].astype(np.float32).sum(0)
        npl = int(np.sum(L >= 0.98 * L.max()))
        r_in = max(3 * fw, 2 * math.sqrt(npl / math.pi), 8.0, r_min or 0.0)
        r_out = min(140.0, 0.7 * h)
        if r_out <= r_in + 10:
            continue
        v = L.ravel()
        m1 = np.bincount(ri, v, minlength=h + 2) / np.maximum(np.bincount(ri, minlength=h + 2), 1)
        ok = v <= 2 * np.maximum(m1[ri], 0) + 1e-30
        m2 = np.bincount(ri[ok], v[ok], minlength=h + 2) / np.maximum(np.bincount(ri[ok], minlength=h + 2), 1)
        sel = (rr.ravel() >= r_in) & (rr.ravel() <= r_out) & (m2[ri] > 0)
        rel = np.clip(v[sel] / m2[ri][sel] - 1, -1, 5)
        prof = np.bincount(th[sel], rel, minlength=360) / np.maximum(np.bincount(th[sel], minlength=360), 1)
        acc += prof; cnt += 1
        if par_etoile:
            chacun.append(prof)
    if par_etoile:
        return (acc / max(cnt, 1)), cnt, chacun
    return (acc / max(cnt, 1)), cnt


def angles_v2(formes, fw, k_pic=4.0, k_jum=2.0, rendre_profil=False):
    prof, cnt = profil_angulaire(formes, fw)
    g = np.exp(-0.5 * (np.arange(-6, 7) / 1.5) ** 2); g /= g.sum()
    sm = np.convolve(np.r_[prof[-6:], prof, prof[:6]], g, mode="same")[6:-6]
    base = float(np.median(sm)); nz = 1.4826 * float(np.median(np.abs(sm - base))) + 1e-12
    pk = [i for i in range(360) if sm[i] > base + k_pic * nz
          and sm[i] == max(sm[(i + d) % 360] for d in range(-4, 5))]
    out = []
    for p_ in sorted(pk, key=lambda i: -sm[i]):
        if all(abs((p_ - q + 180) % 360 - 180) > 4 for q in out):
            out.append(p_)
    # jumelle à 180° : un pic de part et d'autre, l'autre côté au moins à k_jum bruits
    ang = []
    for p_ in out:
        tw = [(p_ + 180 + d) % 360 for d in range(-5, 6)]
        if max(sm[t] for t in tw) > base + k_jum * nz:
            ang.append(p_)
            j = max(tw, key=lambda t: sm[t])
            if all(abs((j - q + 180) % 360 - 180) > 4 for q in ang + out):
                ang.append(j)
    ang = sorted(set(float(a) + 0.5 for a in ang))[:8]
    if rendre_profil:
        return ang, sm, base, nz, cnt
    return ang


def _paire_vue(formes, fw, teeth):
    """Nombre d'étoiles dont le profil angulaire propre a les deux dents (±4°) à plus de 3 bruits."""
    _, cnt, chacun = profil_angulaire(formes, fw, par_etoile=True)
    g = np.exp(-0.5 * (np.arange(-6, 7) / 1.5) ** 2); g /= g.sum()
    n = 0
    for pr in chacun:
        sm = np.convolve(np.r_[pr[-6:], pr, pr[:6]], g, mode="same")[6:-6]
        b0 = float(np.median(sm)); nz = 1.4826 * float(np.median(np.abs(sm - b0))) + 1e-12
        if all(max(sm[(j + d) % 360] for d in range(-4, 5)) > b0 + 3 * nz for j in teeth):
            n += 1
    return n


def _loin(formes, fw, teeth):
    """La paire se voit-elle encore au-delà de 10 FWHM (une aigrette va loin, un filé reste court) ?"""
    pr, cnt = profil_angulaire(formes, fw, r_min=10 * fw)
    if not cnt:
        return False
    b0 = float(np.median(pr)); nz = 1.4826 * float(np.median(np.abs(pr - b0))) + 1e-12
    return all(max(pr[(j + d) % 360] for d in range(-4, 5)) > b0 + 3 * nz for j in teeth)


def _allongement_coeur(formes, fw):
    """1 - b/a du cœur moyen (moments d'ordre 2 dans r <= 2 FWHM, au-dessus de 50 % du pic)."""
    L = np.mean(formes.astype(np.float32), axis=0).sum(0)
    h = L.shape[0] // 2
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1].astype(np.float32)
    w = np.where((np.hypot(xx, yy) <= 2 * fw) & (L > 0.5 * L.max()), L, 0)       # le cœur seul (à 10 % du pic, le pied des aigrettes l'allongeait)
    sw = w.sum()
    mx, my = (w * xx).sum() / sw, (w * yy).sum() / sw
    sxx = (w * (xx - mx) ** 2).sum() / sw; syy = (w * (yy - my) ** 2).sum() / sw; sxy = (w * (xx - mx) * (yy - my)).sum() / sw
    tr, det = sxx + syy, sxx * syy - sxy ** 2
    l1 = tr / 2 + math.sqrt(max(tr * tr / 4 - det, 0)); l2 = tr / 2 - math.sqrt(max(tr * tr / 4 - det, 0))
    return 1 - math.sqrt(max(l2, 1e-12) / max(l1, 1e-12))


def angles_v3(formes, fw, k_dent=2.0, rendre_profil=False):
    """Peigne régulier : une araignée donne des aigrettes également espacées (2 : barre ; 4 ; 6 : 3 branches). Pour
    n = 6, 4, 2 : l'origine qui maximise la somme des dents (chaque dent prise au maximum à ±4°) ; on garde le plus
    grand n dont TOUTES les dents dépassent k_dent bruits (tour 6 : un seuil de pic seul manquait les dents faibles
    de la Rosette et prenait un reste de S002 pour une aigrette)."""
    prof, cnt = profil_angulaire(formes, fw)
    g = np.exp(-0.5 * (np.arange(-6, 7) / 1.5) ** 2); g /= g.sum()
    sm = np.convolve(np.r_[prof[-6:], prof, prof[:6]], g, mode="same")[6:-6]
    base = float(np.median(sm)); nz = 1.4826 * float(np.median(np.abs(sm - base))) + 1e-12
    best = []
    for n in (6, 4, 2):
        sp = 360 / n
        cands = []
        for t0 in range(int(sp)):
            teeth = []
            for k in range(n):
                c = int(round(t0 + k * sp))
                j = max(((c + d) % 360 for d in range(-4, 5)), key=lambda t: sm[t])
                teeth.append(j)
            cands.append((sum(sm[j] for j in teeth), teeth))
        tot, teeth = max(cands, key=lambda c: c[0])
        ok = min(sm[j] for j in teeth) > base + k_dent * nz and max(sm[j] for j in teeth) > base + 4 * nz
        if ok and n == 2:
            # une seule paire : il faut des dents fortes ET fines (un filé ou une étoile allongée donne une bosse large :
            # S007, 200 mm non suivi ; T001 ; E051, lunette)
            def largeur(j):
                hm = base + 0.5 * (sm[j] - base)
                w = 1
                for sgn in (-1, 1):
                    d = 1
                    while d < 30 and sm[(j + sgn * d) % 360] > hm:
                        d += 1
                    w += d - 1
                return w
            ok = min(sm[j] for j in teeth) > base + 4 * nz and max(largeur(j) for j in teeth) <= 8
            if not ok and min(sm[j] for j in teeth) > base + k_dent * nz:
                # tour 7 : une paire VUE SUR PLUSIEURS ÉTOILES (>= 3 et >= 30 %), cœurs non allongés (un filé
                # allonge le cœur, une aigrette non) : T002 et S023, paire horizontale visible sur leurs piles
                vu = _paire_vue(formes, fw, teeth)
                ok = vu >= max(3, 0.3 * cnt) and (_allongement_coeur(formes, fw) < 0.15 or _loin(formes, fw, teeth))
        if ok:
            best = teeth
            break
    ang = sorted(float(a) + 0.5 for a in best)
    if rendre_profil:
        return ang, sm, base, nz, cnt
    return ang


def diag():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for nom, att in NEWTON.items():
        b = BI.charger(nom)
        if b is None:
            continue
        F = b["formes"]
        v1 = BI.angles_aigrettes(F.astype(np.float32), b["fwhm"])
        v2, sm, base, nz, cnt = angles_v3(F, b["fwhm"], rendre_profil=True)
        rows.append((nom, att, b, v1, v2, sm, base, nz, cnt))
        print(f"{nom:10} attendu {att} | avant {len(v1)} {[round(a) for a in v1]} | après {len(v2)} "
              f"{[round(a) for a in v2]} | {cnt} étoiles, fwhm {b['fwhm']:.2f}", flush=True)
    show = [r for r in rows if r[0] in ("S002", "E063", "rosette", "m8_sho")]
    fig, axs = plt.subplots(len(show), 2, figsize=(13, 4.2 * len(show)), gridspec_kw=dict(width_ratios=[1, 1.6]))
    for i, (nom, att, b, v1, v2, sm, base, nz, cnt) in enumerate(show):
        F = b["formes"].astype(np.float32)
        L = np.mean(F, axis=0).sum(0)
        h = L.shape[0] // 2
        z = 140
        c = L[h - z:h + z + 1, h - z:h + z + 1]
        v = np.arcsinh(np.clip(c, 0, None) / (np.percentile(c, 60) + 1e-30)) / np.arcsinh(c.max() / (np.percentile(c, 60) + 1e-30))
        axs[i, 0].imshow(v, cmap="gray")
        for angs, col, rr0 in ((v1, "red", 0.55), (v2, "lime", 0.95)):
            for a in angs:
                t = math.radians(a)
                axs[i, 0].plot([z + 30 * math.cos(t), z + rr0 * z * math.cos(t)],
                               [z + 30 * math.sin(t), z + rr0 * z * math.sin(t)], color=col, lw=1.2)
        axs[i, 0].set_title(f"{nom} : moyenne des {cnt} brillantes (rouge avant, vert après)", fontsize=9)
        axs[i, 0].set_xticks([]); axs[i, 0].set_yticks([])
        ax = axs[i, 1]
        ax.plot(np.arange(360), sm, "k-", lw=1)
        ax.axhline(base + 4 * nz, color="lime", ls="--", lw=0.8, label="dent la plus forte (4 bruits)")
        ax.axhline(base + 2 * nz, color="lime", ls=":", lw=0.8, label="chaque dent (2 bruits)")
        for a in v1:
            ax.axvline(a, color="red", lw=0.8, alpha=0.7)
        for a in v2:
            ax.axvline(a, color="lime", lw=1.6, alpha=0.6)
        ax.set_xlim(0, 360); ax.set_xlabel("angle (deg, 0 = +x, sens image)")
        ax.set_title(f"profil angulaire relatif moyen ; avant {len(v1)} aigrettes, après {len(v2)}", fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle("Aigrettes des Newton : détection avant / après (tour 6)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.canvas.draw()
    out = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    import audit_etoiles as A
    ok, buf = cv2.imencode(".png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR)); assert ok
    buf.tofile(str(A.OUT / "Synth_19_aigrettes_newton.png"))


def ecrire():
    for f in sorted(BI.OUTB.glob("banque2_*.npz")):
        z = np.load(f, allow_pickle=True)
        m = json.loads(str(z["meta"]))
        F = z["formes"]
        m.setdefault("angles_v1", m.get("angles", []))
        m["angles"] = angles_v3(F, m["fwhm"])
        np.savez_compressed(f, formes=F, meta=json.dumps(m))
        print(f.stem, len(m["angles_v1"]), "->", len(m["angles"]), [round(a) for a in m["angles"]], flush=True)


if __name__ == "__main__":
    {"diag": diag, "ecrire": ecrire}[sys.argv[1] if len(sys.argv) > 1 else "diag"]()
