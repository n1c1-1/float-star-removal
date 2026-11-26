# ======================================================================
# resume_paires.py
# Rôle     : Résumé de paires.jsonl (diag_paires.py) par sorte : débruitées / brutes / web, tenues écartées.
# Étape    : diagnostic (25 nov) (voir scripts/LISEZMOI.md)
# Entrées  : paires et sorties de validation
# Sorties  : tables de diagnostic
# Exemple  : python resume_paires.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Résumé de paires.jsonl (diag_paires.py) par sorte : débruitées / brutes / web, tenues écartées. -> paires_resume.txt"""
import json, sys
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
OUT = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Diagnostic écarts – 25 nov")
R = [json.loads(l) for l in open(OUT / "paires.jsonl", encoding="utf-8")]
L = []
pb = [r["paire"] for r in R if "erreur" in r or "manque" in r]
L.append(f"{len(R)} paires lues ; sans mesure : {len(pb)} {pb[:12]}")
for sorte in ("debruitee", "brute", "web"):
    S = [r for r in R if r.get("sorte") == sorte and not r.get("tenue") and "g14_17" in r]
    rat = np.concatenate([r["g14_17"]["ratio"] for r in S]) if S else np.zeros(0)
    wc = np.concatenate([r["g14_17"]["w_centre"] for r in S]) if S else np.zeros(0)
    wm = np.concatenate([r["g14_17"]["w_moy"] for r in S]) if S else np.zeros(0)
    px = np.sum([r["g14_17"]["pix_w"] for r in S], 0) if S else np.zeros(3)
    g = rat > 0.1
    L.append(f"\n== {sorte} : {len(S)} paires, {len(rat)} étoiles G 14-17 au critère de la validation ==")
    if len(rat):
        L.append(f"  part > 10 % du flux restée dans la CIBLE, toutes : {g.mean():.3f}")
        for nom, k in (("poids centre 0", wc == 0), ("poids centre 0,5", (wc > 0) & (wc < 1)), ("poids centre 1", wc >= 1),
                       ("poids centre > 0", wc > 0)):
            L.append(f"  {nom:18s}: {k.mean():.3f} des étoiles ; part > 10 % : {g[k].mean() if k.any() else float('nan'):.3f} (n={int(k.sum())})")
        L.append(f"  étoiles restées (> 10 %) ET poids moyen de l'ouverture > 0 : {(g & (wm > 0)).mean():.3f} de toutes ; "
                 f"au poids moyen >= 0,5 : {(g & (wm >= 0.5)).mean():.3f}")
        L.append(f"  pixels des ouvertures : poids 0 {px[0] / px.sum():.3f} / 0,5 {px[1] / px.sum():.3f} / 1 {px[2] / px.sum():.3f}")
        L.append(f"  flux restant médian (poids centre > 0) : {np.median(rat[wc > 0]) if (wc > 0).any() else float('nan'):.4f} ; "
                 f"p90 {np.percentile(rat[wc > 0], 90) if (wc > 0).any() else float('nan'):.4f}")
    F = [r["fond"] for r in S if "decalage_bruits" in r.get("fond", {})]
    if F:
        dec = np.array([np.mean(f["decalage_bruits"]) for f in F])
        decc = np.array([max(abs(c - np.mean(f["decalage_bruits"])) for c in f["decalage_bruits"]) for f in F])
        def q(v):
            v = np.asarray(v, float)
            return f"méd {np.median(v):.3f} [p10 {np.percentile(v, 10):.3f} ; p90 {np.percentile(v, 90):.3f}]"
        L.append(f"  -- ciel loin des étoiles au poids > 0 ({len(F)} paires, en bruits haute fréquence de l'entrée) --")
        L.append(f"  décalage cible-entrée (luminance)  : {q(dec)} ; |décalage| > 0,1 bruit : {(np.abs(dec) > 0.1).mean():.2f} des paires")
        L.append(f"  écart de couleur (max canal)       : {q(decc)}")
        L.append(f"  |cible-entrée| médian              : {q([f['abs_med_bruits'] for f in F])}")
        L.append(f"  grain cible / entrée               : {q([f['grain_cible_sur_entree'] for f in F])}")
        L.append(f"  part des pixels changés            : {q([f['part_changes'] for f in F])}")
        L.append(f"  taches (écart lissé 4 px > 0,5 b.) : {q([f['taches_part'] for f in F])}")
        L.append(f"  part de ces pixels hors classe 0   : {q([f['classes_non0'] for f in F if f.get('classes_non0') is not None])}")
        pires = sorted(((abs(np.mean(r['fond']['decalage_bruits'])), r['paire']) for r in S if 'decalage_bruits' in r.get('fond', {})), reverse=True)[:5]
        L.append("  plus forts décalages : " + ", ".join(f"{p} {d:.2f}" for d, p in pires))
txt = "\n".join(L)
(OUT / "paires_resume.txt").write_text(txt, encoding="utf-8")
print(txt)
