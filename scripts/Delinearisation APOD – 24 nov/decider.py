# ======================================================================
# decider.py
# Role     : Kept / discarded decision per image, from controle\mesures.jsonl; writes controle\tableau.txt and décisions.json, deletes the .npy files of the discarded ones.
# Step     : 3. De-linearisation and astrometry of the APOD web images (24 Nov) (see scripts/LISEZMOI.md)
# Inputs   : downloaded APOD web images
# Outputs  : linear images, astrometry, tiles
# Example  : python decider.py [arguments]  (see the docstring; paths via the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Kept / discarded decision per image, from controle\mesures.jsonl; writes controle\tableau.txt and décisions.json,
deletes the .npy files of the discarded ones. Rules (thresholds calibrated on calibrer_taille.py: true linear stack, size slope 0.04-0.36)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from outils_apod import CTRL, LIN  # noqa: E402

lignes, dec = [], {}
for l in open(CTRL / "mesures.jsonl", encoding="utf-8"):
    r = json.loads(l)
    n, i, e, o = r["numero"], r.get("info", {}), r.get("etoiles_lin", {}), r.get("etoiles_orig", {})
    ps, rl = e.get("pente_taille"), e.get("r50_rapport")
    raisons, notes = [], []
    if "echec" in r or i.get("repli") or i.get("etoiles_utilisees", 0) < 30:
        raisons.append("methode echoue : trop peu d'etoiles pour tirer la courbe (%s utilisables), repli sur une MTF" % i.get("etoiles_utilisees"))
    else:
        if r["noir_2"] > 0.15 or r["noir_zero"] > 0.05:
            raisons.append("fond ecrase au noir (%.1f %% des pixels)" % (100 * r["noir_2"]))
        if r["ecrete_larges"] > 0.005:
            raisons.append("zone blanchie large (%.2f %% des pixels, plus grande tache %d px)" % (100 * r["ecrete_larges"], r["ecrete_plus_grand_px"]))
        if ps is not None and ps == ps and (ps < 0.06 or (ps < 0.10 and rl < 1.2)):
            raisons.append("etoiles reduites : la taille ne croit pas avec le flux (pente %.2f, rapport brillantes/faibles %.2f)" % (ps, rl))
        if r["verdict_orch"] == "douteuse" and e.get("n_etoiles", 0) < 60:
            raisons.append("douteuse et trop peu d'etoiles mesurables (%d) pour juger la taille" % e.get("n_etoiles", 0))
        if r["noir_2"] > 0.02: notes.append("fond sombre %.1f %%" % (100 * r["noir_2"]))
        if r["ecrete_plus_grand_px"] > 5000 or r["ecrete_larges"] > 0.0003: notes.append("zone ecretee locale (%d px) : poids 0" % r["ecrete_plus_grand_px"])
        if i.get("etoiles_utilisees", 999) < 150: notes.append("peu d'etoiles pour la courbe (%d)" % i.get("etoiles_utilisees"))
        if r["verdict_orch"] == "douteuse": notes.append("douteuse (passe le controle)")
        if i.get("cout_depart") is not None and i.get("cout_depart") == i.get("cout_fin"): notes.append("courbe = depart MTF, non affinee")
    garde = not raisons
    dec[n] = {"gardee": garde, "raison": "; ".join(raisons), "notes": "; ".join(notes)}
    if not garde and (LIN / f"{n:03d}.npy").exists():
        (LIN / f"{n:03d}.npy").unlink()
    lignes.append("%3d | %-8s | %-46s | %s | couleur %s, courbe %s>%s, etoiles courbe %s mesurees %s, FWHM %s px, taille b/f %s>%s pente %s, flux %s dec, noir %.1f%%, ecrete %.2f%% | %s" % (
        n, "GARDEE" if garde else "ECARTEE", r["fichier"][:46], r["verdict_orch"], i.get("couleur"), i.get("cout_depart"), i.get("cout_fin"),
        i.get("etoiles_utilisees"), e.get("n_etoiles"), e.get("fwhm_px"), "%.2f" % o.get("r50_rapport", float("nan")), "%.2f" % (rl if rl is not None else float("nan")),
        "%.2f" % (ps if ps is not None else float("nan")), "%.2f" % (e.get("decades_flux") or float("nan")), 100 * r["noir_2"], 100 * r["ecrete_larges"],
        dec[n]["raison"] or dec[n]["notes"] or "ras"))
ng = sum(d["gardee"] for d in dec.values())
tete = ["Delinearisation APOD 24 nov : %d images, %d gardees, %d ecartees (methode etoiles_auto du 23 nov)." % (len(dec), ng, len(dec) - ng),
        "colonnes : image | decision | fichier | verdict a l'oeil | mesures etoiles_auto | raison de l'ecart ou notes", ""]
(CTRL / "tableau.txt").write_text("\n".join(tete + lignes) + "\n", encoding="utf-8")
(CTRL / "decisions.json").write_text(json.dumps(dec, ensure_ascii=False, indent=1), encoding="utf-8")
print(ng, "gardees", len(dec) - ng, "ecartees")
