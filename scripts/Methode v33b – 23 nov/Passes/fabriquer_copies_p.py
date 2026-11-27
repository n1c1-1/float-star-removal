# ======================================================================
# fabriquer_copies_p.py
# Role     : Builds passe_finale_p.py, passe_finale_t2_p.py, passe_finale_t3_p.py from exact copies of the originals (passe_finale.py, passe_finale_t2.py, passe_finale_t3.py, never modified).
# Step     : 2. Classic Gaia-guided subtraction chain (v33b) (see scripts/LISEZMOI.md)
# Inputs   : stacked image (noise units), astrometry, offline deep Gaia extract
# Outputs  : star-free image, class maps
# Example  : python fabriquer_copies_p.py [arguments]  (see the docstring; paths through the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Builds passe_finale_p.py, passe_finale_t2_p.py, passe_finale_t3_p.py from exact copies of the originals
(passe_finale.py, passe_finale_t2.py, passe_finale_t3.py, never modified). Each replacement must be found exactly
once (otherwise error): nothing is guessed. The computations are unchanged; only the paths, the crop's name,
the angles and the stars shown change (parameters of mv_cfg.py), and the fact that the t3 corrections tied to named positions of decoupe6 only apply to decoupe6."""
from pathlib import Path

D = Path(__file__).resolve().parent


def fab(src, dst, pairs):
    s = (D / src).read_text(encoding="utf-8")
    for old, new in pairs:
        assert s.count(old) == 1, (src, s.count(old), old[:80])
        s = s.replace(old, new)
    (D / dst).write_text(s, encoding="utf-8")


fab("passe_finale.py", "passe_finale_p.py", [
    ("import mesure as M\n", "import mv_cfg as CFG\nimport mesure as M\n"),
    ('BASE = HERE / "sorties" / "hybride_c9" / "rosette_decoupe6"\n', "BASE = CFG.A_OUT\n"),
    ('OUT = HERE / "sorties" / "passe_finale_c9"\n', 'OUT = CFG.B_ROOT / "t1"\n'),
    ('CASE = "rosette_decoupe6"\n', "CASE = CFG.CASE\n"),
    ("SIX = [(1706, 1009), (2043, 968), (2038, 1036), (1762, 1159), (2078, 1220), (1999, 1327)]\n", "SIX = CFG.SIX\n"),
    ("ANGLES = [3.0, 11.5, 183.0, 191.5]", "ANGLES = CFG.ANGLES"),
    ('with open(HERE / "sorties" / "base_piste1" / CASE / "c9_decoupe6_m200" / "etoiles.csv", encoding="utf-8") as f:',
     'with open(CFG.BASE_CSV, encoding="utf-8") as f:'),
])

fab("passe_finale_t2.py", "passe_finale_t2_p.py", [
    ("import passe_finale as pf\nfrom passe_finale import mad, F\n", "import passe_finale_p as pf\nfrom passe_finale_p import mad, F\n"),
    ('OUT = HERE / "sorties" / "passe_finale_c9_t2"\n', 'OUT = pf.CFG.B_ROOT / "t2"\n'),
    ('R1DIR = HERE / "sorties" / "passe_finale_c9"\n', 'R1DIR = pf.CFG.B_ROOT / "t1"\n'),
])

fab("passe_finale_t3.py", "passe_finale_t3_p.py", [
    ("import passe_finale as pf\nfrom passe_finale import mad, F\n", "import passe_finale_p as pf\nfrom passe_finale_p import mad, F\n"),
    ('OUT = HERE / "sorties" / "passe_finale_c9_t3"\n', 'OUT = pf.CFG.B_ROOT / "t3"\n'),
    ('T2 = HERE / "sorties" / "passe_finale_c9_t2"\n', 'T2 = pf.CFG.B_ROOT / "t2"\n'),
    ("FORCED = dict(magenta=(196, 320), vert=(165, 182), bleu=(62, 113), noeud_trou=(84, 96), noeud_point=(83, 90),\n"
     "              sombre1=(272, 154), sombre2=(332, 207))\n",
     "FORCED = dict(magenta=(196, 320), vert=(165, 182), bleu=(62, 113), noeud_trou=(84, 96), noeud_point=(83, 90),\n"
     "              sombre1=(272, 154), sombre2=(332, 207)) if pf.CFG.NAMED_FIXES else dict()   # positions nommees de decoupe6 seulement\n"),
    ("    knot = (np.hypot(xx - 84, yy - 96) < 14) & (np.abs(dog) > 3 * sd)\n",
     "    knot = ((np.hypot(xx - 84, yy - 96) < 14) & (np.abs(dog) > 3 * sd)) if pf.CFG.NAMED_FIXES else np.zeros((H, W), bool)   # noeud nomme de decoupe6\n"),
    ("    for nm, (cx, cy) in dict(x287_y200=(287, 200), x128_y416=(128, 416), x305_y300=(305, 300), x84_y96=(84, 96), x196_y320=(196, 320)).items():\n",
     "    for nm, (cx, cy) in (dict(x287_y200=(287, 200), x128_y416=(128, 416), x305_y300=(305, 300), x84_y96=(84, 96), x196_y320=(196, 320)) if pf.CFG.NAMED_FIXES else {}).items():\n"),
])
print("ok : passe_finale_p.py, passe_finale_t2_p.py, passe_finale_t3_p.py")
