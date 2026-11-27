# ======================================================================
# images_validation_block.py
# Role     : Block appended at the end of Piste1/images.py of a COPY of the chain v33b: registers the 13 validation images.
# Stage    : 15. Classical baseline versus networks on 512 px tiles (26-27 Nov) (see scripts/README.md)
# Inputs   : Validation – 22 nov\liste.py; astrometrie.npz and gaia_dr3_profond.npz copied from the validation
# Outputs  : entries in IMAGES (nothing is written)
# Example  : paste at the end of Piste1/images.py of the chain copy (not run on its own)
# ======================================================================
# Block appended at the end of Piste1/images.py of a COPY of the chain v33b (Methode v33b – 23 nov): registers the validation images.
# The rest of images.py is unchanged.


# 26 Nov (article baseline): the 13 validation images of the network (Validation – 22 nov\liste.py), read in place (same inputs as
# the validation). Their astrometry (Piste1/sorties/<name>/astrometrie.npz) and their deep Gaia (Catalogue Gaia profond – 20 nov/
# gaia_dr3_profond_<name>.npz) are COPIES of F:/float_ia/Validation étoiles – 22 nov/<name>/ (same format as set 2). Kind: "filtres"
# for the two assembled mono narrow-band images, "couleur" for the rest (rule of set 2).
def _validation():
    import sys
    sys.path.insert(0, r"C:\Users\Nicolas\sandbox\starRemoval\Validation – 22 nov")
    from liste import LISTE
    for nom, v in LISTE.items():
        IMAGES[nom] = (Path(v[0]) / "input.npy", v[1], "filtres" if "mono" in nom else "couleur")


_validation()
