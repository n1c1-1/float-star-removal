# ======================================================================
# corriger_mise_a_jour.py
# Rôle     : Module de la chaîne (voir sa docstring ci-dessous et scripts/LISEZMOI.md).
# Étape    : 2. Chaîne classique de soustraction guidée par Gaia (v33b) (voir scripts/LISEZMOI.md)
# Entrées  : image empilée (unités de bruit), astrométrie, extrait Gaia profond hors ligne
# Sorties  : image sans étoiles, cartes de classes
# Exemple  : python corriger_mise_a_jour.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
# Réécrit la mise à jour du modèle dans Piste1/modele_psf_image.py (poids plafonnés, tranches à parts égales dans le terme
# régional, terme régional sans sa partie radiale) ; retire le bloc de débogage. Lancé une fois.
from pathlib import Path
p = Path(__file__).resolve().parents[1] / "Piste1" / "modele_psf_image.py"
L = p.read_text(encoding="utf-8-sig").split("\n")
idx = lambda s: next(i for i, l in enumerate(L) if s in l)
i = idx("Sr = np.zeros((C, nb_, nrb)); Wr = np.zeros((C, nb_, nrb))")
L[i] = "        rad = []                         # par étoile et couleur : (couleur, tranche, région, somme e, nombre, poids)"
i = idx("Sr[c, bsel[i_]] += np.bincount")
L[i] = "                rad.append((c, bsel[i_], reg[i_], np.bincount(ri[ok], weights=v[ok] / F[c], minlength=nrb), np.bincount(ri[ok], minlength=nrb), w))"
del L[i + 1]
i = idx("Ft_list = np.array(Ft_list)")
L[i + 1:i + 1] = [
    "        # poids F^2 / bruit^2 plafonnés à 4 x la médiane de la tranche (sinon 2 ou 3 étoiles très brillantes font la moyenne)",
    "        wcap = {}",
    "        for c in range(C):",
    "            for b in range(nb_):",
    "                ws = [t_[5] for t_ in rad if t_[0] == c and t_[1] == b]",
    "                wcap[c, b] = 4.0 * float(np.median(ws)) if ws else 0.0",
    "        Sr = np.zeros((C, nb_, nrb)); Wr = np.zeros((C, nb_, nrb))",
    "        Wcell = np.zeros((C, ng, nb_))",
    "        for c, b, g_, sk_, nk_, w in rad:",
    "            we = min(w, wcap[c, b])",
    "            Sr[c, b] += we * sk_; Wr[c, b] += we * nk_",
    "        for c, g_, b_, idx_, r_, e_, w in samples:",
    "            Wcell[c, g_, b_] += min(w, wcap[c, b_]) * len(e_)",
]
i = idx("for c, g_, b_, idx, r_, e_, w in samples:")
L[i] = "        for c, g_, b_, idx, r_, e_, w in samples:"
L[i + 1] = ("            e2 = e_ - np.interp(r_, rbins, drho[c, b_])\n"
            "            w = min(w, wcap[c, b_]) / max(Wcell[c, g_, b_], 1e-30)      # chaque tranche compte autant dans le terme régional")
i = idx("D[accN[c, g_].reshape(n, n) < 3] = 0")
L[i] = ("                low = accN[c, g_].reshape(n, n) < 3\n"
        "                D[low] = 0\n"
        "                # terme régional = forme NON radiale seulement (la partie radiale est celle des tranches)\n"
        "                lab_ = np.digitize(rr, np.arange(0, R + 1.01, 0.25))\n"
        "                mr_ = np.asarray(ndi.mean(np.where(low, np.nan, D), labels=lab_, index=np.arange(1, lab_.max() + 1)), float)\n"
        "                D = np.where(low, 0.0, D - np.nan_to_num(mr_)[lab_ - 1])")
i0 = idx('if c == 1 and g_ == ng // 2 and __import__("os").environ.get("MPI_DBG"):')
i1 = idx("S[c, g_] += D")
del L[i0:i1]
p.write_text("\n".join(L), encoding="utf-8")
print("ok")
