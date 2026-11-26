# ======================================================================
# tranches_fines.py
# Rôle     : Module de la chaîne (voir sa docstring ci-dessous et scripts/LISEZMOI.md).
# Étape    : 2. Chaîne classique de soustraction guidée par Gaia (v33b) (voir scripts/LISEZMOI.md)
# Entrées  : image empilée (unités de bruit), astrométrie, extrait Gaia profond hors ligne
# Sorties  : image sans étoiles, cartes de classes
# Exemple  : python tranches_fines.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
# Passe Piste1/modele_psf_image.py à des profils radiaux par demi-magnitude, interpolés en G pour chaque étoile (la forme
# change aussi à l'intérieur d'une tranche de 1-1,5 mag) ; le contrôle reste sur les 4 tranches de RECUL. Lancé une fois.
from pathlib import Path
p = Path(__file__).resolve().parents[1] / "Piste1" / "modele_psf_image.py"
t = p.read_text(encoding="utf-8-sig")


def rep(a, b, n=1):
    global t
    assert t.count(a) == n, (t.count(a), a)
    t = t.replace(a, b)


rep('VERSION = "v7-1"', 'VERSION = "v7-2"')
rep('EDGES = np.array([-np.inf, 12.0, 13.5, 14.5, 15.5, 16.5, np.inf])     # tranches d\'éclat G\n'
    'CHECK_BINS = (1, 2, 3, 4)      # tranches contrôlées : 12-13.5, 13.5-14.5, 14.5-15.5, 15.5-16.5',
    'EDGES = np.r_[-np.inf, 11.0, np.arange(11.5, 17.01, 0.5), np.inf]   # tranches du modèle : demi-magnitudes G\n'
    'CHECK = ((12.0, 13.5), (13.5, 14.5), (14.5, 15.5), (15.5, 16.5))     # tranches contrôlées (celles de RECUL)')
rep('PER_BIN_MAX = int(__import__("os").environ.get("MPI_N", 700))', 'PER_BIN_MAX = int(__import__("os").environ.get("MPI_N", 250))')
rep('''def bin_of(G):
    return np.clip(np.searchsorted(EDGES, np.asarray(G, float), side="right") - 1, 0, len(EDGES) - 2)
''', '''def bin_of(G):
    return np.clip(np.searchsorted(EDGES, np.asarray(G, float), side="right") - 1, 0, len(EDGES) - 2)


def check_of(G):
    for i, (a, b) in enumerate(CHECK):
        if a <= G < b:
            return i
    return -1


def interp_rho(rho_c, gc, G):
    """Profil d'une couleur pour l'éclat G : interpolation linéaire entre les centres des tranches (bornée aux extrêmes)."""
    if G is None:
        return rho_c[-1]
    j = int(np.clip(np.searchsorted(gc, G) - 1, 0, len(gc) - 2))
    t_ = float(np.clip((G - gc[j]) / max(gc[j + 1] - gc[j], 1e-6), 0, 1))
    return (1 - t_) * rho_c[j] + t_ * rho_c[j + 1]
''')
rep('''    def __init__(self, S, rho, centres, os_, R, halo, R_ancien):
        C, ng = S.shape[:2]
        super().__init__([[S[c, g] for g in range(ng)] for c in range(C)], centres, os_, R, halo)
        self.rho = rho
        self.R_ancien = R_ancien

    def blend(self, c, x, y, G=None):
        b = self.rho.shape[1] - 1 if G is None else int(bin_of(G))
        return RM.EPSF.blend(self, c, x, y) + self.rho[c, b]''',
    '''    def __init__(self, S, rho, gc, centres, os_, R, halo, R_ancien):
        C, ng = S.shape[:2]
        super().__init__([[S[c, g] for g in range(ng)] for c in range(C)], centres, os_, R, halo)
        self.rho = rho
        self.gc = gc
        self.R_ancien = R_ancien

    def blend(self, c, x, y, G=None):
        return RM.EPSF.blend(self, c, x, y) + interp_rho(self.rho[c], self.gc, G)''')
rep('''    counts = {f"{EDGES[b]}-{EDGES[b + 1]}": int((bsel == b).sum()) for b in range(len(EDGES) - 1)}''',
    '''    counts = {f"{EDGES[b]}-{EDGES[b + 1]}": int((bsel == b).sum()) for b in range(len(EDGES) - 1)}
    # centre de chaque tranche : G médian de ses étoiles (milieu de la tranche si vide)
    gc = np.array([float(np.median(G[sel][bsel == b])) if (bsel == b).any()
                   else float(np.clip(0.5 * (EDGES[b] + EDGES[b + 1]), 10.5, 17.5)) for b in range(len(EDGES) - 1)])
    gc = np.maximum.accumulate(gc + np.arange(len(gc)) * 1e-6)
    chk = np.array([check_of(G[k]) for k in sel])''')
rep('''    def model_stamps(g_, b_):
        return [S[c, g_] + np.interp(rr, rbins, rho[c, b_]) * taper * (rr <= R) for c in range(C)]''',
    '''    def model_stamps(g_, Gk):
        return [S[c, g_] + np.interp(rr, rbins, interp_rho(rho[c], gc, Gk)) * taper * (rr <= R) for c in range(C)]''')
rep('''        cache_st = {}
        prof = {b: [] for b in range(nb_)}''', '''        prof = {b: [] for b in range(len(CHECK))}''')
rep('''            key = (reg[i_], bsel[i_])
            if key not in cache_st:
                cache_st[key] = model_stamps(*key)
            out = fit_and_residual(k, cache_st[key])''', '''            out = fit_and_residual(k, model_stamps(reg[i_], G[k]))''')
rep('''            prof[bsel[i_]].append(p_)''', '''            if chk[i_] >= 0:
                prof[chk[i_]].append(p_)''')
t = t.replace('''        for b in range(nb_):
            if not prof[b]:
                continue''', '''        for b in range(len(CHECK)):
            if not prof[b]:
                continue''')
rep('''            stat[b] = dict(n=len(a), moy=np.round(mn[:, :CHECK_R + 1], 3).tolist(), se=np.round(se[:, :CHECK_R + 1], 3).tolist())
            if b in CHECK_BINS:
                worst = max(worst, float(np.nanmax(np.abs(mn[:, :CHECK_R + 1]))))''',
    '''            stat[b] = dict(n=len(a), moy=np.round(mn[:, :CHECK_R + 1], 3).tolist(), se=np.round(se[:, :CHECK_R + 1], 3).tolist())
            worst = max(worst, float(np.nanmax(np.abs(mn[:, :CHECK_R + 1]))))''')
rep('''profils={f"{EDGES[b]}-{EDGES[b + 1]}": s for b, s in stat.items()}''', '''profils={f"{CHECK[b][0]}-{CHECK[b][1]}": s for b, s in stat.items()}''')
rep('''        for b in CHECK_BINS:
            if b in stat:
                mn = np.array(stat[b]["moy"])
                lines.append(f"G {EDGES[b]}-{EDGES[b + 1]} ({stat[b]['n']}) r1-4 "''', '''        for b in range(len(CHECK)):
            if b in stat:
                mn = np.array(stat[b]["moy"])
                lines.append(f"G {CHECK[b][0]}-{CHECK[b][1]} ({stat[b]['n']}) r0-4 " + f"{mn[:, 0].mean():+.2f}/"''')
rep('''        st_dbg = []
        for b in CHECK_BINS:
            ms = model_stamps(ng // 2, b)[1]
            fb = Ft_list[Ft_list[:, 3] == b, 2] if len(Ft_list) else []
            st_dbg.append(f"G{EDGES[b]}: F_V med {np.median(fb):.1f} somme {ms.sum() / os_ ** 2:.4f} pic {ms[c0, c0]:.5f}")
        log("[modèle image]   contrôle : " + " ; ".join(st_dbg))''', '''        st_dbg = []
        for gk in (12.75, 14.0, 15.0, 16.0):
            ms = model_stamps(ng // 2, gk)[1]
            st_dbg.append(f"G{gk}: somme {ms.sum() / os_ ** 2:.4f} pic {ms[c0, c0]:.5f}")
        log("[modèle image]   modèle (V, région centrale) : " + " ; ".join(st_dbg))''')
# bloc de débogage de la mise à jour retiré
i0 = t.index('                if c == 1 and b in (1, 3) and __import__("os").environ.get("MPI_DBG"):')
i1 = t.index('        accS = np.zeros((C, ng, n * n))')
t = t[:i0] + t[i1:]
rep('''    rho_img = np.array([[np.interp(rr, rbins, rho[c, b]) * taper * (rr <= R) for b in range(nb_)] for c in range(C)], np.float32)''',
    '''    rho_img = np.array([[np.interp(rr, rbins, rho[c, b]) * taper * (rr <= R) for b in range(nb_)] for c in range(C)], np.float32)
    info_gc = np.round(gc, 3).tolist()''')
rep('''secondes=round(secs), tranches=EDGES.tolist(),''', '''secondes=round(secs), tranches=EDGES.tolist(), centres_G=info_gc, controle=[list(c_) for c_ in CHECK],''')
rep('''    return dict(S=S.astype(np.float32), rho=rho_img, centres=centres, os=os_, R=R, info=info)''',
    '''    return dict(S=S.astype(np.float32), rho=rho_img, gc=gc, centres=centres, os=os_, R=R, info=info)''')
rep('''            return dict(S=z["S"], rho=z["rho"], centres=z["centres"], os=int(z["os"]), R=int(z["R"]), info=info)''',
    '''            return dict(S=z["S"], rho=z["rho"], gc=z["gc"], centres=z["centres"], os=int(z["os"]), R=int(z["R"]), info=info)''')
rep('''    np.savez(f, S=m["S"], rho=m["rho"], centres=m["centres"], os=m["os"], R=m["R"])''',
    '''    np.savez(f, S=m["S"], rho=m["rho"], gc=m["gc"], centres=m["centres"], os=m["os"], R=m["R"])''')
rep('''    return EPSFImage(m["S"].astype(np.float64), m["rho"].astype(np.float64), cen, m["os"], m["R"],''',
    '''    return EPSFImage(m["S"].astype(np.float64), m["rho"].astype(np.float64), m["gc"], cen, m["os"], m["R"],''')
p.write_text(t, encoding="utf-8")
print("ok")
