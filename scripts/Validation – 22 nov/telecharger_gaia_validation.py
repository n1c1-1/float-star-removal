# ======================================================================
# telecharger_gaia_validation.py
# Rôle     : Module de la chaîne (voir sa docstring ci-dessous et scripts/LISEZMOI.md).
# Étape    : 13. Mesure de validation (voir scripts/LISEZMOI.md)
# Entrées  : réseau .pt (ou dossier de sorties) + liste d'images de validation
# Sorties  : json par image, planches, resume.md
# Exemple  : python telecharger_gaia_validation.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
# Copie pour la validation (22 nov) : SRC/OUT vers <DONNEES>/Validation étoiles – 22 nov/<nom>/. Original : # 22 nov soir (jeu 1, images de la bibliotheque) : copie de telecharger_gaia_profond_m8_sho.py, SRC et OUT vers Methode v32 production. Usage : python telecharger_gaia_profond.py <image> [<image> ...]
import sys; sys.stdout.reconfigure(encoding="utf-8")
import sys, time, json, numpy as np, pyvo, requests
from astropy.io import fits
from astropy.wcs import WCS
SRC="<DONNEES>/Validation étoiles – 22 nov/"
OUT="<DONNEES>/Validation étoiles – 22 nov/"
# 22 nov soir : plusieurs serveurs TAP, meme requete (memes colonnes) ; le suivant prend le relais si le premier coupe.
# Variables : GAIA_TAP = heidelberg,esa,vizier (ordre) ; GAIA_TUILE = cote d'une tuile en degres (defaut 0,7 ; 0,35 si ca coupe)
import os
SERVICES = {
 "heidelberg": ("https://dc.zah.uni-heidelberg.de/tap", "gaia.dr3lite", dict(id="source_id", ra="ra", dec="dec", g="phot_g_mean_mag", bp="phot_bp_mean_mag", rp="phot_rp_mean_mag", pmra="pmra", pmde="pmdec")),
 "esa": ("https://gea.esac.esa.int/tap-server/tap", "gaiadr3.gaia_source", dict(id="source_id", ra="ra", dec="dec", g="phot_g_mean_mag", bp="phot_bp_mean_mag", rp="phot_rp_mean_mag", pmra="pmra", pmde="pmdec")),
 "vizier": ("https://tapvizier.cds.unistra.fr/TAPVizieR/tap", '"I/355/gaiadr3"', dict(id="Source", ra="RA_ICRS", dec="DE_ICRS", g="Gmag", bp="BPmag", rp="RPmag", pmra="pmRA", pmde="pmDE")),
}
ORDRE = os.environ.get("GAIA_TAP", "heidelberg,esa,vizier").split(",")
TUILE = float(os.environ.get("GAIA_TUILE", "0.7"))
GMAX=21.0
sess=requests.Session(); sess.headers["User-Agent"]="astro-catalogue-extract/1.0"
def q(make_sql, tries=2):
    """make_sql(table, cols) -> ADQL ; essaie chaque serveur de l'ordre, 2 fois, puis le suivant."""
    for nom in ORDRE:
        url, table, cols = SERVICES[nom]
        svc = pyvo.dal.TAPService(url, session=sess)
        for i in range(tries):
            try:
                run = svc.run_sync if nom == "esa" else svc.run_async     # ESA : les jobs asynchrones sont abandonnes (22 nov), le synchrone repond
                t = run(make_sql(table, cols, nom), maxrec=(3_000_000 if nom == "esa" else 50_000_000)).to_table()
                # noms de colonnes normalises (ceux de dr3lite)
                ren = {cols["id"]: "source_id", cols["ra"]: "ra", cols["dec"]: "dec", cols["g"]: "phot_g_mean_mag", cols["bp"]: "phot_bp_mean_mag",
                       cols["rp"]: "phot_rp_mean_mag", cols["pmra"]: "pmra", cols["pmde"]: "pmdec"}
                for a, b in ren.items():
                    if a != b and a in t.colnames:
                        t.rename_column(a, b)
                print("  serveur", nom, flush=True)
                return t
            except Exception as e:
                print("  echec", nom, i, str(e)[:150], flush=True); time.sleep(20 * 2 ** i)
    raise RuntimeError("failed")
names=sys.argv[1:]
for d in names:
    z=np.load(SRC+d+"/astrometrie.npz",allow_pickle=True)
    c=z['cat'].item(); h=str(z['wcs']); w=WCS(fits.Header.fromstring(h))
    W=c['x'].max(); H=c['y'].max()
    mx=0.05*W+60; my=0.05*H+60
    x0,x1,y0,y1=-mx,W+mx,-my,H+my
    # tiles in pixel space ~ 0.7 deg
    sc=float(z['scale'])/3600; step=int(TUILE/sc)
    xs=list(range(int(x0),int(x1),step))+[int(x1)]; ys=list(range(int(y0),int(y1),step))+[int(y1)]
    parts=[]
    for i in range(len(xs)-1):
        for j in range(len(ys)-1):
            px=[xs[i],xs[i+1],xs[i+1],xs[i]]; py=[ys[j],ys[j],ys[j+1],ys[j+1]]
            ra,de=w.all_pix2world(px,py,0)
            pts=", ".join(f"{a:.7f}, {b:.7f}" for a,b in zip(ra,de))
            ra0, de0 = np.mean(ra), np.mean(de)
            from astropy.coordinates import SkyCoord
            rad = float(SkyCoord(ra0, de0, unit="deg").separation(SkyCoord(ra, de, unit="deg")).deg.max()) * 1.02
            def sql(table, c, nom, pts=pts, ra0=ra0, de0=de0, rad=rad):
                if nom == "esa":   # le polygone sur gaia_source est abandonne par ESA (408) : cercle circonscrit, tri par la zone apres
                    return (f"SELECT {c['id']}, {c['ra']}, {c['dec']}, {c['g']}, {c['bp']}, {c['rp']}, {c['pmra']}, {c['pmde']} FROM {table} "
                            f"WHERE 1=CONTAINS(POINT('ICRS', {c['ra']}, {c['dec']}), CIRCLE('ICRS', {ra0:.7f}, {de0:.7f}, {rad:.5f})) AND {c['g']} < {GMAX}")
                return (f"SELECT {c['id']}, {c['ra']}, {c['dec']}, {c['g']}, {c['bp']}, {c['rp']}, {c['pmra']}, {c['pmde']} FROM {table} "
                        f"WHERE 1=CONTAINS(POINT('ICRS', {c['ra']}, {c['dec']}), POLYGON('ICRS', {pts})) AND {c['g']} < {GMAX}")
            t=q(sql); print(d,i,j,len(t),flush=True); parts.append(t)
            time.sleep(3)
    from astropy.table import vstack, unique
    t=unique(vstack([p for p in parts if len(p)]),keys="source_id")
    ra=np.asarray(t['ra'],float); de=np.asarray(t['dec'],float)
    x,y=w.all_world2pix(ra,de,0)
    m=(x>=x0)&(x<=x1)&(y>=y0)&(y<=y1)
    f=lambda k:np.ma.filled(np.ma.asarray(t[k]).astype(float),np.nan)[m]
    g=f('phot_g_mean_mag')
    np.savez_compressed(OUT+f"{d}/gaia_dr3_profond.npz",Source=np.ma.filled(t['source_id'],-1).astype(np.int64)[m],
        RA_ICRS=ra[m],DE_ICRS=de[m],Gmag=g,BPmag=f('phot_bp_mean_mag'),RPmag=f('phot_rp_mean_mag'),
        pmRA=f('pmra'),pmDE=f('pmdec'),x_pix=x[m],y_pix=y[m],footprint=np.array([x0,x1,y0,y1]))
    inimg=m&(x>=-60)&(x<=W+60)&(y>=-60)&(y<=H+60)
    print("RESULT",d,"n=",int(m.sum()),"faintest G",np.nanmax(g),"n_in_image+60px",int(inimg.sum()),"current in image",len(c['x']),"current faintest",c['g'].max(),flush=True)
