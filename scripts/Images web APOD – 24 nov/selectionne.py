# ======================================================================
# selectionne.py
# Rôle     : Module de la chaîne (voir sa docstring ci-dessous et scripts/LISEZMOI.md).
# Étape    : 3. Collecte des images web APOD (24 nov) (voir scripts/LISEZMOI.md)
# Entrées  : archive APOD (API publique)
# Sorties  : images téléchargées (non publiées), index
# Exemple  : python selectionne.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
# Preselection sur titres (diversite des objets, 2016+, repartition dans le temps). Le tri fin se fait a l'oeil sur les planches.
import json,re,collections
M=r"<DONNEES>\Images web APOD – 24 nov\meta"
d=json.load(open(M+r"\candidats.json",encoding="utf8"))
T=re.compile(r"nebula|galax|cluster|supernova|remnant|\bM ?\d{1,3}\b|NGC|\bIC ?\d|Sh2|Barnard|LDN|vdB|Cocoon|Veil|Rosette|Pleiades|Andromeda|Trifid|Lagoon|Eagle|Heart|Soul|Horsehead|Flame|Helix|Dumbbell|Crab|Bubble|Cave|Cone|Elephant|Pelican|North America|California|Iris|Wizard|Tadpole|Cygnus|Carina|Triangulum|Whirlpool|Sombrero|Pinwheel|Bode|Cigar|Orion|Rho|Cepheus|Cassiopeia",re.I)
BAD=re.compile(r"over |above |sky over|rainbow|landscape|dark matter|simulat|trails|mosaic|panorama|wide|constellation|sky survey|all-sky|map|catalog",re.I)
def cle(t):
    m=re.search(r"\b(M ?\d+|NGC ?\d+|IC ?\d+|Sh2-?\d+|LDN ?\d+|vdB ?\d+|Arp ?\d+)\b",t)
    return m.group(1).replace(" ","").upper() if m else re.sub(r"[^a-z ]","",t.lower())[:25]
def nettoie(a): return re.sub(r"^(Image )?Credit( & (Copyright|License))?:?\s*|^and Copyright:\s*|^& Copyright:\s*","",a).strip()
def prend(liste,n):
    vus=set(); res=[]
    liste=[r for r in liste if r["date"]>="2016" and T.search(r["titre"]) and not BAD.search(r["titre"]) and r["auteur"]]
    pas=max(1,len(liste)//(n*2))
    for r in liste[::pas]:
        k=cle(r["titre"])
        if k in vus: continue
        vus.add(k); res.append(r)
        if len(res)>=n: break
    return res
ama=prend([r for r in d if not r["pro"]],95)
pro=prend([r for r in d if r["pro"] and re.search(r"ESO|NOIRLab|Hubble|Webb|Gemini",r["auteur"])],8)
with open("selection.txt","w",encoding="utf8") as f:
    f.write("# slug|motif  (amateur d'abord, puis pro en complement)\n")
    for r in ama: f.write(f'{r["slug"]}|amateur : {nettoie(r["auteur"])[:60]} ; ciel profond d apres le titre\n')
    for r in pro: f.write(f'{r["slug"]}|pro (complement) : {nettoie(r["auteur"])[:50]}\n')
print(len(ama),len(pro))
