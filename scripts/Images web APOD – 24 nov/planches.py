# ======================================================================
# planches.py
# Rôle     : Module de la chaîne (voir sa docstring ci-dessous et scripts/LISEZMOI.md).
# Étape    : 3. Collecte des images web APOD (24 nov) (voir scripts/LISEZMOI.md)
# Entrées  : archive APOD (API publique)
# Sorties  : images téléchargées (non publiées), index
# Exemple  : python planches.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
# Index CSV + planches de vignettes (12 par planche, numerotees, nom sous la vignette). 1 fil, lecture par python -I.
import os,json,csv,re,html
os.environ["OMP_NUM_THREADS"]="1"
from PIL import Image,ImageDraw,ImageFont
Image.MAX_IMAGE_PIXELS=None
R=r"<DONNEES>\Images web APOD – 24 nov"; B=R+r"\brut"; P=R+r"\planches"
cand={x["slug"]:x for x in json.load(open(R+r"\meta\candidats.json",encoding="utf8"))}
res=[json.loads(l) for l in open(R+r"\meta\telecharges.jsonl",encoding="utf8")]
res=[j for j in res if j["ok"]]
def nettoie(a): return re.sub(r"^(Image )?Credit( & (Copyright|License))?:?\s*|^and Copyright:\s*|^& Copyright:\s*","",a).strip()
with open(R+r"\index.csv","w",encoding="utf-8-sig",newline="") as f:
    w=csv.writer(f,delimiter=";"); w.writerow(["numero","fichier","date APOD","titre","auteur","taille (px)","amateur ou pro","motif du choix"])
    for n,j in enumerate(res,1):
        x=cand[j["slug"]]; j["n"]=n
        w.writerow([n,j["fichier"],x["date"],x["titre"],nettoie(x["auteur"]),f'{j["w"]}x{j["h"]}',"pro" if x["pro"] else "amateur",j["motif"]])
for f_ in os.listdir(P): os.remove(os.path.join(P,f_))
try: fnt=ImageFont.truetype("arial.ttf",15); fb=ImageFont.truetype("arialbd.ttf",22)
except Exception: fnt=fb=ImageFont.load_default()
CW,CH,TH=480,360,44
for p in range(0,len(res),12):
    lot=res[p:p+12]; W=4*CW; H=3*(CH+TH)
    pl=Image.new("RGB",(W,H),(24,24,24)); d=ImageDraw.Draw(pl)
    for k,j in enumerate(lot):
        im=Image.open(os.path.join(B,j["fichier"])); im.draft("RGB",(CW*2,CH*2)); im=im.convert("RGB")
        im.thumbnail((CW-8,CH-8),Image.LANCZOS)
        x0=(k%4)*CW; y0=(k//4)*(CH+TH)
        pl.paste(im,(x0+(CW-im.width)//2,y0+(CH-im.height)//2))
        d.rectangle([x0+4,y0+4,x0+44,y0+32],fill=(0,0,0)); d.text((x0+8,y0+5),str(j["n"]),font=fb,fill=(255,220,0))
        d.text((x0+6,y0+CH+2),j["fichier"][:58],font=fnt,fill=(230,230,230))
        d.text((x0+6,y0+CH+22),f'{j["w"]}x{j["h"]}  '+("pro" if cand[j["slug"]]["pro"] else "amateur"),font=fnt,fill=(150,150,150))
    pl.save(os.path.join(P,f"planche_{p//12+1:02d}.jpg"),quality=88)
print(len(res),"images;",(len(res)+11)//12,"planches")
