# ======================================================================
# telecharge.py
# Role     : Chain module (see its docstring below and scripts/LISEZMOI.md).
# Step     : 3. Collection of the APOD web images (24 Nov) (see scripts/LISEZMOI.md)
# Inputs   : APOD archive (public API)
# Outputs  : downloaded images (not published), index
# Example  : python telecharge.py [arguments]  (see the docstring; paths through the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
# Download of the chosen APOD images (selection.txt: one line = slug|reason). 1 request / 3 s, generic UA.
# Discards (deletes) the images whose long side < 2000 px. Resumable: the files already present are kept.
import json,os,sys,time,subprocess,csv
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
UA="star-removal-research/0.1"
R=r"<DONNEES>\Images web APOD – 24 nov"
B=os.path.join(R,"brut"); M=os.path.join(R,"meta")
D=os.path.dirname(os.path.abspath(sys.argv[0]))
cand={x["slug"]:x for x in json.load(open(os.path.join(M,"candidats.json"),encoding="utf8"))}
sel=[l.rstrip("\n").split("|",1) for l in open(os.path.join(D,"selection.txt"),encoding="utf8") if l.strip() and not l.startswith("#")]
MAXI=int(sys.argv[1]) if len(sys.argv)>1 else 60
res=os.path.join(M,"telecharges.jsonl"); fait={}
if os.path.exists(res):
    for l in open(res,encoding="utf8"): j=json.loads(l); fait[j["slug"]]=j
tmp=os.path.join(B,"tmp.part")
for slug,motif in sel:
    if sum(1 for j in fait.values() if j["ok"])>=MAXI: break
    if slug in fait: continue
    x=cand[slug]; ext=os.path.splitext(x["url"])[1].lower()
    f=os.path.join(B,f'{x["date"]}_{x["nom"]}')
    for k in range(5):
        c=subprocess.run(["curl","-sL","-A",UA,"-D","-","-o",tmp,"-w","%{http_code}",x["url"]],capture_output=True,text=True)
        code=c.stdout.strip()[-3:]
        if code=="200": break
        ra=0
        for l in c.stdout.splitlines():
            if l.lower().startswith("retry-after:"):
                try: ra=int(l.split(":")[1])
                except Exception: pass
        w=ra or 120*(k+1)
        print("http",code,"attente",w,flush=True)
        if code in("404","410"): break
        if k==4: print("ARRET persistant",code,flush=True); sys.exit(2)
        time.sleep(w)
    ok=False; wh=(0,0)
    if code=="200":
        try:
            im=Image.open(tmp); wh=im.size; im.close()
            ok=max(wh)>=2000
        except Exception as e: print("illisible",slug,e,flush=True)
    if ok: os.replace(tmp,f)
    elif os.path.exists(tmp): os.remove(tmp)
    j=dict(slug=slug,ok=ok,fichier=os.path.basename(f) if ok else "",w=wh[0],h=wh[1],motif=motif,http=code)
    fait[slug]=j; open(res,"a",encoding="utf8").write(json.dumps(j,ensure_ascii=False)+"\n")
    print(slug,code,wh,"OK" if ok else "ecartee",flush=True)
    time.sleep(3)
print("fin",sum(1 for j in fait.values() if j["ok"]),"retenues",flush=True)
