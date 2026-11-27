# ======================================================================
# moisson.py
# Role     : Chain module (see its docstring below and scripts/LISEZMOI.md).
# Step     : 3. Collection of the APOD web images (24 Nov) (see scripts/LISEZMOI.md)
# Inputs   : APOD archive (public API)
# Outputs  : downloaded images (not published), index
# Example  : python moisson.py [arguments]  (see the docstring; paths through the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
# Harvest of the APOD metadata through the site's public API (one request / 3 s, generic UA).
import urllib.request,urllib.parse,json,time,os,sys,re
UA="star-removal-research/0.1"
OUT=r"<DONNEES>\Images web APOD – 24 nov\meta"
PAGES=int(sys.argv[1]) if len(sys.argv)>1 else 40
import subprocess
def get(u):
    for k in range(6):
        c=subprocess.run(["curl","-sL","-A",UA,"-D","-","-o",os.path.join(OUT,"tmp.bin"),"-w","%{http_code}",u],capture_output=True,text=True)
        code=c.stdout.strip()[-3:]
        if code=="200": return open(os.path.join(OUT,"tmp.bin"),"rb").read(),None
        if code=="400": return None,None
        ra=0
        for l in c.stdout.splitlines():
            if l.lower().startswith("retry-after:"):
                try: ra=int(l.split(":")[1])
                except Exception: pass
        w=ra or 120*(k+1)
        print("http",code,"attente",w,flush=True)
        if k>=4: print("ARRET persistant",code,flush=True); sys.exit(2)
        time.sleep(w)
    sys.exit(2)
for p in range(1,PAGES+1):
    f=os.path.join(OUT,f"api_{p:03d}.json")
    if os.path.exists(f): continue
    q=urllib.parse.urlencode(dict(per_page=100,page=p,orderby="date",order="desc",search="APOD",_fields="slug,date,title,link,content"))
    b,h=get("https://science.nasa.gov/wp-json/wp/v2/image-article?"+q)
    if b is None: print("fin page",p,flush=True); break
    open(f,"wb").write(b); d=json.loads(b)
    print(p,len(d),d[-1]["date"] if d else "",flush=True)
    if not d: break
    time.sleep(3)
