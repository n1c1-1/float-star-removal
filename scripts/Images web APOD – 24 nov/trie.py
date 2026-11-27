# ======================================================================
# trie.py
# Role     : Chain module (see its docstring below and scripts/LISEZMOI.md).
# Step     : 3. Collection of the APOD web images (24 Nov) (see scripts/LISEZMOI.md)
# Inputs   : APOD archive (public API)
# Outputs  : downloaded images (not published), index
# Example  : python trie.py [arguments]  (see the docstring; paths through the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
# Sorting of the APOD metadata: deep-sky candidates, author, amateur/pro, URL of the large-format image.
import json,glob,re,os,html,csv
M=r"<DONNEES>\Images web APOD – 24 nov\meta"
PRO=re.compile(r"hubble|jwst|webb|stsci|\besa\b|eso\b|nasa|chandra|spitzer|gaia|euclid|noirlab|noao|ctio|kpno|gemini|vlt|alma|subaru|keck|cfht|sdss|wise|galex|nrao|vista|des collab|rubin|lsst|panstarrs|pan-starrs|digitized sky|dss|jpl|caltech|nustar|xmm|fermi|planck|hst|sofia|magellan|skymapper|decam|vera",re.I)
DEEP=re.compile(r"nebula|galax|cluster|supernova|remnant|\bm ?\d{1,3}\b|ngc|\bic ?\d|sh2|barnard|lbn|ldn|vdb|cocoon|veil|rosette|orion|pleiades|andromeda|trifid|lagoon|eagle|heart|soul|horsehead|flame|helix|ring|dumbbell|crab|bubble|cave|cone|elephant|pelican|north america|california|iris|wizard|tadpole|cygnus|carina|nebulosity|dust|molecular cloud|milky way|triangulum|whirlpool|sombrero|pinwheel|bode|cigar|leo triplet|virgo|coma",re.I)
NOT=re.compile(r"\bmoon\b|lunar|eclipse|solar|\bsun\b|sunspot|aurora|comet|saturn|jupiter|mars\b|venus|mercury|neptune|uranus|iss\b|station|meteor|noctilucent|rocket|launch|iceberg|earth|apollo|mission|spacecraft|rover|asteroid|starless|transit|conjunction|planet|artist|illustration|simulation|timelapse|time-lapse|video|animation|panorama|landscape|over the|above the|star trails|startrail",re.I)
rows=[]
for f in sorted(glob.glob(M+r"\api_*.json")):
    for x in json.load(open(f,encoding="utf8")):
        if not x["slug"].startswith("apod-"): continue
        c=x["content"]["rendered"]
        t=html.unescape(re.sub(r"<[^>]+>","",x["title"]["rendered"]))
        t=t.split("–",1)[-1].strip() if "–" in t else t
        urls=re.findall(r'https://assets\.science\.nasa\.gov/content/dam/science/cds/apod/apod/[^"\s?]+\.(?:jpg|jpeg|png)',c,re.I)
        urls=[u for u in urls if "/jcr:content" not in u]
        if not urls: continue
        txt=html.unescape(re.sub(r"\s+"," ",re.sub(r"<[^>]+>"," ",c)))
        i0=txt.rfind("Tomorrow")
        m=re.search(r"Credit(?: & (?:Copyright|License))?:?(.{0,250}?)(?:Authors|$)",txt[i0:] if i0>0 else txt)
        cred=m.group(1).strip() if m else ""
        cred=re.split(r"Date",cred)[0].strip()
        m2=re.search(r"Explanation:(.*?)(?:Tomorrow|Credit|$)",txt); ex=m2.group(1).strip() if m2 else ""
        dm=re.search(r"(\d{4}) (\w+) (\d+)",html.unescape(x["title"]["rendered"]))
        dim=re.findall(r"(\d{3,5})\s*[x×]\s*(\d{3,5})",urls[0])
        rows.append(dict(slug=x["slug"],date=x["date"][:10],titre=t,auteur=cred,url=urls[0],expl=ex,nom=urls[0].rsplit("/",1)[1],
            dim=dim[0] if dim else None))
print(len(rows),"entrees avec image")
out=[]
for r in rows:
    s=r["titre"]+" "+r["expl"][:400]
    if NOT.search(r["titre"]) or not DEEP.search(r["titre"]+" "+r["expl"][:200]): continue
    r["pro"]=bool(PRO.search(r["auteur"]))
    out.append(r)
print(len(out),"candidats;",sum(r["pro"] for r in out),"pro")
json.dump(out,open(os.path.join(M,"candidats.json"),"w",encoding="utf8"),ensure_ascii=False,indent=0)
