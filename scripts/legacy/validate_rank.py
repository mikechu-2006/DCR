import pandas as pd, numpy as np
pd.set_option("display.width", 220)
bm = pd.read_csv("data/interim/beatmaps.csv")
mh = pd.read_csv("data/interim/mania_high.csv")
bm4 = bm[(bm.playmode == 3) & (bm["keys"] == 4)]
m = mh.merge(bm4[["beatmap_id"]], on="beatmap_id", how="inner")
for c in ["count50","count100","count300","countmiss","countgeki","countkatu"]:
    m[c] = pd.to_numeric(m[c], errors="coerce")
tot = m[["count50","count100","count300","countmiss","countgeki","countkatu"]].sum(axis=1)
m["flat"] = (300*(m.count300 + m.countgeki) + 200*m.countkatu + 100*m.count100 + 50*m.count50)/(300*tot)
m["w320"] = (320*m.countgeki + 300*m.count300 + 200*m.countkatu + 100*m.count100 + 50*m.count50)/(320*tot)
def buckets(a):
    return pd.cut(a, [-.01, .70, .80, .90, .95, .99999999, 1.0000001], labels=["D","C","B","A","S","X"])
for f in ["flat","w320"]:
    ct = pd.crosstab(m["rank"].str.replace("H",""), buckets(m[f]))
    print("=== rank vs computed", f, "===")
    print(ct.to_string())
