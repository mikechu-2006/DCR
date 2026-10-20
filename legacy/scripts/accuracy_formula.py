import pandas as pd, numpy as np
pd.set_option("display.width", 220)
bm = pd.read_csv("data/interim/beatmaps.csv")
mh = pd.read_csv("data/interim/mania_high.csv")
bm4 = bm[(bm.playmode == 3) & (bm["keys"] == 4)]
mh4 = mh.merge(bm4[["beatmap_id"]], on="beatmap_id", how="inner")
for c in ["count50","count100","count300","countmiss","countgeki","countkatu"]:
    mh4[c] = pd.to_numeric(mh4[c], errors="coerce")
mh4["total"] = mh4[["count50","count100","count300","countmiss","countgeki","countkatu"]].sum(axis=1)
x = mh4[mh4["rank"].isin(["X","XH"])]
print("rank X/XH rows:", len(x))
print("of those, count300>0:", int((x.count300 > 0).sum()), " count100>0:", int((x.count100>0).sum()),
      " countmiss>0:", int((x.countmiss>0).sum()), " countkatu>0:", int((x.countkatu>0).sum()))
print("sample X rows:")
print(x[["rank","countgeki","count300","countkatu","count100","count50","countmiss","total"]].head(8).to_string())

# candidate accuracy formulas on a random sample
s = mh4.sample(200000, random_state=0).copy()
s["acc320"] = (320*s.countgeki + 300*s.count300 + 200*s.countkatu + 100*s.count100 + 50*s.count50) / (320*s.total)
s["accflat"] = (300*(s.countgeki + s.count300) + 200*s.countkatu + 100*s.count100 + 50*s.count50) / (300*s.total)
print("\nrank vs formula consistency (rank X should be 1.0):")
for f in ["acc320","accflat"]:
    print(f, "mean acc for rank X:", round(s[s['rank'].isin(['X','XH'])][f].mean(), 6),
          "| mean acc for rank S:", round(s[s['rank']=='S'][f].mean(), 6),
          "| mean acc for rank A:", round(s[s['rank']=='A'][f].mean(), 6))
print("\nrank thresholds: X=100 XH=100 S>95 A>90 B>80 C>70")
print("share of rank S with 320-formula <0.95:", round((s[s['rank']=='S'].acc320 < 0.95).mean(), 4))
print("share of rank S with flat-formula <0.95:", round((s[s['rank']=='S'].accflat < 0.95).mean(), 4))
