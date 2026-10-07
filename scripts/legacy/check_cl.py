import pandas as pd, collections, numpy as np
p = pd.read_parquet("data/processed/plays_4k.parquet", columns=["source","mods_raw","date","rate"])
mod = p.source.eq("modern")
print("modern plays:", int(mod.sum()), "legacy plays:", int((~mod).sum()))
c = collections.Counter()
sub = p[mod]
for s in sub.mods_raw.fillna(""):
    for t in s.split("+"):
        if t: c[t] += 1
n = len(sub)
print("\nmodern 4K mods:")
for k,v in c.most_common(25):
    print(f"  {k:5s} {v:8d} {v/n*100:6.3f}%")
# which mods co-occur with CL
cl = sub[sub.mods_raw.fillna("").str.split("+").apply(lambda t: "CL" in t)]
print("\nmodern plays with CL:", len(cl), "| without CL:", n-len(cl))
cc = collections.Counter()
for s in cl.mods_raw:
    for t in s.split("+"):
        if t and t != "CL": cc[t] += 1
print("mods co-occurring with CL (top20):", cc.most_common(20))
# rate mods among CL rows
for tok in ["DT","NC","HT","DC","AS","WU","WD"]:
    print(f"  CL + {tok}: {cc.get(tok,0)}")
# dates
print("\ndate range:", p.date.min(), "->", p.date.max())
print("year counts:", p.date.str[:4].value_counts().sort_index().to_dict())
