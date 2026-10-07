import pandas as pd, re, collections
from pathlib import Path
# all distinct mod acronyms seen anywhere in the modern table
cnt = collections.Counter()
n = 0
for chunk in pd.read_csv("data/interim/scores_detail.csv", usecols=["mods"], chunksize=2_000_000):
    for s in chunk.mods.fillna(""):
        n += 1
        for t in s.split("+"):
            if t:
                cnt[t] += 1
print("rows:", n)
for k, v in cnt.most_common(40):
    print(f"  {k:6s} {v:9d}  {v/n*100:6.3f}%")
# legacy bitmask bits actually used
import numpy as np
h = pd.read_parquet("data/interim_10k/mania_high_4k.parquet", columns=["enabled_mods"])
bits = collections.Counter()
m = h.enabled_mods.to_numpy()
for b in [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288]:
    c = int((m & b != 0).sum())
    if c: bits[b] = c
print("\nlegacy bitmask bits used (mania_high 4k):", bits)
print("any bit >= 1048576:", int((m & ~((1<<20)-1) != 0).sum()))
