import pandas as pd, numpy as np, collections
d = pd.read_parquet("data/interim_10k/mania_high_4k_dated.parquet", columns=["enabled_mods"])
m = d.enabled_mods.to_numpy(np.int64)
n = len(m)
print("rows:", n)
NAMES = {1:"NF",2:"EZ",4:"TD",8:"HD",16:"HR",32:"SD",64:"DT",128:"RX",256:"HT",512:"NC",1024:"FL",
         2048:"AT",4096:"SO",8192:"AP",16384:"PF",32768:"4K",65536:"5K",131072:"6K",262144:"7K",
         524288:"8K",1048576:"FI",2097152:"RD",4194304:"CN",8388608:"TP",16777216:"9K",
         33554432:"CO",67108864:"1K",134217728:"3K",268435456:"2K"}
for bit in sorted(NAMES):
    c = int((m & bit != 0).sum())
    if c: print(f"  {NAMES[bit]:5s} ({bit:>9d}): {c:9d}  {c/n*100:6.3f}%")
# combinations of the high bits
hi = m & ~((1<<20)-1)
print("\nrows with any bit>=2^20:", int((hi!=0).sum()))
print("distinct high-bit values (top):", collections.Counter(hi.tolist()).most_common(6))
