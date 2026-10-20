import pandas as pd, numpy as np
bm = pd.read_csv("data/interim/beatmaps.csv")
bm["star"] = pd.to_numeric(bm.star, errors="coerce")
man = bm[bm.playmode == 3].set_index("beatmap_id")
BINS = [0, 3, 4, 5, 6, 7, 8, 20]; LBL = ["<3", "3-4", "4-5", "5-6", "6-7", "7-8", "8+"]

pc = pd.read_csv("data/interim_random/playcount.csv", dtype={"user_id": np.int32, "beatmap_id": np.int32, "playcount": np.int32})
pc = pc[pc.beatmap_id.isin(man.index)]
pc["star"] = man.star.reindex(pc.beatmap_id).values
pc = pc[pc.star.notna()]
pc["b"] = pd.cut(pc.star, BINS, labels=LBL)
d = pc.groupby("b", observed=False).playcount.sum()
d = d / d.sum()
print(f"[random 10000] mania plays={pc.playcount.sum():,}  star median(weighted by plays)={np.average(pc.star, weights=pc.playcount):.2f}")
print("  " + "  ".join(f"{k}:{v*100:5.1f}%" for k, v in d.items()))

p = pd.read_parquet("data/processed/plays_v2_1k.parquet", columns=["beatmap_id", "rate"])
p = p[p.beatmap_id.isin(man.index)]
p["star"] = man.star.reindex(p.beatmap_id).values
p = p[p.star.notna()]
p["b"] = pd.cut(p.star, BINS, labels=LBL)
e = p.groupby("b", observed=False).size(); e = e / e.sum()
print(f"\n[top 1000] mania plays={len(p):,}  star mean={p.star.mean():.2f} median={p.star.median():.2f}")
print("  " + "  ".join(f"{k}:{v*100:5.1f}%" for k, v in e.items()))
print("\n(注：两边都按谱面原生 star，且都限制在 playmode=3 的 mania 谱面)")

# random sample restricted to the 10k-50k playcount band (mature accounts)
st = None
import sys; sys.path.insert(0, "scripts")
from parse_dump import iter_insert_lines, rows_simple
from pathlib import Path
rows = []
for line in iter_insert_lines(Path("data/raw/extracted_random/2026_09_01_performance_mania_random_10000/osu_user_stats_mania.sql")):
    for r in rows_simple(line):
        if len(r) >= 29: rows.append([int(r[0]), int(r[8])])
s = pd.DataFrame(rows, columns=["user_id", "playcount"])
mature = set(s[(s.playcount > 10000) & (s.playcount <= 50000)].user_id)
pm = pc[pc.user_id.isin(mature)]
if len(pm):
    dm = pm.groupby("b", observed=False).playcount.sum(); dm = dm / dm.sum()
    print(f"\n[random 10000, 只取 10k-50k playcount 的老账号] plays={pm.playcount.sum():,} users={pm.user_id.nunique()}")
    print("  " + "  ".join(f"{k}:{v*100:5.1f}%" for k, v in dm.items()))
