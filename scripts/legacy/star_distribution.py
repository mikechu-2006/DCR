import pandas as pd, numpy as np
d = pd.read_csv("data/interim/beatmap_difficulty_4k.csv")
d["star"] = pd.to_numeric(d.star, errors="coerce")
mods = pd.to_numeric(d.mods, errors="coerce").fillna(0).astype("int64")
d["rate"] = np.where(mods & (64 | 512) != 0, 1, np.where(mods & 256 != 0, 2, 0))
star = d[d.star > 0].sort_values("star").groupby(["beatmap_id", "rate"], as_index=False).star.max()
BINS = [0, 3, 4, 5, 6, 7, 8, 20]
LBL = ["<3", "3-4", "4-5", "5-6", "6-7", "7-8", "8+"]

def dist(plays_path, name, n=None):
    p = pd.read_parquet(plays_path, columns=["beatmap_id", "rate", "acc"])
    p = p.merge(star, on=["beatmap_id", "rate"], how="left")
    p = p[p.star.notna()]
    if n: p = p.sample(min(n, len(p)), random_state=0)
    p["b"] = pd.cut(p.star, BINS, labels=LBL)
    vc = p.b.value_counts(normalize=True).reindex(LBL).fillna(0)
    print(f"\n[{name}] plays={len(p):,}  star median={p.star.median():.2f}  share star<5={float((p.star<5).mean()):.3f}")
    print("  " + "  ".join(f"{l}:{v*100:5.1f}%" for l, v in vc.items()))

dist("data/processed/plays_v2_1k.parquet", "top-1000 players' own plays")
dist("data/processed/plays_v2_10k.parquet", "top-10000 players' own plays", n=3000000)

bm = pd.read_csv("data/interim/beatmaps.csv")
bm["playcount"] = pd.to_numeric(bm.playcount, errors="coerce")
g = bm[(bm.playmode == 3) & (bm["keys"] == 4) & (bm.playcount > 0)]
g = g.assign(b=pd.cut(pd.to_numeric(g.star, errors="coerce"), BINS, labels=LBL))
print("\n[GLOBAL: all players' plays on 4K mania] (osu_beatmaps.playcount)")
print("  " + "  ".join(f"{l}:{v*100:5.1f}%" for l, v in g.groupby("b", observed=False).playcount.sum().pipe(lambda s: s/s.sum()).items()))
