import pandas as pd, numpy as np
d = pd.read_csv("data/interim/beatmap_difficulty_4k.csv")
d["star"] = pd.to_numeric(d.star, errors="coerce")
mods = pd.to_numeric(d.mods, errors="coerce").fillna(0).astype("int64")
d["rate"] = np.where(mods & (64 | 512) != 0, 1, np.where(mods & 256 != 0, 2, 0))
star = d[d.star > 0].sort_values("star").groupby(["beatmap_id", "rate"], as_index=False).star.max()

p = pd.read_parquet("data/processed/plays_v2_1k.parquet", columns=["beatmap_id", "rate", "year", "user_id"])
p = p.merge(star, on=["beatmap_id", "rate"], how="left")
p = p[p.star.notna()]
print("top-1000 players' recorded plays by year:")
print(f"{'year':>5s} {'plays':>9s} {'star_med':>9s} {'share<5':>8s} {'share<4':>8s}")
for y, g in p.groupby("year"):
    print(f"{int(y):5d} {len(g):9,d} {g.star.median():9.2f} {float((g.star<5).mean()):8.3f} {float((g.star<4).mean()):8.3f}")
recent = p[p.year >= 2024]
print(f"\n2024-2026 only: plays={len(recent):,} star_med={recent.star.median():.2f} share<5={float((recent.star<5).mean()):.3f}")
# who are the low-star plays from? distribution across players
low = p[p.star < 5]
print("players contributing low-star plays:", low.user_id.nunique(), "/", p.user_id.nunique())
print("median low-star plays per player:", int(low.groupby('user_id').size().median()))
