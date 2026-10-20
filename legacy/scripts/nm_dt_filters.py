"""Which filter actually removes the NM side? and how many DT plays share a *year* with an NM play?"""
import numpy as np, pandas as pd
cells = pd.read_parquet("data/processed/charts_v2_1k.parquet",
                        columns=["user_id", "beatmap_id", "chart_id", "rate", "primary",
                                 "attempts", "n_scores", "acc_med"])
plays = pd.read_parquet("data/processed/plays_v2_1k.parquet",
                        columns=["user_id", "beatmap_id", "rate", "year"])

def overlap(df, name):
    nm = df[df.rate == 0].groupby("beatmap_id").user_id.apply(set)
    dt = df[df.rate == 1].groupby("beatmap_id").user_id.apply(set)
    common = nm.index.intersection(dt.index)
    sh = np.array([len(nm[m] & dt[m]) / len(dt[m]) for m in common]) if len(common) else np.array([0.0])
    print(f"{name:38s} cells={len(df):8d} DT charts={df[df.rate==1].chart_id.nunique():6d} "
          f"overlap mean={sh.mean():.3f} median={np.median(sh):.3f}")
    return sh.mean()

c = cells
print("=== overlap under different filters (1k data) ===")
overlap(c, "no filter")
overlap(c[c.attempts.fillna(0) > 2], "attempts(player,map) > 2  [current]")
overlap(c[c.n_scores > 2], "n_scores(cell) > 2")
overlap(c[(c.n_scores > 2) & (c.attempts.fillna(0) > 2)], "both")
overlap(c[c.primary], "current primary (attempts>2, acc>=.90)")

print("\n=== how many DT cells have an NM cell on the same map from the SAME year? ===")
yrs = plays.groupby(["user_id", "beatmap_id", "rate"]).year.apply(set).unstack()
yrs = yrs.dropna(subset=[0, 1])
inter = np.array([len(a & b) for a, b in zip(yrs[0], yrs[1])])
print(f"pairs with both rates: {len(yrs)}  share with >=1 common calendar year: {np.mean(inter>0):.3f}")
dt_plays = plays[plays.rate == 1].merge(yrs[[0]].rename(columns={0: "nm_years"}), on=["user_id", "beatmap_id"], how="inner")
same = np.array([y in ys for y, ys in zip(dt_plays.year, dt_plays.nm_years)])
print(f"DT plays whose (player, map, year) also has an NM play: {same.mean():.3f}  (n={len(dt_plays)})")
