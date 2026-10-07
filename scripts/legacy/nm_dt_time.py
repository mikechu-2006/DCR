"""How much NM/DT overlap actually survives the filters, and how far apart in time are the plays?"""
import numpy as np, pandas as pd
pd.set_option("display.width", 200)

cells = pd.read_parquet("data/processed/charts_v2_1k.parquet",
                        columns=["user_id", "beatmap_id", "chart_id", "rate", "primary", "attempts",
                                 "n_scores", "year_first", "year_last"])
plays = pd.read_parquet("data/processed/plays_v2_1k.parquet",
                        columns=["user_id", "beatmap_id", "rate", "year", "doy"])

def overlap(df, name):
    nm = df[df.rate == 0].groupby("beatmap_id").user_id.apply(set)
    dt = df[df.rate == 1].groupby("beatmap_id").user_id.apply(set)
    common = nm.index.intersection(dt.index)
    if not len(common):
        print(name, "no maps"); return
    sh = np.array([len(nm[m] & dt[m]) / len(dt[m]) for m in common])
    print(f"{name:34s} maps={len(common):6d}  DT players also having NM: mean={sh.mean():.3f} "
          f"median={np.median(sh):.3f}  zero-overlap maps={np.mean(sh==0):.3f}")

overlap(cells, "all cells (no filter)")
overlap(cells[cells.primary], "primary cells")

# time structure: per (user, beatmap, rate) span, in continuous years
plays["t"] = plays.year + (plays.doy - 1) / 365.0
span = plays.groupby(["user_id", "beatmap_id", "rate"]).t.agg(["min", "max", "size"]).reset_index()
sp = span.pivot_table(index=["user_id", "beatmap_id"], columns="rate",
                      values=["min", "max", "size"], aggfunc="first")
both = sp.dropna(subset=[("min", 0), ("min", 1)])
print(f"\n(user, beatmap) pairs with both NM and DT plays (no filter): {len(both)}")

prim_keys = set(map(tuple, cells[cells.primary][["user_id", "beatmap_id"]].drop_duplicates().values))
both_prim = both[[k in prim_keys for k in both.index]]
print(f"...of which survive the primary filter: {len(both_prim)}")

for tag, d in (("no filter", both), ("primary", both_prim)):
    a0, a1 = d[("min", 0)].values, d[("max", 0)].values
    b0, b1 = d[("min", 1)].values, d[("max", 1)].values
    same_year = (np.floor(a1) == np.floor(b1)) | (np.floor(a0) == np.floor(b0)) | \
                ((a0 <= b1) & (b0 <= a1))
    gap = np.where(b0 > a1, b0 - a1, np.where(a0 > b1, a0 - b1, 0.0))
    print(f"\n[{tag}] pairs={len(d)}")
    print(f"  NM/DT time spans overlap at all : {same_year.mean():.3f}")
    print(f"  gap between spans (years): median={np.median(gap):.2f} mean={gap.mean():.2f} "
          f"share gap>1y={np.mean(gap>1):.3f} >3y={np.mean(gap>3):.3f}")
    print(f"  NM plays per pair: median={np.median(d[('size',0)]):.0f}  "
          f"DT plays per pair: median={np.median(d[('size',1)]):.0f}")
