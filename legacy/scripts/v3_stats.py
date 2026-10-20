import numpy as np, pandas as pd
cells = pd.read_parquet("data/processed/charts_v2_1k_v3.parquet")
print("cells:", len(cells), "primary:", int(cells.primary.sum()))
prim = cells[cells.primary]
print("\n--- v3 primary: cells per chart by rate ---")
for r, sub in prim.groupby("rate"):
    cc = sub.groupby("chart_id").size(); pp = sub.groupby("chart_id").user_id.nunique()
    print(f"rate={r}: charts={sub.chart_id.nunique():6d} cells={len(sub):7d} "
          f"cells/chart median={cc.median():4.1f} | players/chart median={pp.median():4.0f}")
nm = prim[prim.rate == 0].groupby("beatmap_id").user_id.apply(set)
dt = prim[prim.rate == 1].groupby("beatmap_id").user_id.apply(set)
common = nm.index.intersection(dt.index)
sh = np.array([len(nm[m] & dt[m]) / len(dt[m]) for m in common])
print(f"\nNM/DT overlap (v3 primary): maps={len(common)} mean={sh.mean():.3f} median={np.median(sh):.3f} "
      f"zero={np.mean(sh==0):.3f}")
print("players: NM", prim[prim.rate==0].user_id.nunique(), "DT", prim[prim.rate==1].user_id.nunique(),
      "both", len(set(prim[prim.rate==0].user_id) & set(prim[prim.rate==1].user_id)))
