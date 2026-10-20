import pandas as pd, numpy as np
pd.set_option("display.width", 200)
cells = pd.read_parquet("data/processed/charts_v2_1k.parquet")
print("cells:", len(cells), "| primary:", int(cells.primary.sum()))
prim = cells[cells.primary]
g = prim.groupby("rate")
print("\n--- cells per chart by rate (primary only) ---")
for r, sub in g:
    cc = sub.groupby("chart_id").size()
    pp = sub.groupby("chart_id").user_id.nunique()
    print(f"rate={r}: charts={sub.chart_id.nunique():6d} cells={len(sub):8d} "
          f"cells/chart median={cc.median():5.1f} mean={cc.mean():6.2f} | players/chart median={pp.median():4.0f} mean={pp.mean():5.1f}")

# player overlap between NM and DT chart of the same map
nm = prim[prim.rate == 0].groupby("beatmap_id").user_id.apply(set)
dt = prim[prim.rate == 1].groupby("beatmap_id").user_id.apply(set)
common = nm.index.intersection(dt.index)
print(f"\nmaps with both NM and DT cells: {len(common)}")
ov = []
for m in common:
    a, b = nm[m], dt[m]
    ov.append(len(a & b) / len(b))
ov = np.array(ov)
print("share of DT players that also have an NM cell on the same map:")
print("  mean %.3f median %.3f p25 %.3f p75 %.3f | maps where >50%% overlap: %.3f" %
      (ov.mean(), np.median(ov), np.percentile(ov, 25), np.percentile(ov, 75), (ov > 0.5).mean()))
print("  maps with zero overlap: %.3f" % (ov == 0).mean())

# same for players: how many play both NM and DT at all
ps = prim.groupby("rate").user_id.apply(set)
print("\nplayers with NM cells:", len(ps.get(0, set())), "DT cells:", len(ps.get(1, set())),
      "both:", len(ps.get(0, set()) & ps.get(1, set())))
