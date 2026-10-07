import pandas as pd, numpy as np
bm = pd.read_csv("data/interim/beatmaps.csv")
for c in ["playcount", "passcount", "star"]:
    bm[c] = pd.to_numeric(bm[c], errors="coerce")
mat = set(pd.read_parquet("data/processed/matrix_n1000_chart.parquet", columns=["beatmap_id"]).beatmap_id.unique())
g = bm[bm.beatmap_id.isin(mat)]
g = g[(g.playcount > 0)]
pr = g.passcount / g.playcount
print("n=1000 matrix charts:", len(g), "maps")
print("pass_rate quantiles:", {q: round(float(pr.quantile(q)), 3) for q in [0.1, 0.25, 0.5, 0.75, 0.9]})
print("share <25%%: %.3f  <30%%: %.3f  <20%%: %.3f" % ((pr < .25).mean(), (pr < .30).mean(), (pr < .20).mean()))
print("star quantiles:", {q: round(float(g.star.quantile(q)), 2) for q in [0.1, 0.5, 0.9]})
# same for the full 4K pool for contrast
h = bm[(bm.playmode == 3) & (bm["keys"] == 4) & (bm.playcount > 0)]
pr2 = h.passcount / h.playcount
print("\nall 4K mania: maps", len(h), "| share <25%%: %.3f  <30%%: %.3f | median %.3f" % ((pr2 < .25).mean(), (pr2 < .30).mean(), pr2.median()))
print("median star all 4K: %.2f" % h.star.median())
