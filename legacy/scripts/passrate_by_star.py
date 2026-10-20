import pandas as pd, numpy as np
bm = pd.read_csv("data/interim/beatmaps.csv")
for c in ["playcount", "passcount", "star"]:
    bm[c] = pd.to_numeric(bm[c], errors="coerce")
g = bm[(bm.playmode == 3) & (bm["keys"] == 4) & (bm.playcount > 0) & bm.passcount.notna()].copy()
g["pr"] = g.passcount / g.playcount
bins = [0, 2, 3, 4, 5, 6, 7, 8, 20]
g["bucket"] = pd.cut(g.star, bins)
print("4K mania pass rate by official star bucket:")
print(f"{'star':12s} {'maps':>7s} {'weighted':>9s} {'median':>7s} {'share<25%':>10s} {'plays':>16s}")
for b, sub in g.groupby("bucket", observed=True):
    print(f"{str(b):12s} {len(sub):7d} {sub.passcount.sum()/sub.playcount.sum():9.3f} {sub.pr.median():7.3f} "
          f"{(sub.pr<0.25).mean():10.3f} {sub.playcount.sum():16,.0f}")
print()
for lo, hi in [(5, 20), (6, 20), (7, 20)]:
    s = g[(g.star >= lo)]
    print(f"star>={lo}: maps={len(s)} weighted={s.passcount.sum()/s.playcount.sum():.3f} median={s.pr.median():.3f} share<25%={(s.pr<0.25).mean():.3f}")
