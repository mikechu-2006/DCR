import pandas as pd, numpy as np
bm = pd.read_csv("data/interim/beatmaps.csv")
for c in ["playcount", "passcount", "star", "max_combo", "countTotal"]:
    bm[c] = pd.to_numeric(bm[c], errors="coerce")
bm = bm[(bm.playmode == 3)]
bm["pass_rate"] = bm.passcount / bm.playcount

def rep(name, g, minplays=1000):
    g = g[(g.playcount > 0) & (g.passcount.notna())]
    if len(g) == 0:
        print(name, "empty"); return
    w = g.passcount.sum() / g.playcount.sum()
    print(f"{name:32s} maps={len(g):6d} weighted={w:.3f} median={g.pass_rate.median():.3f} "
          f"share<25%={(g.pass_rate<0.25).mean():.3f} plays={g.playcount.sum():,.0f}")

rep("all mania maps", bm)
rep("all mania (playcount>=1k)", bm[bm.playcount >= 1000])
rep("all mania (playcount>=10k)", bm[bm.playcount >= 10000])
rep("all mania (playcount>=100k)", bm[bm.playcount >= 100000])

for f, col in [("data/processed/charts_v2_10k.parquet", "beatmap_id"),
               ("data/processed/matrix_n1000_chart.parquet", "beatmap_id")]:
    try:
        ids = set(pd.read_parquet(f, columns=[col])[col].unique())
    except Exception as e:
        print(f, "ERR", e); continue
    rep("in " + f.split("/")[-1], bm[bm.beatmap_id.isin(ids)])

m4 = bm[bm["keys"] == 4].copy()
m4["dec"] = pd.qcut(m4.playcount, 10, labels=False, duplicates="drop")
print("\n4K mania by playcount decile:")
for d in sorted(m4.dec.dropna().unique()):
    g = m4[m4.dec == d]
    print(f"  decile {int(d)}: playcount {g.playcount.min():>9.0f}~{g.playcount.max():<10.0f} "
          f"weighted={g.passcount.sum()/g.playcount.sum():.3f} median={g.pass_rate.median():.3f} "
          f"share<25%={(g.pass_rate<0.25).mean():.3f} star_med={g.star.median():.2f}")
