import sys
from pathlib import Path
import pandas as pd, numpy as np
sys.path.insert(0, "scripts")
from parse_dump import iter_insert_lines, rows_simple

D1 = Path("data/raw/extracted/2026_09_01_performance_mania_top_1000")
bm = pd.read_csv("data/interim/beatmaps.csv")
for c in ["playcount", "passcount", "keys"]:
    bm[c] = pd.to_numeric(bm[c], errors="coerce")

acc = {}
for line in iter_insert_lines(D1 / "osu_beatmap_failtimes.sql"):
    for r in rows_simple(line):
        k = (int(r[0]), r[1])
        acc[k] = acc.get(k, 0) + sum(int(x) for x in r[2:] if x.isdigit())
ft = pd.DataFrame([{"beatmap_id": k[0], "type": k[1], "count": v} for k, v in acc.items()])
piv = ft.pivot_table(index="beatmap_id", columns="type", values="count", aggfunc="sum").fillna(0)

def breakdown(name, ids):
    g = bm[bm.beatmap_id.isin(ids)].set_index("beatmap_id").join(piv, how="inner")
    P, F, X = g.playcount.sum(), g.fail.sum(), g.exit.sum()
    print(f"{name:34s} maps={len(g):6d} playcount={P:14,.0f}  pass={g.passcount.sum()/P:.3f} "
          f"fail={F/P:.3f} exit={X/P:.3f} sum={(g.passcount.sum()+F+X)/P:.3f}")

m4 = bm[(bm.playmode == 3) & (bm["keys"] == 4)]
breakdown("all 4K mania", set(m4.beatmap_id))
breakdown("10k dataset maps", set(pd.read_parquet("data/processed/charts_v2_10k.parquet", columns=["beatmap_id"]).beatmap_id.unique()))
breakdown("n=1000 matrix charts", set(pd.read_parquet("data/processed/matrix_n1000_chart.parquet", columns=["beatmap_id"]).beatmap_id.unique()))
top = m4.nlargest(200, "playcount")
breakdown("top-200 4K by playcount", set(top.beatmap_id))
