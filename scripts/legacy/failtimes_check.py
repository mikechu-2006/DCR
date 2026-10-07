import sys
from pathlib import Path
import pandas as pd, numpy as np
sys.path.insert(0, "scripts")
from parse_dump import iter_insert_lines, rows_simple

D = Path("data/raw/extracted/2026_09_01_performance_mania_top_1000")
bm = pd.read_csv("data/interim/beatmaps.csv")
b4 = bm[(bm.playmode == 3) & (bm["keys"] == 4)].copy()
b4["playcount"] = pd.to_numeric(b4.playcount); b4["passcount"] = pd.to_numeric(b4.passcount)
b4 = b4.set_index("beatmap_id")

acc = {}
ncols = set()
for line in iter_insert_lines(D / "osu_beatmap_failtimes.sql"):
    for r in rows_simple(line):
        ncols.add(len(r))
        bid, typ = int(r[0]), r[1]
        tot = sum(int(x) for x in r[2:] if x.isdigit())
        acc[(bid, typ)] = acc.get((bid, typ), 0) + tot
ft = pd.DataFrame([{"beatmap_id": k[0], "type": k[1], "count": v} for k, v in acc.items()])
print("failtimes rows:", len(ft), "| fields per row:", sorted(ncols)[:3], "...")
piv = ft.pivot_table(index="beatmap_id", columns="type", values="count", aggfunc="sum").fillna(0)
j = b4.join(piv, how="inner")
print("4K maps matched:", len(j))
s = j[["playcount", "passcount", "fail", "exit"]].sum()
print(s.to_string())
print()
print("(pass+fail+exit)/playcount = %.3f" % ((s.passcount + s.fail + s.exit) / s.playcount))
print("(pass+fail)/playcount      = %.3f" % ((s.passcount + s.fail) / s.playcount))
print("pass/playcount             = %.3f" % (s.passcount / s.playcount))
print("fail/playcount             = %.3f" % (s.fail / s.playcount))
print("exit/playcount             = %.3f" % (s.exit / s.playcount))
