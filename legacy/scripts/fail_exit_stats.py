import sys
from pathlib import Path
sys.path.insert(0, "scripts")
from parse_dump import iter_insert_lines, rows_simple
import pandas as pd, numpy as np

f = Path("data/raw/extracted/2026_09_01_performance_mania_top_1000/osu_user_stats_mania.sql")
rows = []
for line in iter_insert_lines(f):
    for r in rows_simple(line):
        if len(r) >= 29:
            rows.append([int(r[0]), int(r[8]), int(r[19]), int(r[20])])
d = pd.DataFrame(rows, columns=["user_id", "playcount", "fail_count", "exit_count"])
print("users:", len(d))
for c in ["playcount", "fail_count", "exit_count"]:
    print(f"{c:12s} median {d[c].median():8.0f}  mean {d[c].mean():9.1f}")
print()
print("fail_count / playcount : median %.3f  mean %.3f" % ((d.fail_count/d.playcount).median(), (d.fail_count/d.playcount).mean()))
print("exit_count / playcount : median %.3f  mean %.3f" % ((d.exit_count/d.playcount).median(), (d.exit_count/d.playcount).mean()))
print()
print(d.head(5).to_string(index=False))
