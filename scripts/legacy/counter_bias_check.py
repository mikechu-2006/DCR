"""Is the random sample's implied pass rate inflated by the frozen fail/exit counters?

fail_count / exit_count are legacy-only counters (no modern processor writes them), while playcount
keeps growing. So players whose history is mostly lazer-era have fail+exit ~ 0 -> implied pass ~ 1.
"""
import sys
from pathlib import Path
import pandas as pd, numpy as np
sys.path.insert(0, "scripts")
from parse_dump import iter_insert_lines, rows_simple

for name, d in [("random 10000", Path("data/raw/extracted_random/2026_09_01_performance_mania_random_10000")),
                ("top 1000", Path("data/raw/extracted/2026_09_01_performance_mania_top_1000"))]:
    rows = []
    for line in iter_insert_lines(d / "osu_user_stats_mania.sql"):
        for r in rows_simple(line):
            if len(r) >= 29:
                rows.append([int(r[8]), int(r[19]), int(r[20])])
    df = pd.DataFrame(rows, columns=["playcount", "fail", "exit"])
    df = df[df.playcount > 0].copy()
    df["pass_rate"] = 1 - (df.fail + df.exit) / df.playcount
    df["zero_counters"] = (df.fail + df.exit) == 0
    print(f"\n=== {name} (users={len(df)}) ===")
    print("  share with fail+exit == 0 (implied pass == 100%%): %.3f" % df.zero_counters.mean())
    bins = [0, 100, 500, 2000, 10000, 50000, 10**9]
    df["b"] = pd.cut(df.playcount, bins)
    print(f"  {'playcount':>16s} {'users':>7s} {'agg_pass':>9s} {'median_pass':>12s} {'zero_cnt%':>10s}")
    for b, g in df.groupby("b", observed=True):
        P = g.playcount.sum()
        print(f"  {str(b):>16s} {len(g):7d} {1-(g.fail.sum()+g.exit.sum())/P:9.3f} {g.pass_rate.median():12.3f} {g.zero_counters.mean()*100:10.1f}")
