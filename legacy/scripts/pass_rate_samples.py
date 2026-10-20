"""Implied pass rate (1 - (fail+exit)/playcount) for three player samples."""
import sys
from pathlib import Path
import pandas as pd, numpy as np
sys.path.insert(0, "scripts")
from parse_dump import iter_insert_lines, rows_simple

SAMPLES = [
    ("top 1000", Path("data/raw/extracted/2026_09_01_performance_mania_top_1000")),
    ("top 10000", Path("data/raw/extracted_10k/2026_09_01_performance_mania_top_10000")),
    ("random 10000", Path("data/raw/extracted_random/2026_09_01_performance_mania_random_10000")),
]
out = []
for name, d in SAMPLES:
    p = d / "osu_user_stats_mania.sql"
    if not p.exists():
        print(f"[{name}] missing {p}"); continue
    rows = []
    for line in iter_insert_lines(p):
        for r in rows_simple(line):
            if len(r) >= 29:
                rows.append([int(r[8]), int(r[19]), int(r[20])])
    df = pd.DataFrame(rows, columns=["playcount", "fail", "exit"])
    df = df[df.playcount > 0]
    df["pass_rate"] = 1 - (df.fail + df.exit) / df.playcount
    P, F, X = df.playcount.sum(), df.fail.sum(), df.exit.sum()
    print(f"[{name:12s}] users={len(df):6d} agg_playcount={P:12,.0f} fail={F/P:.3f} exit={X/P:.3f} "
          f"-> agg implied pass={1-(F+X)/P:.3f}")
    print(f"{'':16s} per-user pass_rate: median={df.pass_rate.median():.3f} mean={df.pass_rate.mean():.3f} "
          f"p10={df.pass_rate.quantile(.1):.3f} p90={df.pass_rate.quantile(.9):.3f} "
          f"share<25%={(df.pass_rate<0.25).mean():.3f}")
    print(f"{'':16s} playcount per user: median={df.playcount.median():,.0f}")
    out.append({"sample": name, "users": len(df), "agg_pass": 1 - (F + X) / P,
                "median_user_pass": float(df.pass_rate.median())})
pd.DataFrame(out).to_csv("data/processed/pass_rate_samples.csv", index=False)
print("\nsaved data/processed/pass_rate_samples.csv")
