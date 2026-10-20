"""Validate the pass-rate / playcount numbers three ways."""
import sys
from pathlib import Path
import pandas as pd, numpy as np, re
sys.path.insert(0, "scripts")
from parse_dump import iter_insert_lines, rows_simple

INT = Path("data/interim")
D1 = Path("data/raw/extracted/2026_09_01_performance_mania_top_1000")
D10 = Path("data/raw/extracted_10k/2026_09_01_performance_mania_top_10000")


def stats(prefix, path):
    rows = []
    for line in iter_insert_lines(path / "osu_user_stats_mania.sql"):
        for r in rows_simple(line):
            if len(r) >= 29:
                rows.append([int(r[8]), int(r[19]), int(r[20])])
    d = pd.DataFrame(rows, columns=["playcount", "fail", "exit"])
    P, F, X = d.playcount.sum(), d.fail.sum(), d.exit.sum()
    print(f"[{prefix}] users={len(d):6d} playcount={P:12,.0f} fail={F/P:6.3f} exit={X/P:6.3f} "
          f"-> implied pass = {1-(F+X)/P:6.3f}")
    return d


print("=== 1) user-level counters: top1k vs top10k ===")
stats("top1k", D1)
stats("top10k", D10)

print("\n=== 2) beatmap pass rates by slice ===")
bm = pd.read_csv(INT / "beatmaps.csv")
for c in ["playcount", "passcount", "keys", "playmode"]:
    bm[c] = pd.to_numeric(bm[c], errors="coerce")
bm = bm[(bm.playcount > 0) & bm.passcount.notna()]
bm["pass_rate"] = bm.passcount / bm.playcount
KIND = {0: "osu!std", 1: "taiko", 2: "catch", 3: "mania"}
for m, g in bm.groupby("playmode"):
    print(f"{KIND[int(m)]:8s} maps={len(g):7d} weighted_pass={g.passcount.sum()/g.playcount.sum():.3f} "
          f"median={g.pass_rate.median():.3f} share<25%={ (g.pass_rate<0.25).mean():.3f}")
m4 = bm[(bm.playmode == 3) & (bm.keys == 4)]
print("\n4K mania by popularity decile (playcount):")
m4 = m4.assign(dec=pd.qcut(m4.playcount, 10, labels=False, duplicates="drop"))
for d, g in m4.groupby("dec"):
    print(f"  decile {int(d)} playcount<={g.playcount.max():9.0f} weight={g.playcount.sum()/m4.playcount.sum():.3f} "
          f"weighted_pass={g.passcount.sum()/g.playcount.sum():.3f} median={g.pass_rate.median():.3f}")
print("\n4K mania by key count:")
for k, g in bm[bm.playmode == 3].groupby("keys"):
    if len(g) > 200:
        print(f"  {int(k)}K maps={len(g):6d} weighted_pass={g.passcount.sum()/g.playcount.sum():.3f} median={g.pass_rate.median():.3f}")

print("\n=== 3) raw SQL spot-check (playcount=col21, passcount=col22) ===")
raw = (D1 / "osu_beatmaps.sql").read_text(encoding="utf-8", errors="ignore")
csv = bm.set_index("beatmap_id")
for bid in [20305, 38912, 193127]:
    m = re.search(r"VALUES \(%d," % bid, raw)
    seg = raw[m.start():m.start() + 420]
    fields = re.findall(r"'[^']*'|NULL|[^,()]+", seg)
    print(f"  beatmap {bid}: raw cols 21/22 = {fields[21].strip()}/{fields[22].strip()} | csv = "
          f"{int(csv.loc[bid].playcount)}/{int(csv.loc[bid].passcount)}")
