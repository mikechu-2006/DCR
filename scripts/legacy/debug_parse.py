import pandas as pd, csv
from pathlib import Path
import sys
sys.path.insert(0, "scripts")
from parse_dump import iter_insert_lines, rows_simple

f = Path("data/raw/extracted/2026_09_01_performance_mania_top_1000/osu_scores_mania_high.sql")
tot_grep = 0
tot_parsed = 0
nlines = 0
from collections import Counter
lens = Counter()
for k, line in enumerate(iter_insert_lines(f)):
    nlines += 1
    rows = list(rows_simple(line))
    tot_parsed += len(rows)
    lens.update(len(r) for r in rows)
    if k < 3:
        print("line", k, "rows", len(rows), "first row len", len(rows[0]), rows[0][:6])
print("insert lines:", nlines, "parsed rows:", tot_parsed)
print("row field-length distribution:", lens.most_common(5))

bm = pd.read_csv("data/interim/beatmaps.csv")
mh = pd.read_csv("data/interim/mania_high.csv")
print("beatmaps rows:", len(bm), "mania_high rows:", len(mh))
print("mania_high users:", mh.user_id.nunique(), "beatmaps:", mh.beatmap_id.nunique())
print("beatmap_id overlap: in beatmaps.csv:", mh.beatmap_id.isin(bm.beatmap_id).sum(), "/", len(mh))
print("beatmaps playmode counts:", bm.playmode.value_counts().to_dict())
print("beatmaps keys counts:", bm["keys"].value_counts().head(8).to_dict())
