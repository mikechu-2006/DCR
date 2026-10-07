import sys
from pathlib import Path
sys.path.insert(0, "scripts")
from parse_dump import iter_insert_lines, rows_simple

f = Path("data/raw/extracted_10k/2026_09_01_performance_mania_top_10000/osu_scores_mania_high.sql")
bad = 0
seen = 0
for line in iter_insert_lines(f):
    for r in rows_simple(line):
        seen += 1
        if len(r) != 19:
            if bad < 3:
                print("LEN", len(r), r)
            bad += 1
            continue
        d = r[14]
        try:
            y, m, dd = int(d[1:5]), int(d[6:8]), int(d[9:11])
        except Exception as e:
            if bad < 5:
                print("BAD date field:", repr(d), "| row:", r)
            bad += 1
        if seen > 400000:
            break
    if seen > 400000:
        break
print("seen", seen, "bad", bad)
