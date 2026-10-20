from pathlib import Path
import json, re
p = Path("data/raw/extracted/2026_09_01_performance_mania_top_1000/scores.sql")
with open(p, encoding="utf-8", errors="replace") as f:
    for line in f:
        if line.startswith("INSERT INTO"):
            break
body = line[line.index("VALUES") + 6:].strip()[1:]
if body.endswith(");"):
    body = body[:-2]
row = body.split("),(")[0]
parts = row.split(",")
print("n parts:", len(parts))
print("head:", parts[:12])
raw = ",".join(parts[12:-7])
print("raw json repr (first 300):", repr(raw[:300]))
print("tail:", parts[-7:])
