import re
from pathlib import Path
import pandas as pd
D1 = Path("data/raw/extracted/2026_09_01_performance_mania_top_1000")
bm = pd.read_csv("data/interim/beatmaps.csv")
for c in ["playcount", "passcount"]:
    bm[c] = pd.to_numeric(bm[c], errors="coerce")
csv = bm.set_index("beatmap_id")
raw = (D1 / "osu_beatmaps.sql").read_text(encoding="utf-8", errors="ignore")
print("file chars:", len(raw))
for bid in [20305, 38912, 193127]:
    m = re.search(r"[(,]%d," % bid, raw)
    if not m:
        print(f"  beatmap {bid}: NOT FOUND in dump (only 235k of ~5.8M beatmaps are included)")
        continue
    seg = raw[m.start() + 1:m.start() + 500]
    # split on top-level commas, respecting quotes
    fields, cur, q = [], "", False
    for ch in seg:
        if ch == "'":
            q = not q
        if ch == "," and not q:
            fields.append(cur); cur = ""
        else:
            cur += ch
        if len(fields) > 23:
            break
    print(f"  beatmap {bid}: raw[21]={fields[21].strip()} raw[22]={fields[22].strip()} "
          f"| csv playcount={int(csv.loc[bid].playcount)} passcount={int(csv.loc[bid].passcount)}")
