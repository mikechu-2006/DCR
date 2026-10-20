"""Parse osu_beatmap_difficulty.sql -> per-(beatmap, mods) star rating (mode=3 mania only)."""
from __future__ import annotations
import csv, sys
from pathlib import Path

csv.field_size_limit(10 ** 9)


def main(src: Path, out: Path, bm4_ids: set[int]):
    n = kept = 0
    with open(src, encoding="utf-8", errors="replace") as f, open(out, "w", newline="", encoding="utf-8") as fo:
        wr = csv.writer(fo)
        wr.writerow(["beatmap_id", "mods", "star"])
        for line in f:
            if not line.startswith("INSERT INTO"):
                continue
            body = line[line.index("VALUES") + 6:].strip()[1:]
            if body.endswith(");"):
                body = body[:-2]
            for row in body.split("),("):
                p = row.split(",")
                if len(p) < 5:
                    continue
                n += 1
                if p[1] != "3":          # mode 3 = mania
                    continue
                bid = int(p[0])
                if bid not in bm4_ids:
                    continue
                wr.writerow([bid, p[2], p[3]])
                kept += 1
    print(f"rows={n} kept(4K mania)={kept}")


if __name__ == "__main__":
    import pandas as pd
    bm = pd.read_csv("data/interim/beatmaps.csv")
    ids = set(bm[(bm.playmode == 3) & (bm["keys"] == 4)].beatmap_id.astype("int64"))
    main(Path(sys.argv[1]), Path(sys.argv[2]), ids)
