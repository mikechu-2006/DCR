"""Stream osu_scores_mania_high.sql straight into a compact parquet (no giant intermediate CSV).

Keeps only 4K mania plays and only the columns the pipeline needs.
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from parse_dump import iter_insert_lines, rows_simple  # noqa: E402

COLS = ["score_id", "beatmap_id", "user_id", "c50", "c100", "c300", "cmiss", "c320", "c200",
        "enabled_mods", "year", "doy"]
IDX = [0, 1, 2, 6, 7, 8, 9, 10, 11, 13]
CUM = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
DTYPES = {"score_id": np.int64, "beatmap_id": np.int32, "user_id": np.int32, "c50": np.int16,
          "c100": np.int16, "c300": np.int16, "cmiss": np.int16, "c320": np.int16, "c200": np.int16,
          "enabled_mods": np.int32, "year": np.int16, "doy": np.int16}
FLUSH = 1_000_000


def main(src: Path, out: Path, beatmap_ids: set[int]):
    writer = None
    buf = []
    n = kept = 0
    t0 = time.time()
    for line in iter_insert_lines(src):
        for r in rows_simple(line):
            n += 1
            if int(r[1]) not in beatmap_ids:
                continue
            d = r[14]                       # YYYY-MM-DD HH:MM:SS (quotes already stripped)
            buf.append([r[i] for i in IDX] + [d[0:4], CUM[int(d[5:7]) - 1] + int(d[8:10])])
            kept += 1
            if len(buf) >= FLUSH:
                writer = flush(buf, out, writer)
                buf = []
                print(f"  kept={kept} seen={n} {time.time()-t0:.0f}s", flush=True)
    if buf:
        writer = flush(buf, out, writer)
    if writer:
        writer.close()
    print(f"done rows={n} kept(4K)={kept} {time.time()-t0:.0f}s")


def flush(buf, out, writer):
    df = pd.DataFrame(buf, columns=COLS)
    for c, t in DTYPES.items():
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(t)
    table = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(out, table.schema)
    writer.write_table(table)
    return writer


if __name__ == "__main__":
    import pandas as pd
    bm = pd.read_csv("data/interim/beatmaps.csv")
    ids = set(bm[(bm.playmode == 3) & (bm["keys"] == 4)].beatmap_id.astype("int64"))
    print("4K beatmap ids:", len(ids))
    main(Path(sys.argv[1]), Path(sys.argv[2]), ids)
