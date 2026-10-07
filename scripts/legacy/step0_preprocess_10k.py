#!/usr/bin/env python3
"""Step 0 for the 10k dump: legacy-only play table **with second-precision timestamps**.

Same output contract as `scripts/step0_preprocess.py`:

    player_id, beatmap_id, timestamp, playcount_cur, loss

Differences from step0_preprocess.py:
* the 10k dump has no `scores.sql` and no `osu_beatmaps.sql`, so the 4K beatmap
  whitelist is read from `data/interim/beatmaps.csv` (which fully covers the
  beatmaps referenced by `charts_v2_10k_v3.parquet`);
* output is parquet (16.9 M rows as CSV would be ~850 MB for no benefit).

`playcount_cur` is the 1-based rank of the play inside its (player, beatmap)
cell, ordered by (timestamp, score_id) -- identical definition to step 0.

Usage
-----
    python scripts/step0_preprocess_10k.py \
        --dump-dir data/raw/extracted_10k/2026_09_01_performance_mania_top_10000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_dump import iter_insert_lines, rows_simple  # noqa: E402

MIRROR_BIT = 1 << 30
FLUSH = 1_000_000

# legacy row layout:
#   0 score_id  1 beatmap_id  2 user_id  3 score  4 maxcombo  5 rank
#   6 count50   7 count100    8 count300  9 countmiss 10 countgeki 11 countkatu
#   12 perfect  13 enabled_mods           14 date    15 pp
LEGACY_TOTAL = (6, 7, 8, 9, 10, 11)
LEGACY_NUM = ((10, 300), (8, 300), (11, 200), (7, 100), (6, 50))

COLS = ["player_id", "beatmap_id", "timestamp", "playcount_cur", "loss"]
DTYPES = {"player_id": np.int32, "beatmap_id": np.int32,
          "playcount_cur": np.int32, "loss": np.float32}

STATS: dict[str, int] = {}


def note(k: str, n: int = 1) -> None:
    STATS[k] = STATS.get(k, 0) + n


def loss_of(num: float, total: float) -> float:
    """ACC = (num + 250) / (300 * (total + 1)); loss = log(1 - ACC)."""
    return float(np.log1p(-(num + 250.0) / (300.0 * (total + 1.0))))


def read_4k_ids(beatmaps_csv: Path) -> set[int]:
    bm = pd.read_csv(beatmaps_csv, usecols=["beatmap_id", "keys", "playmode"])
    m = (bm.playmode == 3) & (bm["keys"] == 4)
    return set(bm.loc[m, "beatmap_id"].astype("int64"))


def extract(dump_dir: Path, ids: set[int], scratch: Path) -> Path:
    src = dump_dir / "osu_scores_mania_high.sql"
    if not src.exists():
        raise SystemExit(f"missing {src}")
    scratch.mkdir(parents=True, exist_ok=True)
    out = scratch / "plays_raw.parquet"

    buf: list = []
    writer = None
    n = kept = 0
    t0 = time.time()
    for line in iter_insert_lines(src):
        for r in rows_simple(line):
            n += 1
            bid = int(r[1])
            if bid not in ids:
                continue
            note("legacy_4k")
            if int(r[13]) & ~MIRROR_BIT:          # only CL / CL+MR survive
                note("mod_dropped")
                continue
            total = float(sum(int(r[i]) for i in LEGACY_TOTAL))
            if total <= 0:
                note("empty")
                continue
            num = float(sum(int(r[i]) * w for i, w in LEGACY_NUM))
            buf.append((int(r[2]), bid, int(r[0]), r[14], loss_of(num, total)))
            kept += 1
            if len(buf) >= FLUSH:
                writer = flush(buf, out, writer)
                buf = []
                print(f"  kept={kept:,} seen={n:,} {time.time()-t0:.0f}s", flush=True)
    if buf:
        writer = flush(buf, out, writer)
    if writer is not None:
        writer.close()
    print(f"[extract] seen={n:,} kept={kept:,} in {time.time()-t0:.0f}s", flush=True)
    return out


def flush(buf, out: Path, writer):
    df = pd.DataFrame(buf, columns=["player_id", "beatmap_id", "score_id", "ts", "loss"])
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(out, tbl.schema)
    writer.write_table(tbl)
    return writer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", required=True, type=Path)
    ap.add_argument("--beatmaps-csv", type=Path, default=Path("data/interim/beatmaps.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/step0_10k.parquet"))
    ap.add_argument("--work-dir", type=Path, default=Path("data/interim_10k/step0"))
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--reuse-raw", action="store_true",
                    help="reuse an existing scratch plays_raw.parquet (skip the SQL pass)")
    args = ap.parse_args()

    t0 = time.time()
    ids = read_4k_ids(args.beatmaps_csv)
    print(f"[step0_10k] 4K mania beatmaps: {len(ids):,}", flush=True)

    raw = args.work_dir / "plays_raw.parquet"
    if args.reuse_raw and raw.exists():
        print(f"[step0_10k] reusing {raw}", flush=True)
    else:
        raw = extract(args.dump_dir, ids, args.work_dir)
    plays = pd.read_parquet(raw)
    print(f"[step0_10k] raw kept rows: {len(plays):,}", flush=True)
    plays = plays.rename(columns={"ts": "timestamp"})
    plays["timestamp"] = pd.to_datetime(plays["timestamp"], format="%Y-%m-%d %H:%M:%S")

    key = ["player_id", "beatmap_id"]
    plays = plays.sort_values(key + ["timestamp", "score_id"], kind="stable").reset_index(drop=True)
    plays["playcount_cur"] = plays.groupby(key, sort=False).cumcount() + 1
    print(f"[step0_10k] sorted+cumcount {time.time()-t0:.0f}s", flush=True)

    result = plays[COLS].copy()
    for c, t in DTYPES.items():
        result[c] = result[c].astype(t)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.out, index=False)
    if not args.keep_raw:
        raw.unlink(missing_ok=True)

    # ---- post-conditions (mirrors step0_preprocessing.md section 6) ----------
    grp = result.groupby(key, sort=False)["playcount_cur"]
    contiguous = bool((grp.max() == grp.size()).all()) and bool((grp.min() == 1).all())
    print(f"[step0_10k] rows={len(result):,} players={result.player_id.nunique():,} "
          f"beatmaps={result.beatmap_id.nunique():,} cells={grp.ngroups:,}")
    print(f"[step0_10k] contiguous 1..n={contiguous} | loss<0={bool((result.loss < 0).all())}")
    print(f"[step0_10k] timestamp {result.timestamp.min()} .. {result.timestamp.max()}")
    print(f"[step0_10k] wrote {args.out}  total {time.time()-t0:.0f}s", flush=True)
    print("[step0_10k] filters:", {k: v for k, v in STATS.items()}, flush=True)


if __name__ == "__main__":
    main()
