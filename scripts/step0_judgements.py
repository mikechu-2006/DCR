#!/usr/bin/env python3
"""step0_judgements -- six-grade judgement side table for one dump.

The main step0 play table keeps only log(1-ACC).  This script rescans the
same score tables with the same 4K / CL / dedup policy and writes the six raw
grade counts (MAX/320, 300, 200, 100, 50, miss) for every kept play:

    player_id, beatmap_id, timestamp, score_id,
    n_max, n300, n200, n100, n50, n_miss, n_total,
    loss, source_id

It is a *side table*: the 5-column step0 contract is untouched, and the union /
pairbest steps can attach these counts to the selected best play.

Legacy (osu_scores_mania_high.sql) row layout, see step0_preprocess.py:425:

    0 score_id  1 beatmap_id  2 user_id  3 score  4 maxcombo  5 rank
    6 count50   7 count100    8 count300  9 countmiss 10 countgeki 11 countkatu
    12 perfect  13 enabled_mods 14 date 15 pp

Modern (scores.sql) JSON statistics:
    perfect -> n_max, great -> n300, good -> n200,
    ok -> n100, meh -> n50, miss -> n_miss

Note: MAX and 300 both carry weight 300 in ACC, so this table contains signal
that the existing loss label cannot see.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parse_dump import iter_insert_lines, rows_simple  # noqa: E402
import step0_preprocess as s0  # noqa: E402

N_TAIL = s0.N_TAIL
COLS = ["player_id", "beatmap_id", "timestamp", "score_id",
        "n_max", "n300", "n200", "n100", "n50", "n_miss", "n_total",
        "loss", "source_id"]
SCHEMA = pa.schema([
    ("player_id", pa.int32()),
    ("beatmap_id", pa.int32()),
    ("timestamp", pa.timestamp("ns")),
    ("score_id", pa.int64()),
    ("n_max", pa.int32()),
    ("n300", pa.int32()),
    ("n200", pa.int32()),
    ("n100", pa.int32()),
    ("n50", pa.int32()),
    ("n_miss", pa.int32()),
    ("n_total", pa.int32()),
    ("loss", pa.float32()),
    ("source_id", pa.int8()),
])

STATS: dict[str, int] = {}


def note(key: str, n: int = 1) -> None:
    STATS[key] = STATS.get(key, 0) + n


def _ints(stats: dict) -> tuple[int, int, int, int, int, int]:
    return (int(stats.get("perfect", 0)), int(stats.get("great", 0)),
            int(stats.get("good", 0)), int(stats.get("ok", 0)),
            int(stats.get("meh", 0)), int(stats.get("miss", 0)))


def iter_modern_counts(dump: Path, ids: set[int], allow: list[frozenset],
                       neutral: frozenset):
    """scores.sql -> (user, bid, ts, sid, counts, loss, legacy_id)."""
    src = dump / "scores.sql"
    if not src.exists():
        print("[judge] no scores.sql -> modern branch skipped", flush=True)
        return
    for line in iter_insert_lines(src):
        body = line[line.index("VALUES") + 6:].strip()[1:]
        if body.endswith(");"):
            body = body[:-2]
        for row in body.split("),("):
            parts = row.split(",")
            if len(parts) < 12 + N_TAIL:
                continue
            head, tail = parts[:12], parts[-N_TAIL:]
            note("modern_rows")
            bid = int(head[3])
            if bid not in ids:
                continue
            note("modern_4k")
            if head[2].strip("'") != s0.MANIA_RULESET:
                note("modern_wrong_ruleset")
                continue
            raw = ",".join(parts[12:-N_TAIL]).strip()
            if raw[:1] == "'":
                raw = raw[1:-1]
            data = json.loads(raw.replace('\\"', '"'))
            tokens = frozenset(m["acronym"] for m in data.get("mods", []))
            canon = s0.canonicalize(tokens, neutral)
            if canon not in allow:
                note("modern_mod_dropped")
                continue
            stats = data.get("statistics", {})
            total = float(sum(stats.values()))
            if total <= 0:
                note("modern_empty")
                continue
            num = float(sum(stats.get(k, 0) * w for k, w in s0.CLASSIC))
            counts = _ints(stats)
            if sum(counts) != int(total):
                note("modern_count_mismatch")
            legacy_id = tail[1]
            yield (int(head[1]), bid, tail[4].strip("'"), int(head[0]),
                   counts, s0.loss_of(num, total),
                   int(legacy_id) if legacy_id.isdigit() else -1)


def iter_legacy_counts(dump: Path, ids: set[int], allow: list[frozenset],
                       bits: tuple[tuple[int, str], ...], drop: set[int],
                       neutral: frozenset):
    """osu_scores_mania_high.sql -> (user, bid, ts, sid, counts, loss)."""
    src = dump / "osu_scores_mania_high.sql"
    if not src.exists():
        raise SystemExit(f"missing {src}")
    cache: dict[int, tuple[str, frozenset] | None] = {}
    for line in iter_insert_lines(src):
        for r in rows_simple(line):
            note("legacy_rows")
            bid = int(r[1])
            if bid not in ids:
                continue
            note("legacy_4k")
            mask = int(r[13])
            if mask not in cache:
                mods = s0.decode_legacy_mods(mask, bits)
                if mods is None:
                    cache[mask] = None
                else:
                    tokens = s0.CL_SET | mods
                    cache[mask] = (s0.mod_label(tokens),
                                   s0.canonicalize(tokens, neutral))
            hit = cache[mask]
            if hit is None or hit[1] not in allow:
                note("legacy_mod_dropped")
                continue
            sid = int(r[0])
            if sid in drop:
                note("legacy_deduped")
                continue
            v = [int(r[i]) for i in s0.LEGACY_TOTAL]
            total = float(sum(v))
            if total <= 0:
                note("legacy_empty")
                continue
            num = float(sum(int(r[i]) * w for i, w in s0.LEGACY_NUM))
            # legacy indices: 6 count50, 7 count100, 8 count300,
            # 9 countmiss, 10 countgeki, 11 countkatu
            counts = (int(r[10]), int(r[8]), int(r[11]),
                      int(r[7]), int(r[6]), int(r[9]))
            if sum(counts) != int(total):
                note("legacy_count_mismatch")
            yield (int(r[2]), bid, r[14], sid, counts, s0.loss_of(num, total))


class Writer:
    def __init__(self, out: Path):
        self.out = out
        self.tmp = out.with_name(out.name + ".part")
        self.buf: list[tuple] = []
        self.writer = None

    def add(self, row: tuple) -> None:
        self.buf.append(row)
        if len(self.buf) >= 500_000:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        tbl = pa.Table.from_pylist([
            dict(zip(COLS, r)) for r in self.buf
        ], schema=SCHEMA)
        if self.writer is None:
            self.out.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(self.tmp, SCHEMA)
        self.writer.write_table(tbl)
        self.buf = []

    def close(self) -> None:
        self.flush()
        if self.writer is not None:
            self.writer.close()
            self.tmp.replace(self.out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", required=True, type=Path)
    ap.add_argument("--beatmap-src", type=Path, default=None)
    ap.add_argument("--whitelist", type=Path, default=None)
    ap.add_argument("--tag", default="1k")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--source-id", type=int, default=1,
                    help="int8 source label (1=1k, 2=10k, 3=randommerge)")
    ap.add_argument("--mod-whitelist", default=s0.MOD_WHITELIST)
    ap.add_argument("--neutral-mods", default=s0.NEUTRAL_MODS)
    ap.add_argument("--legacy-mod-bits", default=s0.LEGACY_MOD_BITS)
    ap.add_argument("--mode", default=s0.MANIA_RULESET)
    ap.add_argument("--keys", default="4")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N kept rows (smoke only; modern dedup is incomplete)")
    args = ap.parse_args()

    allow = s0.parse_allow(args.mod_whitelist)
    bits = s0.parse_legacy_bits(args.legacy_mod_bits)
    neutral = s0.parse_names(args.neutral_mods)
    beatmap_src = args.beatmap_src or args.dump_dir
    t0 = time.time()

    ids, meta = s0.read_beatmaps(beatmap_src, args.mode, args.keys)
    print(f"[judge] {args.mode}/{args.keys} beatmaps: {len(ids)}", flush=True)
    if args.whitelist:
        keep, _ = s0.read_whitelist(args.whitelist)
        ids &= keep
        meta = meta[meta.beatmap_id.isin(ids)].reset_index(drop=True)
        print(f"[judge] whitelist -> {len(ids)} beatmaps", flush=True)
    if not ids:
        raise SystemExit("no beatmaps after whitelist")

    w = Writer(args.out)
    n = 0

    # Modern first: it defines the legacy score ids to drop.
    drop: set[int] = set()
    for user, bid, ts, sid, counts, loss, legacy_id in iter_modern_counts(
            args.dump_dir, ids, allow, neutral):
        if legacy_id > 0:
            drop.add(legacy_id)
        w.add((user, bid, pd.Timestamp(ts), sid, *counts, sum(counts),
               float(loss), int(args.source_id)))
        n += 1
        if args.limit and n >= args.limit:
            break
    print(f"[judge] modern kept {n}; legacy drop set {len(drop)}", flush=True)

    if not (args.limit and n >= args.limit):
        for user, bid, ts, sid, counts, loss in iter_legacy_counts(
                args.dump_dir, ids, allow, bits, drop, neutral):
            w.add((user, bid, pd.Timestamp(ts), sid, *counts, sum(counts),
                   float(loss), int(args.source_id)))
            n += 1
            if args.limit and n >= args.limit:
                break

    w.close()
    print(f"[judge] wrote {args.out} rows={n} in {time.time()-t0:.0f}s", flush=True)
    print("[judge] counters: " + " ".join(f"{k}={v}" for k, v in sorted(STATS.items())),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
