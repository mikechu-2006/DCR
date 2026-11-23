#!/usr/bin/env python
"""Merge per-snapshot step0 play tables into one play table (dedupe + renumber).

Why this exists: step1/pairbest rebuilds (player_seg, beatmap) pairs from play
rows, and three of its inputs are computed from the *full* play list -- n_plays,
playcount_cur and the 365-day gap split.  Merging the snapshots at pair level
would freeze those three at snapshot scope and lose every pair whose two plays
landed in different snapshots.  So the merge happens here, at play level, and
step1 runs afterwards on the merged table.

Dedup key: (player_id, beatmap_id, timestamp).  Timestamps have one-second
granularity and a play is longer than a second, so the same player cannot end
two plays of the same chart in the same second; the key identifies a play
across snapshots whether it was seen through the modern or the legacy table.
`loss` is checked for consistency on duplicate keys: the judgement counts of a
play never change, so the same play must carry the same loss everywhere.

Output keeps the 5 columns step1 expects plus three provenance columns:

    player_id, beatmap_id, timestamp, playcount_cur, loss,
    snapshot_first, snapshot_last, n_snapshots

playcount_cur is recomputed on the merged timeline, not taken from any snapshot.

The dedupe/renumber core (`merge_frames`) is shared with
scripts/step0_union_datasets.py: both merges must produce the same play list
semantics, because step1 counts n_plays and segments players on it.

Usage:
  python scripts/step0_merge_snapshots.py \
      --input data/processed/step0_rand_20260713.parquet=20260713 \
      --input data/processed/step0_rand_20260801.parquet=20260801 \
      --input data/processed/step0_rand_20260901.parquet=20260901 \
      --input data/processed/step0_rand_20261105.parquet=20261105 \
      --out data/processed/step0_randommerge.parquet \
      --report data/processed/step0_randommerge.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

KEY = ["player_id", "beatmap_id", "timestamp"]
COLS = ["player_id", "beatmap_id", "timestamp", "playcount_cur", "loss"]


def parse_inputs(items: list[str]) -> list[tuple[str, Path]]:
    out = []
    for item in items:
        if "=" in item:
            path, tag = item.split("=", 1)
        else:
            path, tag = item, Path(item).stem
        out.append((tag, Path(path)))
    return out


def merge_frames(frames: list[pd.DataFrame], tags: list[str], *,
                 prefix: str = "snapshot") -> tuple[pd.DataFrame, dict]:
    """Dedupe play rows on KEY and rebuild playcount_cur on the merged timeline.

    `frames` and `tags` are aligned; each frame only needs COLS.  Provenance
    columns are named after `prefix`: first/last source that saw the play and
    the number of distinct sources (snapshot, dataset, ...).

    Duplicate keys keep the earliest source (argument order), then the lowest
    loss (= best ACC) when one source itself encoded the same play twice.  The
    caller decides whether that is acceptable; stats reports the conflict count.
    """
    if len(frames) != len(tags):
        raise ValueError(f"frames/tags length mismatch: {len(frames)} != {len(tags)}")
    frames = [f[COLS].copy() for f in frames]
    all_df = pd.concat(
        [f.assign(source=tag) for f, tag in zip(frames, tags)], ignore_index=True)
    order = {tag: i for i, tag in enumerate(tags)}
    all_df["_order"] = all_df["source"].map(order).astype("int16")

    dup_mask = all_df.duplicated(subset=KEY, keep=False)
    n_dup_rows = int(dup_mask.sum())
    n_dup_keys = 0
    conflicts = 0
    if n_dup_rows:
        spread = all_df[dup_mask].groupby(KEY, observed=True)["loss"].nunique()
        n_dup_keys = int(len(spread))
        conflicts = int((spread > 1).sum())

    prov_cols = [f"{prefix}_first", f"{prefix}_last", f"n_{prefix}s"]
    prov = (all_df.sort_values("_order")
            .groupby(KEY, observed=True)
            .agg(**{prov_cols[0]: ("source", "first"),
                    prov_cols[1]: ("source", "last"),
                    prov_cols[2]: ("source", "nunique")}))

    # same key, different loss happens inside a single dump too: the modern and the
    # legacy table can encode the same play with slightly different ACC, and step0
    # only links them when legacy_score_id is present.  Keep the earliest source,
    # then the lowest loss (best ACC); the count is reported.
    merged = (all_df.sort_values(KEY + ["_order", "loss"])
              .drop_duplicates(subset=KEY, keep="first"))
    merged = merged.drop(columns=["_order", "playcount_cur", "source"]).set_index(KEY)
    merged = merged.join(prov).reset_index()
    merged = merged.sort_values(["player_id", "timestamp"]).reset_index(drop=True)
    merged["playcount_cur"] = (merged.groupby(["player_id", "beatmap_id"], observed=True)
                               .cumcount() + 1).astype("int32")

    stats = {
        "rows_on_duplicate_keys": n_dup_rows,
        "duplicate_keys": n_dup_keys,
        "loss_conflicts": conflicts,
        "duplicate_play_rule": "earliest source in argument order, then lowest loss (best ACC)",
    }
    return merged[COLS + prov_cols], stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", action="append", required=True, metavar="PATH=TAG")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--strict", action="store_true",
                    help="abort if any dedup key disagrees on loss (default: keep the "
                         "earliest snapshot, then the lowest loss, and report the count)")
    args = ap.parse_args()

    inputs = parse_inputs(args.input)
    frames, per_snapshot, tags = [], {}, []
    for tag, path in inputs:
        if not path.exists():
            print("missing input: " + str(path), file=sys.stderr)
            return 2
        df = pd.read_parquet(path, columns=COLS)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        frames.append(df)
        tags.append(tag)
        per_snapshot[tag] = {
            "plays": int(len(df)),
            "players": int(df["player_id"].nunique()),
            "beatmaps": int(df["beatmap_id"].nunique()),
            "t_min": str(df["timestamp"].min()),
            "t_max": str(df["timestamp"].max()),
        }
        print("[merge] %s: %d plays, %d players" % (tag, len(df), df["player_id"].nunique()))

    merged, stats = merge_frames(frames, tags, prefix="snapshot")
    n_dup_rows = stats["rows_on_duplicate_keys"]
    conflicts = stats["loss_conflicts"]
    if n_dup_rows:
        print("[merge] %d rows on %d duplicate keys, %d loss conflicts"
              % (n_dup_rows, stats["duplicate_keys"], conflicts))
    if conflicts and args.strict:
        print("[merge] aborting: %d keys disagree on loss (remove --strict to keep "
              "the earliest snapshot / lowest loss)" % conflicts, file=sys.stderr)
        return 3
    if conflicts:
        print("[merge] warning: %d keys disagree on loss; kept earliest snapshot / "
              "lowest loss" % conflicts, file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.out, index=False)
    print("[merge] wrote %s (%d plays)" % (args.out, len(merged)))

    players_by_snap = {tag: set(frame["player_id"].unique())
                       for tag, frame in zip(tags, frames)}
    overlap = {}
    for i, a in enumerate(tags):
        for b in tags[i + 1:]:
            inter = len(players_by_snap[a] & players_by_snap[b])
            union = len(players_by_snap[a] | players_by_snap[b])
            overlap[a + "|" + b] = {"intersection": inter,
                                    "jaccard": round(inter / union, 4)}

    all_players = set().union(*players_by_snap.values())
    seen = pd.Series({p: sum(p in s for s in players_by_snap.values())
                      for p in all_players})

    report = {
        "inputs": {tag: str(path) for tag, path in inputs},
        "per_snapshot": per_snapshot,
        "merged": {
            "plays": int(len(merged)),
            "players": int(merged["player_id"].nunique()),
            "beatmaps": int(merged["beatmap_id"].nunique()),
            "pairs": int(merged.groupby(["player_id", "beatmap_id"],
                                        observed=True).ngroups),
            "t_min": str(merged["timestamp"].min()),
            "t_max": str(merged["timestamp"].max()),
            "plays_seen_in_1_snapshot": int((merged["n_snapshots"] == 1).sum()),
            "plays_seen_in_ge2_snapshots": int((merged["n_snapshots"] >= 2).sum()),
        },
        "duplicates": {"rows_on_duplicate_keys": n_dup_rows,
                       "duplicate_keys": stats["duplicate_keys"],
                       "loss_conflicts": conflicts},
        "player_overlap": overlap,
        "players_seen_in_k_snapshots":
            {str(k): int(v) for k, v in seen.value_counts().sort_index().items()},
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print("[merge] wrote %s" % args.report)
    print(json.dumps(report["merged"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
