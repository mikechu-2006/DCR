#!/usr/bin/env python3
"""step0_union_judge -- merge per-source judgement side tables onto the union.

The union play table (step0_full.parquet) has one row per
(player_id, beatmap_id, timestamp).  Each source has a judgement side table
from step0_judgements.py.  This script:

  1. concatenates the source tables (multiple --judge name=path allowed;
     the same name may repeat, e.g. four randommerge snapshots);
  2. dedups by the union key, choosing the row with the lowest loss (the
     union's label is the minimum loss) and, on ties, the source named by
     step0_full.source_first;
  3. left-joins onto step0_full and writes judge_ok / judge_src.

It does not drop unmatched union rows; step1_attach_judgements.py decides
what to do with judge_ok=False.  Output is keyed by the union key and carries
the six counts, n_total, loss and source labels.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

COUNT_COLS = ["n_max", "n300", "n200", "n100", "n50", "n_miss"]
KEY = ["player_id", "beatmap_id", "timestamp"]


def parse_judge(spec: str) -> tuple[str, Path]:
    name, sep, path = spec.partition("=")
    if not sep or not name or not path:
        raise SystemExit(f"--judge expects NAME=PATH, got {spec!r}")
    return name, Path(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", action="append", required=True,
                    help="NAME=PATH; repeat for each source/snapshot")
    ap.add_argument("--plays", required=True, type=Path,
                    help="step0_full.parquet (union play table)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    args = ap.parse_args()

    t0 = time.time()
    judges = [parse_judge(s) for s in args.judge]
    plays = pd.read_parquet(args.plays)
    print(f"[union_judge] union plays {len(plays)}", flush=True)

    frames = []
    for name, path in judges:
        if not path.exists():
            raise SystemExit(f"missing judge table {path}")
        df = pd.read_parquet(path)
        need = set(KEY + COUNT_COLS + ["n_total", "loss"])
        lack = need - set(df.columns)
        if lack:
            raise SystemExit(f"{path}: missing columns {sorted(lack)}")
        df = df[KEY + COUNT_COLS + ["n_total", "loss"]].copy()
        df["judge_src"] = name
        df["player_id"] = df["player_id"].astype("int64")
        df["beatmap_id"] = df["beatmap_id"].astype("int64")
        frames.append(df)
        print(f"[union_judge] {name}: {len(df)} rows from {path}", flush=True)
    judge = pd.concat(frames, ignore_index=True)
    n_judge = len(judge)
    print(f"[union_judge] concatenated {n_judge} judge rows", flush=True)

    # Count conflicts before the dedup: keys with more than one distinct
    # count vector (or loss) across sources.
    uniq = judge.drop_duplicates(KEY + COUNT_COLS + ["n_total", "loss"])
    conflict_keys = int((uniq.groupby(KEY, sort=False).size() > 1).sum())
    n_unique = int(judge.drop_duplicates(KEY).shape[0])
    print(f"[union_judge] unique keys {n_unique}, keys with count/loss conflict "
          f"{conflict_keys}", flush=True)

    # Prefer the lowest loss; on ties, the union's source_first gets rank 0.
    first_map = plays[KEY + ["source_first"]].copy()
    first_map["player_id"] = first_map["player_id"].astype("int64")
    first_map["beatmap_id"] = first_map["beatmap_id"].astype("int64")
    judge = judge.merge(first_map, on=KEY, how="left", validate="many_to_one")
    judge["_first"] = (judge["judge_src"] != judge["source_first"]).astype("int8")
    judge = judge.sort_values(KEY + ["loss", "_first", "judge_src"],
                              kind="stable")
    judge = judge.drop_duplicates(KEY, keep="first").reset_index(drop=True)
    judge = judge.drop(columns=["_first", "source_first"])
    print(f"[union_judge] deduped to {len(judge)} keys", flush=True)

    plays = plays.copy()
    plays["player_id"] = plays["player_id"].astype("int64")
    plays["beatmap_id"] = plays["beatmap_id"].astype("int64")
    out = plays.merge(judge, on=KEY, how="left", validate="one_to_one",
                      indicator=True)
    out["judge_ok"] = out["_merge"].eq("both")
    out = out.drop(columns=["_merge"])
    out["judge_src"] = out["judge_src"].fillna("missing")
    out["loss_delta"] = (out["loss_x"] - out["loss_y"]).abs()
    out = out.rename(columns={"loss_x": "loss"})
    out = out.drop(columns=["loss_y"])

    n = len(out)
    n_ok = int(out.judge_ok.sum())
    print(f"[union_judge] matched {n_ok}/{n} union plays "
          f"({n_ok/n:.4%})", flush=True)

    # Per-source coverage, using the union's source_first label.
    by_src = (out.groupby("source_first")["judge_ok"]
              .agg(["size", "sum"]).reset_index())
    by_src["rate"] = by_src["sum"] / by_src["size"]
    loss_delta = out.loc[out.judge_ok, "loss_delta"]
    report = {
        "plays": str(args.plays),
        "out": str(args.out),
        "n_union": int(n),
        "n_judge_rows": int(n_judge),
        "n_judge_unique_keys": int(n_unique),
        "n_conflict_keys": int(conflict_keys),
        "n_matched": int(n_ok),
        "coverage": float(n_ok / n) if n else None,
        "by_source_first": by_src.to_dict(orient="records"),
        "loss_delta_max": float(loss_delta.max()) if len(loss_delta) else None,
        "loss_delta_mean": float(loss_delta.mean()) if len(loss_delta) else None,
        "n_total_sum_mismatch": int(
            (out.loc[out.judge_ok, COUNT_COLS].sum(axis=1)
             != out.loc[out.judge_ok, "n_total"]).sum()),
        "elapsed_sec": round(time.time() - t0, 1),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".part")
    out.to_parquet(tmp, index=False)
    tmp.replace(args.out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=1, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
