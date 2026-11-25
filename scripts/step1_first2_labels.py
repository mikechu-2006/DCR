#!/usr/bin/env python
"""Build the "best of the first two records" label for step1/pairbest.

This is the Experiment 1 label builder.  It is deliberately self-contained (it
does not import the step1 module) so the same file can run on the HPC checkout
and on a fresh clone.

The rule mirrors step1/pairbest exactly:

  1. player floor: drop players with < --min-plays-per-player step0 records;
  2. 365-day gap split on the FULL step0 table (after the player floor);
  3. for every (new player segment, beatmap), sort by
     (timestamp, playcount_cur), take the first two records,
     loss_first2 = min(loss of those two);
  4. left-join onto the existing step1 pair table; do NOT re-filter pairs.

The main label (historical best) is not touched.  The output adds
loss_first2 / n_first2 / first2_ts / first2_vs_hist and a QC JSON.

Usage
-----
python scripts/step1_first2_labels.py \
    --step0 data/processed/step0_1k.csv \
    --pairs data/processed/step1_1k_pairbest.parquet \
    --out data/processed/step1_1k_pairbest_first2.parquet \
    --json data/processed/step1_1k_pairbest_first2.json

For a quick smoke test, --limit-players N keeps only the first N base players
(after the player floor).  Do not use it for production.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

STEP0_COLS = ["player_id", "beatmap_id", "timestamp", "playcount_cur", "loss"]


def read_step0(path: Path, limit_players: int | None) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path, columns=STEP0_COLS)
    else:
        df = pd.read_csv(path, usecols=STEP0_COLS)
    if limit_players is not None and limit_players > 0:
        keep = np.sort(df["player_id"].unique())[:limit_players]
        df = df[df["player_id"].isin(keep)].copy()
    return df


def assign_epochs(df: pd.DataFrame, gap_days: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Copy of step1_pairbest.assign_epochs; keep the two in sync."""
    n = len(df)
    base = df["player_id"].to_numpy(np.int64)
    bid = df["beatmap_id"].to_numpy(np.int64)
    pc = df["playcount_cur"].to_numpy(np.int64)
    ts = df["timestamp"].to_numpy().astype("datetime64[s]").astype(np.int64)

    order = np.lexsort((pc, bid, ts, base))
    base_s = base[order]
    ts_s = ts[order]

    seg_sorted = np.zeros(n, dtype=np.int32)
    if gap_days > 0 and n > 1:
        gap_seconds = float(gap_days) * 86400.0
        new_seg = np.zeros(n, dtype=bool)
        new_seg[0] = True
        new_seg[1:] = ((base_s[1:] != base_s[:-1])
                       | ((ts_s[1:] - ts_s[:-1]).astype(np.float64) >= gap_seconds))
        cum = np.cumsum(new_seg, dtype=np.int64)
        first = np.r_[0, np.flatnonzero(base_s[1:] != base_s[:-1]) + 1]
        lengths = np.diff(np.r_[first, n])
        seg_sorted = (cum - np.repeat(cum[first], lengths)).astype(np.int32)

    seg = np.empty(n, dtype=np.int32)
    seg[order] = seg_sorted

    seg_max = int(seg_sorted.max()) if n else 0
    span = np.int64(seg_max + 1)
    keys = base * span + seg.astype(np.int64)
    codes, uniq = pd.factorize(keys, sort=True)
    ubase = (uniq // span).astype(np.int64)
    useg = (uniq % span).astype(np.int64)
    fresh = np.empty(uniq.size, dtype=np.int64)
    fresh[useg == 0] = ubase[useg == 0]
    next_id = int(base.max()) + 1
    for j in np.flatnonzero(useg > 0):
        fresh[j] = next_id
        next_id += 1
    pid_new = fresh[codes]

    info = {
        "order": order,
        "n_segments": int(uniq.size),
        "n_bases": int(np.unique(ubase).size),
        "n_split_players": int((np.unique(ubase, return_counts=True)[1] > 1).sum()),
        "n_seg_extra": int(uniq.size - np.unique(ubase).size),
    }
    return pid_new, seg, info


def first_two_best(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (segment, beatmap) with the best of the first two records."""
    d = df.copy()
    d = d.sort_values(["pid", "beatmap_id", "timestamp", "playcount_cur"],
                      kind="mergesort")
    rank = d.groupby(["pid", "beatmap_id"], sort=False).cumcount()
    first2 = (d[rank < 2]
              .groupby(["pid", "beatmap_id"], sort=False)
              .agg(loss_first2=("loss", "min"),
                   first2_ts=("timestamp", "max"),
                   n_first2=("loss", "size"))
              .reset_index())
    return first2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step0", required=True, type=Path)
    ap.add_argument("--pairs", required=True, type=Path,
                    help="step1 pairbest parquet/csv; player_id is the segment id")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--gap-days", type=float, default=365.0)
    ap.add_argument("--min-plays-per-player", type=int, default=20)
    ap.add_argument("--limit-players", type=int, default=None,
                    help="smoke only: keep the first N base players after the floor")
    args = ap.parse_args()
    t0 = time.time()

    s0 = read_step0(args.step0, args.limit_players)
    raw_rows = len(s0)
    raw_players = int(s0["player_id"].nunique())

    # 1) player floor first (same order as step1)
    counts = s0["player_id"].value_counts()
    keep_players = counts.index[counts >= args.min_plays_per_player]
    s0 = s0[s0["player_id"].isin(keep_players)].copy()
    rows_after_floor = len(s0)
    players_after_floor = int(s0["player_id"].nunique())

    # 2) gap split on the full step0
    pid, seg, info = assign_epochs(s0, args.gap_days)
    s0["pid"] = pid.astype(np.int64)
    s0["player_seg"] = seg

    # 3) first two records per pair
    first2 = first_two_best(s0)
    print(f"[first2] step0 {raw_rows:,} -> {rows_after_floor:,} rows after floor; "
          f"{info['n_segments']:,} segments; {len(first2):,} (seg, chart) pairs",
          flush=True)

    # 4) join onto the existing step1 pairs (no re-filtering)
    pairs = (pd.read_parquet(args.pairs) if args.pairs.suffix.lower() == ".parquet"
             else pd.read_csv(args.pairs))
    out = pairs.merge(first2, left_on=["player_id", "beatmap_id"],
                      right_on=["pid", "beatmap_id"], how="left")
    out = out.drop(columns=["pid"])
    out["first2_vs_hist"] = out["loss_first2"] - out["loss"]

    missing = int(out["loss_first2"].isna().sum())
    tol = 1e-5
    ok_cover = missing == 0
    ok_hist = bool((out["loss"] <= out["loss_first2"] + tol).all())
    ok_two = bool((out.loc[out["n_plays"] == 2, "loss_first2"]
                   - out.loc[out["n_plays"] == 2, "loss"]).abs().max() <= tol) \
        if (out["n_plays"] == 2).any() else True
    ok_n = bool((out["n_first2"] >= 1).all())

    acc = 1.0 - np.exp(out["loss"].to_numpy(np.float64))
    acc2 = 1.0 - np.exp(out["loss_first2"].to_numpy(np.float64))
    diff = (out["loss_first2"] - out["loss"]).to_numpy(np.float64)

    report = {
        "step0": str(args.step0), "pairs": str(args.pairs), "out": str(args.out),
        "params": {"gap_days": args.gap_days,
                   "min_plays_per_player": args.min_plays_per_player,
                   "limit_players": args.limit_players},
        "step0_rows": {"raw": int(raw_rows), "after_player_floor": int(rows_after_floor)},
        "players": {"raw": int(raw_players), "after_player_floor": int(players_after_floor),
                    "segments": int(info["n_segments"]),
                    "split_by_gap": int(info["n_split_players"]),
                    "extra_segments": int(info["n_seg_extra"])},
        "pairs": {"rows": int(len(out)), "unique_pairs": int(
            out[["player_id", "beatmap_id"]].drop_duplicates().shape[0]),
            "missing_first2": missing},
        "n_plays": {
            "eq2": int((out["n_plays"] == 2).sum()),
            "eq2_frac": float((out["n_plays"] == 2).mean()),
            "gt2": int((out["n_plays"] > 2).sum()),
            "gt2_frac": float((out["n_plays"] > 2).mean()),
        },
        "labels": {
            "loss_hist_mean": float(out["loss"].mean()),
            "loss_first2_mean": float(out["loss_first2"].mean()),
            "loss_diff_mean": float(diff.mean()),
            "loss_diff_quantiles": {str(q): float(np.quantile(diff, q))
                                    for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)},
            "acc_hist_mean": float(acc.mean()),
            "acc_first2_mean": float(acc2.mean()),
            "first2_acc_le_090": int((acc2 <= 0.90).sum()),
            "first2_acc_le_090_frac": float((acc2 <= 0.90).mean()),
            "first2_acc_gt_0998": int((acc2 > 0.998).sum()),
            "corr_loss": float(np.corrcoef(out["loss"], out["loss_first2"])[0, 1]),
        },
        "checks": {
            "first2_coverage_100pct": bool(ok_cover),
            "hist_le_first2": bool(ok_hist),
            "n_plays_2_equals": bool(ok_two),
            "n_first2_ge_1": bool(ok_n),
            "no_pairs_added": bool(len(out) == len(pairs)),
        },
        "seconds": round(time.time() - t0, 1),
    }
    if not all(report["checks"].values()):
        print(json.dumps(report, indent=1, ensure_ascii=False), flush=True)
        raise SystemExit("first2 QC failed; see report")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(f"[first2] wrote {args.out} ({len(out):,} rows, "
          f"missing={missing}, n_plays=2 {report['n_plays']['eq2_frac']:.1%})",
          flush=True)
    print(json.dumps({k: report[k] for k in ("pairs", "labels", "checks", "seconds")},
                     indent=1, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
