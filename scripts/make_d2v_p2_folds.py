#!/usr/bin/env python
"""Create chart-disjoint P2 fold tables for the D2V residual probe.

The P2-holdout split is pair-random: test pairs share charts with training
pairs.  A 512-d residual fitted on all training pairs can memorise chart-level
noise and transfer it to test pairs on the same charts (the random control
shows this).  This script makes K chart-disjoint folds:

  fold k: train rows = original train rows whose chart is NOT in fold k;
          test rows  = original test rows whose chart IS in fold k;
          everything else dropped.

Then the residual is fitted on other charts and evaluated on held-out charts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--signal-table", default="data/processed/rank_signals_1k.parquet")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_parquet(args.inp)
    sig = pd.read_parquet(args.signal_table, columns=["beatmap_id", "star"])
    star = sig.set_index("beatmap_id")["star"]
    charts = np.sort(df["beatmap_id"].unique())
    s = star.reindex(charts).to_numpy(np.float64)
    if not np.isfinite(s).all():
        s = np.nan_to_num(s, nan=np.nanmedian(s))
    # stratify by star quintile, then round-robin within each stratum
    bins = pd.qcut(s, 5, labels=False, duplicates="drop")
    rng = np.random.default_rng(args.seed)
    fold = np.empty(charts.size, dtype=np.int64)
    for b in sorted(pd.unique(bins)):
        idx = np.flatnonzero(bins == b)
        perm = rng.permutation(idx)
        fold[perm] = np.arange(perm.size) % args.folds
    fold_map = {int(c): int(f) for c, f in zip(charts, fold)}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    info = {"in": str(args.inp), "folds": args.folds, "seed": args.seed,
            "n_charts": int(charts.size), "folds_info": []}
    for k in range(args.folds):
        out = df.copy()
        in_fold = out["beatmap_id"].map(fold_map).to_numpy() == k
        split = np.where((out["split"].to_numpy() == "train") & (~in_fold),
                         "train",
                         np.where((out["split"].to_numpy() == "test") & in_fold,
                                  "test", "dropped"))
        out["split_orig"] = out["split"]
        out["split"] = split
        p = args.out_dir / f"p2_fold{k}.parquet"
        out.to_parquet(p, index=False)
        rec = {"fold": k, "out": str(p),
               "rows": {x: int(y) for x, y in pd.Series(split).value_counts().items()},
               "train_charts": int(out.loc[split == "train", "beatmap_id"].nunique()),
               "test_charts": int(out.loc[split == "test", "beatmap_id"].nunique())}
        info["folds_info"].append(rec)
        print(f"[fold {k}] {rec['rows']} train_charts={rec['train_charts']} "
              f"test_charts={rec['test_charts']}", flush=True)
    (args.out_dir / "folds.json").write_text(
        json.dumps(info, indent=1, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
