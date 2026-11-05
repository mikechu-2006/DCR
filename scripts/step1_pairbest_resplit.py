#!/usr/bin/env python
"""step1_pairbest_resplit -- regenerate the pair-random 70/30 split.

The train/test split is the *last* step of step1/pairbest (docs/data.md
§3.1): the population filters run on all rows first, then the surviving
(player, beatmap) pairs are split pair-random.  Re-splitting the final table
with the same splitter is therefore exactly equivalent to rerunning step1 with
a different --split-seed, without redoing the cleaning.

This script exists for the P2 multi-seed study: the five replicate seeds are
train/test split seeds, not model seeds.  Each output keeps the old labels in
`split_orig` and writes the new labels to `split`.

Example
-------
python scripts/step1_pairbest_resplit.py \
    --in data/processed/step1_1k_pairbest.parquet \
    --out-prefix data/processed/step1_1k_pairbest_p2 \
    --seeds 1,2,3,4,5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from step1_clean import split_rows  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True,
                    help="step1 pairbest table (one row per player-chart pair)")
    ap.add_argument("--out-prefix", default="data/processed/step1_1k_pairbest_p2",
                    help="output parquet: <prefix>_s<seed>.parquet")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--force", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    src = Path(args.inp)
    df = pd.read_parquet(src)
    if "split" not in df.columns:
        raise SystemExit(f"{src}: no split column")
    if not 0.0 < args.train_frac < 1.0:
        raise SystemExit(f"--train-frac must be in (0,1), got {args.train_frac}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    all_info = []
    for seed in seeds:
        out_path = Path(f"{prefix}_s{seed}.parquet")
        info_path = Path(f"{prefix}_s{seed}.json")
        if out_path.exists() and info_path.exists() and not args.force:
            print(f"[skip] {out_path} exists", flush=True)
            all_info.append(json.loads(info_path.read_text()))
            continue
        is_test, info = split_rows(df, "pair-random", args.train_frac, seed)
        out = df.copy()
        out["split_orig"] = df["split"]
        out["split"] = np.where(is_test, "test", "train")
        out.to_parquet(out_path, index=False)
        info.update({
            "seed": int(seed), "train_frac": float(args.train_frac),
            "in": str(src), "out": str(out_path),
            "rows": {k: int(v) for k, v in out["split"].value_counts().items()},
            "n_pairs": int(len(out)),
        })
        info_path.write_text(json.dumps(info, indent=1, ensure_ascii=False) + "\n")
        print(f"[p2 s{seed}] {info['rows']} actual_train_frac="
              f"{info['actual_train_frac']:.4f}", flush=True)
        all_info.append(info)

    summary = Path(f"{prefix}_summary.json")
    summary.write_text(json.dumps({"in": str(src), "splits": all_info},
                                  indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {summary} ({len(all_info)} seeds)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
