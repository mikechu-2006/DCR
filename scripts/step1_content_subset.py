#!/usr/bin/env python
"""Restrict a step1 pairbest table to charts that have a CM3P content vector.

The training scripts drop charts without content internally, but the benchmark /
deployment path fits on a saved table (--fit-all), so the canonical training
table should already be content-covered.  randommerge's published table is this
filtered view (453,406 rows / 20,388 charts); this script makes the same view
for any step1 output and reports what the filter costs.

The filter is row-wise on beatmap_id and never touches the split column, so the
train/test proportions are preserved up to the removed charts.

Usage:
  python scripts/step1_content_subset.py \
      --in data/processed/step1_full_pairbest.parquet \
      --content data/processed/chart_content_cm3p_local.parquet \
      --out data/processed/step1_full_pairbest_content.parquet \
      --json data/processed/step1_full_pairbest_content.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--content", required=True, type=Path,
                    help="content table with at least a beatmap_id column")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.inp) if args.inp.suffix == ".parquet" else pd.read_csv(args.inp)
    content_ids = pd.unique(pd.read_parquet(args.content, columns=["beatmap_id"])["beatmap_id"])
    content_set = set(content_ids.tolist())
    mask = df["beatmap_id"].isin(content_set)
    sub = df.loc[mask].reset_index(drop=True)

    base = df["player_id_base"] if "player_id_base" in df.columns else df["player_id"]
    subs_base = sub["player_id_base"] if "player_id_base" in sub.columns else sub["player_id"]
    split_a = df["split"].value_counts().to_dict() if "split" in df.columns else {}
    split_b = sub["split"].value_counts().to_dict() if "split" in sub.columns else {}
    lost_charts = int(df.loc[~mask, "beatmap_id"].nunique()) if (~mask).any() else 0

    report = {
        "input": str(args.inp), "content": str(args.content), "output": str(args.out),
        "rows": {"before": int(len(df)), "after": int(len(sub)),
                 "dropped": int((~mask).sum()),
                 "kept_frac": round(float(mask.mean()), 6)},
        "charts": {"before": int(df.beatmap_id.nunique()),
                   "after": int(sub.beatmap_id.nunique()),
                   "without_content": lost_charts},
        "players": {"before": int(base.nunique()), "after": int(subs_base.nunique()),
                    "before_segments": int(df.player_id.nunique()),
                    "after_segments": int(sub.player_id.nunique())},
        "split": {"before": {k: int(v) for k, v in split_a.items()},
                  "after": {k: int(v) for k, v in split_b.items()}},
        "checks": {
            "rows_are_subset": bool(len(sub) == int(mask.sum())),
            "row_order_preserved": bool(
                "split" not in df.columns
                or sub["split"].tolist() == df.loc[mask, "split"].tolist()),
            "all_charts_have_content": bool(sub.beatmap_id.isin(content_set).all()),
            "no_rows_added": bool(len(sub) <= len(df)),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_parquet(args.out, index=False)
    print(f"[content-subset] {len(df):,} -> {len(sub):,} rows, "
          f"{df.beatmap_id.nunique():,} -> {sub.beatmap_id.nunique():,} charts")
    print(f"[content-subset] split {report['split']['before']} -> {report['split']['after']}")
    print(f"[content-subset] wrote {args.out}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"[content-subset] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
