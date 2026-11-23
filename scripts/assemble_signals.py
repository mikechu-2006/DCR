#!/usr/bin/env python3
"""Assemble ISOR/Koishi NDJSON into long + wide signal tables.

Long: one row per (beatmap_id, method) with raw/valid/reason.
Wide: beatmap_id + sunny_star + isor_v31_dp + koishi_v5_dan + metadata.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def read_ndjson(pattern: str) -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(pattern)):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # keep the last record per (beatmap_id, method); reruns are append-safe
    df = df.drop_duplicates(["beatmap_id", "method"], keep="last")
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--isor-glob", required=True)
    ap.add_argument("--koishi-glob", default="")
    ap.add_argument("--dataset", required=True, choices=["10k", "randommerge"])
    ap.add_argument("--meta", default=str(ROOT / "data/processed/rank_signals_1k.parquet"))
    ap.add_argument("--out-long", required=True)
    ap.add_argument("--out-wide", required=True)
    args = ap.parse_args()

    frames = []
    isor = read_ndjson(args.isor_glob)
    if not isor.empty:
        frames.append(isor)
    if args.koishi_glob:
        koi = read_ndjson(args.koishi_glob)
        if not koi.empty:
            frames.append(koi)
    if not frames:
        raise SystemExit("no NDJSON rows found")
    long = pd.concat(frames, ignore_index=True)
    long["valid"] = long["valid"].astype(bool)
    long.to_parquet(args.out_long, index=False)

    meta = pd.read_parquet(
        args.meta,
        columns=["beatmap_id", "sunny_star", "daniel_star", "star",
                 "ln_ratio", "n_notes", "patt_mode_tag", "notes_per_sec"],
    ).drop_duplicates("beatmap_id")

    def pivot(method: str, col: str) -> pd.DataFrame:
        sub = long[(long["method"] == method) & long["valid"] & long["raw"].notna()]
        return (sub.set_index("beatmap_id")["raw"].astype(float)
                .rename(col).reset_index())

    wide = meta.copy()
    for method, col in [("isor_v31", "isor_v31_dp"), ("koishi_v5", "koishi_v5_dan")]:
        p = pivot(method, col)
        wide = wide.merge(p, on="beatmap_id", how="left")
    if "isor_v31_dp" not in wide:
        wide["isor_v31_dp"] = float("nan")
    if "koishi_v5_dan" not in wide:
        wide["koishi_v5_dan"] = float("nan")
    wide.to_parquet(args.out_wide, index=False)

    print(f"[assemble] {args.dataset}: long={len(long)} wide={len(wide)} "
          f"isor={wide.isor_v31_dp.notna().sum()} "
          f"koishi={wide.koishi_v5_dan.notna().sum()}")
    print(long.groupby(["method", "reason"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
