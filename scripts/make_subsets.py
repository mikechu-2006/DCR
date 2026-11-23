#!/usr/bin/env python3
"""Build evaluation subsets (ALL / RC20 / RC80 / HARD-RC / KOISHI-NATIVE)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def pairs(df: pd.DataFrame) -> int:
    g = df.groupby("player_id").size()
    return int((g * (g - 1) // 2).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--signals", required=True)
    ap.add_argument("--dataset", required=True, choices=["10k", "randommerge"])
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    step = pd.read_parquet(args.inp)
    cols = ["beatmap_id", "sunny_star", "ln_ratio", "n_notes"]
    sig_all = pd.read_parquet(args.signals)
    if "koishi_v5_dan" in sig_all.columns:
        cols.append("koishi_v5_dan")
    sig = sig_all[cols].copy()
    if "koishi_v5_dan" not in sig.columns:
        sig["koishi_v5_dan"] = float("nan")
    m = step.merge(sig, on="beatmap_id", how="left")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rc20 = m[(m.ln_ratio <= 0.18) & (m.n_notes >= 20)]
    rc80 = m[(m.ln_ratio <= 0.18) & (m.n_notes >= 80)]
    chart = rc80.drop_duplicates("beatmap_id")[["beatmap_id", "sunny_star"]]
    thr = chart.sunny_star.quantile(0.70)
    hard = rc80[rc80.beatmap_id.isin(set(chart.loc[chart.sunny_star >= thr, "beatmap_id"]))]
    koi = m[m.koishi_v5_dan.notna()]
    ln = m[m.ln_ratio > 0.18]

    subsets = {"all": m, "rc20": rc20, "rc80": rc80, "hardrc": hard, "ln": ln}
    if args.dataset == "10k":
        subsets["koishi_native"] = koi

    # only the 9 step1 columns, no join columns
    cols = [c for c in step.columns]
    for name, df in subsets.items():
        df = df[cols].reset_index(drop=True)
        p = out / f"step1_{args.dataset}_{name}.parquet"
        df.to_parquet(p, index=False)
        print(f"{name:14s} rows={len(df):,} charts={df.beatmap_id.nunique():,} "
              f"players={df.player_id.nunique():,} pairs={pairs(df)/1e6:.2f}M -> {p}")
    print(f"HARD-RC threshold={thr:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
