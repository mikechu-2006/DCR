#!/usr/bin/env python
"""Map D2V benchmark embeddings (numeric staging ids) back to benchmark row_id.

d2v_main.extract writes beatmap_id = the numeric file stem in
data/benchmark_d2v_osu (0..1381).  This script joins that back to
data/processed/benchmark_d2v_map.parquet and writes a benchmark content table
with the same row_id contract as benchmark_content_legacy.parquet.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inp", default="data/processed/benchmark_content_d2v.parquet")
    ap.add_argument("--map", default="data/processed/benchmark_d2v_map.parquet")
    ap.add_argument("--out", default="data/processed/benchmark_content_d2v.parquet")
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args(argv)

    raw = pd.read_parquet(args.inp)
    mp = pd.read_parquet(args.map)
    if "embedding" not in raw.columns:
        raise SystemExit(f"{args.inp}: no embedding column")
    if "beatmap_id" in raw.columns:
        raw = raw.rename(columns={"beatmap_id": "d2v_id"})
    if "d2v_id" not in raw.columns:
        raise SystemExit(f"{args.inp}: no beatmap_id/d2v_id column")

    out = mp.merge(raw, on="d2v_id", how="left")
    missing = out["embedding"].isna()
    if missing.any() and not args.allow_missing:
        raise SystemExit(f"{int(missing.sum())} rows missing D2V embedding; "
                         f"ids={out.loc[missing, 'row_id'].head(20).tolist()}")
    out = out.dropna(subset=["embedding"]).reset_index(drop=True)
    dims = {len(x) for x in out["embedding"]}
    if dims != {512}:
        raise SystemExit(f"unexpected embedding dims: {dims}")
    cols = ["row_id", "bid", "d2v_id", "embedding", "n_windows",
            "checksum_ok", "source", "d2v_rev", "osu_path", "osu_md5"]
    out = out[[c for c in cols if c in out.columns]]
    out.to_parquet(args.out, index=False)
    print(f"[finalize] {args.out}: {len(out)} rows, dim=512, "
          f"n_windows mean={out.n_windows.mean():.1f}, "
          f"missing={int(missing.sum())}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
