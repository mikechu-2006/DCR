"""Build plays + chart cells for the top-10000 dump (compact, chunked, ~7 GB RAM safe).

Uses osu_scores_mania_high as the single score source (95% of the top-1000 union; the modern
scores table would need a ~9 GB JSON parse for the remaining ~5%).
Beatmap metadata + per-mod star ratings are reused from the top-1000 dump (same beatmap DB).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

INTERIM = Path("data/interim")
INTERIM10 = Path("data/interim_10k")
OUT = Path("data/processed")

RATE_MASK = 64 | 256 | 512
LEGACY_BITS = [(1, "NF"), (2, "EZ"), (4, "TD"), (8, "HD"), (16, "HR"), (32, "SD"), (64, "DT"),
               (128, "RX"), (256, "HT"), (512, "NC"), (1024, "FL"), (2048, "AT"), (4096, "SO"),
               (8192, "AP"), (16384, "PF"), (32768, "4K"), (65536, "5K"), (131072, "6K"),
               (262144, "7K"), (524288, "8K"), (1048576, "FI"), (2097152, "RD"), (4194304, "CN"),
               (8388608, "TP"), (16777216, "9K"), (33554432, "CO"), (67108864, "1K"),
               (134217728, "3K"), (268435456, "2K")]
DROP_MODS = frozenset({"HD", "FL", "EZ", "HR", "AT", "RX", "AP", "SO"})
MIN_ACC = 0.90


def decode(bitmask: int) -> str:
    return "+".join(name for bit, name in LEGACY_BITS if bitmask & bit)


def main():
    bm = pd.read_csv(INTERIM / "beatmaps.csv")
    bm4 = bm[(bm.playmode == 3) & (bm["keys"] == 4)].copy()
    ids = set(bm4.beatmap_id.astype("int64"))
    print("4K beatmaps:", len(bm4), flush=True)

    plays = pd.read_parquet(INTERIM10 / "mania_high_4k.parquet")
    plays = plays[plays.beatmap_id.isin(ids)].copy()
    print("4K plays (mania_high only):", len(plays), "users:", plays.user_id.nunique(), flush=True)

    mods = plays.enabled_mods.to_numpy()
    plays["rate"] = np.where((mods & (64 | 512)) != 0, 1, np.where((mods & 256) != 0, 2, 0)).astype("int8")
    drop_mask = 1 << 30  # placeholder, replaced below
    drop_mask = (8 | 1024 | 2 | 16 | 2048 | 128 | 8192 | 4096)   # HD FL EZ HR AT RX AP SO
    plays = plays[(mods & drop_mask) == 0].copy()
    print("after dropping HD/FL/EZ/... plays:", len(plays), "rate mix:", plays.rate.value_counts().to_dict(), flush=True)
    plays["chart_id"] = plays.beatmap_id.astype("int64") * 4 + plays.rate

    # NOTE: judgement counts are stored as int16 -- multiply in int64 to avoid overflow
    c320 = plays.c320.to_numpy(dtype=np.int64)
    c300 = plays.c300.to_numpy(dtype=np.int64)
    c200 = plays.c200.to_numpy(dtype=np.int64)
    c100 = plays.c100.to_numpy(dtype=np.int64)
    c50 = plays.c50.to_numpy(dtype=np.int64)
    cmiss = plays.cmiss.to_numpy(dtype=np.int64)
    total = c320 + c300 + c200 + c100 + c50 + cmiss
    keep_rows = total > 0
    plays = plays[keep_rows].copy()
    c320, c300, c200, c100, c50, total = (a[keep_rows] for a in (c320, c300, c200, c100, c50, total))
    plays["total"] = total.astype("int32")
    t = total.astype("float64")
    acc_v2 = (305 * c320 + 300 * c300 + 200 * c200 + 100 * c100 + 50 * c50) / (305 * t)
    acc_flat = (300 * (c300 + c320) + 200 * c200 + 100 * c100 + 50 * c50) / (300 * t)
    plays["acc_v2"] = acc_v2.astype("float32")
    plays["acc_flat"] = acc_flat.astype("float32")
    plays["loss_v2"] = np.log(np.clip(1.0 - acc_v2, 25.0 / (305.0 * t), None)).astype("float32")
    plays["loss_flat"] = np.log(np.clip(1.0 - acc_flat, 25.0 / (300.0 * t), None)).astype("float32")

    keep = ["user_id", "beatmap_id", "chart_id", "rate", "total", "acc_v2", "acc_flat", "loss_v2", "loss_flat"]
    plays = plays[keep]
    plays.to_parquet(OUT / "plays_4k_10k.parquet", index=False)
    print("wrote plays_4k_10k.parquet", plays.shape, flush=True)

    # ---- chunked playcount -> attempts ----
    pc_path = INTERIM10 / "playcount.csv"
    parts = []
    for chunk in pd.read_csv(pc_path, chunksize=4_000_000,
                             dtype={"user_id": np.int32, "beatmap_id": np.int32, "playcount": np.int32}):
        parts.append(chunk[chunk.beatmap_id.isin(ids)])
    pc = pd.concat(parts, ignore_index=True).rename(columns={"playcount": "attempts"})
    del parts
    print("playcount rows for 4K beatmaps:", len(pc), flush=True)

    parts = []
    for i, chunk_users in enumerate(np.array_split(plays.user_id.unique(), 12)):
        sub = plays[plays.user_id.isin(chunk_users)]
        parts.append(sub.groupby(["user_id", "chart_id"], sort=False).agg(
            beatmap_id=("beatmap_id", "first"), rate=("rate", "first"), n_scores=("total", "size"),
            acc_v2_med=("acc_v2", "median"), acc_v2_max=("acc_v2", "max"), acc_v2_mean=("acc_v2", "mean"),
            acc_flat_med=("acc_flat", "median"),
            loss_v2_med=("loss_v2", "median"), loss_v2_max=("loss_v2", "max"),
            loss_v2_mean=("loss_v2", "mean"),
            loss_flat_med=("loss_flat", "median"), total_med=("total", "median")).reset_index())
        print(f"  cells chunk {i+1}/12: {len(parts[-1])}", flush=True)
        del sub
    cells = pd.concat(parts, ignore_index=True)
    del parts
    print("cells:", len(cells), flush=True)
    cells = cells.merge(pc, on=["user_id", "beatmap_id"], how="left")
    cells = cells.merge(bm4[["beatmap_id", "star", "max_combo", "countTotal", "playcount", "passcount", "approved"]],
                        on="beatmap_id", how="left")
    cells["low_acc"] = cells.acc_v2_med < MIN_ACC
    cells["primary"] = ~cells.low_acc
    cells["attempts_per_score"] = cells.attempts / cells.n_scores
    cells.to_parquet(OUT / "charts_4k_10k.parquet", index=False)

    stats = pd.read_csv(INTERIM10 / "user_stats_mania.csv")
    summary = {"plays": int(len(plays)), "users": int(plays.user_id.nunique()),
               "charts": int(cells.chart_id.nunique()), "beatmaps": int(cells.beatmap_id.nunique()),
               "cells": int(len(cells)), "primary_cells": int(cells.primary.sum()),
               "low_acc_share": round(float(cells.low_acc.mean()), 4),
               "rate_mix_plays": {str(k): int(v) for k, v in plays.rate.value_counts().items()},
               "players_in_stats": int(len(stats))}
    (OUT / "charts_4k_10k.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
