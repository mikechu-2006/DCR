"""Dataset v2 (user's rules) -- memory-lean rewrite.

* classic scoring only: modern rows must carry CL; legacy bitmask rows are classic by construction
* ACC = classic (flat) formula, plus one pseudo "250" judgement:
      ACC = (300*(c300+cMAX) + 200*c200 + 100*c100 + 50*c50 + 250) / (300*(total+1))
      L   = log(1 - ACC)
* kept mods: CL, MR, DT, NC, HT, SD, PF, NF   (HD/FL/EZ/HR/FI/RD/... dropped)
* chart = (beatmap_id, rate) with rate 1 = DT/NC (1.5x), rate 2 = HT (0.75x)
* (player, chart) cells with playcount <= 2 dropped
* keeps calendar year + day-of-year for player-year entities with linear drift
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

INTERIM = Path("data/interim")
OUT = Path("data/processed")

# legacy bitmask bits that are allowed: NF(1) SD(32) DT(64) HT(256) NC(512) PF(16384) Mirror(1<<30)
ALLOW_BITS = 1 | 32 | 64 | 256 | 512 | 16384 | (1 << 30)
ALLOW_TOKENS = {"CL", "MR", "DT", "NC", "HT", "SD", "PF", "NF"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plays", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mods-from-bitmask", action="store_true")
    ap.add_argument("--attempts", default=str(INTERIM / "playcount.csv"))
    ap.add_argument("--min-attempts", type=int, default=2,
                    help="osu_user_beatmap_playcount of the (player, map) pair must exceed this")
    ap.add_argument("--min-scores", type=int, default=3,
                    help="recorded plays of the *cell* (player, chart) must be at least this many")
    ap.add_argument("--chunks", type=int, default=16)
    args = ap.parse_args()

    bm = pd.read_csv(INTERIM / "beatmaps.csv")
    bm4 = bm[(bm.playmode == 3) & (bm["keys"] == 4)].copy()
    ids = set(bm4.beatmap_id.tolist())
    print("4K beatmaps:", len(bm4), flush=True)

    d = pd.read_parquet(args.plays)
    d = d[d.beatmap_id.isin(ids)].reset_index(drop=True)
    print("input 4K plays:", len(d), flush=True)

    if args.mods_from_bitmask:
        mask = d.enabled_mods.to_numpy(np.int64, copy=False)
        keep = (mask & ~ALLOW_BITS) == 0
        rate = np.where(mask & (64 | 512) != 0, 1, np.where(mask & 256 != 0, 2, 0)).astype(np.int8)
        print(f"bitmask path: dropped {int((~keep).sum())} plays (HD/FL/EZ/HR/FI/RD/...)", flush=True)
        d = d[keep].reset_index(drop=True)
        rate = rate[keep]
    else:
        col = "mods_raw" if "mods_raw" in d.columns else "mods"
        toks = [set(str(s).split("+")) - {""} for s in d[col].fillna("")]
        keep = np.fromiter((t <= ALLOW_TOKENS for t in toks), dtype=bool, count=len(toks))
        if "source" in d.columns:
            modern = (d.source == "modern").to_numpy()
            keep &= (~modern) | np.fromiter(("CL" in t for t in toks), dtype=bool, count=len(toks))
        rate = np.fromiter((1 if (t & {"DT", "NC"}) else (2 if "HT" in t else 0) for t in toks),
                           dtype=np.int8, count=len(toks))
        print(f"token path: dropped {int((~keep).sum())} plays", flush=True)
        d = d[keep].reset_index(drop=True)
        rate = rate[keep]

    d["rate"] = rate
    d["chart_id"] = d.beatmap_id.to_numpy(np.int64) * 4 + rate

    c320 = d.c320.to_numpy(np.int64); c300 = d.c300.to_numpy(np.int64)
    c200 = d.c200.to_numpy(np.int64); c100 = d.c100.to_numpy(np.int64)
    c50 = d.c50.to_numpy(np.int64); cmiss = d.cmiss.to_numpy(np.int64)
    total = c320 + c300 + c200 + c100 + c50 + cmiss
    num = 300.0 * (c300 + c320) + 200.0 * c200 + 100.0 * c100 + 50.0 * c50 + 250.0
    den = 300.0 * (total + 1.0)
    acc = (num / den).astype(np.float32)
    loss = np.log1p(-acc.astype(np.float64)).astype(np.float32)
    ok = total > 0
    if not ok.all():
        d = d[ok].reset_index(drop=True); acc = acc[ok]; loss = loss[ok]; total = total[ok]
    print("acc range [%.6f, %.6f] | loss mean %.3f std %.3f" %
          (acc.min(), acc.max(), loss.mean(), loss.std()), flush=True)

    if "year" in d.columns:
        year = d.year.to_numpy(np.int16); doy = d.doy.to_numpy(np.int16)
    else:
        ts = pd.to_datetime(d["date"])
        year = ts.dt.year.to_numpy(np.int16); doy = ts.dt.dayofyear.to_numpy(np.int16)
    entity = d.user_id.to_numpy(np.int64) * 100 + (year.astype(np.int64) - 2000)
    plays = pd.DataFrame({
        "user_id": d.user_id.to_numpy(np.int32), "entity_id": entity,
        "beatmap_id": d.beatmap_id.to_numpy(np.int32), "chart_id": d.chart_id.to_numpy(np.int64),
        "rate": d.rate.to_numpy(np.int8), "year": year, "doy": doy,
        "tau": (doy.astype(np.float32) - 1) / 365.0,
        "total": total.astype(np.int32), "acc": acc, "loss": loss})
    del d, c320, c300, c200, c100, c50, cmiss, num, den
    plays.to_parquet(OUT / f"plays_v2_{args.tag}.parquet", index=False)
    print("wrote plays", plays.shape, flush=True)

    pc = pd.read_csv(args.attempts, dtype={"user_id": np.int32, "beatmap_id": np.int32, "playcount": np.int32})
    pc = pc[pc.beatmap_id.isin(ids)].rename(columns={"playcount": "attempts"})
    print("playcount rows (4K):", len(pc), flush=True)

    parts = []
    for chunk in np.array_split(plays.user_id.unique(), args.chunks):
        sub = plays[plays.user_id.isin(chunk)]
        parts.append(sub.groupby(["user_id", "chart_id"], sort=False).agg(
            beatmap_id=("beatmap_id", "first"), rate=("rate", "first"), n_scores=("total", "size"),
            acc_med=("acc", "median"), acc_max=("acc", "max"), acc_mean=("acc", "mean"),
            acc_min=("acc", "min"), loss_med=("loss", "median"), loss_max=("loss", "max"),
            loss_mean=("loss", "mean"), tau_mean=("tau", "mean"),
            year_first=("year", "min"), year_last=("year", "max")).reset_index())
        del sub
    cells = pd.concat(parts, ignore_index=True)
    del parts, plays
    cells = cells.merge(pc, on=["user_id", "beatmap_id"], how="left")
    del pc
    cells = cells.merge(bm4[["beatmap_id", "star", "max_combo", "countTotal", "playcount", "passcount", "approved"]],
                        on="beatmap_id", how="left")
    cells["low_acc"] = cells.acc_med < 0.90
    cells["few_plays"] = cells.attempts.fillna(0) <= args.min_attempts
    cells["few_scores"] = cells.n_scores <= args.min_scores - 1
    cells["primary"] = (~cells.low_acc) & (~cells.few_plays) & (~cells.few_scores)
    cells.to_parquet(OUT / f"charts_v2_{args.tag}.parquet", index=False)
    summary = {"tag": args.tag, "plays": int(len(cells)), "cells": int(len(cells)),
               "primary": int(cells.primary.sum()), "low_acc": int(cells.low_acc.sum()),
               "few_plays": int(cells.few_plays.sum()), "few_scores": int(cells.few_scores.sum()),
               "charts": int(cells.chart_id.nunique()),
               "beatmaps": int(cells.beatmap_id.nunique()), "users": int(cells.user_id.nunique()),
               "rate_mix_cells": {str(k): int(v) for k, v in cells.rate.value_counts().items()}}
    (OUT / f"charts_v2_{args.tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
