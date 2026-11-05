#!/usr/bin/env python
"""step1_p4_split -- build the P4 double-cold-start split.

P4 asks the hardest generalisation question used in this repository: neither the
evaluation players nor the evaluation charts were available when the difficulty
model was fitted.

    players: stratified 60/40 by activity (number of pair rows)
    charts : stratified 60/40 by popularity (number of distinct players)
    train  : train players x train charts            (6 x 6)
    test   : test players  x test charts             (4 x 4)
    off-diagonal rows are labelled "dropped": they are never trained on, but
    they stay in the parquet so the cross-fitted Bradley-Terry ceiling can use
    them (the ceiling is a reference, not a model; see docs/p4_multiseed.md).

The output table keeps the step1 schema and replaces `split` with
train / test / dropped.  `p4_block` and `split_orig` are diagnostic columns;
the model and eval scripts ignore them.

Example
-------
python scripts/step1_p4_split.py \
    --in data/processed/step1_1k_pairbest.parquet \
    --out-prefix data/processed/step1_1k_pairbest_p4 \
    --seeds 1,2,3,4,5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _stratified(ids: np.ndarray, weights: np.ndarray, frac: float,
                seed: int, bins: int) -> np.ndarray:
    """Boolean mask: True = train side, stratified by weight quantiles.

    Ranks (not raw quantiles) are used so tied weights cannot collapse a bin.
    Within a bin the assignment is a random permutation, so the train side has
    the requested fraction of every activity/popularity stratum.
    """
    n = int(ids.size)
    if n == 0:
        raise SystemExit("empty id list")
    order = np.argsort(weights, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    bin_id = np.minimum((rank * bins) // n, bins - 1)

    rng = np.random.default_rng(seed)
    train = np.zeros(n, dtype=bool)
    for b in range(bins):
        idx = np.flatnonzero(bin_id == b)
        if idx.size == 0:
            continue
        idx = rng.permutation(idx)
        n_train = int(round(frac * idx.size))
        if idx.size > 1:  # keep at least one id on each side when possible
            n_train = min(max(n_train, 1), idx.size - 1)
        train[idx[:n_train]] = True
    return train


def build_one(df: pd.DataFrame, seed: int, player_frac: float,
              chart_frac: float, bins: int) -> tuple[pd.DataFrame, dict]:
    if "player_id_base" not in df.columns:
        df = df.copy()
        df["player_id_base"] = df["player_id"]

    base = df["player_id_base"].to_numpy()
    bid = df["beatmap_id"].to_numpy()

    players = np.sort(pd.unique(base))
    charts = np.sort(pd.unique(bid))

    # player activity: rows per player cluster
    player_weight = pd.Series(base).value_counts().reindex(players).to_numpy(np.float64)
    # chart popularity: distinct player clusters per chart (not raw plays)
    chart_weight = (df.groupby("beatmap_id")["player_id_base"].nunique()
                    .reindex(charts).to_numpy(np.float64))

    p_train_mask = _stratified(players, player_weight, player_frac,
                               seed * 2 + 1, bins)
    c_train_mask = _stratified(charts, chart_weight, chart_frac,
                               seed * 2 + 2, bins)

    train_players = players[p_train_mask]
    train_charts = charts[c_train_mask]

    p_train = np.isin(base, train_players)
    c_train = np.isin(bid, train_charts)

    split = np.where(p_train & c_train, "train",
                     np.where((~p_train) & (~c_train), "test", "dropped"))
    block = np.where(p_train & c_train, "train_6x6",
                     np.where((~p_train) & (~c_train), "test_4x4",
                              np.where(p_train, "train_player_test_chart",
                                       "test_player_train_chart")))

    out = df.copy()
    out["split_orig"] = df["split"]
    out["split"] = split
    out["p4_block"] = block

    info = {
        "seed": int(seed),
        "player_frac": float(player_frac),
        "chart_frac": float(chart_frac),
        "bins": int(bins),
        "n_players": int(players.size),
        "n_train_players": int(p_train_mask.sum()),
        "n_test_players": int((~p_train_mask).sum()),
        "n_charts": int(charts.size),
        "n_train_charts": int(c_train_mask.sum()),
        "n_test_charts": int((~c_train_mask).sum()),
        "rows": {k: int(v) for k, v in
                 pd.Series(split).value_counts().items()},
        "blocks": {k: int(v) for k, v in
                   pd.Series(block).value_counts().items()},
        "player_weight_train_mean": float(player_weight[p_train_mask].mean()),
        "player_weight_test_mean": float(player_weight[~p_train_mask].mean()),
        "chart_weight_train_mean": float(chart_weight[c_train_mask].mean()),
        "chart_weight_test_mean": float(chart_weight[~c_train_mask].mean()),
    }
    return out, info


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="step1 pairbest table")
    ap.add_argument("--out-prefix", default="data/processed/step1_1k_pairbest_p4",
                    help="output parquet: <prefix>_s<seed>.parquet")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--player-frac", type=float, default=0.6)
    ap.add_argument("--chart-frac", type=float, default=0.6)
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--force", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    src = Path(args.inp)
    df = pd.read_parquet(src)
    if "split" not in df.columns:
        raise SystemExit(f"{src}: no split column")
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
        out, info = build_one(df, seed, args.player_frac, args.chart_frac,
                              args.bins)
        out.to_parquet(out_path, index=False)
        info.update({"in": str(src), "out": str(out_path)})
        info_path.write_text(json.dumps(info, indent=1, ensure_ascii=False) + "\n")
        print(f"[p4 s{seed}] players {info['n_train_players']}/"
              f"{info['n_test_players']} charts {info['n_train_charts']}/"
              f"{info['n_test_charts']} rows {info['rows']}", flush=True)
        all_info.append(info)

    summary = Path(f"{prefix}_summary.json")
    summary.write_text(json.dumps({"in": str(src), "splits": all_info},
                                  indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {summary} ({len(all_info)} seeds)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
