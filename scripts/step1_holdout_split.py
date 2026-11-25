#!/usr/bin/env python3
"""step1_holdout_split -- H4: double cold start on a fixed chart holdout.

H4 is P4 with the chart side fixed instead of randomly split:

    test charts  = the IDs in --holdout-ids (D2V-unseen charts)
    players      = stratified 60/40 by activity, clustered by player_id_base
    train        = train players x non-holdout charts (minus val charts)
    val          = train players x a 10% chart-disjoint subset of train charts
    test         = test players  x holdout charts
    everything else (off-diagonal blocks) = dropped, kept only for the BT ceiling

Why: for the frozen/unfrozen/random encoder comparison, a test chart that D2V
saw during masked pretraining would give the D2V arms an advantage unrelated to
behavior supervision.  H4 fixes that while keeping the P4 6x6 / 4x4 structure.

Output keeps the step1 schema and replaces split with train/val/test/dropped;
split_orig and h4_block are diagnostics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step1_p4_split import _stratified  # noqa: E402


def read_ids(path: Path) -> list[int]:
    if not path.exists():
        raise SystemExit(f"--holdout-ids {path} not found")
    return [int(x) for x in path.read_text().split() if x.strip()]


def build_one(df: pd.DataFrame, seed: int, holdout: set[int],
              player_frac: float, bins: int, val_chart_frac: float) -> tuple[pd.DataFrame, dict]:
    if "player_id_base" not in df.columns:
        df = df.copy()
        df["player_id_base"] = df["player_id"]

    base = df["player_id_base"].to_numpy()
    bid = df["beatmap_id"].to_numpy()
    players = np.sort(pd.unique(base))
    charts = np.sort(pd.unique(bid))

    test_charts = np.array(sorted(set(charts.tolist()) & holdout), dtype=np.int64)
    train_charts_all = np.array(sorted(set(charts.tolist()) - holdout), dtype=np.int64)
    if test_charts.size == 0:
        raise SystemExit("no holdout chart has any pair in --in")
    if train_charts_all.size == 0:
        raise SystemExit("holdout covers every chart; no train charts")

    player_weight = pd.Series(base).value_counts().reindex(players).to_numpy(np.float64)
    p_train_mask = _stratified(players, player_weight, player_frac,
                               seed * 2 + 1, bins)
    train_players = players[p_train_mask]

    chart_weight = (df.groupby("beatmap_id")["player_id_base"].nunique()
                    .reindex(charts).to_numpy(np.float64))
    train_chart_weight = pd.Series(chart_weight, index=charts).reindex(
        train_charts_all).to_numpy(np.float64)
    if val_chart_frac > 0:
        vmask = _stratified(train_charts_all, train_chart_weight,
                            val_chart_frac, seed * 2 + 3, bins)
        val_charts = train_charts_all[vmask]
        train_charts = train_charts_all[~vmask]
    else:
        val_charts = np.zeros(0, dtype=np.int64)
        train_charts = train_charts_all

    p_train = np.isin(base, train_players)
    c_test = np.isin(bid, test_charts)
    c_val = np.isin(bid, val_charts)
    c_train = np.isin(bid, train_charts)

    split = np.full(len(df), "dropped", dtype=object)
    split[p_train & c_train] = "train"
    split[p_train & c_val] = "val"
    split[(~p_train) & c_test] = "test"

    block = np.full(len(df), "dropped", dtype=object)
    block[p_train & c_train] = "train"
    block[p_train & c_val] = "val"
    block[(~p_train) & c_test] = "test"
    block[p_train & c_test] = "train_player_test_chart"
    block[(~p_train) & c_train] = "test_player_train_chart"
    block[(~p_train) & c_val] = "test_player_val_chart"

    out = df.copy()
    out["split_orig"] = df["split"]
    out["split"] = split
    out["h4_block"] = block

    test_base = set(base[split == "test"].tolist())
    train_base = set(base[(split == "train") | (split == "val")].tolist())
    checks = {
        "test_charts_subset_holdout": bool(set(test_charts.tolist()) <= holdout),
        "test_charts_not_in_train": bool(not (set(test_charts.tolist()) &
                                             set(train_charts.tolist()))),
        "test_charts_not_in_val": bool(not (set(test_charts.tolist()) &
                                           set(val_charts.tolist()))),
        "val_charts_not_in_train": bool(not (set(val_charts.tolist()) &
                                            set(train_charts.tolist()))),
        "test_players_not_in_train": bool(not (test_base & train_base)),
        "test_players_not_in_val": bool(not (test_base &
                                             set(base[split == "val"].tolist()))),
    }
    info = {
        "seed": int(seed),
        "player_frac": float(player_frac),
        "bins": int(bins),
        "val_chart_frac": float(val_chart_frac),
        "n_players": int(players.size),
        "n_train_players": int(p_train_mask.sum()),
        "n_test_players": int((~p_train_mask).sum()),
        "n_charts": int(charts.size),
        "n_train_charts": int(train_charts.size),
        "n_val_charts": int(val_charts.size),
        "n_test_charts": int(test_charts.size),
        "n_holdout_ids": int(len(holdout)),
        "n_holdout_missing": int(len(holdout - set(charts.tolist()))),
        "rows": {k: int(v) for k, v in
                 pd.Series(split).value_counts().items()},
        "blocks": {k: int(v) for k, v in
                   pd.Series(block).value_counts().items()},
        "checks": checks,
    }
    if not all(checks.values()):
        raise SystemExit(f"split checks failed: {checks}")
    return out, info


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="step1 pairbest table")
    ap.add_argument("--holdout-ids", required=True,
                    help="whitespace-separated beatmap ids, e.g. D2V eval_ids.txt")
    ap.add_argument("--out-prefix", default="data/processed/step1_full_holdout")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--player-frac", type=float, default=0.6)
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--val-chart-frac", type=float, default=0.1)
    ap.add_argument("--force", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    src = Path(args.inp)
    df = pd.read_parquet(src)
    if "split" not in df.columns:
        raise SystemExit(f"{src}: no split column")
    holdout = set(read_ids(Path(args.holdout_ids)))
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
        out, info = build_one(df, seed, holdout, args.player_frac,
                              args.bins, args.val_chart_frac)
        out.to_parquet(out_path, index=False)
        info.update({"in": str(src), "out": str(out_path),
                     "holdout_ids": str(args.holdout_ids)})
        info_path.write_text(json.dumps(info, indent=1, ensure_ascii=False) + "\n")
        print(f"[h4 s{seed}] players {info['n_train_players']}/"
              f"{info['n_test_players']} charts {info['n_train_charts']}/"
              f"{info['n_val_charts']}/{info['n_test_charts']} "
              f"rows {info['rows']}", flush=True)
        all_info.append(info)

    summary = Path(f"{prefix}_summary.json")
    summary.write_text(json.dumps({"in": str(src), "splits": all_info},
                                  indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {summary} ({len(all_info)} seeds)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
