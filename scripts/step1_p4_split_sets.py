#!/usr/bin/env python
"""Build a P4 split that holds out whole beatmapsets (all charts of a song).

Experiment 2b: same player 60/40 activity-stratified split as step1_p4_split,
but the chart side groups by beatmapset_id, so no chart of a test song appears
in the 6x6 supervised training block.

Chart-side algorithm:
  * group charts by beatmapset_id (missing set -> singleton -beatmap_id);
  * stratify groups into 5 bins by set popularity (distinct player clusters);
  * inside each bin, assign whole groups to train/test so the *assigned chart
    count* is close to chart_frac (greedy: give each group to the side with the
    larger remaining chart deficit; random tie-break).

The output keeps the step1 schema and adds group_id / split_orig / p4_block.
The observed train/test chart counts can be lower than the assigned counts
because some charts have no train/test player who played them; report both.

Usage
-----
python scripts/step1_p4_split_sets.py \
  --in data/processed/step1_1k_pairbest_nogap.parquet \
  --meta data/processed/beatmap_meta_1k.csv \
  --out-prefix data/processed/step1_1k_pairbest_p4sets \
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
    """Boolean mask: True = train side, stratified by weight quantiles."""
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
        if idx.size > 1:
            n_train = min(max(n_train, 1), idx.size - 1)
        train[idx[:n_train]] = True
    return train


def _bin_ids(weights: np.ndarray, bins: int) -> np.ndarray:
    n = weights.size
    order = np.argsort(weights, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    return np.minimum((rank * bins) // n, bins - 1)


def _greedy_groups(groups: np.ndarray, bin_id: np.ndarray, sizes: np.ndarray,
                   frac: float, seed: int, bins: int) -> np.ndarray:
    """Assign whole groups; True = train; balance assigned chart counts."""
    rng = np.random.default_rng(seed)
    train = np.zeros(groups.size, dtype=bool)
    for b in range(bins):
        idx = np.flatnonzero(bin_id == b)
        if idx.size == 0:
            continue
        total = float(sizes[idx].sum())
        target_tr = frac * total
        target_te = (1.0 - frac) * total
        order = idx[np.lexsort((rng.random(idx.size), -sizes[idx]))]
        tr = 0.0
        te = 0.0
        for j in order:
            if (target_tr - tr) >= (target_te - te):
                train[j] = True
                tr += float(sizes[j])
            else:
                train[j] = False
                te += float(sizes[j])
        if idx.size > 1 and not train[idx].any():
            train[idx[0]] = True
        if idx.size > 1 and train[idx].all():
            train[idx[np.argmin(sizes[idx])]] = False
    return train


def build_one(df: pd.DataFrame, seed: int, player_frac: float,
              chart_frac: float, bins: int) -> tuple[pd.DataFrame, dict]:
    if "player_id_base" not in df.columns:
        df = df.copy()
        df["player_id_base"] = df["player_id"]
    base = df["player_id_base"].to_numpy()
    bid = df["beatmap_id"].to_numpy()
    groups = df["group_id"].to_numpy()

    players = np.sort(pd.unique(base))
    group_ids = np.sort(pd.unique(groups))
    player_weight = pd.Series(base).value_counts().reindex(players).to_numpy(np.float64)
    group_weight = (df.groupby("group_id")["player_id_base"].nunique()
                    .reindex(group_ids).to_numpy(np.float64))
    group_size = (df.groupby("group_id")["beatmap_id"].nunique()
                  .reindex(group_ids).to_numpy(np.float64))

    p_train_mask = _stratified(players, player_weight, player_frac,
                               seed * 2 + 1, bins)
    bin_id = _bin_ids(group_weight, bins)
    g_train_mask = _greedy_groups(group_ids, bin_id, group_size, chart_frac,
                                  seed * 2 + 2, bins)

    train_players = players[p_train_mask]
    train_groups = group_ids[g_train_mask]
    p_train = np.isin(base, train_players)
    c_train = np.isin(groups, train_groups)

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

    assigned_train = np.unique(bid[c_train])
    assigned_test = np.unique(bid[~c_train])
    observed_train = np.unique(bid[split == "train"])
    observed_test = np.unique(bid[split == "test"])
    overlap = set(assigned_train.tolist()) & set(assigned_test.tolist())
    info = {
        "seed": int(seed),
        "player_frac": float(player_frac),
        "chart_frac": float(chart_frac),
        "bins": int(bins),
        "n_players": int(players.size),
        "n_train_players": int(p_train_mask.sum()),
        "n_test_players": int((~p_train_mask).sum()),
        "n_groups": int(group_ids.size),
        "n_train_groups": int(g_train_mask.sum()),
        "n_test_groups": int((~g_train_mask).sum()),
        "assigned_charts": {"train": int(assigned_train.size),
                            "test": int(assigned_test.size),
                            "train_frac": float(assigned_train.size /
                                                (assigned_train.size + assigned_test.size))},
        "observed_charts": {"train": int(observed_train.size),
                            "test": int(observed_test.size)},
        "rows": {k: int(v) for k, v in pd.Series(split).value_counts().items()},
        "blocks": {k: int(v) for k, v in pd.Series(block).value_counts().items()},
        "group_overlap": int(len(overlap)),
        "group_size": {"min": int(group_size.min()), "median": float(np.median(group_size)),
                       "max": int(group_size.max())},
        "player_weight_train_mean": float(player_weight[p_train_mask].mean()),
        "player_weight_test_mean": float(player_weight[~p_train_mask].mean()),
        "group_weight_train_mean": float(group_weight[g_train_mask].mean()),
        "group_weight_test_mean": float(group_weight[~g_train_mask].mean()),
    }
    return out, info


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path,
                    help="beatmap meta csv with beatmap_id, beatmapset_id")
    ap.add_argument("--out-prefix", required=True,
                    help="output parquet: <prefix>_s<seed>.parquet")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--player-frac", type=float, default=0.6)
    ap.add_argument("--chart-frac", type=float, default=0.6)
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--force", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    src = args.inp
    df = pd.read_parquet(src)
    if "split" not in df.columns:
        raise SystemExit(f"{src}: no split column")
    meta = pd.read_csv(args.meta, usecols=["beatmap_id", "beatmapset_id"])
    smap = meta.set_index("beatmap_id")["beatmapset_id"]
    gid = smap.reindex(df["beatmap_id"]).to_numpy()
    df = df.copy()
    missing = pd.isna(gid)
    gid = gid.astype(np.float64)
    gid[missing] = -df.loc[missing, "beatmap_id"].to_numpy(np.float64)
    df["group_id"] = gid.astype(np.int64)
    if missing.any():
        print(f"[warn] {int(missing.sum())} rows / "
              f"{int(df.loc[missing, 'beatmap_id'].nunique())} charts missing "
              f"beatmapset_id; using -beatmap_id as singleton groups", flush=True)

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
        info.update({"in": str(src), "meta": str(args.meta),
                     "out": str(out_path)})
        info_path.write_text(json.dumps(info, indent=1, ensure_ascii=False) + "\n")
        print(f"[set-p4 s{seed}] groups {info['n_train_groups']}/"
              f"{info['n_test_groups']} assigned charts "
              f"{info['assigned_charts']['train']}/"
              f"{info['assigned_charts']['test']} observed "
              f"{info['observed_charts']['train']}/"
              f"{info['observed_charts']['test']} rows {info['rows']}",
              flush=True)
        all_info.append(info)
    summary = Path(f"{prefix}_summary.json")
    summary.write_text(json.dumps({"in": str(src), "meta": str(args.meta),
                                   "splits": all_info},
                                  indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {summary} ({len(all_info)} seeds)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
