#!/usr/bin/env python
"""Step 1 (default, A): pair-best cleaning -- distilled reference implementation.

This is a compact, dependency-light implementation of the default step1 version.
It is meant to be read together with preprocess/README.md; the production entry
point remains scripts/step1_pairbest.py (same rules, same defaults).

Rules, in execution order:
  0. player floor: drop the whole player when its step0 record count is < 20.
  1. player epoch split: within a player, a gap of >= 365 days starts a new
     player. Segment 0 keeps the original id; later segments get fresh ids.
  2. pair best: for each (new player, beatmap), keep the row with the smallest
     loss (= highest ACC). Ties: earliest timestamp, then smallest playcount_cur.
  3. singleton drop: drop pairs with fewer than 2 recorded plays.
  4. high-ACC drop: drop pairs whose best ACC is > 0.998.
  5. low-ACC drop: drop pairs whose best ACC is <= 0.90.
  6. split: pair-random 70/30 on the surviving pairs.

Output: one row per surviving pair, with the same columns as the production
script: player_id, beatmap_id, timestamp, playcount_cur, loss, split,
player_id_base, player_seg, n_plays.

Examples:
  python preprocess/step1_pairbest.py --tag 1k
  python preprocess/step1_pairbest.py --tag 1k --max-best-acc 1.0
  python preprocess/step1_pairbest.py --in data/processed/step0_1k.csv \\
      --out /tmp/step1_1k_pairbest.csv --no-parquet
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["player_id", "beatmap_id", "timestamp", "playcount_cur", "loss"]
OUT_COLS = COLS + ["split", "player_id_base", "player_seg", "n_plays"]
OUT = Path("data/processed")


def resolve_input(tag: str, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise SystemExit(f"--in not found: {path}")
        return path
    for cand in (OUT / f"step0_{tag}.parquet", OUT / f"step0_{tag}.csv"):
        if cand.exists():
            return cand
    raise SystemExit(f"no step0 input found for tag={tag} in {OUT}")


def load(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path, columns=COLS)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        df = pd.read_csv(
            path,
            dtype={"player_id": "int32", "beatmap_id": "int32", "playcount_cur": "int32"},
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
    df["loss"] = df["loss"].astype("float32")
    return df[COLS]


def assign_epochs(df: pd.DataFrame, gap_days: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (new_player_id, segment_index, info), aligned with df.index."""
    n = len(df)
    base = df["player_id"].to_numpy(np.int64)
    bid = df["beatmap_id"].to_numpy(np.int64)
    pc = df["playcount_cur"].to_numpy(np.int64)
    ts = df["timestamp"].to_numpy().astype("datetime64[s]").astype(np.int64)

    order = np.lexsort((pc, bid, ts, base))
    base_s, ts_s = base[order], ts[order]

    new_seg = np.zeros(n, dtype=bool)
    if n:
        new_seg[0] = True
        same_base = base_s[1:] == base_s[:-1]
        if gap_days > 0:
            gap = (ts_s[1:] - ts_s[:-1]).astype(np.float64) >= float(gap_days) * 86400.0
            new_seg[1:] = (~same_base) | gap
        else:
            new_seg[1:] = ~same_base

    # Segment index within each base player. cum is the global cumulative count
    # of segment starts; subtract the count at the player's first row.
    cum = np.cumsum(new_seg, dtype=np.int64)
    starts = np.r_[0, np.flatnonzero(base_s[1:] != base_s[:-1]) + 1]
    lengths = np.diff(np.r_[starts, n])
    first_cum = cum[starts] - 1
    seg_sorted = (cum - np.repeat(first_cum, lengths) - 1).astype(np.int32)

    seg_max = int(seg_sorted.max()) if n else 0
    span = np.int64(seg_max + 1)
    keys = base_s * span + seg_sorted.astype(np.int64)
    codes, uniq = pd.factorize(keys, sort=True)
    ubase = (uniq // span).astype(np.int64)
    useg = (uniq % span).astype(np.int64)

    fresh = np.empty(uniq.size, dtype=np.int64)
    fresh[useg == 0] = ubase[useg == 0]
    next_id = int(base.max()) + 1 if n else 0
    for j in np.flatnonzero(useg > 0):
        fresh[j] = next_id
        next_id += 1
    pid_sorted = fresh[codes]

    pid = np.empty(n, dtype=np.int64)
    seg = np.empty(n, dtype=np.int32)
    pid[order] = pid_sorted
    seg[order] = seg_sorted

    counts = np.unique(ubase, return_counts=True)[1]
    info = {
        "n_bases": int(counts.size),
        "n_segments": int(uniq.size),
        "n_split_players": int((counts > 1).sum()),
        "n_seg_extra": int(uniq.size - counts.size),
    }
    return pid, seg, info


def pair_best(df: pd.DataFrame, pid: np.ndarray, min_plays: int,
              max_best_acc: float, acc_floor: float,
              label_shift: float = 0.0) -> dict:
    n = len(df)
    bid = df["beatmap_id"].to_numpy(np.int64)
    loss = df["loss"].to_numpy(np.float32)
    pc = df["playcount_cur"].to_numpy(np.int64)
    ts = df["timestamp"].to_numpy().astype("datetime64[s]").astype(np.int64)

    k = int(bid.max()) + 1 if n else 1
    pair_key = pid * np.int64(k) + bid
    codes, uniq = pd.factorize(pair_key, sort=False)
    counts = np.bincount(codes, minlength=uniq.size).astype(np.int64)

    keep = counts[codes] >= min_plays
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        raise SystemExit("no pair survives --min-plays-per-pair")

    sub_key = pair_key[idx]
    order = np.lexsort((pc[idx], ts[idx], loss[idx], sub_key))
    idx_sorted = idx[order]
    key_sorted = sub_key[order]
    first = np.r_[0, np.flatnonzero(key_sorted[1:] != key_sorted[:-1]) + 1]
    best_idx = idx_sorted[first]
    n_plays = counts[codes[best_idx]]
    best_loss = loss[best_idx].astype(np.float64)

    upper = 1.0 + label_shift
    thr_high = math.log(upper - max_best_acc) if max_best_acc < upper else -math.inf
    thr_low = math.log(upper - acc_floor)
    keep_high = best_loss >= thr_high
    keep_low = best_loss < thr_low
    keep_acc = keep_high & keep_low

    n_before_acc = int(best_idx.size)
    best_idx = best_idx[keep_acc]
    n_plays = n_plays[keep_acc]
    best_loss = best_loss[keep_acc]

    return {
        "pair_key": pair_key,
        "counts": counts,
        "n_pairs_after_epoch": int(uniq.size),
        "n_pairs_below_min_plays": int((counts < min_plays).sum()),
        "n_rows_below_min_plays": int(counts[counts < min_plays].sum()),
        "n_pairs_ge_min_plays": n_before_acc,
        "n_rows_ge_min_plays": int(counts[counts >= min_plays].sum()),
        "n_single_play_pairs": int((counts == 1).sum()),
        "n_high_acc_dropped": int((~keep_high).sum()),
        "n_low_acc_dropped": int((~keep_low).sum()),
        "best_idx": best_idx,
        "n_plays": n_plays,
        "best_loss": best_loss,
        "threshold_loss": thr_high,
        "acc_floor_loss": thr_low,
    }


def split_pairs(df: pd.DataFrame, method: str, train_frac: float, seed: int) -> tuple[np.ndarray, dict]:
    test_frac = 1.0 - train_frac
    if method == "pair-random":
        k = int(df["beatmap_id"].max()) + 1
        keys = df["player_id"].to_numpy(np.int64) * np.int64(k) + df["beatmap_id"].to_numpy(np.int64)
        units = np.unique(keys)
        is_test_unit = np.random.default_rng(seed).random(units.size) < test_frac
        is_test = is_test_unit[np.searchsorted(units, keys)]
        info = {
            "method": method, "unit": "pair", "n_units": int(units.size),
            "n_train_units": int((~is_test_unit).sum()),
            "n_test_units": int(is_test_unit.sum()),
        }
    elif method == "beatmap-random":
        bids = np.unique(df["beatmap_id"].to_numpy())
        is_test_b = np.random.default_rng(seed).random(bids.size) < test_frac
        is_test = is_test_b[np.searchsorted(bids, df["beatmap_id"].to_numpy())]
        info = {
            "method": method, "unit": "beatmap", "n_units": int(bids.size),
            "n_train_units": int((~is_test_b).sum()),
            "n_test_units": int(is_test_b.sum()),
        }
    else:
        raise SystemExit(f"unknown --split-method {method!r}")
    info["actual_train_frac"] = info["n_train_units"] / max(info["n_units"], 1)
    info["actual_test_frac"] = info["n_test_units"] / max(info["n_units"], 1)
    return is_test, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="1k")
    ap.add_argument("--in", dest="inp", default=None,
                    help="default data/processed/step0_{tag}.{parquet,csv}")
    ap.add_argument("--out", default=None,
                    help="default data/processed/step1_{tag}_pairbest.csv")
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--split-method", default="pair-random",
                    choices=["pair-random", "beatmap-random"])
    ap.add_argument("--split-seed", type=int, default=20260928)
    ap.add_argument("--min-plays-per-pair", type=int, default=2)
    ap.add_argument("--max-best-acc", type=float, default=0.998)
    ap.add_argument("--acc-floor", type=float, default=0.90)
    ap.add_argument("--label-shift", type=float, default=0.0,
                    help="log(1+s-ACC); default 0 (A). 0.001 is the C/D ablation, not the default.")
    ap.add_argument("--gap-days", type=float, default=365.0)
    ap.add_argument("--min-plays-per-player", type=int, default=20)
    ap.add_argument("--no-parquet", action="store_true")
    ap.add_argument("--no-csv", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    if not 0.0 < args.train_frac < 1.0:
        raise SystemExit("--train-frac must be in (0, 1)")
    if args.min_plays_per_pair < 1:
        raise SystemExit("--min-plays-per-pair must be >= 1")
    if not 0.0 < args.max_best_acc <= 1.0:
        raise SystemExit("--max-best-acc must be in (0, 1]")
    if not 0.0 <= args.acc_floor < 1.0:
        raise SystemExit("--acc-floor must be in [0, 1)")
    if args.acc_floor >= args.max_best_acc:
        raise SystemExit("--acc-floor must be < --max-best-acc")
    if args.label_shift < 0:
        raise SystemExit("--label-shift must be >= 0")
    if args.gap_days < 0:
        raise SystemExit("--gap-days must be >= 0")
    if args.min_plays_per_player < 0:
        raise SystemExit("--min-plays-per-player must be >= 0")

    src = resolve_input(args.tag, args.inp)
    out_csv = Path(args.out) if args.out else OUT / f"step1_{args.tag}_pairbest.csv"
    if out_csv.suffix == ".parquet":
        out_csv = out_csv.with_suffix(".csv")
    out_pq, out_json = out_csv.with_suffix(".parquet"), out_csv.with_suffix(".json")

    print(f"[step1/pairbest] tag={args.tag} in={src}", flush=True)
    df_all = load(src)
    if args.label_shift > 0:
        old = df_all["loss"].to_numpy(np.float64)
        df_all["loss"] = np.log(np.exp(old) + args.label_shift).astype(np.float32)
        print(f"[step1/pairbest] label: log(1+{args.label_shift:g}-ACC)", flush=True)

    n_in = len(df_all)
    k0 = int(df_all.beatmap_id.max()) + 1
    raw_keys = df_all.player_id.to_numpy(np.int64) * np.int64(k0) + df_all.beatmap_id.to_numpy(np.int64)
    n_raw_pairs = int(pd.factorize(raw_keys, sort=False)[1].size)
    del raw_keys

    n_low_players = n_low_rows = 0
    df = df_all
    if args.min_plays_per_player > 0:
        plays = df_all.groupby("player_id", sort=False).size()
        low = plays.index[plays.to_numpy() < args.min_plays_per_player]
        n_low_players, n_low_rows = int(len(low)), int(plays.loc[low].sum())
        df = df_all[~df_all.player_id.isin(low)].reset_index(drop=True)
        if len(df) == 0:
            raise SystemExit("--min-plays-per-player dropped every row")

    base = df["player_id"].to_numpy(np.int64)
    pid, seg, epoch_info = assign_epochs(df, args.gap_days)
    bp = pair_best(df, pid, args.min_plays_per_pair, args.max_best_acc,
                   args.acc_floor, args.label_shift)

    keep_idx = np.sort(bp["best_idx"])
    out = df.iloc[keep_idx].reset_index(drop=True)
    out["player_id"] = pid[keep_idx].astype(np.int64)
    out["player_id_base"] = base[keep_idx].astype(np.int64)
    out["player_seg"] = seg[keep_idx].astype(np.int32)
    out["n_plays"] = bp["n_plays"][np.argsort(bp["best_idx"])].astype(np.int64)

    is_test, split_info = split_pairs(out, args.split_method, args.train_frac, args.split_seed)
    out["split"] = np.where(is_test, "test", "train")
    out = out[OUT_COLS]
    n_train, n_test = int((~is_test).sum()), int(is_test.sum())

    # ---- compact checks -------------------------------------------------------
    acc = 1.0 + args.label_shift - np.exp(out["loss"].to_numpy(np.float64))
    pair_dup = int(out.duplicated(["player_id", "beatmap_id"]).sum())
    pairs = (out[["player_id_base", "player_seg", "player_id"]]
             .drop_duplicates().sort_values(["player_id_base", "player_seg"]))
    map_ok = bool(pairs.groupby(["player_id_base", "player_seg"])["player_id"].nunique().max() == 1
                  and pairs["player_id"].nunique() == len(pairs))
    seg0_ok = bool((pairs.loc[pairs.player_seg == 0, "player_id"]
                    == pairs.loc[pairs.player_seg == 0, "player_id_base"]).all())
    fresh_ok = bool((pairs.loc[pairs.player_seg > 0, "player_id"]
                     > int(base.max())).all())
    val_cols = ["beatmap_id", "timestamp", "playcount_cur", "loss"]
    subset_ok = bool(df.iloc[keep_idx][val_cols].reset_index(drop=True)
                     .equals(out[val_cols].reset_index(drop=True)))
    checks = {
        "rows_nonempty": len(out) > 0,
        "one_row_per_player_beatmap": pair_dup == 0,
        "min_plays_per_pair_enforced": bool((out["n_plays"] >= args.min_plays_per_pair).all()),
        "best_acc_at_or_below_max": bool((out["loss"].to_numpy(float) >= bp["threshold_loss"] - 1e-12).all()),
        "min_acc_strictly_above_floor": bool((out["loss"].to_numpy(float) < bp["acc_floor_loss"]).all()),
        "player_ids_unique_per_base_segment": map_ok,
        "segment_zero_keeps_base_id": seg0_ok,
        "later_segments_get_fresh_ids": fresh_ok,
        "split_labels_valid": bool(n_train + n_test == len(out)),
        "actual_train_frac_within_1pct": abs(split_info["actual_train_frac"] - args.train_frac) <= 0.01,
        "loss_negative": bool((out["loss"] < 0).all()),
        "output_rows_are_unmodified_input_rows": subset_ok,
    }

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_csv:
        tmp = out_csv.with_name(out_csv.name + ".part")
        out.to_csv(tmp, index=False, date_format="%Y-%m-%d %H:%M:%S")
        tmp.replace(out_csv)
    if not args.no_parquet:
        tmp = out_pq.with_name(out_pq.name + ".part")
        out.to_parquet(tmp, index=False)
        tmp.replace(out_pq)

    report = {
        "method": "pairbest",
        "label": {"step0": "log(1-ACC)", "shift": args.label_shift,
                  "definition": "log(1+shift-ACC)"},
        "rules": {
            "order": "player_floor -> gap_split -> pair_best -> min_plays_per_pair "
                     "-> max_best_acc -> acc_floor -> split",
            "min_plays_per_player": args.min_plays_per_player,
            "gap_days": args.gap_days,
            "min_plays_per_pair": args.min_plays_per_pair,
            "max_best_acc": args.max_best_acc,
            "acc_floor": args.acc_floor,
            "label_shift": args.label_shift,
            "best": "min loss (= max ACC) within (player_epoch, beatmap)",
            "best_tie_break": "earliest timestamp, then smallest playcount_cur",
        },
        "split": {**split_info, "seed": args.split_seed, "train_frac": args.train_frac,
                  "rows": {"train": n_train, "test": n_test}},
        "funnel": {
            "step0_rows": n_in,
            "step0_pairs": n_raw_pairs,
            "dropped_low_play_rows": n_low_rows,
            "rows_after_player_floor": int(len(df)),
            "player_segments": epoch_info["n_segments"],
            "players_split_by_gap": epoch_info["n_split_players"],
            "extra_segments": epoch_info["n_seg_extra"],
            "pairs_after_epoch_split": bp["n_pairs_after_epoch"],
            "pairs_below_min_plays": bp["n_pairs_below_min_plays"],
            "single_play_pairs": bp["n_single_play_pairs"],
            "pairs_ge_min_plays": bp["n_pairs_ge_min_plays"],
            "dropped_high_acc_pairs": bp["n_high_acc_dropped"],
            "dropped_low_acc_pairs": bp["n_low_acc_dropped"],
            "kept_pairs": int(len(out)),
            "test_rows": n_test,
            "train_rows": n_train,
        },
        "players": {"in": int(df_all.player_id.nunique()),
                    "dropped_low_play": n_low_players,
                    "out": int(out.player_id.nunique()),
                    "out_base_players": int(out.player_id_base.nunique())},
        "beatmaps_in": int(df_all.beatmap_id.nunique()),
        "beatmaps_out": int(out.beatmap_id.nunique()),
        "pairs_out": int(len(out)),
        "n_plays": {"min": int(out["n_plays"].min()),
                    "median": float(out["n_plays"].median()),
                    "mean": float(out["n_plays"].mean()),
                    "max": int(out["n_plays"].max())},
        "loss": {"mean": float(out["loss"].mean()), "sd": float(out["loss"].std()),
                 "min": float(out["loss"].min()), "max": float(out["loss"].max())},
        "acc": {"min": float(acc.min()), "median": float(np.median(acc)),
                "mean": float(acc.mean()), "max": float(acc.max())},
        "checks": checks,
        "seconds": round(time.time() - t0, 1),
    }
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[step1/pairbest] rows={len(out):,} players={report['players']['out']:,} "
          f"beatmaps={report['beatmaps_out']:,} train={n_train:,} test={n_test:,}", flush=True)
    print(f"[step1/pairbest] loss {report['loss']['min']:.4f}..{report['loss']['max']:.4f} "
          f"ACC {report['acc']['min']:.6f}..{report['acc']['max']:.6f}", flush=True)
    bad = [k for k, ok in checks.items() if not ok]
    print(f"[step1/pairbest] checks {len(checks) - len(bad)}/{len(checks)} ok"
          + (f" FAILED={bad}" if bad else ""), flush=True)
    if bad:
        raise SystemExit(f"post-condition failure: {bad}")


if __name__ == "__main__":
    main()
