#!/usr/bin/env python
"""Step 1 — pair-best cleaning + player-epoch split (new cleaning method).

This is a **separate step1 version** (dcr/steps.py: step1/pairbest).  It does not
change step1/clean; the two produce different populations and must not be mixed.

Starting from step0's play table, the rules are, in execution order:

  (0) player floor: --min-plays-per-player (default 20) drops the *whole* player
      when its step0 record count is below the threshold.  This is the same
      population filter as step1/clean rule 0, with a much lower threshold than
      clean's 200 because the pair-level rules already remove most one-off pairs.
  (1) **player epoch split** (--gap-days, default 365): within a player's history
      sorted by timestamp, a gap of at least --gap-days between two consecutive
      records starts a new player.  Each segment gets its own player_id; a player
      with no such gap keeps the original id.  The split is computed on the FULL
      play history *before* any pair filtering, so it is not fooled by the fact
      that only one play per pair survives later.
  (2) **pair best**: for each (new player_id, beatmap_id) pair, keep only the
      record with the smallest loss (= highest ACC).  Ties: earliest timestamp,
      then smallest playcount_cur.
  (3) **singleton pair drop**: --min-plays-per-pair (default 2) drops every pair
      with fewer recorded plays.  With the default 2 this is exactly "a pair
      played only once is removed": a single attempt carries no evidence that
      the retained score is a stable / best-of-many score.
  (4) **high-ACC pair drop**: --max-best-acc (default 0.998) drops the pair when
      its best ACC is strictly greater than the threshold.  Since
      loss = log(1 - ACC), those rows sit on the unbounded negative tail of the
      label and dominate an MSE in loss space; they are "log(1-ACC)-unfriendly".
  (5) **low-ACC pair drop**: --acc-floor (default 0.90) drops the pair when its
      best ACC is at or below the floor (strictly: keep ACC > floor).  This is
      the retained step1/clean rule 1, applied to the pair-best row: the lower
      plays are already gone, so the question is whether the retained best is
      still above the floor.
  (6) **train/test split** on the surviving pairs, using the same splitter as
      step1/clean (pair-random by default, beatmap-random optional, same seed
      and train fraction).  Here the split runs *after* the population filters:
      the observation unit is a pair, so train and test must both be pair-best
      tables.  This is a deliberate difference from step1/clean, where row-level
      cleaning is applied to the train side only.

  --label-shift s (default 0.0) changes the target *after* reading step0:
      loss_new = log(s + exp(loss_step0)) = log(1 + s - ACC)
  With s=0 this is the original log(1-ACC); with s=0.001 it is log(1.001-ACC),
  which has a finite floor log(0.001) = -6.9078 even at ACC=1.  The ACC-based
  filters (max_best_acc / acc_floor) and all ACC checks are computed with
      ACC = 1 + s - exp(loss_new)
  so they stay in ACC space; the shift only changes the label's numerical scale.
  The pair-best choice is unaffected because the transform is strictly
  increasing in ACC (equivalently in the old loss).

The output is compact: only surviving pairs are written (one row per pair), not
the full step0 table.  Extra columns keep the traceability that step1/clean gets
from its "dropped rows stay in the table" contract:

    player_id       new entity id (segment 0 keeps the original id; later
                    segments get fresh ids > max(original player_id))
    beatmap_id      unchanged
    timestamp       timestamp of the kept (best) play
    playcount_cur   original value of the kept play
    loss            original value of the kept play
    split           "train" / "test" (no "dropped": rows were physically removed)
    player_id_base  original step0 player_id
    player_seg      0-based segment index within player_id_base
    n_plays         number of recorded plays of the pair *before* keeping the best

Examples
    python scripts/step1_pairbest.py --tag 1k
    python scripts/step1_pairbest.py --tag 1k --gap-days 0            # no epoch split
    python scripts/step1_pairbest.py --tag 1k --max-best-acc 1.0      # no high-ACC drop
    python scripts/step1_pairbest.py --tag 1k --acc-floor 0.0         # no low-ACC drop
    python scripts/step1_pairbest.py --tag 1k --min-plays-per-pair 1  # keep singletons
    python scripts/step1_pairbest.py --tag 1k --gap-days 730 --min-plays-per-player 200

Output
    data/processed/step1_{tag}_pairbest.csv / .parquet
    data/processed/step1_{tag}_pairbest.json   (funnel + checks)

The script runs its own post-conditions on every invocation; a failure is a
SystemExit.  See legacy/docs/step1_pairbest.md for the measured funnel and caveats.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from step1_clean import COLS, OUT, SORT_KEY, load, resolve_input, split_rows  # noqa: E402

EXTRA_COLS = ["player_id_base", "player_seg", "n_plays"]
OUT_COLS = COLS + ["split"] + EXTRA_COLS
EPS = 1e-12


# --------------------------------------------------------------------------- io
def assign_epochs(df: pd.DataFrame,
                   gap_days: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Split each player's timeline at gaps >= gap_days.

    Returns (player_id_new int64, player_seg int32, info dict) aligned with df's
    index.  info also carries the sort order and the segment counts so the caller
    does not have to materialise a per-row (base, seg) frame on 10k.

    The sort key is (player_id, timestamp, beatmap_id, playcount_cur); it matches
    step1/clean's SORT_KEY and makes the segment boundaries deterministic even
    when two records share a timestamp.  gap_days <= 0 disables the split.
    """
    n = len(df)
    base = df["player_id"].to_numpy(np.int64)
    bid = df["beatmap_id"].to_numpy(np.int64)
    pc = df["playcount_cur"].to_numpy(np.int64)
    ts = df["timestamp"].to_numpy().astype("datetime64[s]").astype(np.int64)

    # Last key is primary: base, then ts, then bid, then pc.
    order = np.lexsort((pc, bid, ts, base))
    base_s = base[order]
    ts_s = ts[order]

    seg_sorted = np.zeros(n, dtype=np.int32)
    if gap_days > 0 and n > 1:
        gap_seconds = float(gap_days) * 86400.0
        new_seg = np.zeros(n, dtype=bool)
        new_seg[0] = True
        new_seg[1:] = ((base_s[1:] != base_s[:-1])
                       | ((ts_s[1:] - ts_s[:-1]).astype(np.float64) >= gap_seconds))
        cum = np.cumsum(new_seg, dtype=np.int64)
        first = np.r_[0, np.flatnonzero(base_s[1:] != base_s[:-1]) + 1]
        lengths = np.diff(np.r_[first, n])
        seg_sorted = (cum - np.repeat(cum[first], lengths)).astype(np.int32)

    seg = np.empty(n, dtype=np.int32)
    seg[order] = seg_sorted

    # Deterministic id map: (base, seg) -> new id.  Segment 0 keeps the original
    # id; later segments get max(base)+1, +2, ... in (base, seg) order.
    seg_max = int(seg_sorted.max()) if n else 0
    span = np.int64(seg_max + 1)
    keys = base * span + seg.astype(np.int64)
    codes, uniq = pd.factorize(keys, sort=True)
    ubase = (uniq // span).astype(np.int64)
    useg = (uniq % span).astype(np.int64)
    fresh = np.empty(uniq.size, dtype=np.int64)
    fresh[useg == 0] = ubase[useg == 0]
    next_id = int(base.max()) + 1
    for j in np.flatnonzero(useg > 0):
        fresh[j] = next_id
        next_id += 1
    pid_new = fresh[codes]

    # Counts from the compact (base, seg) uniques, not from a 13M-row frame.
    ubase_counts = np.unique(ubase, return_counts=True)[1]
    n_bases = int(ubase_counts.size)
    info = {
        "order": order,
        "n_segments": int(uniq.size),
        "n_bases": n_bases,
        "n_split_players": int((ubase_counts > 1).sum()),
        "n_seg_extra": int(uniq.size - n_bases),
    }
    return pid_new, seg, info


def best_per_pair(df: pd.DataFrame, pid: np.ndarray, min_plays: int,
                  max_best_acc: float, acc_floor: float,
                  label_shift: float = 0.0) -> dict:
    """Return the best row of every pair that passes the pair-level rules.

    Keys are (pid, beatmap_id).  Counting happens on *all* rows of the filtered
    df, before the best row is picked; the "one row per pair" output then has the
    original number of plays in n_plays.
    """
    n = len(df)
    bid = df["beatmap_id"].to_numpy(np.int64)
    loss = df["loss"].to_numpy(np.float32)
    pc = df["playcount_cur"].to_numpy(np.int64)
    ts = df["timestamp"].to_numpy().astype("datetime64[s]").astype(np.int64)
    k = int(bid.max()) + 1 if n else 1
    pair_key = pid * np.int64(k) + bid

    # Pair counts via factorize + bincount: O(n) and no MultiIndex over n rows.
    codes, uniq = pd.factorize(pair_key, sort=False)
    counts = np.bincount(codes, minlength=uniq.size).astype(np.int64)
    row_keep = counts[codes] >= min_plays
    idx = np.flatnonzero(row_keep)
    if idx.size == 0:
        raise SystemExit("no pair survives --min-plays-per-pair")

    # Best row per surviving pair.  Sorting only the surviving rows keeps the
    # lexsort cheap on 10k (the singleton pairs are ~70% of all cells).
    sub_key = pair_key[idx]
    order = np.lexsort((pc[idx], ts[idx], loss[idx], sub_key))
    idx_sorted = idx[order]
    key_sorted = sub_key[order]
    first = np.r_[0, np.flatnonzero(key_sorted[1:] != key_sorted[:-1]) + 1]
    best_idx = idx_sorted[first]
    n_plays = counts[codes[best_idx]]
    best_loss = loss[best_idx].astype(np.float64)

    # loss = log(1+s-ACC); ACC thresholds map to
    #   ACC > max_best_acc  <=> loss <  log(1+s-max_best_acc)
    #   ACC <= acc_floor    <=> loss >= log(1+s-acc_floor)
    upper = 1.0 + label_shift
    thr_high = (math.log(upper - max_best_acc)
                if max_best_acc < upper else -math.inf)
    thr_low = math.log(upper - acc_floor)
    keep_high = best_loss >= thr_high          # drop ACC > max_best_acc
    keep_low = best_loss < thr_low             # drop ACC <= acc_floor
    keep = keep_high & keep_low
    n_high_drop = int((~keep_high).sum())
    n_low_drop = int((~keep_low).sum())
    best_idx = best_idx[keep]
    n_plays = n_plays[keep]
    best_loss = best_loss[keep]

    return {
        "pair_key": pair_key,
        "counts": counts,
        "n_pairs_after_epoch": int(uniq.size),
        "n_pairs_below_min_plays": int((counts < min_plays).sum()),
        "n_rows_below_min_plays": int(counts[counts < min_plays].sum()),
        "n_pairs_ge_min_plays": int((counts >= min_plays).sum()),
        "n_rows_ge_min_plays": int(counts[counts >= min_plays].sum()),
        "n_single_play_pairs": int((counts == 1).sum()),
        "n_high_acc_dropped": n_high_drop,
        "n_low_acc_dropped": n_low_drop,
        "n_after_min_plays": int(best_idx.size + n_high_drop + n_low_drop),
        "best_idx": best_idx,
        "n_plays": n_plays,
        "best_loss": best_loss,
        "threshold_loss": thr_high,
        "acc_floor_loss": thr_low,
    }


# ----------------------------------------------------------------- verification
def verify_best_and_counts(df: pd.DataFrame, pid: np.ndarray, pair_key: np.ndarray,
                           out: pd.DataFrame, best_idx: np.ndarray,
                           sample_n: int = 5000) -> dict:
    """Independent groupby check: every kept row is the pair minimum and n_plays
    is the pair's raw record count.

    Full check up to 2M output rows; for larger tables a deterministic sample of
    pairs is re-derived from the raw rows (memory control on the 10k table).
    """
    n_out = len(out)
    if len(df) <= 2_000_000 and n_out <= 2_000_000:
        sel = np.arange(n_out)
        sampled = False
        mask = np.ones(len(df), dtype=bool)
    else:
        rng = np.random.default_rng(20261020)
        sel = np.sort(rng.choice(n_out, size=min(sample_n, n_out), replace=False))
        sampled = True
        keys = pair_key[best_idx[sel]]
        mask = np.isin(pair_key, keys)

    sub = pd.DataFrame({
        "player_id": pid[mask],
        "beatmap_id": df["beatmap_id"].to_numpy()[mask],
        "loss": df["loss"].to_numpy()[mask],
    })
    g = sub.groupby(["player_id", "beatmap_id"], sort=False)
    cnt = g.size()
    mn = g["loss"].min()

    target = out.iloc[sel]
    idx = pd.MultiIndex.from_arrays([target["player_id"].to_numpy(),
                                     target["beatmap_id"].to_numpy()])
    cnt_v = cnt.reindex(idx).to_numpy()
    mn_v = mn.reindex(idx).to_numpy()
    ok_count = bool(not np.isnan(cnt_v).any()
                    and np.array_equal(cnt_v.astype(np.int64),
                                       target["n_plays"].to_numpy(np.int64)))
    ok_best = bool(not np.isnan(mn_v).any()
                   and np.array_equal(mn_v.astype(np.float32),
                                      target["loss"].to_numpy(np.float32)))
    return {"ok": ok_count and ok_best, "ok_count": ok_count, "ok_best": ok_best,
            "checked_pairs": int(len(sel)), "sampled": sampled}


def verify_gap_split(df: pd.DataFrame, pid: np.ndarray, base: np.ndarray,
                     gap_days: float, order: np.ndarray | None = None) -> dict:
    """Every two consecutive raw records inside one new player are < gap_days
    apart, and every boundary between two segments of the same base player is
    >= gap_days apart.  Both directions together mean the segmentation is exact.
    """
    if gap_days <= 0:
        return {"ok": True, "within_bad": 0, "boundary_bad": 0, "enabled": False}
    ts = df["timestamp"].to_numpy().astype("datetime64[s]").astype(np.int64)
    if order is None:
        pc = df["playcount_cur"].to_numpy(np.int64)
        bid = df["beatmap_id"].to_numpy(np.int64)
        order = np.lexsort((pc, bid, ts, base))
    base_s = base[order]
    pid_s = pid[order]
    ts_s = ts[order]
    gap = (ts_s[1:] - ts_s[:-1]).astype(np.float64)
    same_base = base_s[1:] == base_s[:-1]
    same_pid = pid_s[1:] == pid_s[:-1]
    gap_seconds = float(gap_days) * 86400.0
    within_bad = int((same_pid & (gap >= gap_seconds)).sum())
    boundary_bad = int((same_base & ~same_pid & (gap < gap_seconds)).sum())
    return {"ok": within_bad == 0 and boundary_bad == 0,
            "within_bad": within_bad, "boundary_bad": boundary_bad, "enabled": True}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="1k")
    ap.add_argument("--in", dest="inp", default=None,
                    help="默认 data/processed/step0_{tag}.{parquet,csv}")
    ap.add_argument("--out", default=None,
                    help="默认 data/processed/step1_{tag}_pairbest.csv")
    ap.add_argument("--train-frac", type=float, default=0.70,
                    help="划入 train 的比例（test = 1 - 该值）；默认 0.70")
    ap.add_argument("--split-method", default="pair-random",
                    choices=["pair-random", "beatmap-random"],
                    help="pair-random: 对 (player_id, beatmap_id) 有序对均匀随机；"
                         "beatmap-random: 随机挑出谱面，整张谱面进同一侧")
    ap.add_argument("--split-seed", type=int, default=20260928)
    ap.add_argument("--min-plays-per-pair", type=int, default=2,
                    help="每个 (player, beatmap) 对的记录数 < 此值时剔除；默认 2 "
                         "（即只打过一次的对剔除）")
    ap.add_argument("--max-best-acc", type=float, default=0.998,
                    help="对的最高 ACC 严格大于此值时剔除；默认 0.998；1.0 关闭")
    ap.add_argument("--acc-floor", type=float, default=0.90,
                    help="对的最高 ACC <= 此值时剔除（保留 step1/clean 的规则一）；"
                         "默认 0.90；0.0 关闭")
    ap.add_argument("--label-shift", type=float, default=0.0,
                    help="新标签 log(s + exp(loss_step0)) = log(1+s-ACC)；"
                         "默认 0（原 log(1-ACC)）；s=0.001 即 log(1.001-ACC)")
    ap.add_argument("--gap-days", type=float, default=365.0,
                    help="同一玩家连续两条记录间隔 >= 此天数时切成新玩家；"
                         "默认 365；0 关闭")
    ap.add_argument("--min-plays-per-player", type=int, default=20,
                    help="在 step0 全表里记录数 < 此值的玩家整人剔除（先于一切）；"
                         "默认 20；0 关闭")
    ap.add_argument("--no-parquet", action="store_true")
    ap.add_argument("--no-csv", action="store_true",
                    help="不写 CSV（--out 只作为命名基准，parquet/json 照常）")
    args = ap.parse_args()

    t0 = time.time()
    if not 0.0 < args.train_frac < 1.0:
        raise SystemExit(f"--train-frac 必须在 (0, 1) 内，得到 {args.train_frac}")
    if args.min_plays_per_pair < 1:
        raise SystemExit(f"--min-plays-per-pair 必须 >= 1，得到 {args.min_plays_per_pair}")
    if not 0.0 < args.max_best_acc <= 1.0:
        raise SystemExit(f"--max-best-acc 必须在 (0, 1] 内，得到 {args.max_best_acc}")
    if not 0.0 <= args.acc_floor < 1.0:
        raise SystemExit(f"--acc-floor 必须在 [0, 1) 内，得到 {args.acc_floor}")
    if args.label_shift < 0:
        raise SystemExit(f"--label-shift 不能为负，得到 {args.label_shift}")
    if args.acc_floor >= args.max_best_acc:
        raise SystemExit(f"--acc-floor ({args.acc_floor}) 必须小于 --max-best-acc "
                         f"({args.max_best_acc})，否则没有 ACC 能通过")
    if args.gap_days < 0:
        raise SystemExit(f"--gap-days 不能为负，得到 {args.gap_days}")
    if args.min_plays_per_player < 0:
        raise SystemExit(f"--min-plays-per-player 不能为负，得到 {args.min_plays_per_player}")

    src = resolve_input(args.tag, args.inp)
    out_csv = Path(args.out) if args.out else OUT / f"step1_{args.tag}_pairbest.csv"
    if out_csv.suffix == ".parquet":
        out_csv = out_csv.with_suffix(".csv")
    out_pq = out_csv.with_suffix(".parquet")
    out_json = out_csv.with_suffix(".json")

    print(f"[step1/pairbest] tag={args.tag}  in={src}", flush=True)
    df_all = load(src)
    if args.label_shift > 0:
        old_loss = df_all["loss"].to_numpy(np.float64)
        df_all["loss"] = np.log(np.exp(old_loss) + args.label_shift).astype(np.float32)
        print(f"[step1/pairbest] label: loss -> log({args.label_shift:g} + exp(loss)) "
              f"= log({1.0 + args.label_shift:g} - ACC)", flush=True)
    n_in = len(df_all)
    print(f"[step1/pairbest] input rows={n_in:,} players={df_all.player_id.nunique():,} "
          f"beatmaps={df_all.beatmap_id.nunique():,}", flush=True)

    # Number of raw (player, beatmap) cells, before epoching / pair filtering.
    k0 = int(df_all.beatmap_id.max()) + 1
    raw_key_all = (df_all.player_id.to_numpy(np.int64) * k0
                   + df_all.beatmap_id.to_numpy(np.int64))
    n_raw_pairs = int(pd.factorize(raw_key_all, sort=False)[1].size)
    del raw_key_all

    # ---- optional player floor (same population filter as step1/clean rule 0) ----
    n_low_players = n_low_rows = 0
    df = df_all
    if args.min_plays_per_player > 0:
        plays = df_all.groupby("player_id", sort=False).size()
        low = plays.index[plays.to_numpy() < args.min_plays_per_player]
        n_low_players, n_low_rows = int(len(low)), int(plays.loc[low].sum())
        df = df_all[~df_all.player_id.isin(low)].reset_index(drop=True)
        if len(df) == 0:
            raise SystemExit("--min-plays-per-player dropped every row")
        print(f"[step1/pairbest] player floor: {n_low_players:,} players / "
              f"{n_low_rows:,} rows dropped", flush=True)

    # ---- (1) player epoch split ------------------------------------------------
    base = df["player_id"].to_numpy(np.int64)
    pid, seg, ep = assign_epochs(df, args.gap_days)
    n_segments = ep["n_segments"]
    n_split_players = ep["n_split_players"]
    n_seg_extra = ep["n_seg_extra"]
    print(f"[step1/pairbest] gap split: gap>={args.gap_days:g}d -> {n_segments:,} segments "
          f"(+{n_seg_extra:,}), {n_split_players:,} / {ep['n_bases']:,} players split",
          flush=True)

    # ---- (2) pair best + (3) singleton drop + (4) high-ACC drop ----------------
    bp = best_per_pair(df, pid, args.min_plays_per_pair, args.max_best_acc,
                       args.acc_floor, args.label_shift)
    print(f"[step1/pairbest] pair best: {bp['n_pairs_after_epoch']:,} pairs -> "
          f"{bp['n_pairs_ge_min_plays']:,} with >= {args.min_plays_per_pair} plays "
          f"({bp['n_rows_ge_min_plays']:,} rows); dropped "
          f"{bp['n_pairs_below_min_plays']:,} pairs below the floor "
          f"(singletons: {bp['n_single_play_pairs']:,})", flush=True)
    n_before_acc = bp["n_pairs_ge_min_plays"]
    print(f"[step1/pairbest] ACC window: best ACC <= {args.acc_floor:g} -> dropped "
          f"{bp['n_low_acc_dropped']:,}; best ACC > {args.max_best_acc:g} -> dropped "
          f"{bp['n_high_acc_dropped']:,} of {n_before_acc:,} pairs; "
          f"kept {n_before_acc - bp['n_low_acc_dropped'] - bp['n_high_acc_dropped']:,}",
          flush=True)

    best_idx = np.argsort(bp["best_idx"], kind="stable")
    keep_idx = bp["best_idx"][best_idx]
    n_plays = bp["n_plays"][best_idx]

    # ---- build compact output --------------------------------------------------
    out = df.iloc[keep_idx].reset_index(drop=True)
    out["player_id"] = pid[keep_idx].astype(np.int64)
    out["player_id_base"] = base[keep_idx].astype(np.int64)
    out["player_seg"] = seg[keep_idx].astype(np.int32)
    out["n_plays"] = n_plays.astype(np.int64)

    # ---- (5) train/test split on the surviving pairs ---------------------------
    is_test, split_info = split_rows(out, args.split_method, args.train_frac, args.split_seed)
    out["split"] = np.where(is_test, "test", "train")
    out = out[OUT_COLS]
    n_tr, n_te = int((~is_test).sum()), int(is_test.sum())
    print(f"[step1/pairbest] split: method={args.split_method} "
          f"train {n_tr:,} / test {n_te:,} ({n_te / max(len(out), 1):.2%} test) "
          f"seed={args.split_seed}", flush=True)

    # ---- verification ----------------------------------------------------------
    acc = 1.0 + args.label_shift - np.exp(out["loss"].to_numpy(np.float64))
    acc_ok = bool((out["loss"].to_numpy(np.float64) >= bp["threshold_loss"] - EPS).all())
    pair_dup = int(out.duplicated(["player_id", "beatmap_id"]).sum())
    v = verify_best_and_counts(df, pid, bp["pair_key"], out, keep_idx)
    g = verify_gap_split(df, pid, base, args.gap_days, order=ep["order"])

    # player-id map checks
    pairs = (out[["player_id_base", "player_seg", "player_id"]]
             .drop_duplicates().sort_values(["player_id_base", "player_seg"]))
    map_unique = bool(pairs.groupby(["player_id_base", "player_seg"])["player_id"]
                      .nunique().max() == 1
                      and pairs["player_id"].nunique() == len(pairs))
    seg0_ok = bool((pairs.loc[pairs.player_seg == 0, "player_id"]
                    == pairs.loc[pairs.player_seg == 0, "player_id_base"]).all())
    fresh_ok = bool((pairs.loc[pairs.player_seg > 0, "player_id"]
                     > int(base.max())).all())

    # split units disjoint
    if args.split_method == "pair-random":
        kk = int(out.beatmap_id.max()) + 1
        keys_all = (out.player_id.to_numpy(np.int64) * kk
                    + out.beatmap_id.to_numpy(np.int64))
        units_tr = np.unique(keys_all[~is_test])
        units_te = np.unique(keys_all[is_test])
        unit_name = "有序对"
    else:
        units_tr = np.unique(out.beatmap_id.to_numpy()[~is_test])
        units_te = np.unique(out.beatmap_id.to_numpy()[is_test])
        unit_name = "谱面"
    shared = np.intersect1d(units_tr, units_te, assume_unique=True)

    # output rows are an unmodified subset of the input rows (player_id is the
    # only remapped column; every measurement column must be bit-identical)
    val_cols = ["beatmap_id", "timestamp", "playcount_cur", "loss"]
    subset_ok = bool(df.iloc[keep_idx][val_cols].reset_index(drop=True)
                     .equals(out[val_cols].reset_index(drop=True)))

    checks: dict[str, bool] = {
        "rows_nonempty": len(out) > 0,
        "one_row_per_player_beatmap": pair_dup == 0,
        "min_plays_per_pair_enforced": bool((out["n_plays"] >= args.min_plays_per_pair).all()),
        "pair_play_counts_match_input": v["ok_count"],
        "kept_score_is_pair_min_loss": v["ok_best"],
        "best_acc_at_or_below_max": acc_ok,
        "min_acc_strictly_above_floor": bool(
            (out["loss"].to_numpy(np.float64) < bp["acc_floor_loss"]).all()),
        "player_ids_unique_per_base_segment": map_unique,
        "segment_zero_keeps_base_id": seg0_ok,
        "later_segments_get_fresh_ids": fresh_ok,
        "gap_split_is_exact": g["ok"],
        "split_labels_valid": bool(set(out["split"].unique()) <= {"train", "test"}
                                and n_tr + n_te == len(out)),
        "split_units_disjoint": shared.size == 0,
        "actual_train_frac_within_1pct": abs(split_info["actual_train_frac"]
                                             - args.train_frac) <= 0.01,
        "loss_negative": bool((out["loss"] < 0).all()),
        "output_rows_are_unmodified_input_rows": subset_ok,
    }

    # ---- write (atomic) --------------------------------------------------------
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_csv:
        tmp = out_csv.with_name(out_csv.name + ".part")
        out.to_csv(tmp, index=False, date_format="%Y-%m-%d %H:%M:%S")
        tmp.replace(out_csv)
    if not args.no_parquet:
        tmp2 = out_pq.with_name(out_pq.name + ".part")
        out.to_parquet(tmp2, index=False)
        tmp2.replace(out_pq)

    # ---- report ----------------------------------------------------------------
    split_rows_counts = {k: int((out["split"] == k).sum()) for k in ("train", "test")}
    report = {
        "tag": args.tag,
        "input": str(src),
        "output_csv": None if args.no_csv else str(out_csv),
        "output_parquet": None if args.no_parquet else str(out_pq),
        "method": "pairbest",
        "label": {
            "step0": "log(1-ACC)",
            "shift": args.label_shift,
            "definition": "log(shift + exp(step0_loss)) = log(1+shift-ACC)",
            "acc_from_label": "1 + shift - exp(loss)",
        },
        "split": {
            **split_info,
            "seed": args.split_seed,
            "train_frac": args.train_frac,
            "test_frac": 1.0 - args.train_frac,
            "col": "split",
            "labels": ["train", "test"],
            "rows": {**split_rows_counts, "dropped": 0},
            "unit_name": unit_name,
        },
        "rules": {
            "order": "player_floor -> gap_split -> pair_best -> min_plays_per_pair "
                     "-> max_best_acc -> acc_floor -> train_test_split",
            "scope": "population filter on all rows, then split",
            "gap_days": args.gap_days,
            "min_plays_per_pair": args.min_plays_per_pair,
            "max_best_acc": args.max_best_acc,
            "max_best_loss_threshold": bp["threshold_loss"],
            "acc_floor": args.acc_floor,
            "acc_floor_loss_threshold": bp["acc_floor_loss"],
            "label_shift": args.label_shift,
            "min_plays_per_player": args.min_plays_per_player,
            "best": "min loss (= max ACC) within (player_epoch, beatmap)",
            "best_tie_break": "earliest timestamp, then smallest playcount_cur",
            "player_id_mapping": "segment 0 keeps the step0 id; later segments get ids "
                                 "> max(step0 player_id)",
            "playcount_cur": "original value of the kept best row (not recomputed)",
            "n_plays": "record count of the pair before keeping the best",
            "sort_key": SORT_KEY,
        },
        "funnel": {
            "step0_rows": n_in,
            "dropped_low_play_rows": n_low_rows,
            "rows_after_player_floor": int(len(df)),
            "test_rows": n_te,
            "train_candidates": int(len(out)),
            "dropped_acc": bp["n_low_acc_dropped"],
            "dropped_earliest": 0,
            "train_rows": n_tr,
            "dropped_rows": 0,
            "output_rows": int(len(out)),
            "retained_frac": round(len(out) / n_in, 6),
            "step0_pairs": n_raw_pairs,
            "player_segments": n_segments,
            "players_split_by_gap": n_split_players,
            "extra_segments": n_seg_extra,
            "pairs_after_epoch_split": bp["n_pairs_after_epoch"],
            "pairs_below_min_plays": bp["n_pairs_below_min_plays"],
            "rows_below_min_plays": bp["n_rows_below_min_plays"],
            "single_play_pairs": bp["n_single_play_pairs"],
            "pairs_ge_min_plays": bp["n_pairs_ge_min_plays"],
            "rows_ge_min_plays": bp["n_rows_ge_min_plays"],
            "pairs_before_acc_filters": bp["n_pairs_ge_min_plays"],
            "dropped_low_acc_pairs": bp["n_low_acc_dropped"],
            "dropped_high_acc_pairs": bp["n_high_acc_dropped"],
            "kept_pairs": int(len(out)),
            "pair_retention": round(len(out) / max(bp["n_pairs_after_epoch"], 1), 6),
        },
        "players": {
            "in": int(df_all.player_id.nunique()),
            "dropped_low_play": n_low_players,
            "segments_before_pair_filter": n_segments,
            "split_by_gap": n_split_players,
            "out": int(out.player_id.nunique()),
            "out_base_players": int(out.player_id_base.nunique()),
        },
        "beatmaps_in": int(df_all.beatmap_id.nunique()),
        "beatmaps_out": int(out.beatmap_id.nunique()),
        "pairs_out": int(len(out)),
        "n_plays": {
            "min": int(out["n_plays"].min()),
            "median": float(out["n_plays"].median()),
            "mean": float(out["n_plays"].mean()),
            "p90": float(out["n_plays"].quantile(0.90)),
            "max": int(out["n_plays"].max()),
            "total": int(out["n_plays"].sum()),
        },
        "loss": {
            "mean": float(out["loss"].mean()), "sd": float(out["loss"].std()),
            "min": float(out["loss"].min()), "max": float(out["loss"].max()),
        },
        "acc": {
            "min": float(acc.min()), "median": float(np.median(acc)),
            "mean": float(acc.mean()), "max": float(acc.max()),
        },
        "timestamp": {"min": str(out["timestamp"].min()), "max": str(out["timestamp"].max())},
        "verification_sample": v,
        "checks": checks,
        "seconds": round(time.time() - t0, 1),
    }
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[step1/pairbest] output rows={len(out):,} players={report['players']['out']:,} "
          f"beatmaps={report['beatmaps_out']:,} "
          f"n_plays median={report['n_plays']['median']:.0f} max={report['n_plays']['max']:,}",
          flush=True)
    print(f"[step1/pairbest] loss mean {report['loss']['mean']:.4f} sd {report['loss']['sd']:.4f} "
          f"min {report['loss']['min']:.4f} max {report['loss']['max']:.4f} | "
          f"ACC min {report['acc']['min']:.6f} median {report['acc']['median']:.6f} "
          f"max {report['acc']['max']:.6f}", flush=True)
    bad = [k for k, ok in checks.items() if not ok]
    print(f"[step1/pairbest] checks: {len(checks) - len(bad)}/{len(checks)} ok"
          + (f"  FAILED={bad}" if bad else ""), flush=True)
    wrote = ([f"{out_csv} ({out_csv.stat().st_size / 1e6:.1f} MB)"] if not args.no_csv else []) \
        + ([f"{out_pq} ({out_pq.stat().st_size / 1e6:.1f} MB)"] if not args.no_parquet else [])
    print("[step1/pairbest] wrote " + " + ".join(wrote) + f"  {report['seconds']}s", flush=True)
    if bad:
        raise SystemExit(f"post-condition failure: {bad}")


if __name__ == "__main__":
    main()
