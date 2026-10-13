"""Step 1 — train/test 划分 + play 级清洗（新版）。

从 step0 的 play 表出发，两步：

  (0) **划分**：按 --split-method 把样本单位分成 train / test 两组，train 占 --train-frac。
        pair-random    （默认）对所有 (player_id, beatmap_id) 有序对均匀随机；同一对的行
                       要么全在 train、要么全在 test。
        beatmap-random 随机挑出 (1 - train_frac) 比例的 beatmap_id，这些谱面的**所有**行
                       都进 test（谱面级外推，更严格）。
        种子 --split-seed 默认 20260928，与 step2 的 --split-seed 一致：因此 pair-random
        的划分与旧流程（step2 自己划分）逐行相同。
  (1) **清洗**：只作用于划入 train 的行，两条行级删除
        规则 1 ACC 门槛：删除 ACC <= --acc-floor（即 loss >= log(1 - acc_floor)）的游玩。
        规则 2 热身截断：每个 player_id 按时间排序，删除最早的 floor(drop_frac * n) 条，
                        n = 该玩家 train 行经过规则 1 之后剩下的行数。

输出的 split 列（字符串）：

    "train"    划入 train，且通过了清洗
    "test"     划入 test（**原样保留**：清洗是 train 侧的处理，用它去筛 test 就是
               「以处理结果为条件」，见 docs/archive/cleaning_plan.md 3.2）
    "dropped"  划入 train，但被规则 1 或规则 2 删除

两种方法都不重叠：test 侧的行绝不会出现在 train 里。

刻意不做的事（与旧 step1 的区别）：
    * 不做 cell 聚合（不产 loss_med / n_scores / rate）；
    * 不做 (player, beatmap) 的 attempts 或 n_scores 门槛；
    * **不重算 playcount_cur** —— 保留 step0 原值，所以删行后会出现空洞
      （如 5,6,7…），仍表示「这是该 cell 的第 k 条记录」，删过哪些行可追溯；
    * **不再把删掉的行从表里去掉**：输出 = 输入的全部行（顺序与数值都不变），
      只多一列 split，train/test/dropped 三类的行数之和 = step0 行数。

输入  data/processed/step0_{tag}.csv 或 .parquet   (player_id, beatmap_id, timestamp, playcount_cur, loss)
输出  data/processed/step1_{tag}.csv / .parquet   列序 = step0 的 5 列 + split
      data/processed/step1_{tag}.json             split 定义 + 漏斗 + 校验结果

用例
    python scripts/step1_clean.py --tag 1k
    python scripts/step1_clean.py --tag 1k --train-frac 0.7 --split-method pair-random
    python scripts/step1_clean.py --tag 1k --split-method beatmap-random --train-frac 0.5
    python scripts/step1_clean.py --tag 1k --in data/processed/step0_1k.parquet \
        --out experiments_x/artifacts/step1_1k.csv --no-csv
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/processed")
COLS = ["player_id", "beatmap_id", "timestamp", "playcount_cur", "loss"]
SPLIT_COL = "split"
LABELS = ("train", "test", "dropped")

# step0 的排序键是 (user_id, beatmap_id, ts, sid)。sid 不在 play 表里，所以同一秒
# 并列时用 (beatmap_id, playcount_cur) 作确定性 tie-break —— 这是唯一的近似点，
# 见 docs/step1_cleaning.md 5。
SORT_KEY = ["player_id", "timestamp", "beatmap_id", "playcount_cur"]


def resolve_input(tag: str, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise SystemExit(f"--in not found: {p}")
        return p
    for cand in (OUT / f"step0_{tag}.parquet", OUT / f"step0_{tag}.csv"):
        if cand.exists():
            return cand
    raise SystemExit(f"no step0 input found for tag={tag} in {OUT}")


def load(path: Path) -> pd.DataFrame:
    """读 step0 的 play 表，统一 timestamp 为 datetime64、loss 为 float32。"""
    if path.suffix == ".parquet":
        df = pd.read_parquet(path, columns=COLS)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        df = pd.read_csv(
            path,
            dtype={"player_id": "int32", "beatmap_id": "int32", "playcount_cur": "int32"},
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
    # step0 的 --float-dtype 默认 float32；统一回 float32 让两个 tag 的精度一致。
    # 1k 的 CSV 里存的正是 float32 的最短 repr，round-trip 无损。
    df["loss"] = df["loss"].astype("float32")
    return df[COLS]


def pair_keys(player, beatmap, k: int) -> np.ndarray:
    """有序对 key，int64；k = beatmap_id.max() + 1 使编码单射（与 step2 相同）。"""
    return player.astype(np.int64) * np.int64(k) + beatmap.astype(np.int64)


def split_rows(df: pd.DataFrame, method: str, train_frac: float, seed: int) -> tuple[np.ndarray, dict]:
    """返回 (is_test 布尔数组, 说明信息)。单位：pair-random = 有序对，beatmap-random = 谱面。"""
    test_frac = 1.0 - train_frac
    if method == "pair-random":
        k = int(df["beatmap_id"].max()) + 1
        keys = pair_keys(df["player_id"].to_numpy(), df["beatmap_id"].to_numpy(), k)
        units = np.unique(keys)
        is_test_unit = np.random.default_rng(seed).random(units.size) < test_frac
        is_test = is_test_unit[np.searchsorted(units, keys)]      # 每个 key 必在 units 里
        info = {"method": method, "unit": "pair", "n_units": int(units.size),
                "n_train_units": int((~is_test_unit).sum()),
                "n_test_units": int(is_test_unit.sum())}
    elif method == "beatmap-random":
        bids = np.unique(df["beatmap_id"].to_numpy())
        is_test_b = np.random.default_rng(seed).random(bids.size) < test_frac
        is_test = is_test_b[np.searchsorted(bids, df["beatmap_id"].to_numpy())]
        info = {"method": method, "unit": "beatmap", "n_units": int(bids.size),
                "n_train_units": int((~is_test_b).sum()),
                "n_test_units": int(is_test_b.sum())}
    else:
        raise SystemExit(f"unknown --split-method {method!r}")
    info["actual_train_frac"] = info["n_train_units"] / max(info["n_units"], 1)
    info["actual_test_frac"] = info["n_test_units"] / max(info["n_units"], 1)
    return is_test, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="1k")
    ap.add_argument("--in", dest="inp", default=None,
                    help="默认 data/processed/step0_{tag}.{parquet,csv}")
    ap.add_argument("--out", default=None, help="默认 data/processed/step1_{tag}.csv")
    ap.add_argument("--train-frac", type=float, default=0.70,
                    help="划入 train 的比例（test = 1 - 该值）；默认 0.70")
    ap.add_argument("--split-method", default="pair-random",
                    choices=["pair-random", "beatmap-random"],
                    help="pair-random: 对所有 (player_id, beatmap_id) 有序对均匀随机（默认）；"
                         "beatmap-random: 随机挑出 (1-train_frac) 比例的 beatmap_id，"
                         "这些谱面的所有行都进 test")
    ap.add_argument("--split-seed", type=int, default=20260928,
                    help="划分随机种子；默认与 step2 的 --split-seed 相同，"
                         "因此 pair-random 与旧流程逐行一致")
    ap.add_argument("--acc-floor", type=float, default=0.90, help="删除 ACC <= 此值的游玩（仅 train 行）")
    ap.add_argument("--drop-frac", type=float, default=0.30, help="每个玩家删除最早的这一比例（仅 train 行）")
    ap.add_argument("--no-parquet", action="store_true")
    ap.add_argument("--no-csv", action="store_true",
                    help="不写 CSV（--out 只作为命名基准，parquet/json 照常）")
    args = ap.parse_args()

    t0 = time.time()
    if not 0.0 < args.train_frac < 1.0:
        raise SystemExit(f"--train-frac 必须在 (0, 1) 内，得到 {args.train_frac}")
    src = resolve_input(args.tag, args.inp)
    out_csv = Path(args.out) if args.out else OUT / f"step1_{args.tag}.csv"
    if out_csv.suffix == ".parquet":       # --out 是命名基准，避免 CSV/parquet 撞同一路径
        out_csv = out_csv.with_suffix(".csv")
    out_pq = out_csv.with_suffix(".parquet")
    out_json = out_csv.with_suffix(".json")

    print(f"[step1] tag={args.tag}  in={src}", flush=True)
    df = load(src)                      # RangeIndex 0..n-1 —— 后续靠它做精确子集校验
    n_in = len(df)
    log_floor = math.log(1.0 - args.acc_floor)
    print(f"[step1] input rows={n_in:,} players={df.player_id.nunique():,} "
          f"beatmaps={df.beatmap_id.nunique():,}", flush=True)
    print(f"[step1] split: method={args.split_method} train_frac={args.train_frac:g} "
          f"seed={args.split_seed}", flush=True)

    # ---- 规则 0: train/test 划分 ------------------------------------------
    is_test, split_info = split_rows(df, args.split_method, args.train_frac, args.split_seed)
    n_te = int(is_test.sum())
    print(f"[step1] split: {split_info['n_units']:,} {split_info['unit']} -> "
          f"train {split_info['n_train_units']:,} ({split_info['actual_train_frac']:.4%}) / "
          f"test {split_info['n_test_units']:,} ({split_info['actual_test_frac']:.4%}) | "
          f"rows: test {n_te:,}", flush=True)

    # 先全部标成 dropped（= 划进 train，等着看清洗结果），再把 test 覆盖掉。
    split = np.full(n_in, "dropped", dtype=object)
    split[is_test] = "test"

    # ---- 规则 1: ACC floor（仅 train 行）----------------------------------
    d_train = df[~is_test]               # 索引仍是原行号
    keep_acc = d_train["loss"].to_numpy(dtype=np.float64) < log_floor
    n_acc_drop = int((~keep_acc).sum())
    d1 = d_train[keep_acc]
    print(f"[step1] rule 1 (train only): {len(d_train):,} -> dropped {n_acc_drop:,} "
          f"-> {len(d1):,} rows", flush=True)

    # ---- 规则 2: per-player earliest floor(frac * n)（仅 train 行）--------
    d1 = d1.sort_values(SORT_KEY, kind="mergesort")
    counts = d1.groupby("player_id", sort=False).size()          # 每玩家 ACC 筛后的 n
    cut = np.floor(args.drop_frac * counts.to_numpy()).astype(np.int64)   # 每玩家删几条
    cut_row = np.repeat(cut, counts.to_numpy())                  # 摊到行上
    rank = d1.groupby("player_id", sort=False).cumcount().to_numpy()
    keep_time = rank >= cut_row
    n_time_drop = int((~keep_time).sum())
    d2 = d1[keep_time]
    split[d2.index.to_numpy()] = "train"
    print(f"[step1] rule 2 (train only): {len(d1):,} -> dropped {n_time_drop:,} "
          f"-> train {len(d2):,} rows", flush=True)

    out = df.copy()
    out[SPLIT_COL] = split
    tr_mask, te_mask = split == "train", split == "test"
    n_tr, n_dr = int(tr_mask.sum()), int(n_in - int(te_mask.sum()) - int(tr_mask.sum()))
    print(f"[step1] split column: train {n_tr:,} / test {n_te:,} / dropped {n_dr:,}", flush=True)

    # ---- verification -----------------------------------------------------
    acc_out = 1.0 - np.exp(out["loss"].to_numpy(dtype=np.float64)[tr_mask])
    if len(d2):
        kept_pp = d2.groupby("player_id", sort=False).size()
        expect_pp = pd.Series(counts.to_numpy() - cut, index=counts.index)
        per_player_ok = bool(kept_pp.sort_index().equals(expect_pp.sort_index().astype(kept_pp.dtype)))
    else:
        per_player_ok = True

    # 精确子集校验：除了新增的 split 列，输出与输入逐值相同（顺序也不变）
    unmodified = bool(out[COLS].reset_index(drop=True).equals(df[COLS].reset_index(drop=True)))

    # playcount_cur 在 cell 内单调递增：用 groupby.diff() 向量化（旧写法对每 cell 调
    # python lambda，10k 的 700 万个 cell 上非常慢）。step0 已按 (player, beatmap, ts, sid)
    # 排序，所以差分是正确判据。
    pc = out.groupby(["player_id", "beatmap_id"], sort=False)["playcount_cur"]
    pc_monotonic = bool((pc.diff().fillna(1) > 0).all())

    # 划分不重叠：train 侧（含 dropped）与 test 侧的样本单位必须没有交集
    if args.split_method == "pair-random":
        kk = int(df["beatmap_id"].max()) + 1
        keys_all = pair_keys(df["player_id"].to_numpy(), df["beatmap_id"].to_numpy(), kk)
        train_units = np.unique(keys_all[~te_mask])
        test_units = np.unique(keys_all[te_mask])
        unit_name = "有序对"
    else:
        bids = df["beatmap_id"].to_numpy()
        train_units = np.unique(bids[~te_mask])
        test_units = np.unique(bids[te_mask])
        unit_name = "谱面"
    shared = np.intersect1d(train_units, test_units, assume_unique=True)

    checks: dict[str, bool] = {
        "rows_preserved": len(out) == n_in,
        "split_labels_valid": bool(set(np.unique(split).tolist()) <= set(LABELS))
                              and n_tr + n_te + n_dr == n_in,
        "split_units_disjoint": shared.size == 0,
        "actual_train_frac_within_1pct": abs(
            (train_units.size / max(train_units.size + test_units.size, 1)) - args.train_frac) <= 0.01,
        "min_acc_strictly_above_floor_on_train": bool(len(acc_out) == 0 or acc_out.min() > args.acc_floor),
        "loss_negative": bool((out["loss"] < 0).all()),
        "per_player_train_kept_equals_n_minus_floor": per_player_ok,
        "no_dup_player_beatmap_playcount": int(
            out.duplicated(["player_id", "beatmap_id", "playcount_cur"]).sum()) == 0,
        "playcount_cur_ge_1": bool((out["playcount_cur"] >= 1).all()),
        "playcount_cur_monotonic_in_cell": pc_monotonic,
        "output_rows_are_unmodified_input_rows": unmodified,
    }

    # 时间截断不能有交叉：每个玩家被删的最后一条 <= 保留的第一条（同秒并列时取等号）
    dropped = d1[~keep_time]
    if len(dropped) and len(d2):
        j = (dropped.groupby("player_id")["timestamp"].max().to_frame("dmax")
             .join(d2.groupby("player_id")["timestamp"].min().to_frame("kmin"), how="inner"))
        checks["no_time_overlap"] = bool((j["dmax"] <= j["kmin"]).all())
    else:
        checks["no_time_overlap"] = True

    # 信息性指标：删行后 playcount_cur 出现空洞的 cell 数（预期 > 0，证明没被重编号）
    cell_size = out.groupby(["player_id", "beatmap_id"], sort=False).size()
    cell_pcmax = out.groupby(["player_id", "beatmap_id"], sort=False)["playcount_cur"].max()
    cells_with_gaps = int((cell_pcmax > cell_size).sum())

    # ---- write (atomic) ---------------------------------------------------
    out = out.reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_csv:
        tmp = out_csv.with_name(out_csv.name + ".part")
        out.to_csv(tmp, index=False, date_format="%Y-%m-%d %H:%M:%S")
        tmp.replace(out_csv)
    if not args.no_parquet:
        tmp2 = out_pq.with_name(out_pq.name + ".part")
        out.to_parquet(tmp2, index=False)
        tmp2.replace(out_pq)

    # ---- report -----------------------------------------------------------
    cut_q = np.percentile(cut, [50, 75, 90, 99]) if len(cut) else np.zeros(4)
    train_loss = out.loc[tr_mask, "loss"]
    report = {
        "tag": args.tag,
        "input": str(src),
        "output_csv": None if args.no_csv else str(out_csv),
        "output_parquet": None if args.no_parquet else str(out_pq),
        "split": {
            **split_info,
            "seed": args.split_seed,
            "train_frac": args.train_frac,
            "test_frac": 1.0 - args.train_frac,
            "col": SPLIT_COL,
            "labels": list(LABELS),
            "rows": {"train": n_tr, "test": n_te, "dropped": n_dr},
        },
        "rules": {
            "order": "split_then_acc_floor_then_time_cut",
            "cleaning_side": "train only",
            "acc_floor": args.acc_floor,
            "log_loss_threshold": log_floor,
            "drop_frac": args.drop_frac,
            "rounding": "floor",
            "playcount_cur": "preserved_from_step0",
            "sort_key": SORT_KEY,
        },
        "funnel": {
            "step0_rows": n_in,
            "test_rows": n_te,
            "train_candidates": int(len(d_train)),
            "dropped_acc": n_acc_drop,
            "dropped_earliest": n_time_drop,
            "train_rows": n_tr,
            "dropped_rows": n_dr,
            "output_rows": int(len(out)),
            "retained_frac": round(n_tr / n_in, 6),
        },
        "players": {
            "in": int(df.player_id.nunique()),
            "train_out": int(out.loc[tr_mask, "player_id"].nunique()),
            "test_out": int(out.loc[te_mask, "player_id"].nunique()),
            "cut_zero_players": int((cut == 0).sum()),
            "cut_min": int(cut.min()) if len(cut) else 0,
            "cut_median": float(cut_q[0]), "cut_p75": float(cut_q[1]),
            "cut_p90": float(cut_q[2]), "cut_p99": float(cut_q[3]),
            "cut_max": int(cut.max()) if len(cut) else 0,
        },
        "beatmaps_out": int(out.beatmap_id.nunique()),
        "cells_out": int(cell_size.size),
        "cells_with_playcount_gaps": cells_with_gaps,
        "loss": {
            "mean": float(train_loss.mean()), "sd": float(train_loss.std()),
            "min": float(train_loss.min()), "max": float(train_loss.max()),
        },
        "acc": {"min": float(acc_out.min()) if len(acc_out) else None,
                "mean": float(acc_out.mean()) if len(acc_out) else None},
        "playcount_cur": {"mean": float(out["playcount_cur"].mean()),
                          "max": int(out["playcount_cur"].max())},
        "timestamp": {"min": str(out["timestamp"].min()), "max": str(out["timestamp"].max())},
        "checks": checks,
        "seconds": round(time.time() - t0, 1),
    }
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[step1] train rows={n_tr:,} players={report['players']['train_out']:,} "
          f"beatmaps={int(out.loc[tr_mask, 'beatmap_id'].nunique()):,} cells={cell_size.size:,}")
    print(f"[step1] train loss mean {report['loss']['mean']:.4f} sd {report['loss']['sd']:.4f} "
          f"min {report['loss']['min']:.4f} max {report['loss']['max']:.4f}"
          + (f" | min ACC {report['acc']['min']:.6f}" if report["acc"]["min"] is not None else ""))
    print(f"[step1] pc mean {report['playcount_cur']['mean']:.3f} "
          f"max {report['playcount_cur']['max']} | cells with pc gaps {cells_with_gaps:,}")
    bad = [k for k, v in checks.items() if not v]
    print(f"[step1] checks: {len(checks) - len(bad)}/{len(checks)} ok"
          + (f"  FAILED={bad}" if bad else ""))
    wrote = ([f"{out_csv} ({out_csv.stat().st_size / 1e6:.1f} MB)"] if not args.no_csv else []) \
        + ([f"{out_pq} ({out_pq.stat().st_size / 1e6:.1f} MB)"] if not args.no_parquet else [])
    print(f"[step1] wrote " + " + ".join(wrote) + f"  {report['seconds']}s", flush=True)
    if bad:
        raise SystemExit(f"post-condition failure: {bad}")


if __name__ == "__main__":
    main()
