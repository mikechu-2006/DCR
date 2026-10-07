"""Step 1 — play-level cleaning (new, two rules only).

从 step0 的 play 表出发，只做两条**行级**删除，产出 `step1_{tag}.csv`：

    (1) ACC 门槛：删除 ACC <= 0.90 的游玩。
        loss = log(1 - ACC)  =>  ACC <= 0.90  <=>  loss >= log(0.10) = -2.302585
        实现上就是删掉 `loss >= log(1 - acc_floor)` 的行。
    (2) 热身截断：对每个 player_id，按时间排序，删除**最早**的
        floor(drop_frac * n) 条，n = 该玩家在 (1) 之后剩下的行数。

顺序（已确认）：**先筛 ACC，再砍时间**。n 因此是 ACC 筛后的行数。

刻意不做的事（这是与旧 step1 / `build_v2.py` 的关键区别）：
    * 不做 cell 聚合（不产 loss_med / n_scores / rate）；
    * 不做 (player, beatmap) 的 attempts 或 n_scores 门槛；
    * 不做收敛判定 / 分层 / 权重；
    * **不重算 playcount_cur** —— 保留 step0 原值，所以删行后会出现空洞
      （如 5,6,7…）。它仍表示「这是该 (player, beatmap) cell 的第 k 条记录」，
      删掉过哪些行因此可追溯。

输入  data/processed/step0_{tag}.csv 或 .parquet   (player_id, beatmap_id, timestamp, playcount_cur, loss)
输出  data/processed/step1_{tag}.csv     列序与 step0 完全相同
      data/processed/step1_{tag}.parquet 除非 --no-parquet
      data/processed/step1_{tag}.json    漏斗 + 校验结果

用例
    python scripts/step1_clean.py --tag 1k
    python scripts/step1_clean.py --tag 10k
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

# step0 的排序键是 (user_id, beatmap_id, ts, sid)。sid 不在 play 表里，所以同一秒
# 并列时用 (beatmap_id, playcount_cur) 作确定性 tie-break —— 这是唯一的近似点，
# 见 docs/step1_cleaning.md §5。
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="1k")
    ap.add_argument("--in", dest="inp", default=None,
                    help="默认 data/processed/step0_{tag}.{parquet,csv}")
    ap.add_argument("--out", default=None, help="默认 data/processed/step1_{tag}.csv")
    ap.add_argument("--acc-floor", type=float, default=0.90, help="删除 ACC <= 此值的游玩")
    ap.add_argument("--drop-frac", type=float, default=0.30, help="每个玩家删除最早的这一比例")
    ap.add_argument("--no-parquet", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    src = resolve_input(args.tag, args.inp)
    out_csv = Path(args.out) if args.out else OUT / f"step1_{args.tag}.csv"
    out_pq = out_csv.with_suffix(".parquet")
    out_json = out_csv.with_suffix(".json")

    print(f"[step1] tag={args.tag}  in={src}", flush=True)
    df = load(src)                      # RangeIndex 0..n-1 —— 后续靠它做精确子集校验
    n_in = len(df)
    log_floor = math.log(1.0 - args.acc_floor)
    print(f"[step1] input rows={n_in:,} players={df.player_id.nunique():,} "
          f"beatmaps={df.beatmap_id.nunique():,}", flush=True)
    print(f"[step1] rules: drop ACC <= {args.acc_floor:g}  (loss >= {log_floor:.6f})"
          f"  then drop earliest floor({args.drop_frac:g} * n) per player", flush=True)

    # ---- rule 1: ACC floor ------------------------------------------------
    keep_acc = df["loss"].to_numpy(dtype=np.float64) < log_floor
    n_acc_drop = int((~keep_acc).sum())
    d1 = df[keep_acc]                   # 索引仍是原行号
    print(f"[step1] rule 1: dropped {n_acc_drop:,}  -> {len(d1):,} rows", flush=True)

    # ---- rule 2: per-player earliest floor(frac * n) ----------------------
    d1 = d1.sort_values(SORT_KEY, kind="mergesort")
    counts = d1.groupby("player_id", sort=False).size()          # 每玩家 ACC 筛后的 n
    cut = np.floor(args.drop_frac * counts.to_numpy()).astype(np.int64)   # 每玩家删几条
    cut_row = np.repeat(cut, counts.to_numpy())                  # 摊到行上
    rank = d1.groupby("player_id", sort=False).cumcount().to_numpy()
    keep_time = rank >= cut_row
    n_time_drop = int((~keep_time).sum())
    d2 = d1[keep_time]
    print(f"[step1] rule 2: dropped {n_time_drop:,}  -> {len(d2):,} rows", flush=True)

    # ---- verification -----------------------------------------------------
    acc_out = 1.0 - np.exp(d2["loss"].to_numpy(dtype=np.float64))
    kept_pp = d2.groupby("player_id", sort=False).size()
    expect_pp = pd.Series(counts.to_numpy() - cut, index=counts.index)

    # 精确子集校验：输出每一行都必须与 step0 的同一行逐值相同（靠原行号 take 回来）
    unmodified = bool(
        df.take(d2.index.to_numpy()).reset_index(drop=True).equals(
            d2.reset_index(drop=True)[COLS]))

    checks: dict[str, bool] = {
        "rows_add_up": int(len(d2) + n_acc_drop + n_time_drop) == n_in,
        "min_acc_strictly_above_floor": bool(acc_out.min() > args.acc_floor),
        "loss_negative": bool((d2["loss"] < 0).all()),
        "per_player_kept_equals_n_minus_floor": bool(
            kept_pp.sort_index().equals(expect_pp.sort_index().astype(kept_pp.dtype))),
        "no_dup_player_beatmap_playcount": int(
            d2.duplicated(["player_id", "beatmap_id", "playcount_cur"]).sum()) == 0,
        "playcount_cur_ge_1": bool((d2["playcount_cur"] >= 1).all()),
        "playcount_cur_monotonic_in_cell": bool(
            d2.groupby(["player_id", "beatmap_id"], sort=False)["playcount_cur"]
              .apply(lambda s: bool(s.is_monotonic_increasing)).all()),
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
    cell_size = d2.groupby(["player_id", "beatmap_id"], sort=False).size()
    cell_pcmax = d2.groupby(["player_id", "beatmap_id"], sort=False)["playcount_cur"].max()
    cells_with_gaps = int((cell_pcmax > cell_size).sum())

    # ---- write (atomic) ---------------------------------------------------
    d2 = d2.reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_name(out_csv.name + ".part")
    d2.to_csv(tmp, index=False, date_format="%Y-%m-%d %H:%M:%S")
    tmp.replace(out_csv)
    if not args.no_parquet:
        tmp2 = out_pq.with_name(out_pq.name + ".part")
        d2.to_parquet(tmp2, index=False)
        tmp2.replace(out_pq)

    # ---- report -----------------------------------------------------------
    cut_q = np.percentile(cut, [50, 75, 90, 99])
    report = {
        "tag": args.tag,
        "input": str(src),
        "output_csv": str(out_csv),
        "output_parquet": None if args.no_parquet else str(out_pq),
        "rules": {
            "order": "acc_floor_then_time_cut",
            "acc_floor": args.acc_floor,
            "log_loss_threshold": log_floor,
            "drop_frac": args.drop_frac,
            "rounding": "floor",
            "playcount_cur": "preserved_from_step0",
            "sort_key": SORT_KEY,
        },
        "funnel": {
            "step0_rows": n_in,
            "dropped_acc": n_acc_drop,
            "dropped_earliest": n_time_drop,
            "step1_rows": int(len(d2)),
            "retained_frac": round(len(d2) / n_in, 6),
        },
        "players": {
            "in": int(df.player_id.nunique()),
            "out": int(d2.player_id.nunique()),
            "cut_zero_players": int((cut == 0).sum()),
            "cut_min": int(cut.min()),
            "cut_median": float(cut_q[0]), "cut_p75": float(cut_q[1]),
            "cut_p90": float(cut_q[2]), "cut_p99": float(cut_q[3]),
            "cut_max": int(cut.max()),
        },
        "beatmaps_out": int(d2.beatmap_id.nunique()),
        "cells_out": int(cell_size.size),
        "cells_with_playcount_gaps": cells_with_gaps,
        "loss": {
            "mean": float(d2["loss"].mean()), "sd": float(d2["loss"].std()),
            "min": float(d2["loss"].min()), "max": float(d2["loss"].max()),
        },
        "acc": {"min": float(acc_out.min()), "mean": float(acc_out.mean())},
        "playcount_cur": {"mean": float(d2["playcount_cur"].mean()),
                          "max": int(d2["playcount_cur"].max())},
        "timestamp": {"min": str(d2["timestamp"].min()), "max": str(d2["timestamp"].max())},
        "checks": checks,
        "seconds": round(time.time() - t0, 1),
    }
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[step1] rows={len(d2):,} players={d2.player_id.nunique():,} "
          f"beatmaps={d2.beatmap_id.nunique():,} cells={cell_size.size:,}")
    print(f"[step1] loss mean {report['loss']['mean']:.4f} sd {report['loss']['sd']:.4f} "
          f"min {report['loss']['min']:.4f} max {report['loss']['max']:.4f} | "
          f"min ACC {report['acc']['min']:.6f}")
    print(f"[step1] pc mean {report['playcount_cur']['mean']:.3f} "
          f"max {report['playcount_cur']['max']} | cells with pc gaps {cells_with_gaps:,}")
    bad = [k for k, v in checks.items() if not v]
    print(f"[step1] checks: {len(checks) - len(bad)}/{len(checks)} ok"
          + (f"  FAILED={bad}" if bad else ""))
    print(f"[step1] wrote {out_csv} ({out_csv.stat().st_size / 1e6:.1f} MB)"
          + ("" if args.no_parquet else f" + {out_pq.name}") + f"  {report['seconds']}s",
          flush=True)
    if bad:
        raise SystemExit(f"post-condition failure: {bad}")


if __name__ == "__main__":
    main()
