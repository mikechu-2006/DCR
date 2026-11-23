#!/usr/bin/env python
"""Volume / rebuild report for the merged random-snapshot dataset.

Answers, before any weighting decision is made:
  * how much data the merge actually produces;
  * how many pairs were rebuilt (singleton in every snapshot, n_plays>=2 merged);
  * how many pairs are new / lost relative to the per-snapshot pair tables;
  * label quality (ACC window, censoring at 0.998);
  * chart support and a data-side difficulty axis (per-chart bias intercept),
    NOT osu! star.

Usage:
  python scripts/randommerge_report.py \
      --step0 data/processed/step0_randommerge.parquet \
      --pairbest data/processed/step1_randommerge_pairbest.parquet \
      --snapshot data/processed/step0_rand_20260713.parquet \
      --snapshot data/processed/step0_rand_20260801.parquet \
      --snapshot data/processed/step0_rand_20260901.parquet \
      --snapshot data/processed/step0_rand_20261105.parquet \
      --out docs/random_merge_report.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LOW_ACC = 0.90
HIGH_ACC = 0.998


def pair_table(step0: pd.DataFrame) -> pd.DataFrame:
    g = (step0.groupby(["player_id", "beatmap_id"], observed=True)
         .agg(best_loss=("loss", "min"), n_plays=("loss", "size")))
    g["best_acc"] = 1 - np.exp(g["best_loss"])
    return g


def survives(pairs: pd.DataFrame) -> pd.Series:
    return (pairs["n_plays"] >= 2) & (pairs["best_acc"] > LOW_ACC) & (pairs["best_acc"] <= HIGH_ACC)


def fit_bias(train: pd.DataFrame, n_iter: int = 25) -> tuple[float, pd.Series, pd.Series]:
    """loss ~ b0 + bu(player) + bm(chart), alternating means."""
    y = train["loss"].to_numpy(np.float64)
    u = train["player_id"].to_numpy()
    m = train["beatmap_id"].to_numpy()
    uu, ui = np.unique(u, return_inverse=True)
    mm, mi = np.unique(m, return_inverse=True)
    bu = np.zeros(len(uu))
    bm = np.zeros(len(mm))
    b0 = float(y.mean())
    for _ in range(n_iter):
        bu = np.bincount(ui, weights=y - b0 - bm[mi], minlength=len(uu)) / np.bincount(ui, minlength=len(uu))
        bm = np.bincount(mi, weights=y - b0 - bu[ui], minlength=len(mm)) / np.bincount(mi, minlength=len(mm))
        b0 = float((y - bu[ui] - bm[mi]).mean())
    return b0, pd.Series(bu, index=uu), pd.Series(bm, index=mm)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step0", required=True, type=Path)
    ap.add_argument("--pairbest", required=True, type=Path)
    ap.add_argument("--snapshot", action="append", required=True, type=Path,
                    metavar="PATH", help="per-snapshot step0 parquet; repeatable")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    s0 = pd.read_parquet(args.step0)
    pb = pd.read_parquet(args.pairbest)
    snaps = {p.stem.replace("step0_", ""): pd.read_parquet(p) for p in args.snapshot}

    merged_pairs = pair_table(s0)
    merged_surv = survives(merged_pairs)
    out = []
    out.append("# Random 快照合并：数据量报告")
    out.append("")
    out.append("> 自动生成。难度轴用每图截距 bm（数据侧），**不用 star**。")
    out.append("")

    # ---- scale
    per_player_pairs = merged_pairs.groupby("player_id").size()
    out.append("## 1. 规模")
    out.append("")
    out.append("| 项 | 值 |")
    out.append("|---|---:|")
    out.append("| play 行 | {:,} |".format(len(s0)))
    out.append("| 玩家（合并后） | {:,} |".format(s0.player_id.nunique()))
    out.append("| 谱面 | {:,} |".format(s0.beatmap_id.nunique()))
    out.append("| 对（player, beatmap） | {:,} |".format(len(merged_pairs)))
    out.append("| 过规则的对（未扣玩家门槛/空窗） | {:,} |".format(int(merged_surv.sum())))
    out.append("| step1 实际输出（扣门槛+空窗后） | {:,} 对 / {:,} 玩家 |".format(
        len(pb), pb.player_id.nunique()))
    out.append("| 每玩家对数 p10/p50/p90 | {}/{}/{} |".format(
        *[int(x) for x in np.percentile(per_player_pairs, [10, 50, 90])]))
    out.append("| 时间跨度 | {} → {} |".format(s0.timestamp.min(), s0.timestamp.max()))
    out.append("")

    # ---- per snapshot
    out.append("## 2. 单快照 vs 合并")
    out.append("")
    out.append("| 快照 | play | 玩家 | 对 | 过规则对 |")
    out.append("|---|---:|---:|---:|---:|")
    for name, df in snaps.items():
        pp = pair_table(df)
        out.append("| {} | {:,} | {:,} | {:,} | {:,} |".format(
            name, len(df), df.player_id.nunique(), len(pp), int(survives(pp).sum())))
    out.append("")

    # ---- rebuilt
    n_snap = pd.DataFrame(index=merged_pairs.index)
    for name, df in snaps.items():
        n_snap[name] = pair_table(df)["n_plays"]
    appears = n_snap.notna()
    max_n = n_snap.max(axis=1)
    rebuilt = (max_n == 1) & (merged_pairs["n_plays"] >= 2)
    singleton_everywhere = (max_n == 1)
    out.append("## 3. 重建了多少对")
    out.append("")
    out.append("| 指标 | 值 |")
    out.append("|---|---:|")
    out.append("| 每个快照里都只出现 1 次的对 | {:,} |".format(int(singleton_everywhere.sum())))
    out.append("| **其中合并后 n_plays>=2（复活）** | **{:,}** |".format(int(rebuilt.sum())))
    out.append("| 合并后 n_plays>=2 的对 | {:,} |".format(int((merged_pairs.n_plays >= 2).sum())))
    out.append("| 合并后不再过规则的对 | {:,} |".format(int(((max_n >= 2) & ~merged_surv).sum())))
    out.append("")

    # ---- label quality
    acc = merged_pairs["best_acc"]
    out.append("## 4. 标签")
    out.append("")
    out.append("| 项 | 值 |")
    out.append("|---|---:|")
    out.append("| best ACC 中位数 | {:.4f} |".format(float(acc.median())))
    out.append("| ACC>0.998（删失） | {:,} ({:.2%}) |".format(int((acc > HIGH_ACC).sum()), float((acc > HIGH_ACC).mean())))
    out.append("| ACC<=0.90 | {:,} ({:.2%}) |".format(int((acc <= LOW_ACC).sum()), float((acc <= LOW_ACC).mean())))
    out.append("")

    # ---- chart support
    players_per_chart = merged_pairs.groupby("beatmap_id").size()
    out.append("## 5. 每图支持度")
    out.append("")
    out.append("| 每图玩家数 | 谱面数 | 占比 |")
    out.append("|---|---:|---:|")
    bins = [(1, 1), (2, 2), (3, 4), (5, 9), (10, 19), (20, 49), (50, 10 ** 9)]
    for lo, hi in bins:
        n = int(((players_per_chart >= lo) & (players_per_chart <= hi)).sum())
        label = str(lo) if lo == hi else ("{}+".format(lo) if hi > 10 ** 8 else "{}-{}".format(lo, hi))
        out.append("| {} | {:,} | {:.2%} |".format(label, n, n / len(players_per_chart)))
    out.append("")

    # ---- data-side difficulty axis
    train = pb[pb["split"] == "train"] if "split" in pb.columns else pb
    test = pb[pb["split"] == "test"] if "split" in pb.columns else pb.iloc[0:0]
    b0, bu, bm = fit_bias(train)
    tr = train.assign(bm=train.beatmap_id.map(bm))
    te = test.assign(bm=test.beatmap_id.map(bm)).dropna(subset=["bm"])
    tr = tr.dropna(subset=["bm"])
    q = np.quantile(tr["bm"], np.linspace(0, 1, 11))
    tr["bucket"] = pd.cut(tr["bm"], q, duplicates="drop")
    te["bucket"] = pd.cut(te["bm"], q, duplicates="drop")
    out.append("## 6. 数据侧难度轴（每图截距 bm）")
    out.append("")
    out.append("bias 模型 loss ~ b0 + bu + bm 在 train 上拟合；bm 越低越简单。")
    out.append("")
    out.append("| bm 十分位 | train 对 | 谱面 | 每图中位玩家 | test 对 | test RMSE |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for bucket, g in tr.groupby("bucket", observed=True):
        gg = te[te["bucket"] == bucket]
        rmse = float(np.sqrt(np.mean((gg["loss"] - (b0 + gg.player_id.map(bu) + gg.bm)) ** 2))) if len(gg) else float("nan")
        out.append("| {} | {:,} | {:,} | {:.0f} | {:,} | {:.4f} |".format(
            bucket, len(g), g.beatmap_id.nunique(),
            g.groupby("beatmap_id").size().median(), len(gg), rmse))
    out.append("")
    out.append("> bm 只为覆盖率/残差描述，不作预测指标；正式难度序仍以 rank-NLL 为准。")
    out.append("")

    text = "\n".join(out)
    if args.out:
        args.out.write_text(text)
        print("wrote", args.out)
    print(text)

    summary = {
        "plays": int(len(s0)), "players": int(s0.player_id.nunique()),
        "beatmaps": int(s0.beatmap_id.nunique()), "pairs": int(len(merged_pairs)),
        "pairs_survive": int(merged_surv.sum()),
        "rebuilt_pairs": int(rebuilt.sum()),
        "singleton_everywhere": int(singleton_everywhere.sum()),
        "b0": b0, "bm_q": [float(x) for x in q],
    }
    print("\nSUMMARY " + json.dumps(summary))


if __name__ == "__main__":
    main()
