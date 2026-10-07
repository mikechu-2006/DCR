#!/usr/bin/env python
"""Chart-side sparsity in terms of DISTINCT PLAYERS per chart (not plays per chart).

Same source as everything else: step0_1k (all recorded plays) and the step2 split, so the
train-side numbers are the ones a model actually sees.

Outputs
  data/processed/chart_players_distribution_1k.csv   -- the table
  docs/figs/chart_players_distribution_1k.png        -- 2 panels
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROC = Path("data/processed")
FIGS = Path("docs/figs")
plt.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
    "axes.unicode_minus": False, "figure.dpi": 130, "savefig.dpi": 130,
    "axes.grid": True, "grid.alpha": 0.25,
})


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, np.float64))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def dist_table(v: np.ndarray, name: str) -> pd.DataFrame:
    """distribution of charts over distinct-player counts."""
    edges = [0, 1, 2, 3, 4, 5, 8, 10, 20, 50, 100, 200, 500, 10**9]
    labs = ["0", "1", "2", "3", "4", "5-7", "8-9", "10-19", "20-49", "50-99",
            "100-199", "200-499", "500+"]
    tot = v.sum()
    rows = []
    for i, lab in enumerate(labs):
        m = (v >= edges[i]) & (v < edges[i + 1])
        rows.append(dict(playercount_bucket=lab, charts=int(m.sum()),
                         chart_pct=100 * m.mean(),
                         players_sum=int(v[m].sum()),
                         players_pct=100 * v[m].sum() / tot,
                         cum_chart_pct=100 * (v < edges[i + 1]).mean(),
                         cum_players_pct=100 * v[v < edges[i + 1]].sum() / tot))
    df = pd.DataFrame(rows)
    print(f"\n===== {name}：每张谱面的【不同游玩人数】分布 =====")
    print(f"  谱面数 {len(v):,} | 人数 mean {v.mean():.2f} median {np.median(v):.0f} "
          f"p25 {np.percentile(v,25):.0f} p75 {np.percentile(v,75):.0f} "
          f"p95 {np.percentile(v,95):.0f} max {v.max():,} | Gini {gini(v):.4f}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:8.2f}"))
    return df


def main() -> None:
    cols = ["player_id", "beatmap_id", "loss"]
    s0 = pd.read_csv(PROC / "step0_1k.csv", usecols=cols)
    s1 = pd.read_parquet(PROC / "step1_1k.parquet", columns=cols)
    k = int(max(s0.beatmap_id.max(), s1.beatmap_id.max())) + 1
    for df in (s0, s1):
        df["pair"] = df.player_id.astype(np.int64) * k + df.beatmap_id.astype(np.int64)
    up = np.unique(s0.pair.to_numpy())
    tps = set(up[np.random.default_rng(20260928).random(len(up)) < 0.30].tolist())
    te = s0[s0.pair.isin(tps)]

    all_ids = np.unique(s0.beatmap_id.to_numpy())
    p_all = s0.groupby("beatmap_id").player_id.nunique().reindex(all_ids).fillna(0).to_numpy()
    p_te = te.groupby("beatmap_id").player_id.nunique().reindex(all_ids).fillna(0).to_numpy()
    n_play_all = s0.groupby("beatmap_id").size().reindex(all_ids).fillna(0).to_numpy()

    t_all = dist_table(p_all, "step0 全部")
    t_te = dist_table(p_te, "测试侧（被抽中的 30% ordered pairs）")

    # plays per player on the same chart (the other axis of "how much per chart")
    pp = s0.groupby(["beatmap_id", "player_id"]).size()
    print("\n===== 同一 (谱面, 玩家) 的 play 数分布 =====")
    print(f"  cells {len(pp):,} | mean {pp.mean():.2f} median {pp.median():.0f} "
          f"p90 {pp.quantile(.9):.0f} max {pp.max():,}")
    print("  只有 1 条 play 的 cell 占比: %.1f%%" % ((pp == 1).mean() * 100))

    out = PROC / "chart_players_distribution_1k.csv"
    pd.concat([t_all.assign(side="step0_all"), t_te.assign(side="test_30pct")]) \
        .to_csv(out, index=False)
    print("\nwrote", out)

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    ax = axes[0]
    bins = np.logspace(0, np.log10(max(p_all.max(), 2) + 1), 40)
    ax.hist(np.maximum(p_all, 1), bins=bins, alpha=0.6, color="#4c72b0",
            edgecolor="white", lw=0.4, label=f"step0 全部（{len(p_all):,} 张）")
    ax.hist(np.maximum(p_te, 1), bins=bins, alpha=0.6, color="#dd8452",
            edgecolor="white", lw=0.4, label="测试侧 30% pairs")
    ax.set_xscale("log")
    ax.axvline(np.median(p_all), color="#c44e52", ls="--", lw=1.6,
               label=f"中位 = {np.median(p_all):.0f} 人")
    ax.axvline(4, color="black", ls=":", lw=1.6, label="K=4 的最低要求")
    ax.set_xlabel("每张谱面的不同游玩人数（对数轴）")
    ax.set_ylabel("谱面数")
    ax.set_title(f"(a) 一半谱面的游玩人数 ≤ {np.median(p_all):.0f}", fontsize=11)
    ax.legend(fontsize=8.5)

    ax = axes[1]
    x = np.arange(len(t_all))
    ax.bar(x - 0.2, t_all.chart_pct, 0.4, color="#4c72b0", label="谱面占比")
    ax.bar(x + 0.2, t_all.players_pct, 0.4, color="#dd8452", label="人数占比（玩家-谱面次）")
    ax.set_xticks(x); ax.set_xticklabels(t_all.playercount_bucket, rotation=45, ha="right")
    ax.set_xlabel("每张谱面的不同游玩人数")
    ax.set_ylabel("百分比（%）")
    ax.set_title("(b) 人数高度集中：少数谱面吃掉了大部分玩家-谱面次", fontsize=11)
    ax.legend(fontsize=8.5)
    for i in x:
        ax.text(i - 0.2, t_all.chart_pct[i] + 0.8, f"{t_all.chart_pct[i]:.0f}",
                ha="center", fontsize=7, color="#3b5b8c")
        ax.text(i + 0.2, t_all.players_pct[i] + 0.8, f"{t_all.players_pct[i]:.0f}",
                ha="center", fontsize=7, color="#8c564b")
    fig.suptitle("osu!mania 4K · 1k dump — 每张谱面的游玩人数分布", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    FIGS.mkdir(parents=True, exist_ok=True)
    outfig = FIGS / "chart_players_distribution_1k.png"
    fig.savefig(outfig)
    print("wrote", outfig)

    (PROC / "chart_players_stats_1k.json").write_text(json.dumps(dict(
        step0_all=dict(charts=int(len(p_all)), mean=float(p_all.mean()),
                       median=float(np.median(p_all)), p25=float(np.percentile(p_all, 25)),
                       p75=float(np.percentile(p_all, 75)), p95=float(np.percentile(p_all, 95)),
                       max=int(p_all.max()), gini=gini(p_all),
                       frac_lt2=float((p_all < 2).mean()), frac_lt5=float((p_all < 5).mean()),
                       frac_lt10=float((p_all < 10).mean()),
                       share_players_top10pct=float(np.sort(p_all)[::-1][:int(.1*len(p_all))].sum()/p_all.sum())),
        test_30pct=dict(charts=int((p_te > 0).sum()), mean=float(p_te.mean()),
                        median=float(np.median(p_te)),
                        frac_lt5=float((p_te[p_te > 0] < 5).mean())),
        plays_per_cell=dict(cells=int(len(pp)), mean=float(pp.mean()),
                            median=float(pp.median()),
                            frac_single=float((pp == 1).mean()))),
        ensure_ascii=False, indent=2))
    print("wrote", PROC / "chart_players_stats_1k.json")


if __name__ == "__main__":
    main()
