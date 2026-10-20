#!/usr/bin/env python
"""Chart-side sparsity: how many plays does each beatmap actually have, and is that why the
free-latent parameterisation blows up?

Output: docs/figs/chart_playcount_distribution_1k.png  (4 panels) +
        data/processed/chart_playcount_stats_1k.json
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
    "axes.unicode_minus": False,
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

K_EFF = 9.5   # effective independent plays per chart (n/deff, deff=5.4)


def gini(x: np.ndarray) -> float:
    x = np.sort(x.astype(np.float64))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def main() -> None:
    cols = ["player_id", "beatmap_id", "loss"]
    s1 = pd.read_parquet(PROC / "step1_1k.parquet", columns=cols)
    s0 = pd.read_csv(PROC / "step0_1k.csv", usecols=cols)
    meta = pd.read_csv(PROC / "beatmap_meta_1k.csv", usecols=["beatmap_id", "star"])
    k = int(max(s0.beatmap_id.max(), s1.beatmap_id.max())) + 1
    for df in (s0, s1):
        df["pair"] = df.player_id.astype(np.int64) * k + df.beatmap_id.astype(np.int64)
    up = np.unique(s0.pair.to_numpy())
    tps = set(up[np.random.default_rng(20260928).random(len(up)) < 0.30].tolist())
    tr = s1[~s1.pair.isin(tps)]

    n_all = s0.groupby("beatmap_id").size()                 # step0: every recorded play
    n_tr = tr.groupby("beatmap_id").size()                  # step1, train-side pairs
    n_tr = n_tr.reindex(n_all.index).fillna(0).astype(int)  # charts absent from train -> 0
    pl_all = s0.groupby("beatmap_id").player_id.nunique()   # distinct players per chart

    stats = {}
    for nm, n in (("step0_all", n_all), ("train_step1", n_tr)):
        v = n.to_numpy()
        stats[nm] = dict(
            charts=int(len(v)), plays=int(v.sum()), mean=float(v.mean()),
            median=float(np.median(v)),
            p05=float(np.percentile(v, 5)), p25=float(np.percentile(v, 25)),
            p75=float(np.percentile(v, 75)), p95=float(np.percentile(v, 95)),
            max=int(v.max()), gini=gini(v),
            frac_lt2=float((v < 2).mean()), frac_lt5=float((v < 5).mean()),
            frac_lt10=float((v < 10).mean()), frac_lt20=float((v < 20).mean()),
            share_plays_top10pct=float(np.sort(v)[::-1][:int(0.1 * len(v))].sum() / v.sum()),
            share_plays_bot50pct=float(np.sort(v)[:int(0.5 * len(v))].sum() / v.sum()),
        )
    print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)

    v_all = n_all.to_numpy(); v_tr = n_tr.to_numpy()
    star = meta.set_index("beatmap_id").star.reindex(n_all.index).to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))

    # ---- (a) histogram of plays per chart -------------------------------------------
    ax = axes[0, 0]
    bins = np.logspace(0, np.log10(max(v_all.max(), v_tr.max()) + 1), 42)
    ax.hist(v_all, bins=bins, alpha=0.55, label=f"step0 全部 ({len(v_all):,} 张)",
            color="#4c72b0", edgecolor="white", linewidth=0.4)
    ax.hist(np.maximum(v_tr, 0.5), bins=bins, alpha=0.55,
            label=f"训练侧 step1 ({int((v_tr>0).sum()):,} 张有记录)",
            color="#dd8452", edgecolor="white", linewidth=0.4)
    ax.set_xscale("log")
    ax.axvline(np.median(v_tr[v_tr > 0]), color="#c44e52", ls="--", lw=1.6,
               label=f"训练侧中位 = {np.median(v_tr[v_tr>0]):.0f}")
    ax.axvline(4, color="black", ls=":", lw=1.6, label="K=4 的最低要求")
    ax.set_xlabel("每张谱面的 play 数（对数轴）")
    ax.set_ylabel("谱面数")
    ax.set_title("(a) 谱面 play 数分布：一半谱面 ≤ "
                 f"{np.median(v_tr[v_tr>0]):.0f} 条", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right")

    # ---- (b) Lorenz curve -----------------------------------------------------------
    ax = axes[0, 1]
    for nm, v, c in (("step0 全部", v_all, "#4c72b0"), ("训练侧 step1", v_tr, "#dd8452")):
        x = np.sort(v)[::-1]
        cy = np.cumsum(x) / x.sum()
        cx = np.arange(1, len(x) + 1) / len(x)
        ax.plot(cx * 100, cy * 100, lw=2, color=c,
                label=f"{nm}（Gini = {gini(v):.3f}）")
    ax.plot([0, 100], [0, 100], "k--", lw=1, alpha=0.6, label="完全均匀")
    ax.set_xlabel("谱面按 play 数从多到少排序（%）")
    ax.set_ylabel("累计 play 占比（%）")
    ax.set_title("(b) 数据高度集中：前 10% 的谱面占了 "
                 f"{stats['step0_all']['share_plays_top10pct']*100:.0f}% 的 play", fontsize=11)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)

    # ---- (c) plays vs star, with the identifiability threshold ----------------------
    ax = axes[1, 0]
    ok = np.isfinite(star)
    enough = v_tr >= 4
    ax.scatter(star[ok & enough], v_tr[ok & enough], s=5, alpha=0.25, color="#55a868",
               label=f"play ≥ 4，可估 4 维向量（{(ok&enough).sum():,} 张）")
    ax.scatter(star[ok & ~enough], np.maximum(v_tr[ok & ~enough], 0.5), s=5, alpha=0.25,
               color="#c44e52", label=f"play < 4，估不出（{(ok&~enough).sum():,} 张）")
    ax.set_yscale("log")
    ax.axhline(4, color="black", ls=":", lw=1.6)
    ax.set_xlabel("官方 star（2026 快照）")
    ax.set_ylabel("训练侧 play 数（对数轴）")
    ax.set_title("(c) 稀疏谱面遍布整个难度段，不是只集中在简单图", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left")

    # ---- (d) how many charts survive a play threshold -------------------------------
    ax = axes[1, 1]
    ths = [1, 2, 3, 4, 5, 8, 10, 20, 50, 100]
    ch = [(v_tr >= t).sum() for t in ths]
    pl = [(v_tr[v_tr >= t]).sum() for t in ths]
    x = np.arange(len(ths))
    ax.bar(x - 0.2, np.array(ch) / len(v_tr) * 100, 0.4, color="#4c72b0", label="留下的谱面占比")
    ax.bar(x + 0.2, np.array(pl) / v_tr.sum() * 100, 0.4, color="#dd8452", label="留下的 play 占比")
    ax.set_xticks(x); ax.set_xticklabels([str(t) for t in ths])
    ax.set_xlabel("门槛：该谱面至少有这么多条训练 play")
    ax.set_ylabel("百分比（%）")
    ax.set_title("(d) 砍掉稀疏谱面能省参数、几乎不丢数据", fontsize=11)
    ax.legend(fontsize=8.5)
    for i, (c_, p_) in enumerate(zip(ch, pl)):
        ax.text(i + 0.2, p_ + 1.5, f"{p_:.0f}", ha="center", fontsize=7.5, color="#8c564b")
        ax.text(i - 0.2, c_ / len(v_tr) * 100 + 1.5, f"{c_/len(v_tr)*100:.0f}", ha="center",
                fontsize=7.5, color="#3b5b8c")

    fig.suptitle("osu!mania 4K · 1k dump — 谱面侧稀疏度诊断", fontsize=13.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "chart_playcount_distribution_1k.png"
    fig.savefig(out)
    print("wrote", out, flush=True)
    (PROC / "chart_playcount_stats_1k.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False))
    print("wrote", PROC / "chart_playcount_stats_1k.json")


if __name__ == "__main__":
    main()
