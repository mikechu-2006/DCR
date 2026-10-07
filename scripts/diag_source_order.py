"""Diagnostic: what exactly is in osu_scores_mania_high, and does play order carry a selection bias?

回答三个问题：
 1. 每 (user, beatmap) 在 mania_high 里有几条？每 user 有几条？
 2. 1k 的 plays 里 legacy / modern 各占多少，cell 内两种来源怎么分布？
 3. 分开看两种来源，loss vs 时间序百分位 的曲线是否一样？
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("data/processed")
INTERIM10 = Path("data/interim_10k")
ALLOW = {"CL", "MR", "DT", "NC", "HT", "SD", "PF", "NF"}


def q1():
    print("=" * 70)
    print("Q1  10k: mania_high 表结构（每 user / 每 (user,beatmap) 几条）")
    h = pd.read_parquet(INTERIM10 / "mania_high_4k_dated.parquet",
                        columns=["user_id", "beatmap_id", "enabled_mods"])
    print("rows:", len(h), "users:", h.user_id.nunique(), "beatmaps:", h.beatmap_id.nunique())
    per_user = h.groupby("user_id").size()
    print("rows per user: min %d  p25 %d  median %d  p75 %d  max %d" % (
        per_user.min(), per_user.quantile(.25), per_user.median(),
        per_user.quantile(.75), per_user.max()))
    print("  -> p90 %.0f  p99 %.0f" % (per_user.quantile(.90), per_user.quantile(.99)))
    per_pair = h.groupby(["user_id", "beatmap_id"]).size()
    print("rows per (user,beatmap): mean %.2f median %.0f p90 %.0f p99 %.0f max %d" % (
        per_pair.mean(), per_pair.median(), per_pair.quantile(.90), per_pair.quantile(.99),
        per_pair.max()))
    print("  fraction of pairs with >1 row: %.4f" % (per_pair > 1).mean())
    print("  fraction of pairs with >3 rows: %.4f" % (per_pair > 3).mean())
    print("  fraction of pairs with >10 rows: %.5f" % (per_pair > 10).mean())
    del h


def q2():
    print("=" * 70)
    print("Q2  1k: plays_4k.parquet 的来源构成（按 build_v2 的口径复现过滤）")
    p = pd.read_parquet(OUT / "plays_4k.parquet",
                        columns=["user_id", "beatmap_id", "chart_id", "rate", "source",
                                 "date", "mods_raw", "mods_norm", "loss_v2"])
    print("total rows:", len(p), "by source:", p.source.value_counts().to_dict())
    col = "mods_raw"
    toks = p[col].fillna("").map(lambda s: set(str(s).split("+")) - {""})
    keep = toks.map(lambda t: t <= ALLOW)
    modern = (p.source == "modern").to_numpy()
    keep &= (~modern) | toks.map(lambda t: "CL" in t).to_numpy()
    d = p[keep].copy()
    print("after v2 filter:", len(d), "by source:", d.source.value_counts().to_dict())
    print("dropped:", int((~keep).sum()), "of which modern-no-CL:",
          int((~keep & modern).sum()))
    print("legacy rows w/ keymod tokens:",
          int(toks[~keep & (~modern)].map(lambda t: bool(t & {"4K", "5K", "6K", "7K", "8K"})).sum()))
    del p, toks
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["user_id", "chart_id", "date"], kind="stable").reset_index(drop=True)
    key = d.user_id.astype(np.int64) * 100_000_000 + d.chart_id.astype(np.int64)
    codes, _ = pd.factorize(key)
    g = codes.max() + 1
    n = np.bincount(codes, minlength=g)
    pos = np.arange(len(codes)) - np.repeat(np.cumsum(n) - n, n)
    leg = (d.source == "legacy").to_numpy()
    nleg = np.bincount(codes, weights=leg.astype(float), minlength=g)
    print("cells:", g, "| cells with >=1 legacy: %.4f" % (nleg > 0).mean(),
          "| mean legacy per cell %.3f" % nleg.mean())
    sub = n >= 3
    print("cells n>=3:", int(sub.sum()))
    # legacy 在 cell 内的百分位
    pleg = np.where(leg, (pos + 0.5) / n[codes], np.nan)
    valid = np.isfinite(pleg)
    print("legacy plays' percentile inside cell: median %.3f  p25 %.3f  p75 %.3f" % (
        np.nanmedian(pleg), np.nanpercentile(pleg, 25), np.nanpercentile(pleg, 75)))
    print("modern plays' percentile inside cell: median %.3f" %
          np.median(((pos + 0.5) / n[codes])[~leg]))
    # 每 cell 的平均来源百分位
    L = d.loss_v2.to_numpy(np.float64)
    print()
    print("--- 分来源的 loss vs 百分位（cell 内中心化）")
    mean_cell = np.bincount(codes, weights=L, minlength=g) / n
    Lc = L - mean_cell[codes]
    edges = np.linspace(0, 1, 11)
    idx = np.digitize((pos + 0.5) / n[codes], edges[1:-1])
    for name, m in (("legacy", leg), ("modern", ~leg)):
        row = []
        for i in range(10):
            mm = m & (idx == i)
            row.append(np.nan if mm.sum() == 0 else Lc[mm].mean())
        print(f"  {name:7s} n={int(m.sum()):>8d}  " + " ".join(
            "  nan " if np.isnan(v) else f"{v:+.2f}" for v in row))
    # 只保留纯 modern 的 cell，看曲线
    pure_modern = nleg[codes] == 0
    print()
    print("pure-modern cells: %.4f of plays; pure-legacy cells: %.4f" % (
        pure_modern.mean(), (nleg[codes] == n[codes]).mean()))
    for name, m in (("pure-modern", pure_modern), ("mixed", (nleg[codes] > 0) & (nleg[codes] < n[codes])),
                    ("pure-legacy", nleg[codes] == n[codes])):
        mm = m & (n >= 3)
        if mm.sum() < 1000:
            print(f"  {name}: too few ({int(mm.sum())})"); continue
        row = []
        for i in range(10):
            sel = mm & (idx == i)
            row.append(np.nan if sel.sum() == 0 else Lc[sel].mean())
        print(f"  {name:12s} plays={int(mm.sum()):>8d}  " + " ".join(
            "  nan " if np.isnan(v) else f"{v:+.2f}" for v in row))


if __name__ == "__main__":
    q1()
    q2()
