"""Within-cell play-order curve analysis  (v2, sign-corrected).

问题：对同一个 (玩家, 曲目) cell，历次游玩的成绩随「累计游玩次数」怎么变？
累计游玩次数用 cell 内的时间序百分位估计（浮点）：

    pos  = 该成绩在 cell 内按时间排序后的序号 (0-based)
    n    = cell 内被记录的成绩条数
    p    = (pos + 0.5) / n            in (0,1)   -- 百分位
    cum  = p * attempts               -- 估计「到这把为止一共玩了几次」（含失败/重开）

**符号约定（重要）**：loss = log(1-ACC) <= 0，且 **ACC 越高 -> loss 越小（越负）**。
所以：
    slope_p = d(loss)/d(percentile) < 0  =>  越打越好
    loss_centered 递减                  =>  越打越好
    argmin(loss) = 个人最好成绩；argmax(loss) = 个人最差成绩

产出：
    data/processed/order_curve_{tag}{suffix}.json
    data/processed/order_curve_{tag}{suffix}_curves.csv
    data/processed/order_curve_{tag}{suffix}_slope_by_{n,rate,star,density}.csv
    data/processed/order_curve_{tag}{suffix}_position.csv    固定 cell 集合的位置曲线
    data/processed/order_curve_{tag}{suffix}_cells.parquet
    docs/figs/order_{tag}{suffix}_*.png

用例：
    python scripts/play_order_curve.py --tag 1k --tag 10k
    python scripts/play_order_curve.py --tag 1k --unit beatmap
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/processed")
FIG = Path("docs/figs")
KEY_MUL = 100_000_000
RATE_NAME = {0: "NM", 1: "DT", 2: "HT"}
NB_BUCKETS = [(3, 3), (4, 5), (6, 10), (11, 20), (21, 50), (51, 10**9)]
NB_LABELS = ["3", "4-5", "6-10", "11-20", "21-50", "51+"]
STAR_BUCKETS = [(0, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 99)]
STAR_LABELS = ["<3", "3-4", "4-5", "5-6", "6-7", "7+"]
SPAN_BUCKETS = [(-1, 0.0), (0.0, 7 / 365), (7 / 365, 30 / 365), (30 / 365, 1.0), (1.0, 99)]
SPAN_LABELS = ["same-day", "<7d", "7-30d", "30d-1y", ">1y"]
DENS_BUCKETS = [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]
POS_KS = [5, 10, 20, 50]

EDGES_P = np.linspace(0, 1, 21)
LABS_P = [f"{EDGES_P[i]:.2f}-{EDGES_P[i+1]:.2f}" for i in range(20)]
EDGES_C = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 6, 8, 11, 15, 21, 29, 40, 55, 76,
                    105, 145, 200, 275, 380, 1e9])
LABS_C = ["<1.5", "1.5-2.5", "2.5-3.5", "3.5-4.5", "4.5-6", "6-8", "8-11", "11-15",
          "15-21", "21-29", "29-40", "40-55", "55-76", "76-105", "105-145",
          "145-200", "200-275", "275-380", "380+"]


# ------------------------------------------------------------------ vectorised helpers
def slope_within(codes, g, x, y, min_n=3):
    cnt = np.bincount(codes, minlength=g).astype(np.float64)
    sx = np.bincount(codes, weights=x, minlength=g)
    sy = np.bincount(codes, weights=y, minlength=g)
    sxx = np.bincount(codes, weights=x * x, minlength=g)
    sxy = np.bincount(codes, weights=x * y, minlength=g)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = (sxy - sx * sy / cnt) / (sxx - sx * sx / cnt)
    slope[(cnt < min_n) | ~np.isfinite(slope)] = np.nan
    return slope


def curve_by_bin(x, L, A, C, idx, labels):
    g = len(labels)
    cnt = np.bincount(idx, minlength=g)[:g].astype(np.float64)
    def mean(w):
        return np.bincount(idx, weights=w, minlength=g)[:g] / np.maximum(cnt, 1)
    loss_m, acc_m, c_m = mean(L), mean(A), mean(C)
    sd = np.sqrt(np.maximum(mean(L * L) - loss_m ** 2, 0.0))
    rows = []
    for i, lab in enumerate(labels):
        if cnt[i] == 0:
            rows.append(dict(bin=lab, n_plays=0)); continue
        rows.append(dict(bin=lab, n_plays=int(cnt[i]), loss=float(loss_m[i]), acc=float(acc_m[i]),
                         loss_centered=float(c_m[i]), sd_loss=float(sd[i])))
    return pd.DataFrame(rows)


def curve_by_cellmean(x, C, cell, g, idx, labels):
    comb = idx.astype(np.int64) * g + cell
    m = len(labels) * g
    cnt = np.bincount(comb, minlength=m).astype(np.float64)
    s = np.bincount(comb, weights=C, minlength=m)
    cm = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    rows = []
    for i, lab in enumerate(labels):
        seg = cm[i * g:(i + 1) * g]
        seg = seg[np.isfinite(seg)]
        rows.append(dict(bin=lab, n_cells=int(seg.size),
                         loss_centered_cellmean=float(seg.mean()) if seg.size else np.nan,
                         loss_centered_cellmean_se=float(seg.std() / np.sqrt(seg.size))
                         if seg.size > 1 else np.nan))
    return pd.DataFrame(rows)


def head_tail_curve(Lc, A, pos, n, k=20):
    rows = []
    for kk in range(1, k + 1):
        mh = pos == (kk - 1)
        mt = pos == (n - kk)
        rows.append(dict(k=kk, head_n=int(mh.sum()), tail_n=int(mt.sum()),
                         head_loss_c=float(Lc[mh].mean()) if mh.any() else np.nan,
                         head_acc=float(A[mh].mean()) if mh.any() else np.nan,
                         tail_loss_c=float(Lc[mt].mean()) if mt.any() else np.nan,
                         tail_acc=float(A[mt].mean()) if mt.any() else np.nan))
    return pd.DataFrame(rows)


def position_curve(codes, pos, n, Lc, A, K):
    """固定 cell 集合 (n >= K)，第 k 把 / 倒数第 k 把的平均水平。"""
    sel = n[codes] >= K
    pc, pp, la, lc = codes[sel], pos[sel], A[sel], Lc[sel]
    nc = int((n >= K).sum())
    rows = []
    for i in range(K):
        mh = pp == i
        mt = pp == (n[pc] - 1 - i)
        rows.append(dict(K=K, k=i + 1, n_cells=nc,
                         head_acc=float(la[mh].mean()) if mh.any() else np.nan,
                         head_loss_c=float(lc[mh].mean()) if mh.any() else np.nan,
                         head_n=int(mh.sum()),
                         tail_acc=float(la[mt].mean()) if mt.any() else np.nan,
                         tail_loss_c=float(lc[mt].mean()) if mt.any() else np.nan,
                         tail_n=int(mt.sum())))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- main
def run(tag: str, unit: str = "chart"):
    suffix = "" if unit == "chart" else f"_{unit}"
    cells = pd.read_parquet(OUT / f"charts_v2_{tag}_v3.parquet",
                            columns=["user_id", "beatmap_id", "chart_id", "rate", "n_scores",
                                     "attempts", "star", "primary"])
    cells["unit_id"] = (cells.chart_id if unit == "chart" else cells.beatmap_id).astype(np.int64)
    cells["key"] = cells.user_id.astype(np.int64) * KEY_MUL + cells.unit_id
    prim = cells[cells.primary].copy()
    print(f"[{tag}/{unit}] cells={len(cells):,} primary={len(prim):,}", flush=True)

    plays = pd.read_parquet(OUT / f"plays_v2_{tag}_v3.parquet",
                            columns=["user_id", "beatmap_id", "chart_id", "loss", "acc", "year", "doy"])
    plays["unit_id"] = (plays.chart_id if unit == "chart" else plays.beatmap_id).astype(np.int64)
    plays["key"] = plays.user_id.astype(np.int64) * KEY_MUL + plays.unit_id

    meta = prim.groupby("key", sort=False).agg(attempts=("attempts", "max"),
                                               star=("star", "median"),
                                               rate=("rate", "first")).reset_index()
    df = plays.merge(meta, on="key", how="inner")
    del plays
    print(f"[{tag}/{unit}] plays in primary cells={len(df):,}", flush=True)

    df["t"] = df.year.to_numpy(np.float64) + (df.doy.to_numpy(np.float64) - 1.0) / 365.0
    df = df.sort_values(["key", "t"], kind="stable").reset_index(drop=True)

    codes, _ = pd.factorize(df.key.to_numpy(), sort=False)
    g = int(codes.max()) + 1
    n = np.bincount(codes, minlength=g).astype(np.int64)
    pos = np.arange(len(codes), dtype=np.int64) - np.repeat(np.cumsum(n) - n, n)
    n_row = n[codes]
    p = (pos + 0.5) / n_row
    at = df.attempts.to_numpy(np.float64)
    at = np.where(np.isfinite(at) & (at >= 1), at, n_row.astype(np.float64))
    cum = np.clip(p * at, 0.5, None)

    L = df.loss.to_numpy(np.float64)
    A = df.acc.to_numpy(np.float64)
    mean_cell = np.bincount(codes, weights=L, minlength=g) / n
    l2 = np.bincount(codes, weights=L * L, minlength=g) / n
    sd_cell = np.sqrt(np.maximum(l2 - mean_cell ** 2, 0.0))
    Lc = L - mean_cell[codes]

    T = df.t.to_numpy()
    tmin = np.full(g, np.inf); np.minimum.at(tmin, codes, T)
    tmax = np.full(g, -np.inf); np.maximum.at(tmax, codes, T)
    span = tmax - tmin

    star_row = df.star.to_numpy(np.float64)
    rate_row = df.rate.to_numpy(np.int64)
    span_row = span[codes]
    dens_row = n_row / np.maximum(at, 1.0)

    # ---- cell 级
    # loss 最大 = 最差；loss 最小 = 最好
    order_w = np.lexsort((pos, -L, codes))           # 每组第一行 = loss 最大 = 最差
    order_b = np.lexsort((pos, L, codes))            # 每组第一行 = loss 最小 = 最好
    idx_w = order_w[np.searchsorted(codes[order_w], np.arange(g), side="left")]
    idx_b = order_b[np.searchsorted(codes[order_b], np.arange(g), side="left")]
    fs = pd.Series(L).groupby(codes).first().to_numpy()
    ls = pd.Series(L).groupby(codes).last().to_numpy()
    fa = pd.Series(A).groupby(codes).first().to_numpy()
    la = pd.Series(A).groupby(codes).last().to_numpy()

    cs = pd.DataFrame({
        "n": n, "span": span,
        "rate": pd.Series(rate_row).groupby(codes).first().to_numpy(),
        "star": pd.Series(star_row).groupby(codes).first().to_numpy(),
        "attempts": pd.Series(at).groupby(codes).first().to_numpy(),
        "slope_p": slope_within(codes, g, p, L),
        "slope_logcum": slope_within(codes, g, np.log(cum), L),
        "first_loss": fs, "last_loss": ls, "first_acc": fa, "last_acc": la,
        "best_loss": L[idx_b], "worst_loss": L[idx_w],
        "p_best": (pos[idx_b] + 0.5) / n, "p_worst": (pos[idx_w] + 0.5) / n,
        "mean_loss": mean_cell, "sd_loss": sd_cell,
    })
    cs["d_loss"] = cs["last_loss"] - cs["first_loss"]          # <0 = 变好
    cs["d_acc"] = cs["last_acc"] - cs["first_acc"]             # >0 = 变好
    cs["acc_range"] = cs["last_acc"] - cs["first_acc"]
    cs["dens"] = cs.n / np.maximum(cs.attempts, 1.0)
    cs.to_parquet(OUT / f"order_curve_{tag}{suffix}_cells.parquet", index=False)

    def dist(s):
        s = s.dropna()
        return {"mean": float(s.mean()), "median": float(s.median()), "sd": float(s.std()),
                "q10": float(s.quantile(.10)), "q25": float(s.quantile(.25)),
                "q75": float(s.quantile(.75)), "q90": float(s.quantile(.90))}

    summary = {"tag": tag, "unit": unit, "plays": int(len(df)), "cells": g,
               "n_median": float(np.median(n)), "span_years_median": float(np.median(span)),
               "n_dist": {f"{lo}-{hi if hi < 10**8 else '+'}": int(((n >= lo) & (n <= hi)).sum())
                          for lo, hi in NB_BUCKETS}}

    # ---- 曲线
    curves = []
    idx_p = np.digitize(p, EDGES_P[1:-1], right=False)
    t1 = curve_by_bin(p, L, A, Lc, idx_p, LABS_P)
    t1["x"] = (EDGES_P[:-1] + EDGES_P[1:]) / 2
    t1 = t1.merge(curve_by_cellmean(p, Lc, codes, g, idx_p, LABS_P), on="bin", how="left")
    t1.insert(0, "curve", "percentile"); curves.append(t1)

    idx_c = np.digitize(cum, EDGES_C[1:-1], right=False)
    t2 = curve_by_bin(cum, L, A, Lc, idx_c, LABS_C)
    t2["x"] = np.sqrt(EDGES_C[:-1] * EDGES_C[1:])
    t2 = t2.merge(curve_by_cellmean(cum, Lc, codes, g, idx_c, LABS_C), on="bin", how="left")
    t2.insert(0, "curve", "cum_plays"); curves.append(t2)

    t3 = head_tail_curve(Lc, A, pos, n_row, k=20)
    t3.insert(0, "curve", "head_tail_k"); curves.append(t3)

    def grouped(name, masks_labels):
        for lab, m in masks_labels:
            if int(m.sum()) < 200:
                continue
            tb = curve_by_bin(p[m], L[m], A[m], Lc[m], idx_p[m], LABS_P)
            tb["x"] = (EDGES_P[:-1] + EDGES_P[1:]) / 2
            tb.insert(0, "group", lab); tb.insert(0, "curve", name)
            curves.append(tb)

    grouped("by_n", [(lab, (n_row >= lo) & (n_row <= hi))
                     for lab, (lo, hi) in zip(NB_LABELS, NB_BUCKETS)])
    grouped("by_rate", [(rn, rate_row == r) for r, rn in RATE_NAME.items()])
    grouped("by_star", [(lab, (star_row >= lo) & (star_row < hi))
                        for lab, (lo, hi) in zip(STAR_LABELS, STAR_BUCKETS)])
    grouped("by_span", [(lab, (span_row > lo) & (span_row <= hi))
                        for lab, (lo, hi) in zip(SPAN_LABELS, SPAN_BUCKETS)])
    grouped("by_density", [(f"{lo:.1f}-{hi:.1f}", (dens_row >= lo) & (dens_row < hi))
                           for lo, hi in DENS_BUCKETS])
    curves_df = pd.concat(curves, ignore_index=True)
    curves_df.to_csv(OUT / f"order_curve_{tag}{suffix}_curves.csv", index=False)

    # ---- 固定 cell 集合的位置曲线（不受 cell 组成变化干扰）
    pos_tab = pd.concat([position_curve(codes, pos, n, Lc, A, K) for K in POS_KS],
                        ignore_index=True)
    pos_tab.to_csv(OUT / f"order_curve_{tag}{suffix}_position.csv", index=False)

    # ---- 斜率分组表
    def slope_tab(vals, groups):
        rows = []
        for lab, m in groups:
            if int(m.sum()) < 50:
                continue
            rows.append(dict(group=lab, cells=int(m.sum()),
                             slope_p_mean=float(cs.slope_p[m].mean()),
                             slope_p_median=float(cs.slope_p[m].median()),
                             frac_worse=float((cs.slope_p[m] > 0).mean()),
                             d_acc_mean=float(cs.d_acc[m].mean()),
                             d_acc_median=float(cs.d_acc[m].median()),
                             frac_d_acc_pos=float((cs.d_acc[m] > 0).mean()),
                             p_best_median=float(cs.p_best[m].median()),
                             p_worst_median=float(cs.p_worst[m].median()),
                             span_median=float(cs.span[m].median()),
                             n_median=float(cs.n[m].median())))
        return pd.DataFrame(rows)

    st_n = slope_tab(n, [(lab, (n >= lo) & (n <= hi)) for lab, (lo, hi) in zip(NB_LABELS, NB_BUCKETS)])
    rate_c = cs.rate.to_numpy()
    star_c = cs.star.to_numpy()
    dens_c = cs.dens.to_numpy()
    st_rate = slope_tab(rate_c, [(rn, rate_c == r) for r, rn in RATE_NAME.items()])
    st_star = slope_tab(star_c, [(lab, (star_c >= lo) & (star_c < hi))
                                 for lab, (lo, hi) in zip(STAR_LABELS, STAR_BUCKETS)])
    st_dens = slope_tab(dens_c, [(f"{lo:.1f}-{hi:.1f}", (dens_c >= lo) & (dens_c < hi))
                                 for lo, hi in DENS_BUCKETS])
    for name, t in (("n", st_n), ("rate", st_rate), ("star", st_star), ("density", st_dens)):
        t.to_csv(OUT / f"order_curve_{tag}{suffix}_slope_by_{name}.csv", index=False)

    # ---- 汇总
    summary["slope_p"] = dist(cs.slope_p)
    summary["slope_p"].update({"frac_worse": float((cs.slope_p > 0).mean()),
                               "frac_better": float((cs.slope_p < 0).mean()),
                               "frac_abs_lt_0.25": float((cs.slope_p.abs() < 0.25).mean())})
    summary["slope_logcum"] = dist(cs.slope_logcum)
    summary["slope_logcum"]["frac_worse"] = float((cs.slope_logcum > 0).mean())
    summary["d_acc"] = dist(cs.d_acc)
    summary["d_acc"]["frac_pos"] = float((cs.d_acc > 0).mean())
    summary["d_loss"] = dist(cs.d_loss)
    summary["first_acc_mean"] = float(cs.first_acc.mean())
    summary["last_acc_mean"] = float(cs.last_acc.mean())
    summary["acc_gain_mean_pp"] = float(cs.d_acc.mean() * 100)
    summary["p_best_median"] = float(cs.p_best.median())
    summary["p_best_hist"] = [int(v) for v in np.histogram(cs.p_best, bins=np.linspace(0, 1, 11))[0]]
    summary["p_worst_median"] = float(cs.p_worst.median())
    summary["p_worst_hist"] = [int(v) for v in np.histogram(cs.p_worst, bins=np.linspace(0, 1, 11))[0]]
    summary["first_play_loss_centered"] = float(Lc[pos == 0].mean())
    summary["last_play_loss_centered"] = float(Lc[pos == n_row - 1].mean())
    summary["first_play_acc"] = float(A[pos == 0].mean())
    summary["last_play_acc"] = float(A[pos == n_row - 1].mean())

    (OUT / f"order_curve_{tag}{suffix}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

    if unit == "chart":
        try:
            make_figs(tag, suffix, curves_df, cs, pos_tab)
        except Exception:  # noqa
            import traceback; traceback.print_exc()
    return summary


def make_figs(tag, suffix, curves_df, cs, pos_tab):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.axisbelow": True})
    FIG.mkdir(parents=True, exist_ok=True)
    nm = f"{tag}{suffix}"

    # 1. 百分位曲线：loss（越小越好） + ACC
    t = curves_df[curves_df.curve == "percentile"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(t.x, t.acc, "o-", color="#c0392b")
    ax[0].set_xlabel("percentile within cell"); ax[0].set_ylabel("mean ACC")
    ax[0].set_title(f"{nm}: ACC vs play-order percentile")
    ax[1].plot(t.x, t.loss_centered, "o-", label="per-play", color="#c0392b")
    ax[1].plot(t.x, t.loss_centered_cellmean, "s--", label="per-cell", color="#2c7fb8")
    ax[1].axhline(0, color="k", lw=.8)
    ax[1].set_xlabel("percentile"); ax[1].set_ylabel("loss - cell mean  (lower = better)")
    ax[1].legend(); ax[1].set_title(f"{nm}: within-cell centered loss")
    ax[2].plot(t.x, -t.loss_centered, "o-", color="#31a354")
    ax[2].axhline(0, color="k", lw=.8)
    ax[2].set_xlabel("percentile"); ax[2].set_ylabel("-(loss - cell mean)  (higher = better)")
    ax[2].set_title(f"{nm}: skill gain (sign-flipped)")
    fig.tight_layout(); fig.savefig(FIG / f"order_{nm}_1_percentile.png"); plt.close(fig)

    # 2. 累计次数
    t = curves_df[curves_df.curve == "cum_plays"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(t.x, t.acc, "o-", color="#c0392b"); ax[0].set_xscale("log")
    ax[0].set_xlabel("estimated cumulative plays (percentile x attempts)")
    ax[0].set_ylabel("mean ACC"); ax[0].set_title(f"{nm}: ACC vs cumulative play count")
    ax[1].plot(t.x, t.loss_centered, "o-", label="per-play", color="#c0392b")
    ax[1].plot(t.x, t.loss_centered_cellmean, "s--", label="per-cell", color="#2c7fb8")
    ax[1].set_xscale("log"); ax[1].axhline(0, color="k", lw=.8)
    ax[1].set_xlabel("estimated cumulative plays")
    ax[1].set_ylabel("loss - cell mean (lower = better)")
    ax[1].legend(); ax[1].set_title(f"{nm}: centered loss vs play count")
    fig.tight_layout(); fig.savefig(FIG / f"order_{nm}_2_cumcount.png"); plt.close(fig)

    # 3. 固定 cell 集合的位置曲线
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for K, sub in pos_tab.groupby("K"):
        ax[0].plot(sub.k, sub.head_acc, "o-", ms=3, label=f"cells with n>={K}")
    ax[0].set_xlabel("k-th play from the start"); ax[0].set_ylabel("mean ACC")
    ax[0].legend(fontsize=7); ax[0].set_title(f"{nm}: learning curve (fixed cell set)")
    for K, sub in pos_tab.groupby("K"):
        ax[1].plot(sub.k, sub.tail_acc, "s-", ms=3, label=f"cells with n>={K}")
    ax[1].set_xlabel("k-th play from the end"); ax[1].set_ylabel("mean ACC")
    ax[1].legend(fontsize=7); ax[1].set_title(f"{nm}: tail (fixed cell set)")
    fig.tight_layout(); fig.savefig(FIG / f"order_{nm}_3_position.png"); plt.close(fig)

    # 4. 分 cell 大小
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for gg, sub in curves_df[curves_df.curve == "by_n"].groupby("group", sort=False):
        ax[0].plot(sub.x, sub.acc, "o-", label=f"n={gg}", ms=3)
    ax[0].legend(fontsize=7); ax[0].set_xlabel("percentile"); ax[0].set_ylabel("mean ACC")
    ax[0].set_title(f"{nm}: ACC curve by cell size")
    sb = pd.read_csv(OUT / f"order_curve_{nm}_slope_by_n.csv")
    ax[1].bar(sb.group.astype(str), sb.frac_d_acc_pos, color="#31a354", alpha=.85)
    ax[1].set_ylim(0, 1); ax[1].set_ylabel("fraction of cells that improve")
    ax[1].set_xlabel("cell size bucket"); ax[1].set_title(f"{nm}: P(last > first)")
    fig.tight_layout(); fig.savefig(FIG / f"order_{nm}_4_by_n.png"); plt.close(fig)

    # 5. rate / star / density
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for gg, sub in curves_df[curves_df.curve == "by_rate"].groupby("group", sort=False):
        ax[0].plot(sub.x, sub.acc, "o-", label=gg, ms=3)
    ax[0].legend(); ax[0].set_xlabel("percentile"); ax[0].set_ylabel("mean ACC")
    ax[0].set_title(f"{nm}: by rate")
    for gg, sub in curves_df[curves_df.curve == "by_star"].groupby("group", sort=False):
        ax[1].plot(sub.x, sub.acc, "o-", label=f"{gg}*", ms=3)
    ax[1].legend(fontsize=7); ax[1].set_xlabel("percentile"); ax[1].set_ylabel("mean ACC")
    ax[1].set_title(f"{nm}: by difficulty")
    for gg, sub in curves_df[curves_df.curve == "by_density"].groupby("group", sort=False):
        ax[2].plot(sub.x, sub.acc, "o-", label=f"n/attempts {gg}", ms=3)
    ax[2].legend(fontsize=7); ax[2].set_xlabel("percentile"); ax[2].set_ylabel("mean ACC")
    ax[2].set_title(f"{nm}: by record completeness")
    fig.tight_layout(); fig.savefig(FIG / f"order_{nm}_5_groups.png"); plt.close(fig)

    # 6. 斜率 / 位置分布
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    s = cs.slope_p.dropna(); s = s[np.abs(s) < 6]
    ax[0].hist(s, bins=120, color="#2c7fb8")
    ax[0].axvline(0, color="k", lw=1)
    ax[0].axvline(s.median(), color="#c0392b", ls="--", label=f"median {s.median():.2f}")
    ax[0].set_xlabel("d(loss)/d(percentile)  (<0 = improving)"); ax[0].legend()
    ax[0].set_title(f"{nm}: within-cell slope")
    ax[1].hist(cs.p_best.dropna(), bins=20, color="#31a354")
    ax[1].axvline(0.5, color="k", lw=.8, ls=":")
    ax[1].set_xlabel("percentile of the cell's BEST score")
    ax[1].set_title(f"{nm}: where is the personal best?")
    ax[2].hist(cs.d_acc.dropna().clip(-0.05, 0.15) * 100, bins=120, color="#756bb1")
    ax[2].axvline(0, color="k", lw=1)
    ax[2].set_xlabel("ACC(last) - ACC(first), pp")
    ax[2].set_title(f"{nm}: per-cell ACC gain")
    fig.tight_layout(); fig.savefig(FIG / f"order_{nm}_6_slope.png"); plt.close(fig)

    # 7. span
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))
    for gg, sub in curves_df[curves_df.curve == "by_span"].groupby("group", sort=False):
        ax.plot(sub.x, sub.acc, "o-", label=gg, ms=3)
    ax.legend(); ax.set_xlabel("percentile"); ax.set_ylabel("mean ACC")
    ax.set_title(f"{nm}: by calendar span")
    fig.tight_layout(); fig.savefig(FIG / f"order_{nm}_7_span.png"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", action="append", required=True)
    ap.add_argument("--unit", choices=["chart", "beatmap"], default="chart")
    a = ap.parse_args()
    for t in a.tag:
        run(t, a.unit)
