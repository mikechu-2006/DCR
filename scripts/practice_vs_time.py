"""练习效应 vs 时间漂移：到底是谁在压低 loss？

背景
----
`docs/play_order_analysis.md` 发现 cell 内 loss 随「累计游玩次数」近似 log-linear 下降
（ΔACC ≈ 0.0134·ln(n)）。但 cell 内「第 k 把」和「日历时间」是**同向推进**的：
打得越多 ⇒ 越晚 ⇒ 玩家整体水平也可能涨了。两个假设在 cell 内**完全共线**，
单看学习曲线无法区分：

    H_practice  同一张图反复练 ⇒ 读谱/肌肉记忆 ⇒ 变好
    H_drift     玩家实力随时间整体漂移（练别的图、水平自然增长）⇒ 同一张图也变好

本脚本用三组「单因素」设计把二者拆开：

  A. 匹配 n、变 span（时间剂量-反应）
     固定 cell 大小 n（练习剂量固定），比较同日 cell 与跨年 cell 的 ΔACC。
     n=3 的 cell 只有 2 次重复 ⇒ 练习效应几乎为零，此时 ΔACC 全部来自时间。

  B. 匹配 span、变 n（练习剂量-反应）
     只看同日 cell（span=0，时间效应为零），比较 n=3 与 n=21-50 的 ΔACC。
     此时 ΔACC 全部来自练习。

  C. 冷启动首把（practice 固定 = 1 把）
     每个 cell 的第一把 k=1，对**所有** cell 都是「同一练习剂量」。
     首把 loss 随日历年的变化 = 纯时间漂移（player FE + year FE + star 控制）。

  D. cell 内联合面板回归
     loss_centered ~ β·ln(k) + γ·(t − t_first)，cell FE 吸收。
     识别来自跨 cell 的「每单位时间的把数」差异；报 cluster-robust SE。

  E. 分解：把观测到的 +1.95pp 分给练习与时间。

产物
----
data/processed/practice_vs_time_{tag}.json
data/processed/practice_vs_time_{tag}_{matched_n_span,practice_curve,curve_by_span,
                                   first_attempt_year,slope_by_span}.csv
docs/figs/practice_vs_time_{tag}_*.png

用例
----
python scripts/practice_vs_time.py --tag 10k --tag 1k
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

NB_BUCKETS = [(3, 3), (4, 5), (6, 10), (11, 20), (21, 50), (51, 10 ** 9)]
NB_LABELS = ["3", "4-5", "6-10", "11-20", "21-50", "51+"]
SPAN_BUCKETS = [(-1e-9, 0.0), (0.0, 7 / 365), (7 / 365, 30 / 365), (30 / 365, 1.0), (1.0, 99.0)]
SPAN_LABELS = ["same-day", "<7d", "7-30d", "30d-1y", ">1y"]


# ------------------------------------------------------------------ helpers
def slope_within(codes, g, x, y, min_n=3):
    cnt = np.bincount(codes, minlength=g).astype(np.float64)
    sx = np.bincount(codes, weights=x, minlength=g)
    sy = np.bincount(codes, weights=y, minlength=g)
    sxx = np.bincount(codes, weights=x * x, minlength=g)
    sxy = np.bincount(codes, weights=x * y, minlength=g)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = (sxy - sx * sy / cnt) / (sxx - sx * sx / cnt)
    s[(cnt < min_n) | ~np.isfinite(s)] = np.nan
    return s


def demean_by(v, codes, g):
    m = np.bincount(codes, weights=v, minlength=g) / np.maximum(
        np.bincount(codes, minlength=g), 1)
    return v - m[codes]


def twoway_fe(y, g1, g2, n1, n2, iters=120):
    """交替投影吸收两组固定效应。

    返回 (残差, 第二组效应)。效应必须**逐轮累加**——收敛后残差在两组内均值都为 0，
    若在收敛后直接取组均值会得到全 0（早期版本就踩了这个坑）。
    两组效应之和只定义到差一个常数，故把 e2 做计数加权零均值化。
    """
    y = y.astype(np.float64).copy()
    e1 = np.zeros(n1); e2 = np.zeros(n2)
    c1 = np.bincount(g1, minlength=n1).astype(np.float64)
    c2 = np.bincount(g2, minlength=n2).astype(np.float64)
    for _ in range(iters):
        d1 = np.bincount(g1, weights=y, minlength=n1) / np.maximum(c1, 1)
        y -= d1[g1]; e1 += d1
        d2 = np.bincount(g2, weights=y, minlength=n2) / np.maximum(c2, 1)
        y -= d2[g2]; e2 += d2
    e2 = e2 - float((e2 * c2).sum() / c2.sum())
    return y, e2


def ols_cluster(X, y, cluster_codes, n_cluster):
    """OLS + cluster-robust (by cluster) 协方差。X 已含常数之外的列。"""
    XtX = X.T @ X
    XtXi = np.linalg.pinv(XtX)
    b = XtXi @ (X.T @ y)
    e = y - X @ b
    # cluster meat：逐 cluster 求和后再平方
    k = X.shape[1]
    meat = np.zeros((k, k))
    for j in range(k):
        Xj = X[:, j]
        for l in range(j, k):
            z = Xj * e * X[:, l]
            s = np.bincount(cluster_codes, weights=z, minlength=n_cluster)
            meat[j, l] = meat[l, j] = float((s * s).sum())
    V = XtXi @ meat @ XtXi
    se = np.sqrt(np.maximum(np.diag(V), 0))
    return b, se


def dist(s):
    s = pd.Series(s).dropna()
    if len(s) == 0:
        return {}
    return {"mean": float(s.mean()), "median": float(s.median()), "sd": float(s.std()),
            "q10": float(s.quantile(.10)), "q90": float(s.quantile(.90))}


# ------------------------------------------------------------------ main
def run(tag: str):
    print(f"===== {tag} =====", flush=True)
    cells = pd.read_parquet(
        OUT / f"charts_v2_{tag}_v3.parquet",
        columns=["user_id", "chart_id", "rate", "n_scores", "attempts", "star", "primary"])
    cells = cells[cells.primary].copy()
    cells["key"] = cells.user_id.astype(np.int64) * KEY_MUL + cells.chart_id

    plays = pd.read_parquet(
        OUT / f"plays_v2_{tag}_v3.parquet",
        columns=["user_id", "chart_id", "year", "doy", "loss", "acc"])
    plays["key"] = plays.user_id.astype(np.int64) * KEY_MUL + plays.chart_id
    plays = plays.merge(cells[["key", "star", "rate", "attempts"]], on="key", how="inner")
    del cells
    plays["t"] = plays.year.to_numpy(np.float64) + (plays.doy.to_numpy(np.float64) - 1.0) / 365.0
    plays = plays.sort_values(["key", "t"], kind="stable").reset_index(drop=True)
    print(f"plays in primary cells = {len(plays):,}", flush=True)

    codes, _ = pd.factorize(plays.key.to_numpy(), sort=False)
    g = int(codes.max()) + 1
    n = np.bincount(codes, minlength=g).astype(np.int64)
    pos = np.arange(len(codes), dtype=np.int64) - np.repeat(np.cumsum(n) - n, n)
    n_row = n[codes]

    L = plays.loss.to_numpy(np.float64)
    A = plays.acc.to_numpy(np.float64)
    T = plays.t.to_numpy(np.float64)
    star_row = plays.star.to_numpy(np.float64)
    rate_row = plays.rate.to_numpy(np.int64)
    uid = plays.user_id.to_numpy()
    yid = plays.year.to_numpy(np.int16)

    k = (pos + 1).astype(np.float64)
    lk = np.log(k)
    t0 = np.full(g, np.inf)
    np.minimum.at(t0, codes, T)
    t1 = np.full(g, -np.inf)
    np.maximum.at(t1, codes, T)
    span = t1 - t0
    span_row = span[codes]
    dt = T - t0[codes]

    cell_mean_L = np.bincount(codes, weights=L, minlength=g) / n
    cell_mean_A = np.bincount(codes, weights=A, minlength=g) / n
    Lc = L - cell_mean_L[codes]
    Ac = A - cell_mean_A[codes]

    # cell 级首末
    first_loss = np.zeros(g); last_loss = np.zeros(g)
    first_acc = np.zeros(g); last_acc = np.zeros(g)
    first_t = t0.copy()
    first_idx = np.zeros(g, dtype=np.int64)
    sel0 = pos == 0
    first_idx[codes[sel0]] = np.nonzero(sel0)[0]
    first_loss = L[first_idx]; first_acc = A[first_idx]
    last_idx = np.zeros(g, dtype=np.int64)
    selL = pos == n_row - 1
    last_idx[codes[selL]] = np.nonzero(selL)[0]
    last_loss = L[last_idx]; last_acc = A[last_idx]
    d_acc = last_acc - first_acc

    cell_star = np.zeros(g); cell_rate = np.zeros(g, dtype=np.int64)
    cell_star[codes] = star_row
    cell_rate[codes] = rate_row

    span_bin = np.digitize(span, [b[1] for b in SPAN_BUCKETS[:-1]], right=True)
    n_bin = np.digitize(n, [b[1] for b in NB_BUCKETS[:-1]], right=True)

    out = {"tag": tag, "cells": int(g), "plays": int(len(plays)),
           "n_median": float(np.median(n)), "span_median": float(np.median(span))}

    # ------------------------------------------------------------------ A/B
    # 匹配 n × span 表
    rows = []
    for ni, nlab in enumerate(NB_LABELS):
        for si, slab in enumerate(SPAN_LABELS):
            m = (n_bin == ni) & (span_bin == si)
            if m.sum() < 200:
                continue
            rows.append(dict(n_bucket=nlab, span_bucket=slab, cells=int(m.sum()),
                             mean_n=float(n[m].mean()), mean_span=float(span[m].mean()),
                             d_acc_mean_pp=float(d_acc[m].mean() * 100),
                             d_acc_median_pp=float(np.median(d_acc[m]) * 100),
                             frac_d_acc_pos=float((d_acc[m] > 0).mean()),
                             first_acc=float(first_acc[m].mean()),
                             last_acc=float(last_acc[m].mean())))
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / f"practice_vs_time_{tag}_matched_n_span.csv", index=False)
    print(tab.to_string(index=False), flush=True)

    # 时间剂量-反应：每个 n 桶里，>1y 相对 same-day 的 ΔACC 增量
    dose_rows = []
    for ni, nlab in enumerate(NB_LABELS):
        sub = tab[tab.n_bucket == nlab]
        if sub.empty:
            continue
        base = sub[sub.span_bucket == "same-day"]
        base_v = float(base.d_acc_mean_pp.iloc[0]) if len(base) else np.nan
        for _, r in sub.iterrows():
            dose_rows.append(dict(n_bucket=nlab, span_bucket=r.span_bucket, cells=r.cells,
                                  d_acc_mean_pp=r.d_acc_mean_pp,
                                  delta_vs_sameday_pp=r.d_acc_mean_pp - base_v))
    dose = pd.DataFrame(dose_rows)
    dose.to_csv(OUT / f"practice_vs_time_{tag}_span_dose.csv", index=False)

    # ------------------------------------------------------------------ B 纯练习
    # 同日 cell（span==0）：时间效应 = 0，只有练习
    m_same = (span_row == 0)
    x = lk - np.bincount(codes, weights=lk, minlength=g)[codes] / n_row
    # cell 内去均值（用精确 cell 均值）
    lk_c = lk - (np.bincount(codes, weights=lk, minlength=g) / n)[codes]
    tt_c = T - (np.bincount(codes, weights=T, minlength=g) / n)[codes]

    def slope_acc(xx, mm):
        num = float((xx[mm] * Ac[mm]).sum()); den = float((xx[mm] ** 2).sum())
        return num / den if den > 0 else np.nan

    def slope_loss(xx, mm):
        num = float((xx[mm] * Lc[mm]).sum()); den = float((xx[mm] ** 2).sum())
        return num / den if den > 0 else np.nan

    beta_acc_same = slope_acc(lk_c, m_same)
    beta_acc_all = slope_acc(lk_c, np.ones(len(L), bool))
    m_long = (span_row > 1.0)
    beta_acc_long = slope_acc(lk_c, m_long)
    beta_acc_short = slope_acc(lk_c, (span_row > 0) & (span_row <= 30 / 365))

    # 匹配 n∈{3,4,5} 的纯练习斜率（否则各 span 组的 k 范围不同，不可比）
    small_n = (n >= 3) & (n <= 5)
    slope_by_span = []
    for si, slab in enumerate(SPAN_LABELS):
        mc = (span_bin == si) & small_n
        mm = mc[codes]
        if mm.sum() < 500:
            continue
        slope_by_span.append(dict(
            span_bucket=slab, cells=int(mc.sum()),
            plays=int(mm.sum()),
            mean_n=float(n_row[mm].mean()), mean_span=float(span_row[mm].mean()),
            beta_acc_per_lnk_matched=slope_acc(lk_c, mm),
            beta_loss_per_lnk_matched=slope_loss(lk_c, mm)))
    sbs = pd.DataFrame(slope_by_span)
    sbs.to_csv(OUT / f"practice_vs_time_{tag}_slope_by_span.csv", index=False)
    print(sbs.to_string(index=False), flush=True)

    # 位置曲线（同日 vs 跨年，匹配 n>=10）
    K = 10
    pos_rows = []
    for slab, m in [("same-day", span_row == 0), ("<30d", (span_row > 0) & (span_row <= 30 / 365)),
                    ("30d-1y", (span_row > 30 / 365) & (span_row <= 1.0)), (">1y", span_row > 1.0)]:
        mm = m & (n_row >= K)
        for kk in range(1, K + 1):
            s = mm & (pos == kk - 1)
            if s.sum() < 100:
                continue
            pos_rows.append(dict(span_bucket=slab, k=kk, n=int(s.sum()),
                                 acc=float(A[s].mean()), loss_c=float(Lc[s].mean())))
    pc = pd.DataFrame(pos_rows)
    pc.to_csv(OUT / f"practice_vs_time_{tag}_practice_curve.csv", index=False)

    # ------------------------------------------------------------------ D 联合面板
    X = np.column_stack([lk_c, tt_c])
    XtX = X.T @ X
    XtXi = np.linalg.pinv(XtX)
    b_loss = XtXi @ (X.T @ Lc)
    b_acc = XtXi @ (X.T @ Ac)
    # cluster-robust by player
    gcode, ng = pd.factorize(uid, sort=False)
    se_loss, se_acc = None, None
    try:
        _, se_loss = ols_cluster(X, Lc, gcode, len(ng))
        _, se_acc = ols_cluster(X, Ac, gcode, len(ng))
    except Exception as e:  # noqa
        print("cluster SE failed:", e, flush=True)

    # 灵活版：练习项用 k 哑变量（不假设 log-linear），再叠加时间项
    KMAX = 10
    kd = np.zeros((len(L), KMAX - 1))
    for kk in range(2, KMAX + 1):
        kd[:, kk - 2] = (k >= kk).astype(np.float64)   # 累积哑变量：k>=2, k>=3, ...
    kd_c = kd - np.array([np.bincount(codes, weights=kd[:, j], minlength=g)[codes] / n_row
                          for j in range(KMAX - 1)]).T
    Xf = np.column_stack([kd_c, tt_c])
    bf_loss = np.linalg.pinv(Xf.T @ Xf) @ (Xf.T @ Lc)
    bf_acc = np.linalg.pinv(Xf.T @ Xf) @ (Xf.T @ Ac)

    # ---------------------------------------------------------------- G 相邻把配对回归
    # Δ = loss(j+1) - loss(j)，同时 Δt = t(j+1) - t(j)。
    # 模型：Δloss = α_j + γ·Δt。α_j 是「第 j→j+1 把的练习收益」，γ 是每年的时间漂移。
    # 识别完全来自「同一对相邻把之间隔了多久」的跨 cell 差异，**不需要假设学习曲线的函数形式**。
    same_cell = codes[1:] == codes[:-1]
    jdx = pos[:-1] + 1
    dL = L[1:] - L[:-1]
    dA = A[1:] - A[:-1]
    dT = T[1:] - T[:-1]
    mp = same_cell & (jdx <= 9)
    jj = jdx[mp]
    Yl, Ya, Xt = dL[mp], dA[mp], dT[mp]
    # 设计矩阵：[α_1..α_9 哑变量] + Δt
    Dj = np.zeros((len(jj), 9))
    Dj[np.arange(len(jj)), np.clip(jj, 1, 9) - 1] = 1.0
    Xg = np.column_stack([Dj, Xt])
    bg_loss = np.linalg.pinv(Xg.T @ Xg) @ (Xg.T @ Yl)
    bg_acc = np.linalg.pinv(Xg.T @ Xg) @ (Xg.T @ Ya)
    # 只保留 Δt=0（同日相邻把）的对照：α_j 的另一种估法
    alpha_dt0_loss = [float(Yl[jj == j].mean()) for j in range(1, 10)]
    alpha_dt0_acc = [float(Ya[jj == j].mean()) for j in range(1, 10)]
    # 只用 Δt>0 的配对做同样回归，检验 α_j 是否稳定
    mpos = Xt > 1e-9
    bg_loss_pos = np.linalg.pinv(Xg[mpos].T @ Xg[mpos]) @ (Xg[mpos].T @ Yl[mpos])
    bg_acc_pos = np.linalg.pinv(Xg[mpos].T @ Xg[mpos]) @ (Xg[mpos].T @ Ya[mpos])
    # cluster-robust SE（按玩家）
    gcode_p, ng_p = pd.factorize(uid[:-1][mp], sort=False)
    _, se_g_loss = ols_cluster(Xg, Yl, gcode_p, len(ng_p))
    _, se_g_acc = ols_cluster(Xg, Ya, gcode_p, len(ng_p))
    pair = {
        "n_pairs": int(mp.sum()), "n_pairs_dt_gt0": int(mpos.sum()),
        "frac_dt_zero": float((Xt <= 1e-9).mean()),
        "gamma_loss_per_year": float(bg_loss[-1]),
        "gamma_loss_per_year_se": float(se_g_loss[-1]),
        "gamma_acc_per_year": float(bg_acc[-1]),
        "gamma_acc_per_year_se": float(se_g_acc[-1]),
        "gamma_loss_per_year_dt_gt0": float(bg_loss_pos[-1]),
        "gamma_acc_per_year_dt_gt0": float(bg_acc_pos[-1]),
        "alpha_loss_j1_9": [float(v) for v in bg_loss[:9]],
        "alpha_acc_j1_9": [float(v) for v in bg_acc[:9]],
        "alpha_loss_j1_9_dt0_only": alpha_dt0_loss,
        "alpha_acc_j1_9_dt0_only": alpha_dt0_acc,
        "mean_dt_all_pairs": float(Xt.mean()),
    }
    # 两阶段：α_j 只由「同日相邻把」估计（最干净），再用 Δt>0 的配对估 γ
    a0_acc = np.asarray(alpha_dt0_acc, dtype=np.float64)
    a0_loss = np.asarray(alpha_dt0_loss, dtype=np.float64)
    resid_acc = Ya[mpos] - a0_acc[np.clip(jj[mpos], 1, 9) - 1]
    resid_loss = Yl[mpos] - a0_loss[np.clip(jj[mpos], 1, 9) - 1]
    X2 = np.column_stack([np.ones(int(mpos.sum())), Xt[mpos]])
    b2_acc = np.linalg.pinv(X2.T @ X2) @ (X2.T @ resid_acc)
    b2_loss = np.linalg.pinv(X2.T @ X2) @ (X2.T @ resid_loss)
    pair["stage2_gamma_acc_per_year"] = float(b2_acc[1])
    pair["stage2_gamma_loss_per_year"] = float(b2_loss[1])
    pair["stage2_intercept_acc_pp"] = float(b2_acc[0] * 100)
    pair["stage2_n_pairs"] = int(mpos.sum())
    print("[pair regression]", json.dumps(pair, indent=2), flush=True)
    # 分 Δt 桶的 ΔACC（不假设线性，直接看剂量-反应）
    dt_edges = [0, 1 / 365, 7 / 365, 30 / 365, 90 / 365, 180 / 365, 1.0, 2.0, 99.0]
    dt_labs = ["0", "<1d", "1-7d", "7-30d", "1-3m", "3-6m", "0.5-1y", "1-2y", ">2y"]
    di = np.digitize(Xt, dt_edges[1:-1], right=True)
    dt_rows = []
    for d, lab in enumerate(dt_labs):
        m = di == d
        if m.sum() < 500:
            continue
        # 用 α_j 校正掉「练习收益」后剩下的就是时间部分
        resid = Ya[m] - Dj[m] @ bg_acc[:9]
        dt_rows.append(dict(dt_bucket=lab, pairs=int(m.sum()), mean_dt=float(Xt[m].mean()),
                            mean_d_acc_pp=float(Ya[m].mean() * 100),
                            d_acc_after_practice_pp=float(resid.mean() * 100)))
    dtt = pd.DataFrame(dt_rows)
    dtt.to_csv(OUT / f"practice_vs_time_{tag}_pair_dt.csv", index=False)
    print(dtt.to_string(index=False), flush=True)

    # --- 为什么配对回归与匹配 n 设计给出的 γ 差 3 倍？逐 n 桶做配对回归
    pair_by_n = []
    n_bin_p = n_bin[codes[:-1][mp]]
    for ni, nlab in enumerate(NB_LABELS):
        m = n_bin_p == ni
        if m.sum() < 5000:
            continue
        Dn = np.zeros((int(m.sum()), 9))
        Dn[np.arange(int(m.sum())), np.clip(jj[m], 1, 9) - 1] = 1.0
        Xn = np.column_stack([Dn, Xt[m]])
        bn = np.linalg.pinv(Xn.T @ Xn) @ (Xn.T @ Ya[m])
        pair_by_n.append(dict(n_bucket=nlab, pairs=int(m.sum()),
                              mean_dt=float(Xt[m].mean()),
                              gamma_acc_per_year=float(bn[-1]),
                              gamma_loss_per_year=float(
                                  (np.linalg.pinv(Xn.T @ Xn) @ (Xn.T @ Yl[m]))[-1])))
    pbn = pd.DataFrame(pair_by_n)
    pbn.to_csv(OUT / f"practice_vs_time_{tag}_pair_gamma_by_n.csv", index=False)
    print(pbn.to_string(index=False), flush=True)

    # --- n==3 的逐对明细：同一 n 下，把第 1→2 把、2→3 把分开看
    n3_rows = []
    m3 = (n_row == 3)
    if m3.sum() > 0:
        idx3 = np.nonzero(m3)[0].reshape(-1, 3)
        a1, a2, a3 = A[idx3[:, 0]], A[idx3[:, 1]], A[idx3[:, 2]]
        l1, l2, l3 = L[idx3[:, 0]], L[idx3[:, 1]], L[idx3[:, 2]]
        sp3 = span[codes[idx3[:, 0]]]
        sb3 = span_bin[codes[idx3[:, 0]]]
        for si, slab in enumerate(SPAN_LABELS):
            s = sb3 == si
            if s.sum() < 300:
                continue
            n3_rows.append(dict(span_bucket=slab, cells=int(s.sum()),
                                mean_span=float(sp3[s].mean()),
                                d12_acc_pp=float((a2[s] - a1[s]).mean() * 100),
                                d23_acc_pp=float((a3[s] - a2[s]).mean() * 100),
                                d13_acc_pp=float((a3[s] - a1[s]).mean() * 100),
                                d12_loss=float((l2[s] - l1[s]).mean()),
                                d23_loss=float((l3[s] - l2[s]).mean()),
                                first_acc=float(a1[s].mean())))
    n3t = pd.DataFrame(n3_rows)
    n3t.to_csv(OUT / f"practice_vs_time_{tag}_n3_pairs.csv", index=False)
    print(n3t.to_string(index=False), flush=True)

    # --- n==3 + 首把水平分层：排除「同日 cell 首把本来就高、天花板压缩了提升」的混淆
    level_adj = None
    if m3.sum() > 0:
        c3 = codes[idx3[:, 0]]
        sb3 = span_bin[c3]
        d13 = a3 - a1
        qs = np.quantile(a1, np.linspace(0, 1, 11))
        lb = np.clip(np.digitize(a1, qs[1:-1]), 0, 9)
        Xs = np.zeros((len(a1), len(SPAN_LABELS) + 10))
        for si in range(len(SPAN_LABELS)):
            Xs[sb3 == si, si] = 1.0
        Xs[np.arange(len(a1)), len(SPAN_LABELS) + lb] = 1.0
        ok = Xs.sum(1) > 0
        b_lvl = np.linalg.lstsq(Xs[ok], d13[ok], rcond=None)[0]
        # 各 span 层的加权平均首把水平（检查分层后是否还差很多）
        lvl_mean = {slab: float(a1[sb3 == si].mean()) for si, slab in enumerate(SPAN_LABELS)}
        level_adj = {"span_effect_acc_pp_vs_sameday": {
            slab: float((b_lvl[si] - b_lvl[0]) * 100) for si, slab in enumerate(SPAN_LABELS)},
            "raw_first_acc_by_span": lvl_mean,
            "mean_span_by_bucket": {slab: float(span[c3][sb3 == si].mean())
                                    for si, slab in enumerate(SPAN_LABELS)}}
        print("[level-adjusted n=3 span effect]", json.dumps(level_adj, indent=2), flush=True)

        # --- 最优设计：n=3，同一步练习（1→2 或 2→3），控制起始水平，只看时间间隔
        t3 = T[idx3[:, 0]]
        t3b = T[idx3[:, 1]]
        t3c = T[idx3[:, 2]]
        step_res = {}
        for name, dstep, dt_step, lvl in (
                ("step_1to2", a2 - a1, t3b - t3, a1),
                ("step_2to3", a3 - a2, t3c - t3b, a2)):
            qs2 = np.quantile(lvl, np.linspace(0, 1, 11))
            lb2 = np.clip(np.digitize(lvl, qs2[1:-1]), 0, 9)
            D = np.zeros((len(lvl), 11))
            D[np.arange(len(lvl)), lb2] = 1.0
            D[:, 10] = dt_step
            bb = np.linalg.lstsq(D, dstep, rcond=None)[0]
            step_res[name] = {
                "gamma_acc_per_year": float(bb[10]),
                "gamma_acc_pp_per_year": float(bb[10] * 100),
                "level_dummies_acc_pp": [float(v * 100) for v in bb[:10]],
                "mean_dt": float(dt_step.mean()),
                "n": int(len(dstep)),
            }
        level_adj["same_step_level_controlled"] = step_res
        print("[same-step, level-controlled gamma]",
              json.dumps({k: {"pp_per_year": round(v["gamma_acc_pp_per_year"], 4)}
                          for k, v in step_res.items()}), flush=True)

    # 单独回归（展示共线造成的混淆）
    def one_col(xx, yy):
        return float((xx * yy).sum() / (xx * xx).sum())

    joint = {
        "n_plays": int(len(L)),
        "beta_loss_per_lnk": float(b_loss[0]),
        "gamma_loss_per_year": float(b_loss[1]),
        "beta_acc_per_lnk": float(b_acc[0]),
        "gamma_acc_per_year": float(b_acc[1]),
        "beta_acc_per_lnk_se": None if se_acc is None else float(se_acc[0]),
        "gamma_acc_per_year_se": None if se_acc is None else float(se_acc[1]),
        "beta_loss_per_lnk_se": None if se_loss is None else float(se_loss[0]),
        "gamma_loss_per_year_se": None if se_loss is None else float(se_loss[1]),
        "flex_gamma_acc_per_year": float(bf_acc[-1]),
        "flex_gamma_loss_per_year": float(bf_loss[-1]),
        "flex_k_profile_loss": [float(v) for v in bf_loss[:-1]],
        "practice_only_acc_per_lnk": float(beta_acc_all),
        "practice_only_sameday_acc_per_lnk": float(beta_acc_same),
        "practice_only_longspan_acc_per_lnk": float(beta_acc_long),
        "practice_only_shortspan_acc_per_lnk": float(beta_acc_short),
        "time_only_acc_per_year": float(one_col(tt_c, Ac)),
        "time_only_loss_per_year": float(one_col(tt_c, Lc)),
        "corr_within_lk_dt": float(np.corrcoef(lk_c, tt_c)[0, 1]),
        "mean_within_sd_lk": float(lk_c.std()),
        "mean_within_sd_dt": float(tt_c.std()),
        "pair": pair,
    }
    print(json.dumps(joint, indent=2), flush=True)

    # ------------------------------------------------------------------ C 冷启动首把
    fa = pd.DataFrame({"user_id": uid[first_idx], "year": yid[first_idx],
                       "chart_id": plays.chart_id.to_numpy()[first_idx],
                       "t": T[first_idx], "loss": L[first_idx], "acc": A[first_idx],
                       "star": star_row[first_idx], "rate": rate_row[first_idx]})
    # 谱面的「诞生年」= 该谱面在所有玩家里的最早游玩年（≈ 上架年）
    birth = plays.groupby("chart_id").year.min()
    fa["chart_birth"] = fa.chart_id.map(birth).astype(float)
    fa["chart_age"] = fa.year - fa.chart_birth
    # 先回归掉 star 与 rate（谱面难度构成会随时间变）
    Z = np.column_stack([np.ones(len(fa)), fa.star.to_numpy(),
                         fa.star.to_numpy() ** 2, (fa.rate.to_numpy() == 1).astype(float),
                         (fa.rate.to_numpy() == 2).astype(float)])
    bz = np.linalg.lstsq(Z, fa.loss.to_numpy(), rcond=None)[0]
    fa["loss_r"] = fa.loss.to_numpy() - Z @ bz
    mean_om = float((1 - fa.acc).mean())

    def drift_on(sub, label):
        g1, ng1 = pd.factorize(sub.user_id.to_numpy(), sort=False)
        g2, yr_labels = pd.factorize(sub.year.to_numpy(), sort=True)
        ng2 = len(yr_labels)
        if ng2 < 3:
            return None
        _, eff2 = twoway_fe(sub.loss_r.to_numpy(), g1, g2, len(ng1), ng2)
        cnt2 = np.bincount(g2, minlength=ng2)
        t = pd.DataFrame({"year": np.asarray(yr_labels, dtype=np.int64),
                          "n_first_attempts": cnt2, "loss_effect": eff2})
        t["acc_effect_pp"] = -t.loss_effect * mean_om * 100
        t["raw_mean_acc"] = t.year.map(sub.groupby(sub.year.astype(np.int64)).acc.mean())
        t["raw_mean_star"] = t.year.map(sub.groupby(sub.year.astype(np.int64)).star.mean())
        keep = t.n_first_attempts.to_numpy() > 500
        slope = float("nan")
        if keep.sum() >= 3:
            yy = t.year.to_numpy(np.float64)[keep] - t.year.to_numpy(np.float64)[keep].mean()
            ww = np.sqrt(t.n_first_attempts.to_numpy(np.float64)[keep])
            bb = np.linalg.lstsq(np.column_stack([np.ones(keep.sum()), yy]) * ww[:, None],
                                 t.loss_effect.to_numpy()[keep] * ww, rcond=None)[0]
            slope = float(bb[1])
        return t, slope, label

    # 主口径：全部首把
    main = drift_on(fa, "all")
    year_tab, drift_loss_per_year, _ = main
    drift_acc_per_year = -drift_loss_per_year * mean_om * 100
    print(f"[first-attempt drift] loss/yr={drift_loss_per_year:.4f} "
          f"-> acc {drift_acc_per_year:+.3f} pp/yr", flush=True)
    print(year_tab.to_string(index=False), flush=True)
    year_tab.to_csv(OUT / f"practice_vs_time_{tag}_first_attempt_year.csv", index=False)

    # 稳健性：老玩家（>=30 张图首把、>=3 个年份）+ 难度窗口 4-6★
    robust = {}
    nfa = fa.groupby("user_id").year.nunique()
    heavy = set(nfa[nfa >= 3].index)
    sub_heavy = fa[fa.user_id.isin(heavy)]
    r1 = drift_on(sub_heavy, "players>=3yrs")
    if r1:
        robust["players_multi_year"] = {
            "n": int(len(sub_heavy)), "loss_per_year": r1[1],
            "acc_pp_per_year": -r1[1] * mean_om * 100}
    sub_mid = fa[(fa.star >= 4) & (fa.star <= 6)]
    r2 = drift_on(sub_mid, "star4-6")
    if r2:
        robust["star_4_6"] = {"n": int(len(sub_mid)), "loss_per_year": r2[1],
                              "acc_pp_per_year": -r2[1] * mean_om * 100}
    # 只用「谱面上架已满 1 年」的首把：排除新图潮带来的选图构成变化
    sub_old = fa[fa.chart_age >= 1]
    r3 = drift_on(sub_old, "chart_age>=1")
    if r3:
        robust["chart_age_ge_1y"] = {"n": int(len(sub_old)), "loss_per_year": r3[1],
                                     "acc_pp_per_year": -r3[1] * mean_om * 100}
    # 只用「上架已满 2 年」
    sub_old2 = fa[fa.chart_age >= 2]
    r4 = drift_on(sub_old2, "chart_age>=2")
    if r4:
        robust["chart_age_ge_2y"] = {"n": int(len(sub_old2)), "loss_per_year": r4[1],
                                     "acc_pp_per_year": -r4[1] * mean_om * 100}
    print(json.dumps(robust, indent=2), flush=True)

    # 朴素（无 FE）首把漂移，作对照
    An = np.column_stack([np.ones(len(fa)), fa.t.to_numpy() - fa.t.mean()])
    bn = np.linalg.lstsq(An, fa.loss.to_numpy(), rcond=None)[0]
    drift_naive_loss = float(bn[1])

    out.update({
        "joint": joint,
        "drift_first_attempt_loss_per_year": drift_loss_per_year,
        "drift_first_attempt_acc_pp_per_year": drift_acc_per_year,
        "drift_first_attempt_loss_per_year_naive": drift_naive_loss,
        "drift_first_attempt_acc_pp_per_year_naive": float(-drift_naive_loss * mean_om * 100),
        "drift_robust": robust,
        "pair_by_n": pbn.to_dict("records"),
        "n3_pairs": n3t.to_dict("records"),
        "level_adjusted": level_adj,
        "sameday_beta_acc_per_lnk": beta_acc_same,
        "slope_by_span_matched_n": sbs.to_dict("records"),
        "first_acc_mean": float(first_acc.mean()),
        "last_acc_mean": float(last_acc.mean()),
        "d_acc_mean_pp": float(d_acc.mean() * 100),
        "d_acc_by_span": {slab: float(d_acc[span_bin == si].mean() * 100)
                          for si, slab in enumerate(SPAN_LABELS)
                          if (span_bin == si).sum() > 0},
        "span_share_by_n": {},
    })

    # ------------------------------------------------------------------ E 分解
    mean_lk = float(np.log(n).mean())
    mean_span = float(span.mean())
    obs = float(d_acc.mean() * 100)
    # 口径 1：纯练习（同日 cell 斜率） + 首把漂移
    p1 = beta_acc_same * mean_lk * 100
    t1 = drift_acc_per_year * mean_span
    # 口径 2：cell 内联合面板
    p2 = joint["beta_acc_per_lnk"] * mean_lk * 100
    t2 = joint["gamma_acc_per_year"] * mean_span
    # 口径 3：匹配 n 的 span 增量（n=3 桶，>1y 相对 same-day）
    d3 = None
    try:
        b = tab[(tab.n_bucket == "3") & (tab.span_bucket == "same-day")].iloc[0]
        l = tab[(tab.n_bucket == "3") & (tab.span_bucket == ">1y")].iloc[0]
        d3 = float((l.d_acc_mean_pp - b.d_acc_mean_pp) / max(l.mean_span - b.mean_span, 1e-9))
    except Exception:
        pass
    # 口径 4（首选）：相邻把配对回归。练习收益 = Σ_{j=1}^{n-1} α_j，时间 = γ × span
    al = np.asarray(bg_acc[:9], dtype=np.float64)
    jlist = np.arange(1, int(n.max()) + 1)
    al_ext = np.where(jlist <= 9, al[np.clip(jlist, 1, 9) - 1], al[-1])
    cum = np.concatenate([[0.0], np.cumsum(al_ext)])           # cum[j] = k=1..j+1 的练习收益
    practice_pp = float(cum[np.clip(n - 1, 0, len(cum) - 1)].mean() * 100)
    time_pp = float(pair["gamma_acc_per_year"] * mean_span * 100)
    out["decomp"] = {
        "mean_ln_n": mean_lk, "mean_span_years": mean_span,
        "observed_d_acc_pp": obs,
        "spec1_sameday_plus_drift": {"practice_pp": float(p1), "time_pp": float(t1),
                                     "practice_share": float(p1 / (p1 + t1)) if p1 + t1 else None},
        "spec2_joint_panel": {"practice_pp": float(p2), "time_pp": float(t2),
                              "practice_share": float(p2 / (p2 + t2)) if p2 + t2 else None},
        "spec3_matched_n_span_slope_pp_per_year": d3,
        "spec4_pair_regression": {"practice_pp": practice_pp, "time_pp": time_pp,
                                  "practice_share": float(practice_pp / (practice_pp + time_pp))
                                  if practice_pp + time_pp else None,
                                  "total_pp": practice_pp + time_pp},
    }
    print(json.dumps(out["decomp"], indent=2), flush=True)

    # ------------------------------------------------------------------ F 估计量自检
    out["selftest"] = simulate_check(seed=0)
    print("[selftest]", json.dumps(out["selftest"], indent=2), flush=True)

    # 每个 n 桶里 span 造成的 ΔACC 增量（share of span effect）
    for ni, nlab in enumerate(NB_LABELS):
        sub = tab[tab.n_bucket == nlab]
        if len(sub) < 2:
            continue
        base = sub[sub.span_bucket == "same-day"]
        if not len(base):
            continue
        out["span_share_by_n"][nlab] = float(
            sub.d_acc_mean_pp.max() - float(base.d_acc_mean_pp.iloc[0]))

    (OUT / f"practice_vs_time_{tag}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    try:
        make_figs(tag, tab, pc, year_tab, dose, joint, dtt, n3t)
    except Exception:  # noqa
        import traceback; traceback.print_exc()
    return out


def _sim_once(seed, n_cells, gamma_loss, shape):
    rng = np.random.default_rng(seed)
    n = np.clip(np.round(np.exp(rng.normal(np.log(4.0), 0.55, n_cells))), 3, 60).astype(int)
    g = len(n)
    t0 = rng.uniform(0, 11, g)
    span = np.clip(0.05 * (n - 3) * rng.uniform(0.2, 3.0, g) + rng.exponential(0.3, g), 0, 6)
    codes = np.repeat(np.arange(g), n)
    n_row = np.repeat(n, n)
    pos = np.concatenate([np.arange(v) for v in n])
    lk = np.log(pos + 1.0)
    alpha = rng.uniform(0.3, 3.0, g)[codes]
    tt = (t0[codes] + span[codes] * (pos / np.maximum(n_row - 1, 1)) ** alpha
          + rng.normal(0, 0.02, len(codes)))
    if shape == "log":
        practice = -0.45 * lk
    else:                       # 饱和型练习曲线（真值与 log-linear 不符）
        prof = np.concatenate([[0.0], -0.30 * np.cumsum(1.0 / np.arange(1, 61) ** 0.7)])
        practice = prof[pos]
    y = (rng.normal(0, 0.8, g)[codes] + practice + gamma_loss * tt
         + rng.normal(0, 0.6, len(codes)))
    return n, codes, n_row, pos, lk, tt, y


def simulate_check(seed=0):
    """自检：已知真值下，两个估计量能不能还原 γ（时间漂移）。

    shape='log'  练习曲线确实是 log-linear —— 联合面板回归应当无偏。
    shape='sat'  练习曲线饱和（真值不是 log-linear）—— 联合面板回归的 γ 会被污染，
                 而相邻把配对回归（α_j 完全自由）应当仍然无偏。这正是我选配对回归的理由。
    """
    res = {}
    for shape in ("log", "sat"):
        n, codes, n_row, pos, lk, tt, y = _sim_once(seed, 150000, -0.15, shape)
        g = len(n)
        lk_c, tt_c, y_c = demean_by(lk, codes, g), demean_by(tt, codes, g), demean_by(y, codes, g)
        X = np.column_stack([lk_c, tt_c])
        b = np.linalg.pinv(X.T @ X) @ (X.T @ y_c)
        # 配对回归
        same = codes[1:] == codes[:-1]
        jj = (pos[:-1] + 1)[same]
        dY = (y[1:] - y[:-1])[same]
        dT = (tt[1:] - tt[:-1])[same]
        m = jj <= 9
        Dj = np.zeros((int(m.sum()), 9))
        Dj[np.arange(int(m.sum())), np.clip(jj[m], 1, 9) - 1] = 1.0
        Xg = np.column_stack([Dj, dT[m]])
        bg = np.linalg.pinv(Xg.T @ Xg) @ (Xg.T @ dY[m])
        res[shape] = {"true_gamma": -0.15,
                      "joint_est_gamma": float(b[1]),
                      "pair_est_gamma": float(bg[-1])}
    return res


def make_figs(tag, tab, pc, year_tab, dose, joint, dtt=None, n3t=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.axisbelow": True,
                         "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False})
    FIG.mkdir(parents=True, exist_ok=True)

    # 1. 匹配 n × span：ΔACC 热图/折线
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2))
    for nlab in NB_LABELS:
        sub = tab[tab.n_bucket == nlab]
        if sub.empty:
            continue
        xs = [SPAN_LABELS.index(s) for s in sub.span_bucket]
        ax[0].plot(xs, sub.d_acc_mean_pp, "o-", label=f"n={nlab}")
        ax[1].plot(xs, sub.frac_d_acc_pos, "o-", label=f"n={nlab}")
    for a in ax:
        a.set_xticks(range(len(SPAN_LABELS)))
        a.set_xticklabels(SPAN_LABELS, rotation=20)
        a.legend(fontsize=7)
    ax[0].set_ylabel("ΔACC = ACC(last) − ACC(first), pp")
    ax[0].set_title(f"{tag}: 时间剂量-反应（固定 n）")
    ax[1].set_ylabel("P(变好)")
    ax[1].set_title(f"{tag}: 固定 n，跨度的作用")
    fig.tight_layout(); fig.savefig(FIG / f"practice_vs_time_{tag}_1_matched.png"); plt.close(fig)

    # 2. 练习曲线（同日 vs 跨年，匹配 n>=10）
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for slab, sub in pc.groupby("span_bucket", sort=False):
        ax[0].plot(sub.k, sub.acc, "o-", label=slab)
        ax[1].plot(sub.k, sub.loss_c, "o-", label=slab)
    ax[0].set_xlabel("cell 内第 k 把"); ax[0].set_ylabel("mean ACC")
    ax[0].set_title(f"{tag}: 练习剂量相同、跨度不同 (n≥10)")
    ax[0].legend()
    ax[1].set_xlabel("cell 内第 k 把"); ax[1].set_ylabel("loss − cell mean")
    ax[1].set_title(f"{tag}: 同上（loss 尺度）"); ax[1].legend()
    fig.tight_layout(); fig.savefig(FIG / f"practice_vs_time_{tag}_2_curve_by_span.png"); plt.close(fig)

    # 3. 首把漂移
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    yt = year_tab[year_tab.n_first_attempts > 500]
    ax[0].plot(yt.year, yt.acc_effect_pp, "o-", color="#c0392b")
    ax[0].axhline(0, color="k", lw=.8)
    ax[0].set_xlabel("年份"); ax[0].set_ylabel("首把 ACC 固定效应, pp")
    ax[0].set_title(f"{tag}: 冷启动首把的水平漂移（player+year FE）")
    ax2 = ax[1]
    ax2.plot(yt.year, yt.raw_mean_acc, "o-", color="#2c7fb8", label="raw mean ACC")
    ax2.set_xlabel("年份"); ax2.set_ylabel("raw mean ACC")
    ax2b = ax2.twinx()
    ax2b.plot(yt.year, yt.raw_mean_star, "s--", color="#31a354", label="mean star")
    ax2b.set_ylabel("mean star", color="#31a354")
    ax2.set_title(f"{tag}: 首把原始均值（含选图构成变化）")
    fig.tight_layout(); fig.savefig(FIG / f"practice_vs_time_{tag}_3_drift.png"); plt.close(fig)

    # 4. span 剂量
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.2))
    for nlab in NB_LABELS:
        sub = dose[dose.n_bucket == nlab]
        if sub.empty:
            continue
        xs = [SPAN_LABELS.index(s) for s in sub.span_bucket]
        ax.plot(xs, sub.delta_vs_sameday_pp, "o-", label=f"n={nlab}")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(range(len(SPAN_LABELS))); ax.set_xticklabels(SPAN_LABELS, rotation=20)
    ax.set_ylabel("ΔACC 相对同日 cell 的增量, pp")
    ax.set_title(f"{tag}: 时间漂移的增量（练习剂量已固定）")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / f"practice_vs_time_{tag}_4_span_dose.png"); plt.close(fig)

    # 5. 相邻把的 ΔACC vs Δt（不假设函数形式的核心剂量-反应）
    if dtt is not None and len(dtt):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        ax[0].plot(dtt.mean_dt, dtt.mean_d_acc_pp, "o-", color="#c0392b",
                   label="raw ΔACC")
        ax[0].plot(dtt.mean_dt, dtt.d_acc_after_practice_pp, "s--", color="#2c7fb8",
                   label="扣掉练习收益后")
        ax[0].axhline(0, color="k", lw=.8)
        ax[0].set_xscale("symlog", linthresh=0.005)
        ax[0].set_xlabel("相邻两条成绩的时间间隔 Δt (年)")
        ax[0].set_ylabel("ΔACC (pp)")
        ax[0].legend(); ax[0].set_title(f"{tag}: 时间间隔的剂量-反应")
        ax[1].plot(dtt.mean_dt, dtt.d_acc_after_practice_pp, "s-", color="#2c7fb8")
        ax[1].axhline(0, color="k", lw=.8)
        ax[1].set_xscale("symlog", linthresh=0.005)
        ax[1].set_xlabel("Δt (年)"); ax[1].set_ylabel("ΔACC after practice (pp)")
        ax[1].set_title(f"{tag}: 斜率 ≈ 时间漂移速度")
        fig.tight_layout(); fig.savefig(FIG / f"practice_vs_time_{tag}_5_dt_dose.png")
        plt.close(fig)

    # 6. n=3 逐对
    if n3t is not None and len(n3t):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        xs = np.arange(len(n3t))
        ax[0].bar(xs - 0.2, n3t.d12_acc_pp, 0.4, label="第1→2把", color="#c0392b")
        ax[0].bar(xs + 0.2, n3t.d23_acc_pp, 0.4, label="第2→3把", color="#2c7fb8")
        ax[0].set_xticks(xs); ax[0].set_xticklabels(n3t.span_bucket, rotation=20)
        ax[0].set_ylabel("ΔACC (pp)"); ax[0].legend()
        ax[0].set_title(f"{tag}: n=3，练习剂量固定，只看时间间隔")
        ax[1].plot(xs, n3t.first_acc, "o-", color="#31a354")
        ax[1].set_xticks(xs); ax[1].set_xticklabels(n3t.span_bucket, rotation=20)
        ax[1].set_ylabel("首把 ACC")
        ax[1].set_title(f"{tag}: 各 span 组的首把水平（选择偏差检查）")
        fig.tight_layout(); fig.savefig(FIG / f"practice_vs_time_{tag}_6_n3.png")
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", action="append", required=True)
    a = ap.parse_args()
    for t in a.tag:
        run(t)
