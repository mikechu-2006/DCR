"""遗忘曲线：cell 内成绩对「练习间隔」的依赖

背景
----
`play_order_analysis` 发现 cell 内 loss 随**累计游玩次数**近似 log-linear 下降
（ΔACC ≈ 0.0134·ln n），并在几十把后饱和。但「次数」把 10 年前的那把和昨天的那把
一视同仁。用户观察：**很久以前的游玩影响不大，当天的游玩影响很大**。

本脚本检验并量化这个假设，做法是把「次数」换成带遗忘的**有效练习存量**：

    E_i = Σ_{j<i} K(t_i − t_j)              # 有效存量（单位：等价于几把"新鲜"练习）
    loss_i = a_c + β · ln(1 + E_i)          # 学习曲线对存量仍然 log-linear

K(Δ) 候选（Δ 单位：天）：

    none        K = 1                        → E = k−1，退化成原来的「次数」模型
    exp(τ)      K = exp(−Δ/τ)                → 半衰期 τ·ln2
    pow(τ,d)    K = (1 + Δ/τ)^(−d)           → 人类记忆的经典幂律遗忘（Wixted 1991）
    2exp        K = w·exp(−Δ/τ_s) + (1−w)·exp(−Δ/τ_l)

K(Δ) 的读法：**Δ 天前的那把练习，等价于 K(Δ) 把「刚刚打的」练习**。

识别策略（关键）
----------------
cell 内「第 k 把」与「日历时间」共线（corr(ln k, t) ≈ 0.58），所以单看学习曲线
无法区分「遗忘」和「玩家水平整体漂移」。本脚本用三条腿：

  1. **非参数剂量-反应**（主证据）：相邻两把的 ΔACC 对 Δt 分箱，同时控制
     「第几把 k」「上一把的水平」和一个线性时间项 γ·Δt。**短 Δt 尺度上漂移可忽略**
     （γ ≈ 0.15 pp/年 ⇒ 1 天只有 0.0004 pp），所以 1 分钟→30 天这一段的变化
     **只能是遗忘**。这正是需要**秒级时间戳**（`step0_*.csv/parquet`）而不是
     (year,doy) 的原因。
  2. **核函数拟合**：在 cell FE 下拟合 β、γ 与核参数，比较 none/exp/pow/2exp 的
     within-cell R² 与 AIC，并扫出 τ 的可识别性曲线（平坦 ⇒ τ 不可识别）。
  3. **模拟自检**：造一份已知 τ 的数据，验证估计量能否还原。

产物
----
data/processed/forgetting_{tag}.json
data/processed/forgetting_{tag}_{gap_dist,dt_reg,dt_reg_loss,by_k_dt,dt_profile,
                                 kernel_curve,kernel_scan}.csv
docs/figs/forgetting_{tag}_*.png

用例
----
python scripts/forgetting_curve.py --tag 1k --tag 10k
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/processed")
FIG = Path("docs/figs")
SEC_DAY = 86400.0
DAYS_YEAR = 365.25
RATE_NM = 0

# Δt 分箱（天）
DT_EDGES = [0, 10 / 1440, 60 / 1440, 6 / 24, 1, 3, 7, 30, 90, 182, 365, 730, 1e9]
DT_LABS = ["<10min", "10-60min", "1-6h", "6-24h", "1-3d", "3-7d", "7-30d",
           "1-3m", "3-6m", "6-12m", "1-2y", "2-5y", ">5y"]
DT_REF = 0                     # 参考组 = <10min（每个 k 都有足够样本）
KMAX_PAIR = 9
NLB = 10                       # 水平分层（十分位）


# ------------------------------------------------------------------ helpers
def center(v, codes, g, n):
    return v - (np.bincount(codes, weights=v, minlength=g) / n)[codes]


def r2(y, resid):
    return 1.0 - float((resid ** 2).sum()) / float((y ** 2).sum())


def kernel(dt, kind, p):
    if kind == "none":
        return np.ones_like(dt, dtype=np.float64)
    if kind == "exp":
        return np.exp(-dt / p["tau"])
    if kind == "pow":
        return (1.0 + dt / p["tau"]) ** (-p["d"])
    if kind == "2exp":
        return p["w"] * np.exp(-dt / p["tau_s"]) + (1.0 - p["w"]) * np.exp(-dt / p["tau_l"])
    raise ValueError(kind)


def half_life(kind, p):
    if kind == "exp":
        return p["tau"] * np.log(2)
    if kind == "pow":
        return p["tau"] * (2 ** (1.0 / p["d"]) - 1.0)
    if kind == "2exp":
        return p["tau_s"] * np.log(2)
    return np.inf


def ols_cluster(X, y, cl, ncl):
    """OLS + 按 cluster 的稳健协方差。"""
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ (X.T @ y)
    e = y - X @ b
    k = X.shape[1]
    meat = np.zeros((k, k))
    for j in range(k):
        Xj = X[:, j]
        for l in range(j, k):
            s = np.bincount(cl, weights=Xj * e * X[:, l], minlength=ncl)
            meat[j, l] = meat[l, j] = float((s * s).sum())
    V = XtXi @ meat @ XtXi
    return b, np.sqrt(np.maximum(np.diag(V), 0))


def design(k, db, nb, lb=None, cont=None, kmax=KMAX_PAIR):
    cols, names = [], []
    for kk in range(2, kmax + 1):
        cols.append((k == kk).astype(np.float64)); names.append(f"k={kk}")
    for b in range(nb):
        if b == DT_REF:
            continue
        cols.append((db == b).astype(np.float64)); names.append(DT_LABS[b])
    if lb is not None:
        for b in range(1, NLB):
            cols.append((lb == b).astype(np.float64)); names.append(f"lvl{b}")
    if cont is not None:
        cols.append(cont); names.append("dt_cont")
    X = np.column_stack(cols)
    keep = X.any(axis=0)                     # 丢掉全零列（空分箱）
    return X[:, keep], [n for n, kp in zip(names, keep) if kp]


# ------------------------------------------------------------------ loading
def read_step0(tag: str) -> pd.DataFrame:
    pq = OUT / f"step0_{tag}.parquet"
    csv = OUT / f"step0_{tag}.csv"
    if pq.exists():
        df = pd.read_parquet(pq)
        print(f"[{tag}] read {pq.name}: {len(df):,} rows", flush=True)
    else:
        df = pd.read_csv(csv, parse_dates=["timestamp"])
        print(f"[{tag}] read {csv.name}: {len(df):,} rows", flush=True)
    return df


def prepare(tag: str, min_n: int):
    plays = read_step0(tag)
    plays["chart_id"] = plays.beatmap_id.astype(np.int64) * 4 + RATE_NM

    ch = pd.read_parquet(OUT / f"charts_v2_{tag}_v3.parquet",
                         columns=["user_id", "chart_id", "n_scores", "attempts",
                                  "star", "rate", "primary"])
    ch = ch[ch.rate == RATE_NM]
    print(f"[{tag}] charts(NM)={len(ch):,} primary={int(ch.primary.sum()):,}", flush=True)

    df = plays.merge(ch[["user_id", "chart_id", "n_scores", "attempts", "star", "primary"]],
                     left_on=["player_id", "chart_id"], right_on=["user_id", "chart_id"],
                     how="inner")
    del plays
    agg = (df.groupby(["player_id", "chart_id"], sort=False)
           .agg(pc=("playcount_cur", "max"), ns=("n_scores", "first")))
    print(f"[{tag}] playcount_cur vs n_scores mismatch = "
          f"{int((agg.pc != agg.ns).sum()):,} / {len(agg):,} cells", flush=True)
    del agg

    df = df[df.primary & (df.n_scores >= min_n)].copy()
    df = df.sort_values(["player_id", "chart_id", "timestamp"], kind="stable").reset_index(drop=True)
    print(f"[{tag}] analysis set: {len(df):,} plays in "
          f"{df.groupby(['player_id','chart_id'], sort=False).ngroups:,} cells", flush=True)
    return df


# ------------------------------------------------------------------ core
def build(df):
    key = df.player_id.to_numpy(np.int64) * 100_000_000 + df.chart_id.to_numpy(np.int64)
    _, codes = np.unique(key, return_inverse=True)
    codes = np.asarray(codes, dtype=np.int64)
    g = int(codes.max()) + 1
    n = np.bincount(codes, minlength=g).astype(np.int64)
    pos = np.arange(len(codes), dtype=np.int64) - np.repeat(np.cumsum(n) - n, n)
    n_row = n[codes]

    ts = df.timestamp.to_numpy().astype("datetime64[s]").astype(np.int64).astype(np.float64)
    t_day = ts / SEC_DAY
    L = df.loss.to_numpy(np.float64)
    A = 1.0 - np.exp(L)

    same = codes[1:] == codes[:-1]
    d_day = np.full(len(L), np.nan); d_day[1:] = np.where(same, t_day[1:] - t_day[:-1], np.nan)
    dL = np.full(len(L), np.nan); dL[1:] = np.where(same, L[1:] - L[:-1], np.nan)
    dA = np.full(len(L), np.nan); dA[1:] = np.where(same, A[1:] - A[:-1], np.nan)

    return dict(codes=codes, g=g, n=n, pos=pos, n_row=n_row, t_day=t_day, t_year=t_day / DAYS_YEAR,
                L=L, A=A, d_day=d_day, dL=dL, dA=dA,
                Lc=center(L, codes, g, n), Ac=center(A, codes, g, n),
                tc=center(t_day / DAYS_YEAR, codes, g, n),
                uid=df.player_id.to_numpy())


def build_pairs(C, max_n):
    n = C["n"]
    keep = (n >= 2) & (n <= max_n)
    starts = np.concatenate([[0], np.cumsum(n)])
    ii, jj = [], []
    for c in np.nonzero(keep)[0]:
        s, e = starts[c], starts[c + 1]
        m = e - s
        if m < 2:
            continue
        idx = np.arange(1, m)
        cnt = idx
        i_loc = np.repeat(idx, cnt)
        j_loc = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        ii.append(s + i_loc); jj.append(s + j_loc)
    i_idx = np.concatenate(ii).astype(np.int32)
    j_idx = np.concatenate(jj).astype(np.int32)
    dt = (C["t_day"][i_idx] - C["t_day"][j_idx]).astype(np.float32)
    print(f"  pairs={len(i_idx):,} (max_n={max_n})", flush=True)
    return i_idx, j_idx, dt


def fit_one(C, i_idx, dt, kind, p, use_drift, gamma_fixed=None):
    K = kernel(dt.astype(np.float64), kind, p)
    E = np.bincount(i_idx, weights=K, minlength=len(C["L"]))
    xc = center(np.log1p(E), C["codes"], C["g"], C["n"])
    if gamma_fixed is not None:                       # 漂移外部固定
        y = C["Lc"] - gamma_fixed * C["tc"]
        b = np.array([float((xc * y).sum() / (xc * xc).sum())])
        resid = y - b[0] * xc
        return dict(sse=float((resid ** 2).sum()), r2=r2(C["Lc"], C["Lc"] - xc * b[0]),
                    beta=float(b[0]), gamma=float(gamma_fixed), n_par=1 + len(p))
    X = np.column_stack([xc, C["tc"]]) if use_drift else xc[:, None]
    b = np.linalg.pinv(X.T @ X) @ (X.T @ C["Lc"])
    resid = C["Lc"] - X @ b
    return dict(sse=float((resid ** 2).sum()), r2=r2(C["Lc"], resid), beta=float(b[0]),
                gamma=float(b[1]) if use_drift else None, n_par=len(b) + len(p))


def scan(C, i_idx, dt, kind, grid, use_drift, gamma_fixed=None):
    best = None
    for p in grid:
        try:
            r = fit_one(C, i_idx, dt, kind, p, use_drift, gamma_fixed)
        except (OverflowError, FloatingPointError):
            continue
        if not np.isfinite(r["sse"]):
            continue
        r["p"] = p
        if best is None or r["sse"] < best["sse"]:
            best = r
    return best


def exp_grid():
    return [{"tau": float(t)} for t in np.geomspace(0.005, 3e5, 60)]


def pow_grid():
    return [{"tau": float(t), "d": float(d)}
            for t in np.geomspace(0.02, 1e5, 22)
            for d in (0.1, 0.15, 0.25, 0.35, 0.5, 0.7, 1.0)]


def twoexp_grid():
    out = []
    for ts_ in np.geomspace(0.002, 20.0, 9):
        for tl in np.geomspace(1.0, 3e4, 10):
            if tl <= ts_ * 5:
                continue
            for w in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
                out.append({"tau_s": float(ts_), "tau_l": float(tl), "w": float(w)})
    return out


def aic(sse, n_obs, k):
    return float(n_obs * np.log(max(sse, 1e-300) / n_obs) + 2 * k)


# ------------------------------------------------------------------ main
def run(tag: str, min_n: int, max_n: int):
    print(f"===== {tag} =====", flush=True)
    df = prepare(tag, min_n)
    C = build(df)
    out = {"tag": tag, "min_n": min_n, "max_n_pairs": max_n,
           "plays": int(len(C["L"])), "cells": int(C["g"]),
           "n_median": float(np.median(C["n"])), "n_mean": float(C["n"].mean())}

    # ---------------------------------------------------------- A. 间隔分布
    m = np.isfinite(C["d_day"]) & (C["d_day"] > 0)
    dd = C["d_day"][m]
    di = np.digitize(dd, DT_EDGES[1:-1], right=True)
    gap_rows = [dict(bucket=lab, pairs=int((di == i).sum()),
                     share=float((di == i).mean()),
                     median_days=float(np.median(dd[di == i])) if (di == i).sum() else np.nan)
                for i, lab in enumerate(DT_LABS)]
    gap = pd.DataFrame(gap_rows)
    gap.to_csv(OUT / f"forgetting_{tag}_gap_dist.csv", index=False)
    print("[gap distribution]\n" + gap.to_string(index=False), flush=True)
    out["gap_dist"] = gap_rows
    out["frac_sub_hour"] = float((dd < 1 / 24).mean())
    out["frac_same_day"] = float((dd < 1).mean())

    # ---------------------------------------------- B. 非参数剂量-反应（主证据）
    sel = np.isfinite(C["d_day"]) & (C["d_day"] >= 0) & (C["pos"] >= 1) & (C["pos"] <= KMAX_PAIR)
    kk = C["pos"][sel].astype(np.int64)
    dts = C["d_day"][sel]
    db = np.digitize(dts, DT_EDGES[1:-1], right=True)
    acc_prev = C["A"][np.nonzero(sel)[0] - 1]        # 上一把的水平（注意 sel 是布尔掩码）
    qs = np.quantile(acc_prev, np.linspace(0, 1, NLB + 1))
    lb = np.clip(np.digitize(acc_prev, qs[1:-1]), 0, NLB - 1)
    cl, ncl = pd.factorize(C["uid"][sel], sort=False)
    y_raw = C["dA"][sel] * 100.0
    yl_raw = C["dL"][sel]

    reg = {}
    specs = [("konly", None, None), ("klevel", lb, None), ("kleveldrift", lb, dts / DAYS_YEAR)]
    outcomes = [("acc", y_raw, "ΔACC (pp)"), ("loss", yl_raw, "Δloss"),
                ("level", C["Lc"][sel], "cell-centered loss（水平）")]
    for name, y, ylab in outcomes:
        for spec_name, lb_arg, cont in specs:
            X, names = design(kk, db, len(DT_LABS), lb=lb_arg, cont=cont)
            b, se = ols_cluster(X, y, cl, len(ncl))
            rows = [dict(term=nm, coef=float(v), se=float(s), t=float(v / s) if s > 0 else np.nan)
                    for nm, v, s in zip(names, b, se)]
            reg[f"{name}_{spec_name}"] = rows
            pd.DataFrame(rows).to_csv(OUT / f"forgetting_{tag}_dt_reg_{name}_{spec_name}.csv",
                                      index=False)
            dt_rows = [r for r in rows if r["term"] in DT_LABS]
            print(f"[{ylab} ~ {spec_name}]  n={int(sel.sum()):,}", flush=True)
            print("   " + "  ".join(f"{r['term']}:{r['coef']:+.3f}±{r['se']:.3f}"
                                    for r in dt_rows), flush=True)
    out["dt_regression"] = reg

    # 稳健性：把上一把水平限制在窄窗内，消掉天花板差异后看 <10min 效应
    narrow = (acc_prev >= 0.96) & (acc_prev <= 0.99)
    Xn, nmn = design(kk[narrow], db[narrow], len(DT_LABS))
    bn, sen = ols_cluster(Xn, y_raw[narrow], cl[narrow], len(ncl))
    nar = [dict(term=a, coef=float(v), se=float(s))
           for a, v, s in zip(nmn, bn, sen) if a in DT_LABS]
    pd.DataFrame(nar).to_csv(OUT / f"forgetting_{tag}_dt_reg_narrow.csv", index=False)
    print(f"[窄窗 ACC∈[0.96,0.99] ΔACC ~ k + Δt]  n={int(narrow.sum()):,}", flush=True)
    print("   " + "  ".join(f"{r['term']}:{r['coef']:+.3f}±{r['se']:.3f}" for r in nar), flush=True)
    out["dt_regression_narrow"] = nar

    # 组成诊断：每个 Δt 桶里上一把的水平 / 原始 ΔACC（判断控制变量是否在硬撑）
    comp = []
    for i, lab in enumerate(DT_LABS):
        s = db == i
        if s.sum() < 100:
            continue
        comp.append(dict(bucket=lab, pairs=int(s.sum()),
                         mean_dt_days=float(dts[s].mean()),
                         mean_acc_prev=float(acc_prev[s].mean()),
                         raw_d_acc_pp=float(y_raw[s].mean()),
                         raw_d_loss=float(yl_raw[s].mean()),
                         mean_k=float(kk[s].mean())))
    compdf = pd.DataFrame(comp)
    compdf.to_csv(OUT / f"forgetting_{tag}_dt_composition.csv", index=False)
    print("[Δt 桶组成诊断]\n" + compdf.to_string(index=False), flush=True)

    # 明细表（k × Δt）
    dA_sel, dL_sel = y_raw, yl_raw
    byk = []
    for k in range(1, KMAX_PAIR + 1):
        for i, lab in enumerate(DT_LABS):
            s = (kk == k) & (db == i)
            if s.sum() < 100:
                continue
            byk.append(dict(k=k, bucket=lab, pairs=int(s.sum()),
                            mean_dt_days=float(dts[s].mean()),
                            d_acc_pp=float(dA_sel[s].mean()),
                            se_acc_pp=float(dA_sel[s].std() / np.sqrt(s.sum())),
                            d_loss=float(dL_sel[s].mean())))
    bykdf = pd.DataFrame(byk)
    bykdf.to_csv(OUT / f"forgetting_{tag}_by_k_dt.csv", index=False)

    # ---------------------------------------------------------- C. 会话结构
    sess = []
    for thr_min, thr_lab in ((30, "30min"), (360, "6h"), (1440, "1d")):
        brk = (~np.isfinite(C["d_day"])) | (C["d_day"] * 1440.0 > thr_min)
        sid = np.cumsum(brk) - 1
        per_cell = pd.Series(sid).groupby(C["codes"]).nunique()
        sess.append(dict(thr=thr_lab, sessions=int(sid.max() + 1),
                         sessions_per_cell_mean=float(per_cell.mean()),
                         frac_cells_single_session=float((per_cell == 1).mean())))
    out["sessions"] = sess
    print("[sessions]", json.dumps(sess), flush=True)

    # ------------------------------------------------------------ D. 核函数拟合
    use_row = C["n_row"] <= max_n
    C_fit = build(df[use_row])
    print(f"  kernel-fit set: {len(C_fit['L']):,} plays "
          f"({int((~use_row).sum()):,} dropped, n>{max_n})", flush=True)
    out["kernel_fit_plays"] = int(len(C_fit["L"]))
    i_idx, j_idx, dt_pairs = build_pairs(C_fit, max_n)
    C = C_fit
    n_obs = int(len(C["L"]))

    # 外部漂移锚：配对回归量级（practice_vs_time ≈ 0.15 pp/年 @ACC≈0.97 ⇒ loss 约 0.26/年）
    gamma_anchor = -0.25

    fits = {}
    fits["none"] = fit_one(C, i_idx, dt_pairs, "none", {}, False)
    fits["none"]["p"] = {}
    fits["none"]["half_life_days"] = np.inf
    fits["none+drift"] = fit_one(C, i_idx, dt_pairs, "none", {}, True)
    fits["none+drift"]["p"] = {}
    fits["none+drift"]["half_life_days"] = np.inf
    for drift in (False, True):
        for kind, grid in (("exp", exp_grid()), ("pow", pow_grid()), ("2exp", twoexp_grid())):
            key = f"{kind}{'+drift' if drift else ''}"
            b = scan(C, i_idx, dt_pairs, kind, grid, drift)
            if b is not None:
                b["half_life_days"] = half_life(kind, b["p"]); fits[key] = b
    for kind, grid in (("exp", exp_grid()), ("2exp", twoexp_grid())):
        key = f"{kind}+driftfix"
        b = scan(C, i_idx, dt_pairs, kind, grid, False, gamma_fixed=gamma_anchor)
        if b is not None:
            b["half_life_days"] = half_life(kind, b["p"]); fits[key] = b
    base = fits["none"]
    for key, b in fits.items():
        b["aic"] = aic(b["sse"], n_obs, b["n_par"])
        b["delta_aic_vs_none"] = b["aic"] - aic(base["sse"], n_obs, base["n_par"])
        b["delta_r2_vs_none"] = b["r2"] - base["r2"]
    out["kernel_fits"] = fits
    print("[kernel fits]")
    for k, v in fits.items():
        g = v["gamma"]
        print(f"  {k:14s} r2={v['r2']:.5f} (+{v['delta_r2_vs_none']:+.5f}) "
              f"dAIC={v['delta_aic_vs_none']:+.0f} beta={v['beta']:.4f} "
              f"gamma={'—' if g is None else f'{g:+.3f}'} hl={v['half_life_days']:.3g}d",
              flush=True)

    # τ 可识别性：扫 R²(τ)
    scan_rows = []
    for kind, grid in (("exp", exp_grid()), ("pow", pow_grid())):
        for p in grid:
            try:
                r = fit_one(C, i_idx, dt_pairs, kind, p, False)
            except Exception:  # noqa
                continue
            scan_rows.append(dict(kernel=kind, **p, sse=r["sse"], r2=r["r2"],
                                  beta=r["beta"], half_life_days=half_life(kind, p)))
    pd.DataFrame(scan_rows).to_csv(OUT / f"forgetting_{tag}_kernel_scan.csv", index=False)

    # 遗忘曲线输出（K(Δ)）
    curve = []
    for k, v in fits.items():
        if not v.get("p"):
            continue
        kind = k.split("+")[0]
        for d in [0, 10 / 1440, 1 / 24, 6 / 24, 1, 3, 7, 30, 90, 182, 365, 730, 1825, 3650]:
            curve.append(dict(kernel=k, dt_days=d,
                              K=float(kernel(np.array([d]), kind, v["p"])[0]),
                              half_life_days=v["half_life_days"]))
    pd.DataFrame(curve).to_csv(OUT / f"forgetting_{tag}_kernel_curve.csv", index=False)

    # ------------------------------------------------------------ E. 模拟自检
    out["selftest"] = simulate_check(seed=0, max_n=max_n)
    print("[selftest]", json.dumps(out["selftest"], indent=2), flush=True)

    (OUT / f"forgetting_{tag}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    try:
        make_figs(tag, bykdf, reg, fits, scan_rows)
    except Exception:  # noqa
        import traceback; traceback.print_exc()
    return out


# ------------------------------------------------------------------ selftest
def simulate_check(seed=0, max_n=60):
    """已知遗忘核下，估计量能否还原 τ？（cell 内 k 与 t 共线，所以要同时看带/不带漂移）"""
    rng = np.random.default_rng(seed)
    n_cells = 40000
    tau_true, beta_true, gamma_true = 30.0, -0.35, -0.05
    n = np.clip(np.round(np.exp(rng.normal(np.log(8.0), 0.6, n_cells))), 5, max_n).astype(int)
    g = len(n)
    codes = np.repeat(np.arange(g), n)
    starts = np.concatenate([[0], np.cumsum(n)])
    rate = rng.uniform(0.2, 40.0, g)
    gap = rng.exponential(rate[codes])
    cum = np.cumsum(gap)
    base_ = np.repeat(np.concatenate([[0.0], cum[starts[1:-1] - 1]]), n)
    t_days = cum - base_
    t = t_days / DAYS_YEAR
    a_c = rng.normal(0, 1.2, g)[codes]
    i_idx, _, dtp = build_pairs(dict(n=n, t_day=t_days), max_n)
    E = np.bincount(i_idx, weights=np.exp(-dtp.astype(np.float64) / tau_true), minlength=len(t))
    L = a_c + beta_true * np.log1p(E) + gamma_true * t + rng.normal(0, 0.5, len(t))
    C = dict(codes=codes, g=g, n=n, t_day=t_days, L=L,
             Lc=center(L, codes, g, n), tc=center(t, codes, g, n))
    res = {"true": {"tau_days": tau_true, "beta": beta_true, "gamma_per_year": gamma_true}}
    for drift in (False, True):
        b = scan(C, i_idx, dtp, "exp", exp_grid(), drift)
        res[f"est_drift_{drift}"] = {"tau_days": float(b["p"]["tau"]), "beta": b["beta"],
                                     "gamma": b["gamma"]}
    return res


# ------------------------------------------------------------------ figures
def make_figs(tag, bykdf, reg, fits, scan_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.axisbelow": True,
                         "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False})
    FIG.mkdir(parents=True, exist_ok=True)
    order = [l for l in DT_LABS]

    # 1. ΔACC vs Δt：分 k 的原始曲线 + 回归后的 δ_b（控制 k/水平/漂移）
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for k in sorted(bykdf.k.unique()):
        sub = bykdf[bykdf.k == k]
        xi = [order.index(b) for b in sub.bucket]
        ax[0].errorbar(xi, sub.d_acc_pp, yerr=sub.se_acc_pp, fmt="o-", ms=3, lw=1,
                       alpha=.75, capsize=2, label=f"k={k}")
    ax[0].axhline(0, color="k", lw=.8)
    ax[0].set_xticks(range(len(order))); ax[0].set_xticklabels(order, rotation=45, ha="right")
    ax[0].set_ylabel("ΔACC (pp)"); ax[0].set_xlabel("相邻两把的间隔 Δt")
    ax[0].legend(fontsize=6, ncol=2)
    ax[0].set_title(f"{tag}: 每把的收益 vs 间隔（按第 k 把）")

    for nm, col, lab in (("acc_konly", "#7f8c8d", "ΔACC（只控 k）"),
                         ("acc_kleveldrift", "#c0392b", "ΔACC（控 k+水平+漂移）"),
                         ("loss_kleveldrift", "#2c7fb8", "Δloss（控 k+水平+漂移）")):
        rows = [r for r in reg[nm] if r["term"] in DT_LABS]
        xi = [order.index(r["term"]) for r in rows]
        ax[1].errorbar(xi, [r["coef"] for r in rows], yerr=[r["se"] for r in rows],
                       fmt="o-", color=col, capsize=2, ms=3, label=lab)
    ax[1].axhline(0, color="k", lw=.8)
    ax[1].set_xticks(range(len(order))); ax[1].set_xticklabels(order, rotation=45, ha="right")
    ax[1].set_ylabel("相对 <10min 的额外收益"); ax[1].set_xlabel("Δt")
    ax[1].legend(fontsize=7); ax[1].set_title(f"{tag}: 控制后 Δt 的净效应（±SE）")
    fig.tight_layout(); fig.savefig(FIG / f"forgetting_{tag}_1_dt_dose.png"); plt.close(fig)

    # 2. 核函数 + τ 可识别性
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    ds = np.geomspace(1e-3, 1e4, 500)
    for k, v in fits.items():
        if not v.get("p"):
            continue
        ax[0].plot(ds, kernel(ds, k.split("+")[0], v["p"]), lw=1.6,
                   label=f"{k}  (R²={v['r2']:.4f}, ΔAIC={v['delta_aic_vs_none']:+.0f})")
    ax[0].set_xscale("log"); ax[0].set_ylim(0, 1.02)
    ax[0].set_xlabel("间隔 Δ (天)"); ax[0].set_ylabel("K(Δ) = 等价的新鲜练习把数")
    ax[0].legend(fontsize=6.5); ax[0].set_title(f"{tag}: 拟合出的遗忘曲线")
    sc = pd.DataFrame(scan_rows)
    e = sc[sc.kernel == "exp"].sort_values("tau")
    if len(e):
        ax[1].plot(e.tau, e.r2, "o-", ms=3, color="#31a354")
        ax[1].set_xscale("log")
        ax[1].axhline(fits["none"]["r2"], color="#c0392b", ls="--", lw=1,
                      label="无遗忘（次数模型）")
        ax[1].set_xlabel("τ (天)"); ax[1].set_ylabel("within-cell R²")
        ax[1].legend(fontsize=7)
        ax[1].set_title(f"{tag}: exp 核 τ 的可识别性")
    fig.tight_layout(); fig.savefig(FIG / f"forgetting_{tag}_2_kernel.png"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", action="append", required=True)
    ap.add_argument("--min-n", type=int, default=5)
    ap.add_argument("--max-n", type=int, default=60)
    a = ap.parse_args()
    for t in a.tag:
        run(t, a.min_n, a.max_n)
