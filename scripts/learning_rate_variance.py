"""学习率的异质性：per-cell 学习率随谱面 / 玩家变化多大？（loss 单位）

cell 内学习率定义（x = ln(cum)，cum = percentile × attempts）：

    beta_loss = d(loss) / d ln(cum)      < 0 = 越打越好
    beta_acc  = d(ACC)  / d ln(cum)      > 0 = 越打越好

因为 loss = log(1-ACC)，d(loss)/d(ACC) = -1/(1-ACC)，所以
**beta_loss ≈ -beta_acc / (1-ACC)**：同一个 ACC 增益，在 ACC 越高的 cell 里
loss 变化越大。也就是说 beta_loss 的离散度里有一大块是 ACC 水平的确定性后果，
不是真实的"学习速度"差异。

方差分解（player × chart 交叉随机效应，每 cell 一条观测）：
用「同玩家不同谱面」与「同谱面不同玩家」的成对协方差无偏估计 σ²_player / σ²_chart
（估计噪声在跨 cell 协方差里不贡献，所以不需要额外扣除）：

    Var(beta) = σ²_player + σ²_chart + σ²_resid + σ²_noise
    σ²_noise  = mean(SE(beta)²)          # OLS 斜率的理论标准误

产出：
    data/processed/learning_rate_{tag}.json
    data/processed/learning_rate_{tag}_by_chart.csv   （cell 数 >= 10 的谱面）
    data/processed/learning_rate_{tag}_by_player.csv
    docs/figs/lr_{tag}_*.png

用例：
    python scripts/learning_rate_variance.py --tag 1k --tag 10k
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
MIN_N = 5          # 只对 n >= MIN_N 的 cell 估斜率


# --------------------------------------------------------------------- helpers
def group_ols(codes, g, x, y, min_n=3):
    """每个 cell 内 y 对 x 的 OLS 斜率 + 理论 SE。"""
    cnt = np.bincount(codes, minlength=g).astype(np.float64)
    sx = np.bincount(codes, weights=x, minlength=g)
    sy = np.bincount(codes, weights=y, minlength=g)
    sxx = np.bincount(codes, weights=x * x, minlength=g)
    sxy = np.bincount(codes, weights=x * y, minlength=g)
    syy = np.bincount(codes, weights=y * y, minlength=g)
    with np.errstate(invalid="ignore", divide="ignore"):
        sxx_c = sxx - sx * sx / cnt
        slope = (sxy - sx * sy / cnt) / sxx_c
        sse = syy - sy * sy / cnt - slope ** 2 * sxx_c
    sse = np.maximum(sse, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.sqrt((sse / np.maximum(cnt - 2, 1)) / sxx_c)
    bad = (cnt < min_n) | ~np.isfinite(slope) | ~np.isfinite(se) | (sxx_c <= 0)
    slope[bad] = np.nan
    se[bad] = np.nan
    return slope, se, sxx_c


def pair_cov(vals, groups, G, grand_mean):
    """同组内「不同成员」的成对协方差（无偏估计 σ²_group）。"""
    d = vals - grand_mean
    s = np.bincount(groups, weights=d, minlength=G)
    ss = np.bincount(groups, weights=d * d, minlength=G)
    k = np.bincount(groups, minlength=G).astype(np.float64)
    pair_sum = (s ** 2 - ss) / 2.0
    pair_cnt = k * (k - 1) / 2.0
    if pair_cnt.sum() <= 0:
        return np.nan, 0
    return float(pair_sum.sum() / pair_cnt.sum()), int(pair_cnt.sum())


def decompose(vals, player_codes, chart_codes, n_players, n_charts, noise_var, label):
    m = np.isfinite(vals)
    v = vals[m]
    total = float(np.var(v))
    gm = float(v.mean())
    s2p, npair_p = pair_cov(v, player_codes[m], n_players, gm)
    s2c, npair_c = pair_cov(v, chart_codes[m], n_charts, gm)
    resid = total - s2p - s2c
    return {"label": label, "n_cells": int(m.sum()), "total_var": total,
            "sd": float(np.sqrt(max(total, 0))),
            "var_player": s2p, "var_chart": s2c,
            "var_resid_incl_noise": resid, "var_noise": noise_var,
            "var_resid_excl_noise": max(resid - noise_var, np.nan),
            "pairs_player": npair_p, "pairs_chart": npair_c,
            "share_player": s2p / total if total > 0 else np.nan,
            "share_chart": s2c / total if total > 0 else np.nan,
            "share_noise": noise_var / total if total > 0 else np.nan,
            "share_resid": (resid - noise_var) / total if total > 0 else np.nan,
            "reliability": (total - noise_var) / total if total > 0 else np.nan}


# --------------------------------------------------------------------- main
def run(tag: str, min_n: int = MIN_N):
    cells = pd.read_parquet(OUT / f"charts_v2_{tag}_v3.parquet",
                            columns=["user_id", "chart_id", "rate", "n_scores", "attempts",
                                     "star", "primary", "countTotal"])
    cells["key"] = cells.user_id.astype(np.int64) * KEY_MUL + cells.chart_id.astype(np.int64)
    prim = cells[cells.primary]
    print(f"[{tag}] primary cells={len(prim):,}", flush=True)

    plays = pd.read_parquet(OUT / f"plays_v2_{tag}_v3.parquet",
                            columns=["user_id", "chart_id", "loss", "acc", "year", "doy"])
    plays["key"] = plays.user_id.astype(np.int64) * KEY_MUL + plays.chart_id.astype(np.int64)
    meta = prim[["key", "attempts", "star", "rate", "countTotal"]]
    df = plays.merge(meta, on="key", how="inner")
    del plays
    df["t"] = df.year.to_numpy(np.float64) + (df.doy.to_numpy(np.float64) - 1.0) / 365.0
    df = df.sort_values(["key", "t"], kind="stable").reset_index(drop=True)

    codes, uniq = pd.factorize(df.key.to_numpy(), sort=False)
    g = int(codes.max()) + 1
    n = np.bincount(codes, minlength=g).astype(np.int64)
    pos = np.arange(len(codes), dtype=np.int64) - np.repeat(np.cumsum(n) - n, n)
    n_row = n[codes]
    p = (pos + 0.5) / n_row
    at = df.attempts.to_numpy(np.float64)
    at = np.where(np.isfinite(at) & (at >= 1), at, n_row.astype(np.float64))
    cum = np.clip(p * at, 0.5, None)
    x = np.log(cum)

    L = df.loss.to_numpy(np.float64)
    A = df.acc.to_numpy(np.float64)

    b_loss, se_loss, sxxc = group_ols(codes, g, x, L, min_n)
    b_acc, se_acc, _ = group_ols(codes, g, x, A, min_n)

    mean_acc = np.bincount(codes, weights=A, minlength=g) / n
    mean_loss = np.bincount(codes, weights=L, minlength=g) / n
    fs = pd.Series(L).groupby(codes).first().to_numpy()
    ls = pd.Series(L).groupby(codes).last().to_numpy()
    fa = pd.Series(A).groupby(codes).first().to_numpy()
    la = pd.Series(A).groupby(codes).last().to_numpy()
    T = df.t.to_numpy()
    tmin = np.full(g, np.inf); np.minimum.at(tmin, codes, T)
    tmax = np.full(g, -np.inf); np.maximum.at(tmax, codes, T)

    player = df.user_id.to_numpy()
    chart = df.chart_id.to_numpy()
    pcode, puniq = pd.factorize(pd.Series(player).groupby(codes).first().to_numpy(), sort=False)
    ccode, cuniq = pd.factorize(pd.Series(chart).groupby(codes).first().to_numpy(), sort=False)

    cs = pd.DataFrame({
        "user_id": puniq[pcode], "chart_id": cuniq[ccode], "n": n,
        "attempts": pd.Series(at).groupby(codes).first().to_numpy(),
        "star": pd.Series(df.star.to_numpy(np.float64)).groupby(codes).first().to_numpy(),
        "rate": pd.Series(df.rate.to_numpy(np.int64)).groupby(codes).first().to_numpy(),
        "countTotal": pd.Series(df.countTotal.to_numpy(np.float64)).groupby(codes).first().to_numpy(),
        "mean_acc": mean_acc, "mean_loss": mean_loss,
        "first_acc": fa, "last_acc": la, "first_loss": fs, "last_loss": ls,
        "span": tmax - tmin,
        "b_loss": b_loss, "se_loss": se_loss, "b_acc": b_acc, "se_acc": se_acc,
        "pcode": pcode, "ccode": ccode,
    })
    cs["d_acc"] = cs.last_acc - cs.first_acc
    cs["d_loss"] = cs.last_loss - cs.first_loss
    cs["xrange"] = np.nan   # 见下方
    # cum 的跨度（对数尺度）
    xmin = np.full(g, np.inf); np.minimum.at(xmin, codes, x)
    xmax = np.full(g, -np.inf); np.maximum.at(xmax, codes, x)
    cs["xrange"] = xmax - xmin
    cs.to_parquet(OUT / f"learning_rate_{tag}_cells.parquet", index=False)

    ok = np.isfinite(b_loss)
    summary = {"tag": tag, "min_n": min_n, "cells_total": g, "cells_used": int(ok.sum())}

    def dist(v):
        v = pd.Series(np.asarray(v, dtype=np.float64))
        v = v[np.isfinite(v)]
        return {"mean": float(v.mean()), "median": float(v.median()), "sd": float(v.std()),
                "q05": float(v.quantile(.05)), "q25": float(v.quantile(.25)),
                "q75": float(v.quantile(.75)), "q95": float(v.quantile(.95))}

    summary["b_loss"] = dist(cs.b_loss.to_numpy())
    summary["b_acc"] = dist(cs.b_acc.to_numpy())
    summary["frac_b_loss_neg"] = float((cs.b_loss.to_numpy()[ok] < 0).mean())
    summary["frac_b_acc_pos"] = float((cs.b_acc.to_numpy()[ok] > 0).mean())
    summary["xrange_median"] = float(np.nanmedian(cs.xrange))
    summary["se_loss_median"] = float(np.nanmedian(cs.se_loss))
    summary["se_acc_median"] = float(np.nanmedian(cs.se_acc))

    # ---- 噪声地板（按 n 分层）
    noise = []
    for lo, hi, lab in [(5, 5, "n=5"), (6, 7, "n=6-7"), (8, 10, "n=8-10"), (11, 15, "n=11-15"),
                        (16, 25, "n=16-25"), (26, 10**9, "n>=26")]:
        m = ok & (cs.n >= lo) & (cs.n <= hi)
        if m.sum() < 50:
            continue
        noise.append(dict(bucket=lab, cells=int(m.sum()),
                          se_loss_median=float(np.median(cs.se_loss.to_numpy()[m])),
                          sd_b_loss=float(np.std(cs.b_loss.to_numpy()[m])),
                          sd_b_acc=float(np.std(cs.b_acc.to_numpy()[m])),
                          se_acc_median=float(np.median(cs.se_acc.to_numpy()[m]))))
    summary["noise_by_n"] = noise

    # ---- 方差分解
    dec = []
    v_loss = cs.b_loss.to_numpy()
    v_acc = cs.b_acc.to_numpy()
    dec.append(decompose(v_loss, cs.pcode.to_numpy(), cs.ccode.to_numpy(), len(puniq), len(cuniq),
                         float(np.nanmean(cs.se_loss.to_numpy() ** 2)), "b_loss (all)"))
    dec.append(decompose(v_acc, cs.pcode.to_numpy(), cs.ccode.to_numpy(), len(puniq), len(cuniq),
                         float(np.nanmean(cs.se_acc.to_numpy() ** 2)), "b_acc (all)"))

    # 只保留噪声较小的 cell（n>=8）再分解一次
    for minn in (8, 12):
        m = cs.n.to_numpy() >= minn
        vv = np.where(m, v_loss, np.nan)
        dec.append(decompose(vv, cs.pcode.to_numpy(), cs.ccode.to_numpy(), len(puniq), len(cuniq),
                             float(np.nanmean(np.where(m, cs.se_loss.to_numpy() ** 2, np.nan))),
                             f"b_loss (n>={minn})"))
        vv = np.where(m, v_acc, np.nan)
        dec.append(decompose(vv, cs.pcode.to_numpy(), cs.ccode.to_numpy(), len(puniq), len(cuniq),
                             float(np.nanmean(np.where(m, cs.se_acc.to_numpy() ** 2, np.nan))),
                             f"b_acc (n>={minn})"))
    summary["decomposition"] = dec

    # ---- b_loss 有多少是 ACC 水平的确定性后果？
    # 理论关系 b_loss ≈ -b_acc/(1-ACC)；直接看 b_loss 对 log(1/(1-ACC)) 的回归
    z = np.log(1.0 / (1.0 - cs.mean_acc.to_numpy()))
    m = ok & np.isfinite(z) & (z > 0)
    A_ = np.vstack([z[m], np.ones(m.sum())]).T
    coef, *_ = np.linalg.lstsq(A_, v_loss[m], rcond=None)
    pred = A_ @ coef
    ss_res = float(((v_loss[m] - pred) ** 2).sum())
    ss_tot = float(((v_loss[m] - v_loss[m].mean()) ** 2).sum())
    summary["b_loss_vs_acc_level"] = {
        "n": int(m.sum()), "slope_on_log1over1mACC": float(coef[0]), "intercept": float(coef[1]),
        "r2": 1 - ss_res / ss_tot,
        "corr": float(np.corrcoef(z[m], v_loss[m])[0, 1])}
    summary["corr_b_loss_b_acc"] = float(np.corrcoef(v_loss[ok], v_acc[ok])[0, 1])
    summary["corr_b_loss_star"] = float(np.corrcoef(cs.star.to_numpy()[ok], v_loss[ok])[0, 1])
    summary["corr_b_acc_star"] = float(np.corrcoef(cs.star.to_numpy()[ok], v_acc[ok])[0, 1])
    summary["corr_b_loss_meanacc"] = float(np.corrcoef(cs.mean_acc.to_numpy()[ok], v_loss[ok])[0, 1])
    summary["corr_b_loss_span"] = float(np.corrcoef(cs.span.to_numpy()[ok], v_loss[ok])[0, 1])

    # ---- 谱面元数据能解释多少 b_loss 的方差？
    feats = {"star": cs.star.to_numpy(), "rate": cs.rate.to_numpy().astype(float),
             "log_notes": np.log(np.maximum(cs.countTotal.to_numpy(), 1.0)),
             "mean_acc": cs.mean_acc.to_numpy(),
             "log_attempts": np.log(np.maximum(cs.attempts.to_numpy(np.float64), 1.0))}
    for name, key in (("star_only", ["star"]), ("meta", ["star", "rate", "log_notes"]),
                      ("meta+level", ["star", "rate", "log_notes", "mean_acc"]),
                      ("meta+level+attempts", ["star", "rate", "log_notes", "mean_acc", "log_attempts"])):
        mm = ok.copy()
        for f in key:
            mm &= np.isfinite(feats[f])
        X = np.column_stack([feats[k][mm] for k in key] + [np.ones(int(mm.sum()))])
        yv = v_loss[mm]
        coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
        resid = yv - X @ coef
        summary.setdefault("meta_r2", {})[name] = {
            "n": int(mm.sum()), "r2": float(1 - (resid ** 2).sum() / ((yv - yv.mean()) ** 2).sum()),
            "coef": {k: float(c) for k, c in zip(key, coef[:-1])}}

    # ---- 按 star / rate
    star = cs.star.to_numpy()
    rows = []
    for lab, lo, hi in [("<3", 0, 3), ("3-4", 3, 4), ("4-5", 4, 5), ("5-6", 5, 6),
                        ("6-7", 6, 7), ("7+", 7, 99)]:
        m = ok & (star >= lo) & (star < hi)
        if m.sum() < 50:
            continue
        rows.append(dict(group=lab, cells=int(m.sum()),
                         b_loss_mean=float(cs.b_loss.to_numpy()[m].mean()),
                         b_loss_median=float(np.median(cs.b_loss.to_numpy()[m])),
                         b_acc_mean=float(cs.b_acc.to_numpy()[m].mean()) * 1e2,
                         mean_acc=float(cs.mean_acc.to_numpy()[m].mean()),
                         frac_neg=float((cs.b_loss.to_numpy()[m] < 0).mean())))
    by_star = pd.DataFrame(rows)
    by_star.to_csv(OUT / f"learning_rate_{tag}_by_star.csv", index=False)

    # ---- 按谱面 / 玩家聚合（只保留 cell 数够的）
    def agg(group_col, name, min_cells=10):
        gg = cs[ok].groupby(group_col).agg(
            cells=("b_loss", "size"), b_loss_mean=("b_loss", "mean"), b_loss_median=("b_loss", "median"),
            b_acc_mean=("b_acc", "mean"), mean_acc=("mean_acc", "mean"),
            star=("star", "median"), n_med=("n", "median")).reset_index()
        gg = gg[gg.cells >= min_cells]
        gg["b_loss_se"] = cs[ok].groupby(group_col)["b_loss"].std().reindex(gg[group_col]).to_numpy() / np.sqrt(gg.cells)
        gg.to_csv(OUT / f"learning_rate_{tag}_by_{name}.csv", index=False)
        return gg
    by_chart = agg("chart_id", "chart")
    by_player = agg("user_id", "player")

    # 谱面/玩家均值自身的离散度（用 cell 数加权的"真值"部分）
    summary["by_chart"] = {"charts": int(len(by_chart)),
                           "b_loss_mean_sd": float(by_chart.b_loss_mean.std()),
                           "b_acc_mean_sd_x100": float(by_chart.b_acc_mean.std()) * 1e2,
                           "q05_b_loss": float(by_chart.b_loss_mean.quantile(.05)),
                           "q95_b_loss": float(by_chart.b_loss_mean.quantile(.95))}
    summary["by_player"] = {"players": int(len(by_player)),
                            "b_loss_mean_sd": float(by_player.b_loss_mean.std()),
                            "b_acc_mean_sd_x100": float(by_player.b_acc_mean.std()) * 1e2,
                            "q05_b_loss": float(by_player.b_loss_mean.quantile(.05)),
                            "q95_b_loss": float(by_player.b_loss_mean.quantile(.95))}

    # ---- 谱面/玩家的"学习率"均值是否与可观测特征相关
    if len(by_chart) > 20:
        summary["by_chart"]["corr_mean_b_loss_star"] = float(
            np.corrcoef(by_chart.star.fillna(0), by_chart.b_loss_mean)[0, 1])
        summary["by_chart"]["corr_mean_b_loss_n_med"] = float(
            np.corrcoef(by_chart.n_med.fillna(0), by_chart.b_loss_mean)[0, 1])
    if len(by_player) > 20:
        summary["by_player"]["corr_mean_b_loss_meanacc"] = float(
            np.corrcoef(by_player.mean_acc, by_player.b_loss_mean)[0, 1])
        summary["by_player"]["corr_mean_b_acc_meanacc"] = float(
            np.corrcoef(by_player.mean_acc, by_player.b_acc_mean)[0, 1])

    (OUT / f"learning_rate_{tag}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

    try:
        make_figs(tag, cs, ok, by_chart, by_player)
    except Exception:
        import traceback; traceback.print_exc()
    return summary


def make_figs(tag, cs, ok, by_chart, by_player):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.axisbelow": True})
    FIG.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    v = cs.b_loss.to_numpy(); v = v[np.isfinite(v)]
    ax[0].hist(v[(v > -6) & (v < 6)], bins=160, color="#2c7fb8")
    ax[0].axvline(0, color="k", lw=1)
    ax[0].axvline(np.median(v), color="#c0392b", ls="--", label=f"median {np.median(v):.3f}")
    ax[0].set_xlabel("beta_loss = d(loss)/d ln(cum)"); ax[0].legend()
    ax[0].set_title(f"{tag}: per-cell learning rate (loss)")
    a = cs.b_acc.to_numpy() * 100; a = a[np.isfinite(a)]
    ax[1].hist(a[(a > -10) & (a < 10)], bins=160, color="#31a354")
    ax[1].axvline(0, color="k", lw=1)
    ax[1].axvline(np.median(a), color="#c0392b", ls="--", label=f"median {np.median(a):.3f}")
    ax[1].set_xlabel("beta_acc (pp per ln cum)"); ax[1].legend()
    ax[1].set_title(f"{tag}: per-cell learning rate (ACC)")
    z = np.log(1 / (1 - cs.mean_acc.to_numpy()))
    ax[2].hexbin(z[ok], cs.b_loss.to_numpy()[ok], gridsize=60, cmap="viridis", bins="log")
    ax[2].set_xlabel("log(1/(1-mean ACC))  ->  loss sensitivity")
    ax[2].set_ylabel("beta_loss")
    ax[2].set_title(f"{tag}: why loss-space rates spread")
    fig.tight_layout(); fig.savefig(FIG / f"lr_{tag}_1_rates.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(by_chart.b_loss_mean, bins=80, color="#756bb1")
    ax[0].set_xlabel("per-chart mean beta_loss"); ax[0].set_title(
        f"{tag}: {len(by_chart)} charts (>=10 cells)")
    ax[1].hist(by_player.b_loss_mean, bins=80, color="#d95f02")
    ax[1].set_xlabel("per-player mean beta_loss"); ax[1].set_title(
        f"{tag}: {len(by_player)} players (>=10 cells)")
    fig.tight_layout(); fig.savefig(FIG / f"lr_{tag}_2_groups.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    m = ok
    ax[0].scatter(cs.star.to_numpy()[m], cs.b_loss.to_numpy()[m], s=1, alpha=.03, color="#2c7fb8")
    ax[0].set_ylim(-6, 3); ax[0].set_xlabel("star"); ax[0].set_ylabel("beta_loss")
    ax[0].set_title(f"{tag}: beta_loss vs difficulty")
    ax[1].scatter(cs.mean_acc.to_numpy()[m], cs.b_loss.to_numpy()[m], s=1, alpha=.03, color="#31a354")
    ax[1].set_ylim(-6, 3); ax[1].set_xlabel("cell mean ACC"); ax[1].set_ylabel("beta_loss")
    ax[1].set_title(f"{tag}: beta_loss vs level")
    fig.tight_layout(); fig.savefig(FIG / f"lr_{tag}_3_scatter.png"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", action="append", required=True)
    ap.add_argument("--min-n", type=int, default=MIN_N)
    a = ap.parse_args()
    for t in a.tag:
        run(t, a.min_n)
