#!/usr/bin/env python
"""诊断：冻结谱面特征能解释多少"经验谱面难度"，以及这个目标本身的可信度。

目标（不是模型产物，直接由成绩算）：
    adj_ij      = loss_ij - 玩家 i 的平均 loss          # 去掉玩家水平
    chart_j     = mean_i(adj_ij)  over n_j 条 play      # 谱面侧残差难度
只保留 n_j >= --min-plays 的谱面（play 太少的谱面估计噪声盖过信号）。

两件事：
  1. **split-half 可信度**：把每张谱面的 play 随机劈成两半，各自算 chart_j，
     两半之间的相关系数经 Spearman-Brown 校正 = 全样本 chart_j 的可信度。
     任何"只看谱面特征"的模型，解释力的上限就是它（R2 <= reliability^2）。
  2. **ridge + 按谱面分组 K 折 R2**：每组特征能解释多少 chart_j 的方差。
     因为因变量有测量噪声，观察到的 R2 是"衰减后"的；除以可信度即近似真值。

    python scripts/diag_map_analyser_charteffect.py \\
        --plays data/processed/step1_1k.parquet \\
        --feat data/processed/map_analyser_1k.parquet \\
        --msd data/processed/minacalc_msd_1k.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CORE = ["stream", "chordstream", "jacks", "coordination", "density", "wildcard"]
GROUPS = {
    "minacalc7": [f"msd_{s}" for s in ("stream", "jumpstream", "handstream", "stamina",
                                       "jackspeed", "chordjack", "technical")],
    "ett": [f"ett_{v}_{s}" for v in ("068", "070", "072", "0723", "074", "075")
            for s in ("overall", "stream", "jumpstream", "handstream", "stamina",
                      "jackspeed", "chordjack", "technical")],
    "est": ["sunny_star", "sunny_variety", "sunny_acc_scalar", "sunny_spikiness",
            "sunny_switches", "daniel_star", "azusa_star", "azusa_numeric",
            "azusa_raw_numeric", "roxy_star", "roxy_raw_numeric", "interlude_star",
            "sunnywin_ln_star"],
    "patt": (["patt_hb_ratio", "patt_sv_amount", "patt_n_clusters", "patt_n_specific"]
             + [f"patt_{p}_{k}" for p in CORE
                for k in ("n", "amount_s", "bpm_max", "importance", "mult")]),
    "struct": ["od", "n_notes", "n_holds", "ln_ratio", "duration_s", "n_rows",
               "chord_ratio", "notes_per_sec", "hold_ms_per_note", "bpm_median",
               "bpm_max", "n_timing_points", "n_breaks", "break_s"],
}


def ridge_cv(X, y, folds, lams=(0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)):
    """按 fold 分组的 K 折 ridge；返回 (best_lambda, best_R2)。"""
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = (X - mu) / sd
    yc = y - y.mean()
    best = (-np.inf, None)
    for lam in lams:
        preds = np.zeros_like(yc)
        for f in range(int(folds.max()) + 1):
            te = folds == f
            tr = ~te
            A = Z[tr].T @ Z[tr] + lam * np.eye(Z.shape[1])
            w = np.linalg.solve(A, Z[tr].T @ yc[tr])
            preds[te] = Z[te] @ w
        r2 = 1.0 - float(((yc - preds) ** 2).sum() / (yc ** 2).sum())
        if r2 > best[0]:
            best = (r2, lam)
    return best[1], best[0]


def split_half_reliability(plays: pd.DataFrame, rng, min_plays: int) -> dict:
    """chart effect 的 split-half 可信度（Spearman-Brown 校正到全样本）。

    plays 需要 beatmap_id / adj 两列。每张谱面的 play 随机排序后对半劈开，两半各算
    一次 chart effect；两半之间的相关 r 经 2r/(1+r) 校正 = 全样本 chart effect 的可信度。
    """
    out = {}
    n = plays.groupby("beatmap_id")["adj"].transform("size")
    sub = plays[n >= 2 * min_plays]
    if sub.beatmap_id.nunique() < 50:
        return out
    shuffled = sub.sample(frac=1.0, random_state=int(rng.integers(1 << 30)))
    cnt = shuffled.groupby("beatmap_id")["adj"].transform("size")
    rank = shuffled.groupby("beatmap_id").cumcount()
    half = cnt // 2
    ea = shuffled[rank < half].groupby("beatmap_id")["adj"].mean()
    eb = shuffled[rank >= half].groupby("beatmap_id")["adj"].mean()
    r = float(ea.corr(eb))
    sb = 2 * r / (1 + r)
    out.update({"n_charts": int(ea.size), "r_half": round(r, 4),
                "reliability_sb": round(sb, 4), "r2_ceiling": round(sb ** 2, 4)})
    return out


def main():
    ap = argparse.ArgumentParser(description="特征组 -> 经验谱面难度的解释力与可信度上限")
    ap.add_argument("--plays", required=True, help="step1 的 play 表（需 split 列）")
    ap.add_argument("--feat", required=True, help="map_analyser 特征 parquet")
    ap.add_argument("--msd", default=None, help="minacalc MSD parquet（minacalc7 对照）")
    ap.add_argument("--min-plays", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.plays, columns=["player_id", "beatmap_id", "loss", "split"])
    df["adj"] = df.loss - df.groupby("player_id")["loss"].transform("mean")
    chart = df.groupby("beatmap_id")["adj"].agg(["mean", "size"]).rename(
        columns={"mean": "chart_effect", "size": "n"})
    print(f"[data] {len(df):,} plays / {len(chart):,} charts；"
          f"adj-loss sd {df.adj.std():.4f}")

    fold_charts = chart[chart.n >= args.min_plays]
    print(f"[data] n>={args.min_plays}: {len(fold_charts):,} charts "
          f"({fold_charts.n.sum():,} plays, {fold_charts.n.sum() / len(df):.1%})；"
          f"chart_effect sd {fold_charts.chart_effect.std():.4f}")

    rng = np.random.default_rng(0)
    rel = split_half_reliability(df[["beatmap_id", "adj"]], rng, args.min_plays)
    print(f"[reliability] split-half r={rel.get('r_half')} -> Spearman-Brown "
          f"{rel.get('reliability_sb')} (R2 ceiling {rel.get('r2_ceiling')})")

    feat = pd.read_parquet(args.feat).set_index("beatmap_id")
    if args.msd:
        feat = feat.join(pd.read_parquet(args.msd).set_index("beatmap_id"), rsuffix="_minacalc")
    joined = fold_charts.join(feat, how="inner")
    print(f"[join] {len(joined):,} charts")
    y = joined.chart_effect.to_numpy(np.float64)
    folds = rng.integers(0, args.folds, len(joined))

    out = {"n_plays": int(len(df)), "n_charts": int(len(chart)),
           "min_plays": args.min_plays, "n_charts_used": int(len(joined)),
           "adj_sd": float(df.adj.std()), "chart_effect_sd": float(y.std()),
           "reliability": rel, "groups": {}}
    for name, cols in GROUPS.items():
        cols = [c for c in cols if c in joined.columns]
        if not cols:
            print(f"[skip] {name}: 无对应列")
            continue
        X = joined[cols].to_numpy(np.float64)
        X = np.where(np.isfinite(X), X, np.nanmedian(X, axis=0))
        lam, r2 = ridge_cv(X, y, folds)
        out["groups"][name] = {"n_cols": len(cols), "lambda": lam, "r2": round(r2, 4)}
        print(f"{name:10s} d={len(cols):3d}  R2={r2:.4f}  (lambda={lam})")

    all_cols = [c for g in ("ett", "est", "patt", "struct") for c in GROUPS[g]
                if c in joined.columns]
    X = joined[all_cols].to_numpy(np.float64)
    X = np.where(np.isfinite(X), X, np.nanmedian(X, axis=0))
    lam, r2 = ridge_cv(X, y, folds)
    out["groups"]["all"] = {"n_cols": len(all_cols), "lambda": lam, "r2": round(r2, 4)}
    print(f"{'all':10s} d={len(all_cols):3d}  R2={r2:.4f}  (lambda={lam})")

    skip = {"chart_effect", "n"}
    # 只列覆盖率 >=50% 的列：稀疏列（如 roxy_numeric，95% 为空）在"有值的子集"上
    # 相关再高也没法当特征，列进 top 只会误导。
    corr = {c: float(joined[c].corr(joined.chart_effect)) for c in joined.columns
            if c not in skip and pd.api.types.is_numeric_dtype(joined[c])
            and joined[c].nunique() > 1 and joined[c].notna().mean() >= 0.5}
    top = sorted(corr.items(), key=lambda kv: -abs(kv[1]))[:20]
    out["top_corr"] = [{"col": c, "corr": round(v, 4)} for c, v in top]
    print("\n与经验难度相关性最强（|r| 前 20；正 = 该特征越大谱面越难）:")
    for c, v in top:
        print(f"  {c:28s} {v:+.4f}")

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
