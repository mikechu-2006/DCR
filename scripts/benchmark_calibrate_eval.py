#!/usr/bin/env python
"""5-fold chart-group LOCV calibration and benchmark metrics.

Primary calibration: isotonic regression (monotone, no equal-Dan-spacing
assumption), fitted on 4 folds and applied out-of-fold.  Linear and constant
calibrations are reported as controls.

    python scripts/benchmark_calibrate_eval.py \
        --pred-dir results/benchmark/pred \
        --out-dir results/benchmark/eval
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

RUNS = {
    "RM_L": {"display": "DCR-CM3P-RM-Linear",
             "method": "CM3P + external linear (randommerge)",
             "arm": "legacy"},
    "M1_L": {"display": "DCR-CM3P-XMLP-Legacy", "method": "CM3P + external MLP",
             "arm": "legacy"},
    "M1_C": {"display": "DCR-CM3P-XMLP-Canon", "method": "CM3P + external MLP",
             "arm": "canon"},
    "M2_L": {"display": "DCR-CM3P-MLP-Legacy", "method": "CM3P MLP h64 L2",
             "arm": "legacy"},
    "M2_C": {"display": "DCR-CM3P-MLP-Canon", "method": "CM3P MLP h64 L2",
             "arm": "canon"},
}


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def metrics(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    rho = spearmanr(y, p).statistic if len(y) > 2 else np.nan
    tau = kendalltau(y, p).statistic if len(y) > 2 else np.nan
    return {
        "n": int(len(y)),
        "mae": mae(y, p),
        "rmse": rmse(y, p),
        "bias": float(np.mean(y - p)),
        "r2": float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
        if np.var(y) > 0 else np.nan,
        "spearman": float(rho) if np.isfinite(rho) else None,
        "kendall": float(tau) if np.isfinite(tau) else None,
    }


def make_folds(df, n_splits=5, seed=0):
    y = df["expected"].to_numpy(dtype=float)
    # 5 quantile bins; duplicates='drop' keeps this robust to the 14.0 mass
    try:
        bins = pd.qcut(y, q=n_splits, labels=False, duplicates="drop")
    except ValueError:
        bins = np.zeros(len(y), dtype=int)
    strata = df["family"].astype(str) + "_" + pd.Series(bins).astype(str)
    if strata.value_counts().min() < n_splits:
        strata = df["family"].astype(str)
    if strata.value_counts().min() < n_splits:
        return None, GroupKFold(n_splits=n_splits)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return strata.to_numpy(), splitter


def fit_oof(df, column, kind, folds, splitter, strata):
    y = df["expected"].to_numpy(dtype=float)
    x = df[column].to_numpy(dtype=float)
    oof = np.full(len(df), np.nan)
    for tr, te in splitter.split(x.reshape(-1, 1), strata, groups=df["chart_group"]):
        if kind == "constant":
            pred = np.full(len(te), float(np.mean(y[tr])))
        elif kind == "isotonic":
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(x[tr], y[tr])
            pred = ir.predict(x[te])
        elif kind == "linear":
            lr = LinearRegression()
            lr.fit(x[tr].reshape(-1, 1), y[tr])
            pred = lr.predict(x[te].reshape(-1, 1))
        else:
            raise ValueError(kind)
        oof[te] = pred
    return oof


def fit_external_oof(df, cols, folds, splitter, strata):
    y = df["expected"].to_numpy(dtype=float)
    X = df[cols].to_numpy(dtype=float)
    oof = np.full(len(df), np.nan)
    for tr, te in splitter.split(X, strata, groups=df["chart_group"]):
        lr = LinearRegression()
        lr.fit(X[tr], y[tr])
        oof[te] = lr.predict(X[te])
    return oof


def bootstrap(df, methods, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    groups = df["chart_group"].astype(str).to_numpy()
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    y = df["expected"].to_numpy(dtype=float)
    vals = {k: [] for k in methods}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in pick])
        yy = y[idx]
        for k in methods:
            vals[k].append(mae(yy, methods[k][idx]))
    out = {}
    for k, v in vals.items():
        v = np.asarray(v)
        out[k] = {"mae_mean": float(v.mean()),
                  "mae_ci_lo": float(np.percentile(v, 2.5)),
                  "mae_ci_hi": float(np.percentile(v, 97.5))}
    return out


def per_group(df, col, pred_col):
    rows = []
    for key in ["family", "pattern"]:
        for val, sub in df.groupby(key, dropna=False):
            rows.append({"group": key, "value": str(val), **metrics(
                sub["expected"].to_numpy(), sub[pred_col].to_numpy())})
    sub = df.assign(_band=pd.cut(df["expected"], [-np.inf, 11, 15, np.inf],
                                 labels=["low", "mid", "high"], right=False))
    for val, s in sub.groupby("_band", observed=False):
        if len(s):
            rows.append({"group": "difficulty", "value": str(val), **metrics(
                s["expected"].to_numpy(), s[pred_col].to_numpy())})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", default="results/benchmark/pred")
    ap.add_argument("--manifest", default="data/processed/benchmark_manifest.parquet")
    ap.add_argument("--signals", default="data/processed/benchmark_signals.parquet")
    ap.add_argument("--benchmark", default=".cache/benchmark/VSRG-DanEstimation-Benchmark")
    ap.add_argument("--out-dir", default="results/benchmark/eval")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--runs", default="M1_L,M1_C,M2_L,M2_C",
                    help="comma-separated RUNS keys to evaluate; the default "
                         "reproduces the original four-combination eval")
    args = ap.parse_args(argv)

    run_keys = [s.strip() for s in args.runs.split(",") if s.strip()]
    unknown = [k for k in run_keys if k not in RUNS]
    if unknown:
        raise SystemExit(f"--runs: unknown keys {unknown}; choose from {list(RUNS)}")
    if not run_keys:
        raise SystemExit("--runs is empty")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(args.manifest)
    signals = pd.read_parquet(args.signals)

    frames = []
    for key in run_keys:
        path = Path(args.pred_dir) / f"{key}.parquet"
        if not path.exists():
            raise SystemExit(f"missing prediction {path}")
        d = pd.read_parquet(path)[["row_id", "d_m"]].rename(columns={"d_m": key})
        frames.append(d)
    preds = frames[0]
    for f in frames[1:]:
        preds = preds.merge(f, on="row_id", how="inner")
    df = manifest.merge(signals, on="row_id", how="left", suffixes=("", "_sig"))
    df = df.merge(preds, on="row_id", how="inner")
    print(f"[data] {len(df)} rows with all {len(run_keys)} run predictions", flush=True)

    for col in ("sunny_star", "msd_overall"):
        if df[col].isna().any():
            raise SystemExit(f"{col} has {int(df[col].isna().sum())} missing rows")

    strata, splitter = make_folds(df, n_splits=args.folds, seed=args.seed)
    folds = np.full(len(df), -1, dtype=int)
    for k, (_, te) in enumerate(splitter.split(
            df[[c for c in run_keys]].to_numpy(), strata, groups=df["chart_group"])):
        folds[te] = k
    df["fold"] = folds
    if (folds < 0).any():
        raise SystemExit("fold assignment incomplete")
    print("[folds] " + str(pd.Series(folds).value_counts().sort_index().to_dict()), flush=True)

    # calibrations
    oof = pd.DataFrame({"row_id": df["row_id"], "expected": df["expected"],
                        "family": df["family"], "pattern": df["pattern"],
                        "chart_group": df["chart_group"], "fold": df["fold"]})
    results = {}
    for key in run_keys:
        oof[key + "_d"] = df[key].to_numpy()
        for kind in ("isotonic", "linear"):
            p = fit_oof(df, key, kind, folds, splitter, strata)
            oof[f"{key}_{kind}"] = p
            results[f"{key}_{kind}"] = metrics(df["expected"], p)
        results[f"{key}_raw"] = metrics(df["expected"], df[key])
    # baselines
    for name, col, kind in (
            ("constant", None, "constant"),
            ("sunny_iso", "sunny_star", "isotonic"),
            ("sunny_lin", "sunny_star", "linear"),
            ("msd_iso", "msd_overall", "isotonic"),
            ("msd_lin", "msd_overall", "linear")):
        if col is None:
            p = np.full(len(df), np.nan)
            y = df["expected"].to_numpy()
            for tr, te in splitter.split(df[["sunny_star"]].to_numpy(), strata,
                                          groups=df["chart_group"]):
                p[te] = y[tr].mean()
        else:
            p = fit_oof(df, col, kind, folds, splitter, strata)
        oof[name] = p
        results[name] = metrics(df["expected"], p)
    p_ext = fit_external_oof(df, ["sunny_star", "msd_overall"], folds, splitter, strata)
    oof["external_lin"] = p_ext
    results["external_lin"] = metrics(df["expected"], p_ext)

    # bootstrap CIs and paired deltas vs M1_L isotonic
    methods = {k: df[k].to_numpy() for k in run_keys}
    methods.update({k: oof[k].to_numpy() for k in
                    ("constant", "sunny_iso", "msd_iso", "external_lin")})
    methods.update({f"{k}_isotonic": oof[f"{k}_isotonic"].to_numpy() for k in run_keys})
    boot = bootstrap(df, methods, n_boot=args.bootstrap, seed=args.seed)
    for k, v in boot.items():
        if k in results:
            results[k].update({"mae_ci_lo": v["mae_ci_lo"], "mae_ci_hi": v["mae_ci_hi"]})
    base = methods.get("M1_L_isotonic")
    if base is not None:
        for k in run_keys:
            p = oof[f"{k}_isotonic"].to_numpy()
            d = np.abs(df["expected"].to_numpy() - p) - np.abs(df["expected"].to_numpy() - base)
            rng = np.random.default_rng(args.seed)
            groups = df["chart_group"].astype(str).to_numpy()
            uniq = np.unique(groups)
            idx_by = {g: np.flatnonzero(groups == g) for g in uniq}
            ds = []
            for _ in range(max(1, args.bootstrap)):
                pick = rng.choice(uniq, len(uniq), replace=True)
                idx = np.concatenate([idx_by[g] for g in pick])
                ds.append(float(np.mean(d[idx])))
            results[f"{k}_isotonic"]["delta_vs_M1_L_mae"] = float(np.mean(d))
            results[f"{k}_isotonic"]["delta_vs_M1_L_ci_lo"] = float(np.percentile(ds, 2.5))
            results[f"{k}_isotonic"]["delta_vs_M1_L_ci_hi"] = float(np.percentile(ds, 97.5))

    oof.to_parquet(out_dir / "oof.parquet", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                          encoding="utf-8")

    # per-group table for the primary calibration
    rows = []
    for k in run_keys:
        for rec in per_group(oof, "family", f"{k}_isotonic"):
            rec["run"] = k
            rows.append(rec)
    pd.DataFrame(rows).to_csv(out_dir / "per_group.csv", index=False)

    # summary csv/md
    table = []
    for k, v in results.items():
        table.append({"method": k, **v})
    pd.DataFrame(table).to_csv(out_dir / "summary.csv", index=False)
    lines = ["# VSRG Dan benchmark — 5-fold chart-group LOCV", "",
             "Calibration: isotonic (primary), linear (control), out-of-fold.",
             "All rows use the same 5 chart-group folds.", "",
             "| method | MAE | RMSE | bias | Spearman | Kendall | MAE 95% CI |",
             "|---|---:|---:|---:|---:|---:|---|"]
    order = [f"{k}_isotonic" for k in run_keys] + [f"{k}_linear" for k in run_keys] + [
        "constant", "sunny_iso", "msd_iso", "external_lin"]
    for k in order:
        if k not in results:
            continue
        r = results[k]
        ci = (f"[{r['mae_ci_lo']:.4f}, {r['mae_ci_hi']:.4f}]"
              if r.get("mae_ci_lo") is not None else "")
        lines.append(f"| {k} | {r['mae']:.4f} | {r['rmse']:.4f} | {r['bias']:+.4f} | "
                     f"{r['spearman']:.4f} | {r['kendall']:.4f} | {ci} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)

    # published benchmark algorithms on the same rows
    pub_rows = []
    bench = Path(args.benchmark) / "results"
    for name in ("Sunny", "Daniel", "XDC1", "Koishi", "Azusa", "ISOR",
                 "Roxy", "Companella", "Mixed", "P1h"):
        path = bench / f"{name}.csv"
        if not path.exists():
            continue
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        raw["row_id"] = [f"b{i:04d}" for i in range(len(raw))]
        raw = raw[raw["row_id"].isin(df["row_id"])].copy()
        got = pd.to_numeric(raw["got"], errors="coerce")
        mask = got.notna()
        if mask.sum() == 0:
            continue
        rec = metrics(raw.loc[mask, "expected"].astype(float).to_numpy(),
                      got[mask].to_numpy())
        rec["algorithm"] = name
        rec["coverage"] = float(mask.mean())
        pub_rows.append(rec)
    pd.DataFrame(pub_rows).to_csv(out_dir / "published_baselines.csv", index=False)
    if pub_rows:
        lines += ["", "## Published benchmark algorithms (same rows, as-is)", "",
                  "| algorithm | n | coverage | MAE | RMSE | bias | Spearman |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for r in pub_rows:
            lines.append(f"| {r['algorithm']} | {r['n']} | {r['coverage']:.1%} | "
                         f"{r['mae']:.4f} | {r['rmse']:.4f} | {r['bias']:+.4f} | "
                         f"{r['spearman']:.4f} |")
        (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # candidate CSVs (OOF isotonic; documented as cross-fitted)
    cand_dir = out_dir / "candidates"
    cand_dir.mkdir(exist_ok=True)
    raw = pd.read_csv(Path(args.benchmark) / "samples/data.csv", dtype=str,
                      keep_default_na=False)
    raw["row_id"] = [f"b{i:04d}" for i in range(len(raw))]
    for k in run_keys:
        cfg = RUNS[k]
        got = oof.set_index("row_id")[f"{k}_isotonic"]
        out = raw.copy()
        got_vals = []
        for r in out["row_id"]:
            value = got.get(r, np.nan)
            got_vals.append(f"{float(value):.4f}" if np.isfinite(value) else "Missing")
        out["got"] = got_vals
        delta, dabs = [], []
        for e, g in zip(out["expected"], out["got"]):
            if g == "Missing":
                delta.append("")
                dabs.append("")
            else:
                d = float(e) - float(g)
                delta.append(f"{d:.4f}")
                dabs.append(f"{abs(d):.4f}")
        out["delta"] = delta
        out["deltaAbs"] = dabs
        out = out[["bid", "name", "pattern", "subPattern", "expected", "got",
                   "delta", "deltaAbs"]]
        out.to_csv(cand_dir / f"{cfg['display']}.csv", index=False)
    print(f"[done] {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    raise SystemExit(main())
