#!/usr/bin/env python
"""Breakdown / sensitivity report for the VSRG benchmark RM-L run.

Reads the combined OOF table produced by scripts/benchmark_calibrate_eval.py
and writes RC, family, overlap, difficulty-band, range and paired-delta tables.

    python scripts/benchmark_breakdown.py \
        --oof results/benchmark/eval_rm/oof.parquet \
        --manifest data/processed/benchmark_manifest.parquet \
        --signals data/processed/benchmark_signals.parquet \
        --model results/benchmark/models/RM_L_full.pt \
        --train-signals results/benchmark/rm_linear/train/RM_L_signals.parquet \
        --out-dir results/benchmark/eval_rm/breakdown

The calibration itself is NOT refit here: the OOF columns are already
cross-fitted on all 1,382 rows.  Subsets only slice the evaluation rows.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_eval_module():
    path = ROOT / "scripts/benchmark_calibrate_eval.py"
    spec = importlib.util.spec_from_file_location("bench_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def group_bootstrap(df, methods, n_boot=1000, seed=0):
    """Chart-group bootstrap of MAE; methods maps label -> prediction array."""
    rng = np.random.default_rng(seed)
    groups = df["chart_group"].astype(str).to_numpy()
    uniq = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in uniq}
    y = df["expected"].to_numpy(dtype=float)
    vals = {k: [] for k in methods}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[g] for g in pick])
        yy = y[idx]
        for k, p in methods.items():
            vals[k].append(float(np.mean(np.abs(yy - p[idx]))))
    return {k: (float(np.mean(v)), float(np.percentile(v, 2.5)),
                float(np.percentile(v, 97.5))) for k, v in vals.items()}


def paired_bootstrap(df, a, b, n_boot=1000, seed=0):
    """Group bootstrap of mean(|y-a| - |y-b|); positive => a is worse."""
    rng = np.random.default_rng(seed)
    groups = df["chart_group"].astype(str).to_numpy()
    uniq = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in uniq}
    y = df["expected"].to_numpy(dtype=float)
    d = np.abs(y - a) - np.abs(y - b)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[g] for g in pick])
        vals.append(float(np.mean(d[idx])))
    return float(np.mean(d)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def md_table(df, floatfmt=".4f"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append(f"{float(v):{floatfmt}}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def metrics_row(df, col, subset, label, mod):
    rec = {"subset": subset, "method": label, "n": int(len(df))}
    if col not in df.columns:
        return None
    m = mod.metrics(df["expected"], df[col])
    for k in ("mae", "rmse", "bias", "spearman"):
        rec[k] = m[k]
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof", required=True)
    ap.add_argument("--manifest", default="data/processed/benchmark_manifest.parquet")
    ap.add_argument("--signals", default="data/processed/benchmark_signals.parquet")
    ap.add_argument("--model", required=True,
                    help="RM_L checkpoint; train_chart_ids defines the overlap mask")
    ap.add_argument("--train-signals", required=True,
                    help="training signal parquet (20,388 charts) for the range check")
    ap.add_argument("--train-signal-col", default="cm3p_linear_extra")
    ap.add_argument("--out-dir", default="results/benchmark/eval_rm/breakdown")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mod = load_eval_module()

    oof = pd.read_parquet(args.oof)
    manifest = pd.read_parquet(args.manifest)
    signals = pd.read_parquet(args.signals)
    payload = torch.load(args.model, map_location="cpu", weights_only=False)
    train_ids = set(int(x) for x in payload["train_chart_ids"])
    train_sig = pd.read_parquet(args.train_signals)

    df = oof.merge(manifest[["row_id", "bid"]], on="row_id", how="left")
    df = df.merge(signals[["row_id", "msd_overall"]], on="row_id", how="left")
    if df["msd_overall"].isna().any():
        raise SystemExit("msd_overall has missing rows after merge")
    df["bid_num"] = pd.to_numeric(df["bid"], errors="coerce")
    df["seen"] = df["bid_num"].isin(train_ids)
    df["expected_band"] = pd.cut(df["expected"], [-np.inf, 11, 15, np.inf],
                                 labels=["low", "mid", "high"], right=False)
    q = pd.qcut(df["msd_overall"], 4, labels=False, duplicates="drop")
    df["msd_band"] = "Q" + (q.astype(int) + 1).astype(str)

    for c in ("RM_L_isotonic", "RM_L_d", "M1_L_isotonic", "sunny_iso", "msd_iso"):
        if c not in df.columns:
            raise SystemExit(f"{args.oof}: missing column {c}")

    methods = [
        ("RM-L", "RM_L_isotonic"),
        ("M1-L", "M1_L_isotonic"),
        ("Sunny", "sunny_iso"),
        ("MSD", "msd_iso"),
    ]
    methods = [(lab, col) for lab, col in methods if col in df.columns]

    subsets = {
        "all": df,
        "clean": df[~df.seen],
        "rc": df[df.family == "rc"],
        "clean_rc": df[(~df.seen) & (df.family == "rc")],
        "ln": df[df.family == "ln"],
        "course": df[df.family == "course"],
        "msd_hard": df[df.msd_overall >= 32.0],
    }
    subsets = {k: v.reset_index(drop=True) for k, v in subsets.items() if len(v)}

    # ---------------------------------------------------------- subset metrics
    rows = []
    for name, sub in subsets.items():
        for lab, col in methods:
            rec = metrics_row(sub, col, name, lab, mod)
            if rec is not None:
                rows.append(rec)
    pd.DataFrame(rows).to_csv(out_dir / "subset_metrics.csv", index=False)

    # --------------------------------------------------------------- RC table
    rc_rows = []
    for name in ("rc", "clean_rc"):
        sub = subsets.get(name)
        if sub is None:
            continue
        ci = group_bootstrap(sub, {lab: sub[col].to_numpy(float)
                                   for lab, col in methods},
                             args.bootstrap, args.seed)
        for lab, col in methods:
            m = mod.metrics(sub["expected"], sub[col])
            rc_rows.append({"subset": name, "method": lab, "n": m["n"],
                            "mae": m["mae"], "mae_ci_lo": ci[lab][1],
                            "mae_ci_hi": ci[lab][2], "spearman": m["spearman"]})
    pd.DataFrame(rc_rows).to_csv(out_dir / "rc_summary.csv", index=False)

    # ----------------------------------------------------------- family table
    fam_rows = []
    for name in ("rc", "ln", "course", "clean"):
        sub = subsets.get(name)
        if sub is None:
            continue
        for lab, col in methods:
            m = mod.metrics(sub["expected"], sub[col])
            fam_rows.append({"subset": name, "method": lab, "n": m["n"],
                             "mae": m["mae"], "rmse": m["rmse"],
                             "spearman": m["spearman"]})
    pd.DataFrame(fam_rows).to_csv(out_dir / "family_summary.csv", index=False)

    # -------------------------------------------------------- overlap table
    ov_rows = []
    for name in ("all", "clean"):
        sub = subsets.get(name)
        if sub is None:
            continue
        ci = group_bootstrap(sub, {lab: sub[col].to_numpy(float)
                                   for lab, col in methods},
                             args.bootstrap, args.seed)
        m_rm = mod.metrics(sub["expected"], sub["RM_L_isotonic"])
        m_m1 = mod.metrics(sub["expected"], sub["M1_L_isotonic"])
        delta, lo, hi = paired_bootstrap(sub, sub["RM_L_isotonic"].to_numpy(float),
                                         sub["M1_L_isotonic"].to_numpy(float),
                                         args.bootstrap, args.seed)
        for lab, col in methods:
            m = mod.metrics(sub["expected"], sub[col])
            ov_rows.append({"subset": name, "method": lab, "n": m["n"],
                            "mae": m["mae"], "mae_ci_lo": ci[lab][1],
                            "mae_ci_hi": ci[lab][2],
                            "delta_vs_M1_L": (m["mae"] - m_m1["mae"]),
                            "delta_ci_lo": lo if lab == "RM-L" else np.nan,
                            "delta_ci_hi": hi if lab == "RM-L" else np.nan})
    pd.DataFrame(ov_rows).to_csv(out_dir / "overlap_sensitivity.csv", index=False)

    # ------------------------------------------------------ difficulty bands
    band_rows = []
    for band_key, band_col in (("expected", "expected_band"), ("msd", "msd_band")):
        for band in df[band_col].dropna().unique():
            sub = df[df[band_col] == band].reset_index(drop=True)
            for lab, col in methods:
                m = mod.metrics(sub["expected"], sub[col])
                band_rows.append({"band_type": band_key, "band": str(band),
                                  "method": lab, "n": m["n"], "mae": m["mae"],
                                  "spearman": m["spearman"]})
    sub = df[df.msd_overall >= 32.0].reset_index(drop=True)
    for lab, col in methods:
        m = mod.metrics(sub["expected"], sub[col])
        band_rows.append({"band_type": "msd_fixed", "band": ">=32",
                          "method": lab, "n": m["n"], "mae": m["mae"],
                          "spearman": m["spearman"]})
    pd.DataFrame(band_rows).to_csv(out_dir / "difficulty_band_summary.csv",
                                   index=False)

    # -------------------------------------------------------- paired deltas
    pd_rows = []
    comparisons = [
        ("RM-L vs M1-L", "RM_L_isotonic", "M1_L_isotonic"),
        ("RM-L vs Sunny", "RM_L_isotonic", "sunny_iso"),
        ("RM-L vs MSD", "RM_L_isotonic", "msd_iso"),
    ]
    for cname, ca, cb in comparisons:
        for sname in ("all", "clean", "rc", "clean_rc", "msd_hard"):
            sub = subsets.get(sname)
            if sub is None or ca not in sub.columns or cb not in sub.columns:
                continue
            delta, lo, hi = paired_bootstrap(sub, sub[ca].to_numpy(float),
                                             sub[cb].to_numpy(float),
                                             args.bootstrap, args.seed)
            pd_rows.append({"comparison": cname, "subset": sname, "n": len(sub),
                            "delta_mae": delta, "delta_ci_lo": lo,
                            "delta_ci_hi": hi})
    pd.DataFrame(pd_rows).to_csv(out_dir / "paired_delta.csv", index=False)

    # ------------------------------------------- paired deltas by band
    band_pd = []
    for band_key, band_col in (("expected", "expected_band"), ("msd", "msd_band")):
        for band in df[band_col].dropna().unique():
            sub = df[df[band_col] == band].reset_index(drop=True)
            for cname, ca, cb in comparisons:
                delta, lo, hi = paired_bootstrap(sub, sub[ca].to_numpy(float),
                                                 sub[cb].to_numpy(float),
                                                 args.bootstrap, args.seed)
                band_pd.append({"band_type": band_key, "band": str(band),
                                "comparison": cname, "n": len(sub),
                                "delta_mae": delta, "delta_ci_lo": lo,
                                "delta_ci_hi": hi})
    sub = df[df.msd_overall >= 32.0].reset_index(drop=True)
    for cname, ca, cb in comparisons:
        delta, lo, hi = paired_bootstrap(sub, sub[ca].to_numpy(float),
                                         sub[cb].to_numpy(float),
                                         args.bootstrap, args.seed)
        band_pd.append({"band_type": "msd_fixed", "band": ">=32",
                        "comparison": cname, "n": len(sub),
                        "delta_mae": delta, "delta_ci_lo": lo,
                        "delta_ci_hi": hi})
    pd.DataFrame(band_pd).to_csv(out_dir / "paired_delta_by_band.csv", index=False)

    # --------------------------------------------------------- head to head
    h2h_rows = []
    for cname, ca, cb in comparisons:
        for sname in ("all", "clean", "rc", "clean_rc", "msd_hard"):
            sub = subsets.get(sname)
            if sub is None or ca not in sub.columns or cb not in sub.columns:
                continue
            y = sub["expected"].to_numpy(float)
            ea = np.abs(y - sub[ca].to_numpy(float))
            eb = np.abs(y - sub[cb].to_numpy(float))
            h2h_rows.append({"comparison": cname, "subset": sname, "n": len(sub),
                             "a_wins": int((ea < eb).sum()),
                             "b_wins": int((ea > eb).sum()),
                             "ties": int((ea == eb).sum()),
                             "a_mae": float(ea.mean()),
                             "b_mae": float(eb.mean())})
    pd.DataFrame(h2h_rows).to_csv(out_dir / "head_to_head.csv", index=False)

    # ---------------------------------------------------------- range check
    if args.train_signal_col not in train_sig.columns:
        candidates = [c for c in train_sig.columns if c != "beatmap_id"]
        if not candidates:
            raise SystemExit(f"{args.train_signals}: no signal column")
        args.train_signal_col = candidates[0]
    tr = train_sig.loc[train_sig["beatmap_id"].isin(train_ids),
                       args.train_signal_col].to_numpy(float)
    be = df["RM_L_d"].to_numpy(float)
    lo, hi = float(tr.min()), float(tr.max())
    p1, p99 = (float(x) for x in np.percentile(tr, [1, 99]))
    masks = {
        "below_min": be < lo,
        "above_max": be > hi,
        "outside_minmax": (be < lo) | (be > hi),
        "below_p1": be < p1,
        "above_p99": be > p99,
        "outside_p1p99": (be < p1) | (be > p99),
    }
    rng_rows = []
    for name, arr in (("train", tr), ("benchmark", be)):
        for q in (1, 5, 50, 95, 99):
            rng_rows.append({"series": name, "stat": f"p{q}",
                             "value": float(np.percentile(arr, q))})
        rng_rows += [{"series": name, "stat": "min", "value": float(arr.min())},
                     {"series": name, "stat": "max", "value": float(arr.max())}]
    for name, mask in masks.items():
        rng_rows.append({"series": "benchmark", "stat": f"share_{name}",
                         "value": float(mask.mean())})
        if mask.any():
            rng_rows.append({"series": "benchmark", "stat": f"mae_{name}",
                             "value": float(np.mean(np.abs(
                                 df["expected"].to_numpy(float)[mask]
                                 - df["RM_L_isotonic"].to_numpy(float)[mask])))})
            rng_rows.append({"series": "benchmark", "stat": f"n_{name}",
                             "value": int(mask.sum())})
    pd.DataFrame(rng_rows).to_csv(out_dir / "range_diagnostic.csv", index=False)

    # ------------------------------------------------------------- summary.md
    def read(name):
        p = out_dir / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    lines = ["# RM-L breakdown / sensitivity", "",
             f"OOF: {args.oof}", "",
             f"Rows: {len(df)}; seen in randommerge training: {int(df.seen.sum())}; "
             f"clean: {int((~df.seen).sum())}", ""]
    for title, name in (
            ("Subset metrics", "subset_metrics.csv"),
            ("RC", "rc_summary.csv"),
            ("Family", "family_summary.csv"),
            ("Overlap sensitivity", "overlap_sensitivity.csv"),
            ("Difficulty bands", "difficulty_band_summary.csv"),
            ("Paired deltas (positive = RM-L worse)", "paired_delta.csv"),
            ("Paired deltas by band", "paired_delta_by_band.csv"),
            ("Head to head", "head_to_head.csv")):
        tab = read(name)
        if not tab.empty:
            lines += [f"## {title}", "", md_table(tab), ""]
    rng_tab = read("range_diagnostic.csv")
    if not rng_tab.empty:
        lines += ["## d_m range diagnostic", "", md_table(rng_tab, ".6f"), ""]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
