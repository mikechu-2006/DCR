#!/usr/bin/env python
"""Compare D2V-L against RM-L / M1-L / Sunny / MSD on the same benchmark folds.

    python scripts/benchmark_d2v_compare.py \
        --oof results/benchmark/eval_d2v/oof.parquet \
        --model results/benchmark/models/D2V_L_full.pt \
        --train-signals results/benchmark/d2v_linear/train/D2V_L_signals.parquet \
        --out-dir results/benchmark/eval_d2v/compare
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, rel: str):
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-signals", required=True)
    ap.add_argument("--train-signal-col", default="cm3p_linear_extra")
    ap.add_argument("--manifest", default="data/processed/benchmark_manifest.parquet")
    ap.add_argument("--signals", default="data/processed/benchmark_signals.parquet")
    ap.add_argument("--out-dir", default="results/benchmark/eval_d2v/compare")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    bd = load_module("bd", "scripts/benchmark_breakdown.py")
    ev = load_module("ev", "scripts/benchmark_calibrate_eval.py")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    oof = pd.read_parquet(args.oof)
    man = pd.read_parquet(args.manifest)[["row_id", "bid"]]
    sig = pd.read_parquet(args.signals)[["row_id", "msd_overall"]]
    df = oof.merge(man, on="row_id", how="left").merge(sig, on="row_id", how="left")
    if df["msd_overall"].isna().any():
        raise SystemExit("msd_overall missing after merge")
    payload = torch.load(args.model, map_location="cpu", weights_only=False)
    train_ids = set(int(x) for x in payload["train_chart_ids"])
    df["seen"] = pd.to_numeric(df["bid"], errors="coerce").isin(train_ids)
    df["expected_band"] = pd.cut(df["expected"], [-np.inf, 11, 15, np.inf],
                                 labels=["low", "mid", "high"], right=False)
    q = pd.qcut(df["msd_overall"], 4, labels=False, duplicates="drop")
    df["msd_band"] = "Q" + (q.astype(int) + 1).astype(str)

    methods = [
        ("D2V-L", "D2V_L_isotonic"),
        ("RM-L", "RM_L_isotonic"),
        ("M1-L", "M1_L_isotonic"),
        ("Sunny", "sunny_iso"),
        ("MSD", "msd_iso"),
    ]
    methods = [(lab, col) for lab, col in methods if col in df.columns]
    if not any(lab == "D2V-L" for lab, _ in methods):
        raise SystemExit(f"{args.oof}: missing D2V_L_isotonic")

    subsets = {
        "all": df,
        "clean": df[~df.seen],
        "rc": df[df.family == "rc"],
        "clean_rc": df[(~df.seen) & (df.family == "rc")],
        "ln": df[df.family == "ln"],
        "course": df[df.family == "course"],
        "msd_ge32": df[df.msd_overall >= 32.0],
    }
    for b in ("low", "mid", "high"):
        subsets[f"expected_{b}"] = df[df.expected_band == b]
    for b in ("Q1", "Q2", "Q3", "Q4"):
        subsets[f"msd_{b}"] = df[df.msd_band == b]
    subsets = {k: v.reset_index(drop=True) for k, v in subsets.items() if len(v)}

    rows = []
    for name, sub in subsets.items():
        for lab, col in methods:
            m = ev.metrics(sub["expected"], sub[col])
            rows.append({"subset": name, "method": lab, "n": m["n"],
                         "mae": m["mae"], "rmse": m["rmse"],
                         "bias": m["bias"], "spearman": m["spearman"]})
    pd.DataFrame(rows).to_csv(out_dir / "subset_metrics.csv", index=False)

    comparisons = [
        ("D2V-L vs RM-L", "D2V_L_isotonic", "RM_L_isotonic"),
        ("D2V-L vs M1-L", "D2V_L_isotonic", "M1_L_isotonic"),
        ("D2V-L vs Sunny", "D2V_L_isotonic", "sunny_iso"),
        ("D2V-L vs MSD", "D2V_L_isotonic", "msd_iso"),
    ]
    pd_rows, h2h_rows = [], []
    for cname, ca, cb in comparisons:
        for sname, sub in subsets.items():
            if ca not in sub.columns or cb not in sub.columns:
                continue
            delta, lo, hi = bd.paired_bootstrap(
                sub, sub[ca].to_numpy(float), sub[cb].to_numpy(float),
                args.bootstrap, args.seed)
            pd_rows.append({"comparison": cname, "subset": sname, "n": len(sub),
                            "delta_mae": delta, "ci_lo": lo, "ci_hi": hi})
            y = sub["expected"].to_numpy(float)
            ea = np.abs(y - sub[ca].to_numpy(float))
            eb = np.abs(y - sub[cb].to_numpy(float))
            h2h_rows.append({"comparison": cname, "subset": sname, "n": len(sub),
                             "a_wins": int((ea < eb).sum()),
                             "b_wins": int((ea > eb).sum()),
                             "ties": int((ea == eb).sum()),
                             "a_mae": float(ea.mean()), "b_mae": float(eb.mean())})
    pd.DataFrame(pd_rows).to_csv(out_dir / "paired_delta.csv", index=False)
    pd.DataFrame(h2h_rows).to_csv(out_dir / "head_to_head.csv", index=False)

    # Optional range diagnostic for the D2V raw d_m.
    rng_rows = []
    train_sig = pd.read_parquet(args.train_signals)
    if args.train_signal_col not in train_sig.columns:
        cand = [c for c in train_sig.columns if c != "beatmap_id"]
        args.train_signal_col = cand[0]
    tr = train_sig.loc[train_sig["beatmap_id"].isin(train_ids),
                       args.train_signal_col].to_numpy(float)
    be = df["D2V_L_d"].to_numpy(float)
    lo, hi = float(tr.min()), float(tr.max())
    p1, p99 = (float(x) for x in np.percentile(tr, [1, 99]))
    for name, arr in (("train", tr), ("benchmark", be)):
        for p in (1, 5, 50, 95, 99):
            rng_rows.append({"series": name, "stat": f"p{p}",
                             "value": float(np.percentile(arr, p))})
        rng_rows += [{"series": name, "stat": "min", "value": float(arr.min())},
                     {"series": name, "stat": "max", "value": float(arr.max())}]
    masks = {
        "above_max": be > hi,
        "above_p99": be > p99,
        "outside_minmax": (be < lo) | (be > hi),
        "outside_p1p99": (be < p1) | (be > p99),
    }
    for name, mask in masks.items():
        rng_rows.append({"series": "benchmark", "stat": f"share_{name}",
                         "value": float(mask.mean())})
        if mask.any():
            rng_rows.append({"series": "benchmark", "stat": f"mae_{name}",
                             "value": float(np.mean(np.abs(
                                 df["expected"].to_numpy(float)[mask]
                                 - df["D2V_L_isotonic"].to_numpy(float)[mask])))})
    pd.DataFrame(rng_rows).to_csv(out_dir / "range_diagnostic.csv", index=False)

    lines = ["# D2V-L benchmark comparison", "",
             f"OOF: {args.oof}", "",
             f"Rows: {len(df)}; seen in training charts: {int(df.seen.sum())}; "
             f"clean: {int((~df.seen).sum())}", ""]
    for title, name in (
            ("Subset metrics", "subset_metrics.csv"),
            ("Paired deltas (positive = D2V-L worse)", "paired_delta.csv"),
            ("Head to head", "head_to_head.csv"),
            ("D2V d_m range", "range_diagnostic.csv")):
        tab = pd.read_csv(out_dir / name)
        lines += [f"## {title}", "", md_table(tab), ""]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
