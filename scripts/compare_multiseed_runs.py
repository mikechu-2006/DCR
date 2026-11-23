#!/usr/bin/env python
"""compare_multiseed_runs -- paired comparison of two run_p2_p4_multiseed runs.

Only the four embedding-based methods are compared (the fixed per-chart
signals do not depend on the content table; they are reported separately as a
consistency check -- identical numbers across runs prove the split files and
the signal table are the same).

Usage:
    python scripts/compare_multiseed_runs.py \
        --baseline results/p4_multiseed \
        --variant  results/p4_multiseed_canon \
        --out      results/p4_multiseed_canon/vs_legacy.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EMB_METHODS = [
    ("cm3p_linear_extra", "CM3P + external 线性"),
    ("cm3p_mlp_extra", "CM3P + external MLP"),
    ("cm3p_linear", "CM3P 线性 (λ1e-3)"),
    ("cm3p_mlp_h64_L2", "CM3P MLP h64 L2"),
]
FIXED = [("sunny_star", "Sunny"), ("daniel_star", "Daniel"), ("star", "star")]
PROTOCOLS = [("p2", "P2（新 chart）"), ("p4", "P4（新玩家 + 新 chart）")]


def load(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "summary.csv"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    df = pd.read_csv(path)
    need = {"protocol", "seed", "method", "signal", "nll", "pearson", "spearman",
            "captured", "ceiling_nll"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{path}: missing columns {sorted(missing)}")
    return df


def fmt(mu: float, sd: float) -> str:
    return f"{mu:.5f} ± {sd:.5f}" if np.isfinite(mu) else "—"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    a_dir, b_dir = Path(args.baseline), Path(args.variant)
    a, b = load(a_dir), load(b_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for proto, _ in PROTOCOLS:
        ca = a[(a.protocol == proto) & (a.signal == "cm3p_linear")].set_index("seed")
        cb = b[(b.protocol == proto) & (b.signal == "cm3p_linear")].set_index("seed")
        if ca.empty or cb.empty:
            continue
        common = ca.index.intersection(cb.index)
        if len(common) and not np.allclose(ca.loc[common, "ceiling_nll"],
                                           cb.loc[common, "ceiling_nll"]):
            raise SystemExit(f"{proto}: ceiling NLL differs between runs; "
                             "the splits/settings are not comparable")

    rows, md = [], []
    md += [f"# content ablation: {b_dir.name} vs {a_dir.name}", "",
           "Paired over the same train/test split seeds (model seed fixed at 0). "
           "Δ < 0 means the variant has the lower (better) NLL.", ""]

    for proto, title in PROTOCOLS:
        if a[a.protocol == proto].empty or b[b.protocol == proto].empty:
            continue
        md += [f"## {title}", ""]
        for proto_rows in (rows,):
            pass
        md += ["| 方法 | baseline NLL | variant NLL | ΔNLL (variant−baseline) "
               "| seeds 变好 | ΔPearson | ΔSpearman |", "|---|---|---|---|---|---|---|"]
        for sig, label in EMB_METHODS:
            sa = a[(a.protocol == proto) & (a.signal == sig)].set_index("seed").sort_index()
            sb = b[(b.protocol == proto) & (b.signal == sig)].set_index("seed").sort_index()
            common = sa.index.intersection(sb.index)
            if len(common) == 0:
                continue
            d = (sb.loc[common, "nll"] - sa.loc[common, "nll"]).to_numpy()
            dp = (sb.loc[common, "pearson"] - sa.loc[common, "pearson"]).to_numpy()
            ds = (sb.loc[common, "spearman"] - sa.loc[common, "spearman"]).to_numpy()
            md.append(
                f"| {label} | {fmt(sa.nll.mean(), sa.nll.std(ddof=1))} "
                f"| {fmt(sb.nll.mean(), sb.nll.std(ddof=1))} "
                f"| {d.mean():+.5f} ± {d.std(ddof=1):.5f} "
                f"| {int((d < 0).sum())}/{len(d)} "
                f"| {dp.mean():+.4f} | {ds.mean():+.4f} |")
            rows.append({
                "protocol": proto, "signal": sig, "method": label,
                "baseline_nll_mean": sa.nll.mean(), "baseline_nll_std": sa.nll.std(ddof=1),
                "variant_nll_mean": sb.nll.mean(), "variant_nll_std": sb.nll.std(ddof=1),
                "delta_nll_mean": d.mean(), "delta_nll_std": d.std(ddof=1),
                "n_seeds": len(d), "n_better": int((d < 0).sum()),
                "delta_pearson_mean": dp.mean(), "delta_spearman_mean": ds.mean(),
                **{f"delta_seed_{int(s)}": float(x) for s, x in zip(common, d)},
            })
        md.append("")

        md += ["fixed signals（不依赖 content 表，只作一致性检查）:", "",
               "| 信号 | baseline | variant | 最大 |Δ| |", "|---|---|---|---|"]
        for sig, label in FIXED:
            sa = a[(a.protocol == proto) & (a.signal == sig)].set_index("seed").sort_index()
            sb = b[(b.protocol == proto) & (b.signal == sig)].set_index("seed").sort_index()
            common = sa.index.intersection(sb.index)
            if len(common) == 0:
                continue
            diff = (sb.loc[common, "nll"] - sa.loc[common, "nll"]).abs().max()
            md.append(f"| {label} | {fmt(sa.nll.mean(), sa.nll.std(ddof=1))} "
                      f"| {fmt(sb.nll.mean(), sb.nll.std(ddof=1))} | {diff:.2e} |")
        md.append("")

    df = pd.DataFrame(rows)
    csv_path = out.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"-> {out} / {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
