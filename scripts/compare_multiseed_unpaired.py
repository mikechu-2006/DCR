#!/usr/bin/env python
"""compare_multiseed_unpaired -- descriptive comparison of two runs on different splits.

Unlike `compare_multiseed_runs.py`, this script deliberately does NOT pair seeds and
does NOT require equal ceilings.  It is for "how much did the data/split change
move the numbers?" checks, e.g.

    new legacy control (new 6,819-chart splits)
        vs  results/p4_multiseed (old 6,828-chart splits)

The two runs may have different train/test splits and ceilings, so only the
mean +/- std per method and the difference of those means are reported.  Do not
read the result as a paired per-seed delta; for that use compare_multiseed_runs.py.

Usage:
    python scripts/compare_multiseed_unpaired.py \
        --baseline results/p4_multiseed \
        --variant  results/p4_multiseed_audio/legacy \
        --out      results/p4_multiseed_audio/legacy_vs_p4_multiseed.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = [
    ("cm3p_linear_extra", "CM3P + external 线性"),
    ("cm3p_mlp_extra", "CM3P + external MLP"),
    ("cm3p_linear", "CM3P 线性 (λ1e-3)"),
    ("cm3p_mlp_h64_L2", "CM3P MLP h64 L2"),
    ("sunny_star", "Sunny"),
    ("daniel_star", "Daniel"),
    ("star", "star"),
]
PROTOCOLS = [("p2", "P2（新 chart）"), ("p4", "P4（新玩家 + 新 chart）")]


def load(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "summary.csv"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    df = pd.read_csv(path)
    need = {"protocol", "seed", "method", "signal", "nll", "pearson",
            "spearman", "captured", "ceiling_nll"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{path}: missing columns {sorted(missing)}")
    return df


def fmt(mu: float, sd: float) -> str:
    return f"{mu:.5f} ± {sd:.5f}" if np.isfinite(mu) else "—"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline-label", default=None)
    ap.add_argument("--variant-label", default=None)
    args = ap.parse_args(argv)

    a_dir, b_dir = Path(args.baseline), Path(args.variant)
    a, b = load(a_dir), load(b_dir)
    a_label = args.baseline_label or a_dir.name
    b_label = args.variant_label or b_dir.name
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    md = [f"# unpaired comparison: {b_label} vs {a_label}", "",
          "This script never pairs seeds and does not require equal ceilings.  It is "
          "intended for runs on different split sets; if the ceilings match, the two "
          "runs share splits and the mean difference equals the paired mean difference, "
          "but the table is still descriptive.  If the ceilings differ, only the "
          "difference of the 5-seed means is meaningful.",
          ""]

    for proto, title in PROTOCOLS:
        if a[a.protocol == proto].empty or b[b.protocol == proto].empty:
            continue
        ca = a[(a.protocol == proto) & (a.signal == "cm3p_linear")].set_index("seed")
        cb = b[(b.protocol == proto) & (b.signal == "cm3p_linear")].set_index("seed")
        common = ca.index.intersection(cb.index)
        md += [f"## {title}", ""]
        if len(common):
            same = bool(np.allclose(ca.loc[common, "ceiling_nll"],
                                    cb.loc[common, "ceiling_nll"]))
            md += [
                f"- ceiling NLL: {a_label} **{ca.loc[common, 'ceiling_nll'].mean():.5f} ± "
                f"{ca.loc[common, 'ceiling_nll'].std(ddof=1):.5f}**; "
                f"{b_label} **{cb.loc[common, 'ceiling_nll'].mean():.5f} ± "
                f"{cb.loc[common, 'ceiling_nll'].std(ddof=1):.5f}** "
                f"（{'ceilings match; same split files' if same else 'ceilings differ; different split files'}）",
                "",
            ]
        md += ["| 方法 | " + a_label + " NLL | " + b_label + " NLL | Δmean "
               "| ΔPearson | ΔSpearman |", "|---|---|---|---|---|---|"]

        for sig, label in METHODS:
            sa = a[(a.protocol == proto) & (a.signal == sig)].set_index("seed").sort_index()
            sb = b[(b.protocol == proto) & (b.signal == sig)].set_index("seed").sort_index()
            common = sa.index.intersection(sb.index)
            if len(common) == 0:
                continue
            am, asd = sa.loc[common, "nll"].mean(), sa.loc[common, "nll"].std(ddof=1)
            bm, bsd = sb.loc[common, "nll"].mean(), sb.loc[common, "nll"].std(ddof=1)
            dp = sb.loc[common, "pearson"].mean() - sa.loc[common, "pearson"].mean()
            ds = sb.loc[common, "spearman"].mean() - sa.loc[common, "spearman"].mean()
            md.append(f"| {label} | {fmt(am, asd)} | {fmt(bm, bsd)} "
                      f"| {bm - am:+.5f} | {dp:+.4f} | {ds:+.4f} |")
            rows.append({
                "protocol": proto, "signal": sig, "method": label,
                "baseline_nll_mean": am, "baseline_nll_std": asd,
                "variant_nll_mean": bm, "variant_nll_std": bsd,
                "delta_nll_mean": bm - am, "n_seeds": len(common),
                "delta_pearson_mean": dp, "delta_spearman_mean": ds,
                "baseline_ceiling_mean": ca.loc[common, "ceiling_nll"].mean(),
                "variant_ceiling_mean": cb.loc[common, "ceiling_nll"].mean(),
            })
        md.append("")

    if not rows:
        raise SystemExit("no common protocol/method/seed found")
    df = pd.DataFrame(rows)
    csv_path = out.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"-> {out} / {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
