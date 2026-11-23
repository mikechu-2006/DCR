#!/usr/bin/env python3
"""Summarize ki_eval rank_eval JSONs into CSV + Markdown."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(args.results, recursive=True)):
        if not p.endswith(".json"):
            continue
        j = json.loads(Path(p).read_text())
        tag = Path(p).parent.name + "/" + Path(p).stem
        sigs = j.get("signals", {})
        sunny = sigs.get("sunny_star", {}).get("nll_per_instance")
        for name, s in sigs.items():
            if "error" in s:
                continue
            nll = s.get("nll_per_instance")
            rows.append({
                "table": tag,
                "signal": name,
                "n_rows": s.get("n_rows"),
                "n_players": s.get("n_players"),
                "n_maps": s.get("n_maps"),
                "n_pairs": s.get("n_pairs"),
                "n_instances": s.get("n_instances_used"),
                "nll": nll,
                "delta_nll_vs_sunny": None if (nll is None or sunny is None or name == "sunny_star") else nll - sunny,
                "captured": s.get("captured"),
                "pearson_bhat": s.get("pearson_bhat"),
                "rho_bhat": s.get("rho_bhat"),
                "lw_spearman": (s.get("per_player") or {}).get("lw"),
                "macro_spearman": (s.get("per_player") or {}).get("macro"),
                "coverage_rows": s.get("coverage_rows"),
                "coverage_maps": s.get("coverage_maps"),
                "ceiling_nll": s.get("ceiling_nll"),
                "input": j.get("meta", {}).get("in"),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no result rows")
    df = df.sort_values(["table", "nll"])
    df.to_csv(args.out_csv, index=False)

    lines = ["# Koishi / ISOR rank evaluation", ""]
    for table, g in df.groupby("table", sort=True):
        lines.append(f"## {table}")
        lines.append("")
        lines.append("| signal | n_rows | n_maps | n_pairs | NLL | ΔNLL vs Sunny | Pearson(d,b) | Spearman(d,b) | lw ρ | macro ρ | coverage maps |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in g.iterrows():
            def f(x, nd=4):
                return "" if x is None or pd.isna(x) else f"{x:.{nd}f}"
            lines.append("| " + " | ".join([
                str(r["signal"]), str(r["n_rows"]), str(r["n_maps"]),
                str(r["n_pairs"]), f(r["nll"], 5), f(r["delta_nll_vs_sunny"], 5),
                f(r["pearson_bhat"], 3), f(r["rho_bhat"], 3),
                f(r["lw_spearman"], 3), f(r["macro_spearman"], 3),
                f(r["coverage_maps"], 3),
            ]) + " |")
        lines.append("")
    Path(args.out_md).write_text("\n".join(lines))
    print(df.to_string(index=False))
    print(f"wrote {args.out_csv} / {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
