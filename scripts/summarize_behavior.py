#!/usr/bin/env python3
"""summarize_behavior -- collect run metrics and paired deltas into one table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def run_tag(path: Path, root: Path) -> dict:
    rel = path.relative_to(root).parts
    return {"rel": "/".join(rel[:-1]), "run": rel[-2] if len(rel) >= 2 else ""}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results/behavior")
    ap.add_argument("--out-dir", default="results/behavior/summary")
    args = ap.parse_args()
    root = Path(args.root); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in sorted(root.glob("**/metrics.json")):
        try:
            m = json.loads(p.read_text())
        except Exception:
            continue
        m.update(run_tag(p, root))
        m["path"] = str(p)
        rows.append(m)
    df = pd.DataFrame(rows)
    if len(df):
        keys = [c for c in ("task", "aux", "freeze", "init") if c in df.columns]
        num = [c for c in ("best_val_nll", "pair_views", "row_views", "steps")
               if c in df.columns]
        agg = df.groupby(keys, dropna=False)[num].agg(["mean", "std", "count"])
        agg.columns = ["_".join(c) for c in agg.columns]
        agg.reset_index().to_csv(out / "runs.csv", index=False)
        md = ["# behavior runs", "", f"n={len(df)}", ""]
        cols = keys + [c for c in ("best_val_nll_mean", "best_val_nll_std") if c in agg.columns]
        md.append("| " + " | ".join(cols) + " |")
        md.append("|" + "|".join(["---"] * len(cols)) + "|")
        for _, r in agg.reset_index().iterrows():
            md.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        (out / "runs.md").write_text("\n".join(md) + "\n")

    prows = []
    for p in sorted(root.glob("**/paired*.json")):
        try:
            j = json.loads(p.read_text())
        except Exception:
            continue
        j["path"] = str(p)
        prows.append(j)
    pdf = pd.DataFrame(prows)
    if len(pdf):
        pdf.to_csv(out / "paired.csv", index=False)
    print(f"[summary] runs={len(df)} paired={len(pdf)} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
