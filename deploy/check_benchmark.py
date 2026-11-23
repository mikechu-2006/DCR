#!/usr/bin/env python
"""Print the deploy-function output distribution and in-sample MAE on benchmark.

This is the check the packaging side should run before judging the output scale.
It uses only pandas/numpy and deploy/dan_predict.py (no sklearn/torch).

    python deploy/check_benchmark.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy"))

from dan_predict import cm3p_to_dan, d2v_to_dan  # noqa: E402

CASES = [
    ("cm3p", cm3p_to_dan,
     "data/processed/benchmark_content_legacy.parquet"),
    ("d2v", d2v_to_dan,
     "data/processed/benchmark_content_d2v.parquet"),
]


def main() -> int:
    sig = pd.read_parquet(ROOT / "data/processed/benchmark_signals.parquet").set_index("row_id")
    for kind, fn, content_path in CASES:
        path = ROOT / content_path
        if not path.exists():
            print(f"[skip] {kind}: missing {content_path}")
            continue
        c = pd.read_parquet(path).set_index("row_id")
        common = sig.index.intersection(c.index)
        vals = []
        for rid in common:
            C = np.asarray(c.loc[rid, "embedding"], dtype=float)
            vec = list(C) + [float(sig.loc[rid, "msd_overall"]),
                             float(sig.loc[rid, "sunny_star"])]
            vals.append(fn(vec))
        v = np.asarray(vals, dtype=float)
        y = sig.loc[common, "expected"].to_numpy(float)
        q = np.percentile(v, [0, 25, 50, 75, 99, 100])
        print(f"{kind}: n={len(v)} min={q[0]:.4f} p25={q[1]:.4f} p50={q[2]:.4f} "
              f"p75={q[3]:.4f} p99={q[4]:.4f} max={q[5]:.4f} "
              f"mae={np.mean(np.abs(v - y)):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
