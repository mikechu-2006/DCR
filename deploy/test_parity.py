#!/usr/bin/env python
"""Parity harness for deploy/dan_predict.py against the Python reference.

For each content type:
  * build [C, msd_overall, sunny_star] for all 1,382 benchmark rows;
  * run the two-function module;
  * fit the same full-VSRG isotonic calibrator with sklearn on the raw d_m;
  * report max abs difference and MAE.

The D2V case is skipped until data/processed/benchmark_content_d2v.parquet and
results/benchmark/pred/D2V_L.parquet exist.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy"))

from dan_predict import cm3p_to_dan, d2v_to_dan, VECTOR_DIM, _raw_d  # noqa: E402

CASES = [
    ("cm3p", cm3p_to_dan,
     "results/benchmark/pred/RM_L.parquet",
     "data/processed/benchmark_content_legacy.parquet"),
    ("d2v", d2v_to_dan,
     "results/benchmark/pred/D2V_L.parquet",
     "data/processed/benchmark_content_d2v.parquet"),
]


def main() -> int:
    sig = pd.read_parquet(ROOT / "data/processed/benchmark_signals.parquet")
    sig = sig.set_index("row_id")
    rows = []
    for kind, fn, pred_path, content_path in CASES:
        pred = ROOT / pred_path
        content = ROOT / content_path
        if not pred.exists() or not content.exists():
            print(f"[skip] {kind}: missing {pred_path} or {content_path}")
            continue
        c = pd.read_parquet(content).set_index("row_id")
        p = pd.read_parquet(pred).set_index("row_id")
        common = sig.index.intersection(c.index).intersection(p.index)
        if len(common) != 1382:
            print(f"[warn] {kind}: only {len(common)} common rows")

        y = sig.loc[common, "expected"].to_numpy(float)
        d = p.loc[common, "d_m"].to_numpy(float)
        vecs = []
        for rid in common:
            C = np.asarray(c.loc[rid, "embedding"], dtype=float)
            vec = list(C) + [float(sig.loc[rid, "msd_overall"]),
                             float(sig.loc[rid, "sunny_star"])]
            if len(vec) != VECTOR_DIM:
                raise SystemExit(f"{kind}: vector len {len(vec)}")
            vecs.append(vec)
        got = np.array([fn(v) for v in vecs], dtype=float)
        raw = np.array([_raw_d(kind, v) for v in vecs], dtype=float)

        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(d, y)
        ref = ir.predict(d)
        diff = np.abs(got - ref)
        raw_diff = np.abs(raw - d)
        rows.append({
            "content": kind,
            "n": len(common),
            "max_abs_d": float(raw_diff.max()),
            "max_abs_dan": float(diff.max()),
            "mae_vs_expected": float(np.mean(np.abs(y - got))),
            "reference_mae": float(np.mean(np.abs(y - ref))),
        })
        print(f"[{kind}] n={len(common)} max|Δd|={raw_diff.max():.3e} "
              f"max|Δdan|={diff.max():.3e} "
              f"mae={np.mean(np.abs(y - got)):.4f} ref={np.mean(np.abs(y - ref)):.4f}")
        try:
            fn([0.0] * (VECTOR_DIM - 1))
        except ValueError:
            print(f"[{kind}] length guard OK")
        else:
            raise SystemExit(f"{kind}: length guard did not fire")
    print(pd.DataFrame(rows).to_string(index=False))
    ok = all(r["max_abs_d"] < 1e-5 and r["max_abs_dan"] < 1e-3 for r in rows)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
