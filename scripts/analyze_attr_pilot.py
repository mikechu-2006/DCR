#!/usr/bin/env python
"""Collect the 1-seed attribute-channel pilot results into one markdown table.

Reads the per-arm summary.csv files produced by scripts/run_p2_p4_multiseed.py
(single split seed 1) and prints the content-linear and content+external-linear
NLL plus the paired delta against the matching 8ch control arm:

  Protocol S (continuation from the 8ch holdout ckpt, 5 epochs):
      control a0cont; arms a12/a20/a27; protocols p4, p4_hold, p2
  Protocol D (from scratch on the mapset-disjoint split, 20 epochs):
      control a0d;    arms a12d/a20d/a27d; protocol setp4 (+ CM3P legacy ref)

Usage: python scripts/analyze_attr_pilot.py [--out results/d2v_attr_run/pilot_table.md]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LINEAR = "CM3P 线性 (λ1e-3)"
EXT = "CM3P + external 线性"

S_ARMS = ["a0cont", "a12", "a20", "a27"]
D_ARMS = ["a0d", "a12d", "a20d", "a27d"]
S_PROTOCOLS = ["p4", "p4_hold", "p2"]
D_PROTOCOLS = ["setp4"]


def _read(path: Path):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty or "method" not in df.columns:
        return None
    return df.set_index("method")


def _rows_for(arm: str, base: Path, protocol: str):
    df = _read(base / arm / protocol / "summary.csv")
    if df is None:
        return None
    out = {"arm": arm, "protocol": protocol}
    for key, label in (("linear", LINEAR), ("ext", EXT)):
        if label in df.index:
            r = df.loc[label]
            out["nll_" + key] = float(r["nll"])
            out["captured_" + key] = float(r.get("captured", float("nan")))
            out["pearson_" + key] = float(r.get("pearson", float("nan")))
            out["n_maps"] = int(r.get("n_maps", 0))
    ceils = df["ceiling_nll"].dropna().unique() if "ceiling_nll" in df.columns else []
    out["ceiling"] = float(ceils[0]) if len(ceils) else float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/d2v_attr_run/pilot_table.md"))
    args = ap.parse_args()

    rows = []
    for arm in S_ARMS:
        for proto in S_PROTOCOLS:
            r = _rows_for(arm, ROOT / "results/d2v_attr", proto)
            if r:
                rows.append({"stage": "S", **r})
    for arm in D_ARMS:
        for proto in D_PROTOCOLS:
            r = _rows_for(arm, ROOT / "results/d2v_attr_d_cached", proto)
            if r:
                rows.append({"stage": "D", **r})
    ref = _rows_for("cm3p_legacy", ROOT / "results/d2v_attr_d_cached", "setp4")
    if ref:
        rows.append({"stage": "D", **{**ref, "arm": "CM3P-legacy"}})
    df = pd.DataFrame(rows)
    if df.empty:
        print("no summary.csv found yet")
        return 1

    df["d_linear"] = float("nan")
    df["d_ext"] = float("nan")
    for stage, control in (("S", "a0cont"), ("D", "a0d")):
        for proto in df["protocol"].unique():
            sub = df[(df.stage == stage) & (df.protocol == proto)]
            ctl = sub[sub.arm == control]
            if ctl.empty:
                continue
            c = ctl.iloc[0]
            mask = (df.stage == stage) & (df.protocol == proto)
            df.loc[mask, "d_linear"] = df.loc[mask, "nll_linear"] - c["nll_linear"]
            df.loc[mask, "d_ext"] = df.loc[mask, "nll_ext"] - c["nll_ext"]

    cols = ["stage", "protocol", "arm", "nll_linear", "d_linear", "nll_ext", "d_ext",
            "captured_linear", "pearson_ext", "n_maps", "ceiling"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].sort_values(["stage", "protocol", "arm"])
    lines = ["# Attribute-channel pilot (1 split seed = 1)", "",
             "NLL = pairwise Bradley-Terry rank-NLL (lower is better). "
             "d_* = arm minus its 8ch control (a0cont for S, a0d for D). "
             "Single split seed: no CI; treat |d| < 0.005 as noise "
             "(5-seed MDE ~0.005; 1 seed is weaker still).", "",
             "~~~", df.to_markdown(index=False, floatfmt=".5f"), "~~~", ""]
    if ref:
        lines.append("CM3P-legacy row = the frozen CM3P legacy 512d content table on the "
                     "same set-P4 split (linear/linear_extra; MLP rows are in its summary.csv).")
    text = "\n".join(lines)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
