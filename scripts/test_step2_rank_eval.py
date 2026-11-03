#!/usr/bin/env python
"""End-to-end checks for scripts/step2_rank_eval.py on synthetic pairbest data.

Run:  python scripts/test_step2_rank_eval.py

Covers the ceiling fit/eval fix: --ceiling-fit all fits b on every row and
evaluates on the --split rows; --ceiling-fit split is the legacy test-only
path.  Also covers --common-maps and --strict-pairs.  No data/ files needed.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "step2_rank_eval.py"
FAILS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""),
          flush=True)
    if not ok:
        FAILS.append(name)


def make_data(tmp: Path, n_players=400, n_maps=40, per=12, seed=0):
    rng = np.random.default_rng(seed)
    b = rng.normal(0.0, 1.0, n_maps)
    rows = []
    for p in range(n_players):
        maps = rng.choice(n_maps, per, replace=False)
        loss = b[maps] - rng.logistic(0.0, 0.7, size=per)   # lower = better
        rows.extend((p, int(m), float(l)) for m, l in zip(maps, loss))
    df = pd.DataFrame(rows, columns=["player_id", "beatmap_id", "loss"])
    df["player_id_base"] = df["player_id"]
    rng = np.random.default_rng(seed + 1)
    df["split"] = np.where(rng.random(len(df)) < 0.7, "train", "test")
    step1 = tmp / "step1.parquet"
    df.to_parquet(step1, index=False)

    good = b + rng.normal(0.0, 0.5, n_maps)
    sparse = good.copy()
    sparse[: n_maps // 2] = np.nan
    sig = pd.DataFrame({"beatmap_id": np.arange(n_maps), "good": good,
                        "sparse": sparse})
    sig_path = tmp / "signals.csv"
    sig.to_csv(sig_path, index=False)
    return step1, sig_path, df, b


def run(step1: Path, sig: Path, out: Path, *args: str) -> dict:
    cmd = [sys.executable, str(RUNNER), "--in", str(step1),
           "--signal-table", str(sig), "--bootstrap", "0", "--no-cache",
           "--out", str(out), *args]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"runner failed\n{' '.join(cmd)}\n{p.stdout}\n{p.stderr}")
    return json.loads(out.read_text())


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        step1, sig_path, df, b = make_data(tmp)
        n_test = int((df.split == "test").sum())
        n_test_maps = int(df.loc[df.split == "test", "beatmap_id"].nunique())

        # 1) fixed all-row ceiling: b from every row, scored on test rows
        d = run(step1, sig_path, tmp / "a.json", "--signals", "good",
                "--split", "test", "--ceiling-fit", "all", "--folds", "2")
        s = d["signals"]["good"]
        check("all-row ceiling uses every row", s["ceiling_fit_rows"] == len(df),
              f"{s['ceiling_fit_rows']} vs {len(df)}")
        check("all-row ceiling is a valid upper bound", s["captured"] <= 1.0 + 1e-9,
              f"captured={s['captured']:.4f} ceiling={s['ceiling_nll']:.4f}")
        check("all-row ceiling coverage is reported", 0.0 < s["ceiling_coverage"] <= 1.0,
              f"coverage={s['ceiling_coverage']:.3f}")

        # 2) legacy test-only ceiling: same signal, but b fitted on test rows only
        d2 = run(step1, sig_path, tmp / "b.json", "--signals", "good",
                 "--split", "test", "--ceiling-fit", "split", "--folds", "2")
        s2 = d2["signals"]["good"]
        check("split ceiling uses only the split rows",
              s2["ceiling_fit_rows"] == n_test,
              f"{s2['ceiling_fit_rows']} vs {n_test}")
        check("split ceiling uses only the split charts",
              s2["ceiling_fit_maps"] == n_test_maps,
              f"{s2['ceiling_fit_maps']} vs {n_test_maps}")

        # 3) common map set: the sparse signal restricts both signals to the
        #    charts where it is defined
        d3 = run(step1, sig_path, tmp / "c.json", "--signals", "good,sparse",
                 "--split", "test", "--ceiling-fit", "all", "--folds", "2",
                 "--common-maps")
        g, sp = d3["signals"]["good"], d3["signals"]["sparse"]
        check("common-maps restricts both signals", g["n_maps"] == sp["n_maps"] == 20,
              f"good={g['n_maps']} sparse={sp['n_maps']}")
        check("common-maps gives one pair set", g["n_instances"] == sp["n_instances"],
              f"{g['n_instances']} vs {sp['n_instances']}")
        check("common-maps gives one ceiling",
              abs(g["ceiling_nll"] - sp["ceiling_nll"]) < 1e-12,
              f"{g['ceiling_nll']:.6f} vs {sp['ceiling_nll']:.6f}")

        # 4) strict pairs: the signal NLL is measured on exactly the ceiling pairs
        d4 = run(step1, sig_path, tmp / "d.json", "--signals", "good",
                 "--split", "test", "--ceiling-fit", "all", "--folds", "2",
                 "--strict-pairs")
        s4 = d4["signals"]["good"]
        check("strict pairs match the ceiling count", s4["n_strict"] == s4["ceiling_n"],
              f"{s4['n_strict']} vs {s4['ceiling_n']}")
        check("strict mode is the primary metric",
              s4["strict_pairs"] and abs(s4["captured"] - s4["captured_strict"]) < 1e-12,
              f"captured={s4['captured']:.4f} strict={s4['captured_strict']:.4f}")
        check("strict mode keeps the all-pair comparison",
              np.isfinite(s4["captured_all"]),
              f"captured_all={s4['captured_all']:.4f}")

    print(f"\n{len(FAILS)} failure(s)")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
