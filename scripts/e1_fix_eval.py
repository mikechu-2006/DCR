#!/usr/bin/env python3
"""Run the E1_fix strict-pair eval + paired deltas for all arms of one split.

This is an orchestration helper, not a new metric implementation: it calls
`step2_rank_eval.py` and `rank_paired_delta.py` and checks that every arm
shares the same chart/strict-pair set.

Example:
  python scripts/e1_fix_eval.py \
    --runs runs/behavior/e1_fix/legacy/s1 \
    --split-table data/processed/step1_full_holdout_s1.parquet \
    --baseline-dir results/behavior/baselines/h4_s1 \
    --out-dir runs/behavior/e1_fix/legacy/s1/eval
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("unfrozen_k4", "random", "frozen_samehead")
PAIRS = (
    ("unfrozen_k4", "frozen_samehead"),
    ("unfrozen_k4", "random"),
    ("unfrozen_k4", "d2v_linear"),
    ("random", "d2v_linear"),
)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--split-table", required=True, type=Path)
    ap.add_argument("--baseline-dir", type=Path, default=None)
    ap.add_argument("--baseline-col", default="d2v_linear_h4_s1")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--cache-dir", default="data/processed/rank_eval_cache")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=1,
                    help="H4 split seed; used to find <run>/s<N>/d_test.parquet")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse existing eval/paired JSON instead of rerunning")
    return ap.parse_args(argv)


def run(cmd: list[str]) -> None:
    print("[eval]", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=ROOT, check=True)


def signal_path(runs: Path, arm: str, baseline_dir: Path | None,
                 split_seed: int) -> Path | None:
    if arm == "d2v_linear":
        return baseline_dir / "d2v_linear.parquet" if baseline_dir else None
    for rel in (Path("score") / "d_test.parquet", Path("d_test.parquet"),
                Path(f"s{split_seed}") / "score" / "d_test.parquet",
                Path(f"s{split_seed}") / "d_test.parquet"):
        p = runs / arm / rel
        if p.exists():
            return p
    return None


def main(argv=None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = {arm: signal_path(args.runs, arm, args.baseline_dir, args.split_seed)
             for arm in ARMS + ("d2v_linear",)}
    paths = {k: v for k, v in paths.items() if v is not None}
    if "unfrozen_k4" not in paths or "frozen_samehead" not in paths:
        raise SystemExit(f"missing required d_test under {args.runs}")
    print("[eval] signals:", {k: str(v) for k, v in paths.items()}, flush=True)

    cols = {arm: (args.baseline_col if arm == "d2v_linear" else "d")
            for arm in paths}
    evals = {}
    for arm, path in paths.items():
        out = args.out_dir / f"eval_{arm}.json"
        if not (args.reuse and out.exists()):
            run([sys.executable, "scripts/step2_rank_eval.py",
                 "--in", args.split_table, "--signal-table", path,
                 "--signals", cols[arm], "--split", "test",
                 "--ceiling-fit", "all", "--strict-pairs",
                 "--cache-dir", args.cache_dir,
                 "--bootstrap", args.bootstrap, "--seed", args.seed,
                 "--out", out])
        evals[arm] = json.loads(out.read_text())
    sig = {a: e["signals"][cols[a]] for a, e in evals.items()}
    base = sig["unfrozen_k4"]
    for arm, s in sig.items():
        if s["n_maps"] != base["n_maps"] or s["n_strict"] != base["n_strict"]:
            raise SystemExit(
                f"{arm}: chart/strict set differs from unfrozen_k4 "
                f"({s['n_maps']}/{s['n_strict']} vs "
                f"{base['n_maps']}/{base['n_strict']}); cannot form a main-table delta")
        if abs(s["ceiling_nll"] - base["ceiling_nll"]) > 1e-12:
            raise SystemExit(f"{arm}: ceiling differs from unfrozen_k4")
    cache = base["ceiling_cache"]
    if not cache:
        raise SystemExit("unfrozen_k4 eval has no ceiling_cache; rerun without --no-cache")

    paired = {}
    for a, b in PAIRS:
        if a not in paths or b not in paths:
            continue
        out = args.out_dir / f"paired_{a}_vs_{b}.json"
        if not (args.reuse and out.exists()):
            extra = ["--allow-subset"] if "d2v_linear" in (a, b) else []
            run([sys.executable, "scripts/rank_paired_delta.py",
                 "--in", args.split_table,
                 "--a", paths[a], "--b", paths[b],
                 "--a-col", cols[a], "--b-col", cols[b],
                 "--split", "test", "--strict-pairs",
                 "--ceiling-cache", cache,
                 "--expected-n-strict", base["n_strict"],
                 "--bootstrap", args.bootstrap, "--seed", args.seed,
                 *extra, "--out", out])
        paired[f"{a}_vs_{b}"] = json.loads(out.read_text())

    summary = {"runs": str(args.runs), "split_table": str(args.split_table),
               "signals": {a: {k: s.get(k) for k in
                               ("nll_per_instance", "captured_strict", "n_maps",
                                "n_strict", "ceiling_nll", "ceiling_cache")}
                           for a, s in sig.items()},
               "paired": paired}
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    lines = ["# E1_fix strict-pair summary", "",
             "| signal | NLL | captured | n_strict |", "|---|---:|---:|---:|"]
    for arm, s in sig.items():
        lines.append(f"| {arm} | {s['nll_per_instance']:.6f} | "
                     f"{s['captured_strict']:.4f} | {s['n_strict']} |")
    lines += ["", "| comparison | delta | player CI | chart CI |",
              "|---|---:|---|---|"]
    for name, p in paired.items():
        lines.append(f"| {name} | {p['delta']:+.6f} | "
                     f"[{p['ci_player'][0]:+.6f}, {p['ci_player'][1]:+.6f}] | "
                     f"[{p['ci_chart'][0]:+.6f}, {p['ci_chart'][1]:+.6f}] |")
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
