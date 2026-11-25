#!/usr/bin/env python3
"""Evaluate E2/E3 arms with the same strict-pair protocol and paired CIs.

Example:
  python scripts/behavior_arm_eval.py \
    --runs runs/behavior/e2 --split-table data/processed/step1_full_holdout_judge_s1.parquet \
    --arms rank,judge,judge_shuffled,single_grade_miss \
    --pairs judge:rank,judge_shuffled:rank,single_grade_miss:rank,judge:judge_shuffled \
    --out-dir runs/behavior/e2/eval
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--split-table", required=True, type=Path)
    ap.add_argument("--arms", required=True,
                    help="comma-separated arm directory names")
    ap.add_argument("--pairs", required=True,
                    help="comma-separated a:b comparisons")
    ap.add_argument("--split-seed", type=int, default=1)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--cache-dir", default="data/processed/rank_eval_cache")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reuse", action="store_true")
    return ap.parse_args(argv)


def run(cmd):
    print("[eval]", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=ROOT, check=True)


def signal_path(runs: Path, arm: str, split_seed: int) -> Path:
    for rel in (Path(f"s{split_seed}") / "d_test.parquet",
                Path("d_test.parquet")):
        p = runs / arm / rel
        if p.exists():
            return p
    raise SystemExit(f"missing d_test.parquet for arm {arm} under {runs}")


def main(argv=None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    pairs = [tuple(p.split(":")) for p in args.pairs.split(",") if p.strip()]
    paths = {a: signal_path(args.runs, a, args.split_seed) for a in arms}
    evals = {}
    for a in arms:
        out = args.out_dir / f"eval_{a}.json"
        if not (args.reuse and out.exists()):
            run([sys.executable, "scripts/step2_rank_eval.py",
                 "--in", args.split_table, "--signal-table", paths[a],
                 "--signals", "d", "--split", "test",
                 "--ceiling-fit", "all", "--strict-pairs",
                 "--cache-dir", args.cache_dir,
                 "--bootstrap", args.bootstrap, "--seed", args.seed,
                 "--out", out])
        evals[a] = json.loads(out.read_text())
    sig = {a: e["signals"]["d"] for a, e in evals.items()}
    ref = sig[arms[0]]
    for a, s in sig.items():
        if (s["n_maps"] != ref["n_maps"] or s["n_strict"] != ref["n_strict"]
                or abs(s["ceiling_nll"] - ref["ceiling_nll"]) > 1e-12):
            raise SystemExit(f"{a}: chart/strict/ceiling set differs from {arms[0]}")
    cache = ref["ceiling_cache"]
    if not cache:
        raise SystemExit("no ceiling_cache; rerun without --no-cache")
    paired = {}
    for a, b in pairs:
        out = args.out_dir / f"paired_{a}_vs_{b}.json"
        if not (args.reuse and out.exists()):
            run([sys.executable, "scripts/rank_paired_delta.py",
                 "--in", args.split_table, "--a", paths[a], "--b", paths[b],
                 "--a-col", "d", "--b-col", "d", "--split", "test",
                 "--strict-pairs", "--ceiling-cache", cache,
                 "--expected-n-strict", ref["n_strict"],
                 "--bootstrap", args.bootstrap, "--seed", args.seed,
                 "--out", out])
        paired[f"{a}_vs_{b}"] = json.loads(out.read_text())
    qa = {}
    for a in arms:
        p = args.runs / a / f"s{args.split_seed}" / "events_qa.json"
        if p.exists():
            qa[a] = json.loads(p.read_text())
    summary = {"runs": str(args.runs), "split_table": str(args.split_table),
               "signals": {a: {k: s.get(k) for k in
                               ("nll_per_instance", "captured_strict", "n_maps",
                                "n_strict", "ceiling_nll", "ceiling_cache")}
                           for a, s in sig.items()},
               "paired": paired, "events_qa": qa}
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    lines = ["# E2/E3 strict-pair summary", "",
             "| signal | NLL | captured | n_strict |", "|---|---:|---:|---:|"]
    for a, s in sig.items():
        lines.append(f"| {a} | {s['nll_per_instance']:.6f} | "
                     f"{s['captured_strict']:.4f} | {s['n_strict']} |")
    lines += ["", "| comparison | delta | player CI | chart CI |",
              "|---|---:|---|---|"]
    for name, p in paired.items():
        lines.append(f"| {name} | {p['delta']:+.6f} | "
                     f"[{p['ci_player'][0]:+.6f}, {p['ci_player'][1]:+.6f}] | "
                     f"[{p['ci_chart'][0]:+.6f}, {p['ci_chart'][1]:+.6f}] |")
    if qa:
        lines += ["", "## Event proportion QA (val)", "",
                  "| arm | event NLL | baseline NLL | delta | max-rate rho | miss rho |",
                  "|---|---:|---:|---:|---:|---:|"]
        for a, q in qa.items():
            sp = q.get("spearman", {})
            lines.append(f"| {a} | {q['event_nll']:.5f} | {q['baseline_nll']:.5f} | "
                         f"{q['delta_vs_baseline']:+.5f} | "
                         f"{sp.get('n_max')} | {sp.get('n_miss')} |")
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
