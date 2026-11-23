#!/usr/bin/env python
"""run_p2_p4_multiseed -- split-seed replicates of the P2 and P4 experiments.

The replicate seed is a TRAIN/TEST SPLIT seed, not a model seed:

  P2  pair-random 70/30 resplit of the surviving step1 pairs
      (scripts/step1_pairbest_resplit.py).  The 5 chart-disjoint folds use
      --chart-seed 0 and the models use --seed 0, so the only thing that moves
      across replicates is which pairs are in train/test.
  P4  stratified 60/40 player x chart split (scripts/step1_p4_split.py):
      train on the 6x6 block, test on the 4x4 block, off-diagonal rows are
      "dropped" (never trained; kept only for the BT ceiling).  Models use
      --seed 0.

Methods (the seed is fixed for all of them):
    CM3P + external linear      (linear_extra, lam=1e-3)
    CM3P + external MLP         (mlp_extra, h64 L2, dropout 0.1, 98 epochs)
    CM3P linear                 (linear, lam=1e-3)
    CM3P MLP h64 L2             (mlp, h64 L2, dropout 0.1, 98 epochs)
    Sunny / Daniel / star       (fixed per-chart signals)

The official evaluation is scripts/step2_rank_eval.py with --ceiling-fit all,
--common-maps and --strict-pairs; Pearson/Spearman are the per-chart
`pearson_bhat` / `rho_bhat` against the cross-fitted BT b.

Examples
--------
# full study, one process (about 1.5-2 h on 6 CPU threads)
python scripts/run_p2_p4_multiseed.py --protocols p2,p4 --seeds 1,2,3,4,5

# two parallel halves (4 threads each) in separate shells/jobs
python scripts/run_p2_p4_multiseed.py --seeds 1,3,5 --threads 4
python scripts/run_p2_p4_multiseed.py --seeds 2,4   --threads 4

# aggregate whatever exists
python scripts/run_p2_p4_multiseed.py --aggregate-only

# fast end-to-end smoke test (2 chart folds, 2 epochs, 1 seed)
python scripts/run_p2_p4_multiseed.py --quick --seeds 99
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
STEP1 = ROOT / "data/processed/step1_1k_pairbest.parquet"
P2_PREFIX = ROOT / "data/processed/step1_1k_pairbest_p2"
P4_PREFIX = ROOT / "data/processed/step1_1k_pairbest_p4"
CONTENT = ROOT / "data/processed/chart_content_cm3p_local.parquet"
SIGNALS = ROOT / "data/processed/rank_signals_1k.parquet"
CACHE = ROOT / "data/processed/rank_eval_cache"

METHODS = {
    "linear": {
        "display": "CM3P 线性 (λ1e-3)",
        "signal": "cm3p_linear",
        "args": ["--model", "linear", "--lam", "1e-3"],
    },
    "linear_extra": {
        "display": "CM3P + external 线性",
        "signal": "cm3p_linear_extra",
        "args": ["--model", "linear_extra", "--lam", "1e-3"],
    },
    "mlp": {
        "display": "CM3P MLP h64 L2",
        "signal": "cm3p_mlp_h64_L2",
        "args": ["--model", "mlp", "--lam", "1e-2", "--hid", "64", "--depth", "2",
                 "--dropout", "0.1", "--lr", "3e-3"],
    },
    "mlp_extra": {
        "display": "CM3P + external MLP",
        "signal": "cm3p_mlp_extra",
        "args": ["--model", "mlp_extra", "--lam", "1e-2", "--hid", "64", "--depth", "2",
                 "--dropout", "0.1", "--lr", "3e-3"],
    },
}
METHOD_ORDER = ["linear_extra", "mlp_extra", "linear", "mlp"]
FIXED = [("sunny_star", "Sunny"), ("daniel_star", "Daniel"), ("star", "star")]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _rel(p) -> str:
    """Relative path for commands/JSONs; runs use cwd=ROOT."""
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def _real_split_path(protocol: str, seed: int,
                     p2_prefix: Path = P2_PREFIX,
                     p4_prefix: Path = P4_PREFIX) -> Path:
    # prefixes are already the full stem; the split scripts append _s<seed>
    prefix = p2_prefix if protocol == "p2" else p4_prefix
    return Path(f"{prefix}_s{seed}.parquet")


def run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as fh:
        fh.write(" ".join(cmd) + "\n\n")
        fh.flush()
        r = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-30:])
        raise SystemExit(f"command failed ({r.returncode}): {' '.join(cmd)}\n{tail}")


def ensure_splits(seeds: list[int], protocols: list[str], force: bool,
                  p2_prefix: Path = P2_PREFIX,
                  p4_prefix: Path = P4_PREFIX) -> None:
    """Create the default splits on demand; custom prefixes must already exist."""
    if "p2" in protocols:
        missing = [s for s in seeds
                   if not _real_split_path("p2", s, p2_prefix, p4_prefix).exists()]
        if missing and p2_prefix.resolve() != P2_PREFIX.resolve():
            raise SystemExit(f"--split-p2 {p2_prefix}: missing seeds {missing}")
        if missing or force:
            run([PY, "scripts/step1_pairbest_resplit.py",
                 "--in", _rel(STEP1), "--out-prefix", _rel(p2_prefix),
                 "--seeds", ",".join(str(s) for s in seeds)]
                + (["--force"] if force else []),
                Path("/tmp/p2_resplit.log"))
    if "p4" in protocols:
        missing = [s for s in seeds
                   if not _real_split_path("p4", s, p2_prefix, p4_prefix).exists()]
        if missing and p4_prefix.resolve() != P4_PREFIX.resolve():
            raise SystemExit(f"--split-p4 {p4_prefix}: missing seeds {missing}")
        if missing or force:
            run([PY, "scripts/step1_p4_split.py",
                 "--in", _rel(STEP1), "--out-prefix", _rel(p4_prefix),
                 "--seeds", ",".join(str(s) for s in seeds)]
                + (["--force"] if force else []),
                Path("/tmp/p4_split.log"))


def train(protocol: str, seed: int, method: str, out_dir: Path,
          threads: int, epochs: int, p2_folds: int, force: bool,
          content: Path, split: Path, content_col: str | None = None) -> None:
    cfg = METHODS[method]
    name = cfg["signal"]
    run_dir = out_dir / protocol / f"s{seed}"
    sig = run_dir / "signals" / f"{name}.parquet"
    js = run_dir / "signals" / f"{name}.json"
    if sig.exists() and js.exists() and not force:
        _log(f"{protocol} s{seed} {method}: exists, skip")
        return
    cmd = [PY, "scripts/step2_cm3p_rank.py",
           "--in", _rel(split),
           "--content-path", _rel(content),
           "--signal-table", _rel(SIGNALS),
           "--extra-signals", "msd_overall,sunny_star",
           "--val-frac", "0", "--seed", "0", "--threads", str(threads),
           "--name", name,
           "--out-signals", _rel(sig), "--out-json", _rel(js)]
    if content_col:
        cmd += ["--content-col", str(content_col)]
    if protocol == "p2":
        cmd += ["--chart-only", "--chart-folds", str(p2_folds), "--chart-seed", "0"]
    cmd += list(cfg["args"])
    if method in ("mlp", "mlp_extra"):
        cmd += ["--epochs", str(epochs)]
    _log(f"{protocol} s{seed} {method}: train")
    run(cmd, out_dir / "logs" / f"{protocol}_s{seed}_{method}.log")


def assemble(protocol: str, seed: int, out_dir: Path,
             models: list[str], n_charts: int) -> Path:
    run_dir = out_dir / protocol / f"s{seed}"
    model_sigs = [run_dir / "signals" / f"{METHODS[m]['signal']}.parquet"
                  for m in models]
    base = None
    for p, m in zip(model_sigs, models):
        if not p.exists():
            raise SystemExit(f"missing {p}")
        name = METHODS[m]["signal"]
        # chart-disjoint runs also write a chart_fold column; keep only the signal
        df = pd.read_parquet(p)[["beatmap_id", name]]
        if base is None:
            base = df
        else:
            base = base.merge(df, on="beatmap_id", how="inner")
    fixed = pd.read_parquet(SIGNALS)[["beatmap_id", "sunny_star",
                                      "daniel_star", "star"]]
    out = base.merge(fixed, on="beatmap_id", how="inner")
    if len(out) != n_charts:
        raise SystemExit(f"{protocol} s{seed}: assembled {len(out)} charts, "
                         f"expected {n_charts} (from the split file)")
    cols = [METHODS[m]["signal"] for m in models] + [c for c, _ in FIXED]
    missing = [c for c in cols if c not in out.columns]
    if missing:
        raise SystemExit(f"{protocol} s{seed}: assembled table missing {missing}; "
                         f"have {list(out.columns)}")
    if out[cols].isna().any().any():
        raise SystemExit(f"{protocol} s{seed}: NaN in signal table")
    dest = run_dir / "signals.parquet"
    out.to_parquet(dest, index=False)
    _log(f"{protocol} s{seed}: signals {len(out)} charts x {len(cols)} cols "
         f"({', '.join(cols)}) -> {dest}")
    return dest


def evaluate(protocol: str, seed: int, out_dir: Path, force: bool,
             models: list[str], split: Path, force_eval: bool = False) -> None:
    run_dir = out_dir / protocol / f"s{seed}"
    dest = run_dir / "rank_eval.json"
    if dest.exists() and not force and not force_eval:
        _log(f"{protocol} s{seed}: eval exists, skip (pass --force-eval to redo)")
        return
    sig = run_dir / "signals.parquet"
    if not sig.exists():
        raise SystemExit(f"missing {sig}")
    names = [METHODS[m]["signal"] for m in models] + [c for c, _ in FIXED]
    cmd = [PY, "scripts/step2_rank_eval.py",
           "--in", _rel(split),
           "--signal-table", _rel(sig),
           "--signals", ",".join(names),
           "--split", "test", "--ceiling-fit", "all",
           "--common-maps", "--strict-pairs",
           "--folds", "2", "--bootstrap", "0", "--seed", "0",
           "--cache-dir", _rel(CACHE),
           "--out", _rel(dest)]
    _log(f"{protocol} s{seed}: eval")
    run(cmd, out_dir / "logs" / f"{protocol}_s{seed}_eval.log")


def _mean_std(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    sd = float(a.std(ddof=1)) if a.size > 1 else 0.0
    return float(a.mean()), sd


def aggregate(out_dir: Path, protocols: list[str], seeds: list[int]) -> None:
    rows = []
    for protocol in protocols:
        for seed in seeds:
            path = out_dir / protocol / f"s{seed}" / "rank_eval.json"
            if not path.exists():
                continue
            j = json.loads(path.read_text())
            for m in METHOD_ORDER:
                cfg = METHODS[m]
                s = j["signals"].get(cfg["signal"])
                if s is None:
                    continue
                rows.append({
                    "protocol": protocol, "seed": seed, "method": cfg["display"],
                    "signal": cfg["signal"],
                    "nll": s["nll_per_instance"],
                    "pearson": s["pearson_bhat"],
                    "spearman": s["rho_bhat"],
                    "captured": s["captured"],
                    "ceiling_nll": s["ceiling_nll"],
                    "n_maps": s["n_maps"],
                    "n_instances_used": s["n_instances_used"],
                })
            for col, label in FIXED:
                s = j["signals"].get(col)
                if s is None:
                    continue
                rows.append({
                    "protocol": protocol, "seed": seed, "method": label,
                    "signal": col,
                    "nll": s["nll_per_instance"],
                    "pearson": s["pearson_bhat"],
                    "spearman": s["rho_bhat"],
                    "captured": s["captured"],
                    "ceiling_nll": s["ceiling_nll"],
                    "n_maps": s["n_maps"],
                    "n_instances_used": s["n_instances_used"],
                })
    if not rows:
        raise SystemExit("no rank_eval.json found")
    long = pd.DataFrame(rows).sort_values(["protocol", "method", "seed"])
    long.to_csv(out_dir / "summary.csv", index=False)

    mean_rows = []
    order = [METHODS[m]["display"] for m in METHOD_ORDER] + [l for _, l in FIXED]
    for protocol in protocols:
        for method in order:
            sub = long[(long.protocol == protocol) & (long.method == method)]
            if sub.empty:
                continue
            rec = {"protocol": protocol, "method": method,
                   "n_seeds": int(sub.seed.nunique())}
            for col in ("nll", "pearson", "spearman", "captured", "ceiling_nll"):
                mu, sd = _mean_std(sub[col].tolist())
                rec[f"{col}_mean"] = mu
                rec[f"{col}_std"] = sd
            mean_rows.append(rec)
    mean = pd.DataFrame(mean_rows)
    mean.to_csv(out_dir / "summary_mean_std.csv", index=False)

    def fmt(mu: float, sd: float, nd: int) -> str:
        if not np.isfinite(mu):
            return "—"
        return f"{mu:.{nd}f} ± {sd:.{nd}f}"

    lines = ["# P2 / P4 multi-seed summary", "",
             "Seeds are train/test split seeds; model seed is fixed at 0.", ""]
    for protocol, title in (("p2", "P2（新 chart）"),
                            ("p4", "P4（新玩家 + 新 chart，6×6 / 4×4）")):
        sub = mean[mean.protocol == protocol]
        if sub.empty:
            continue
        ceil = long[long.protocol == protocol].groupby("seed").ceiling_nll.first()
        lines += [f"## {title}", "",
                  f"ceiling NLL = {ceil.mean():.5f} ± "
                  f"{(ceil.std(ddof=1) if len(ceil) > 1 else 0.0):.5f} "
                  f"（{len(ceil)} seeds）", "",
                  "| 方法 | test NLL | Pearson(d,b) | Spearman(d,b) |",
                  "|---|---|---|---|"]
        for method in order:
            r = sub[sub.method == method]
            if r.empty:
                continue
            r = r.iloc[0]
            lines.append(f"| {method} | {fmt(r.nll_mean, r.nll_std, 5)} | "
                         f"{fmt(r.pearson_mean, r.pearson_std, 4)} | "
                         f"{fmt(r.spearman_mean, r.spearman_std, 4)} |")
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    _log(f"aggregate -> {out_dir/'summary.csv'} / summary_mean_std.csv / summary.md")
    print("\n".join(lines), flush=True)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocols", default="p2,p4")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--models", default=",".join(METHOD_ORDER))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--split-p2", default=str(P2_PREFIX.relative_to(ROOT)),
                    help="P2 split prefix; the split scripts append _s<seed>.parquet "
                         "(default: the main 6,828-chart splits)")
    ap.add_argument("--split-p4", default=str(P4_PREFIX.relative_to(ROOT)),
                    help="P4 split prefix (default: the main splits)")
    ap.add_argument("--content-path", default=str(CONTENT.relative_to(ROOT)),
                    help="chart content table used by the CM3P models "
                         "(default: the legacy local table; pass "
                         "data/processed/chart_content_cm3p_local_canon.parquet "
                         "for the canon crop)")
    ap.add_argument("--content-col", default=None,
                    help="content column passed to step2_cm3p_rank.py; use "
                         "'features' for the T3 pool tables")
    ap.add_argument("--signal-table", default=str(SIGNALS.relative_to(ROOT)),
                    help="chart-level signal table (fixed signals + extra signals)")
    ap.add_argument("--cache-dir", default=str(CACHE.relative_to(ROOT)),
                    help="step2_rank_eval cache dir")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--p2-folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=98)
    ap.add_argument("--quick", action="store_true",
                    help="2 chart folds, 2 epochs, results/p4_multiseed_quick")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--skip-train", action="store_true",
                    help="skip training, only assemble existing signals")
    ap.add_argument("--assemble-only", action="store_true",
                    help="alias of --skip-train: assemble + eval existing signals")
    ap.add_argument("--force-eval", action="store_true",
                    help="re-run step2_rank_eval even if rank_eval.json exists")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--force", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    global SIGNALS, CACHE
    args = parse_args(argv)
    SIGNALS = Path(args.signal_table)
    if not SIGNALS.is_absolute():
        SIGNALS = ROOT / SIGNALS
    CACHE = Path(args.cache_dir)
    if not CACHE.is_absolute():
        CACHE = ROOT / CACHE
    if not SIGNALS.exists():
        raise SystemExit(f"--signal-table does not exist: {SIGNALS}")
    protocols = [s.strip() for s in args.protocols.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    models = [s.strip() for s in args.models.split(",") if s.strip()]
    for m in models:
        if m not in METHODS:
            raise SystemExit(f"unknown method {m!r}; choose from {list(METHODS)}")

    content = Path(args.content_path)
    if not content.is_absolute():
        content = ROOT / content
    if not content.exists():
        raise SystemExit(f"--content-path does not exist: {content}")
    p2_prefix, p4_prefix = Path(args.split_p2), Path(args.split_p4)
    if not p2_prefix.is_absolute():
        p2_prefix = ROOT / p2_prefix
    if not p4_prefix.is_absolute():
        p4_prefix = ROOT / p4_prefix

    out_dir = Path(args.out_dir) if args.out_dir else (
        ROOT / ("results/p4_multiseed_quick" if args.quick else "results/p4_multiseed"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        args.p2_folds = min(args.p2_folds, 2)
        args.epochs = min(args.epochs, 2)

    skip_train = args.skip_train or args.assemble_only
    if not args.aggregate_only:
        ensure_splits(seeds, protocols, args.force, p2_prefix, p4_prefix)
        for protocol in protocols:
            for seed in seeds:
                split = _real_split_path(protocol, seed, p2_prefix, p4_prefix)
                if not split.exists():
                    raise SystemExit(f"missing split file: {split}")
                n_charts = int(pd.read_parquet(
                    split, columns=["beatmap_id"])["beatmap_id"].nunique())
                _log(f"=== {protocol} seed {seed} ({n_charts:,} charts) ===")
                if not skip_train:
                    for m in models:
                        train(protocol, seed, m, out_dir, args.threads,
                              args.epochs, args.p2_folds, args.force, content, split,
                              args.content_col)
                assemble(protocol, seed, out_dir, models, n_charts)
                if not args.skip_eval:
                    evaluate(protocol, seed, out_dir, args.force, models, split,
                             force_eval=args.force_eval)
        # aggregate over the seeds this process handled
    aggregate(out_dir, protocols, seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
