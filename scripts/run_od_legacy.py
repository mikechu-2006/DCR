#!/usr/bin/env python
"""Legacy L0-L3: A0 + log_n / OD on linear_extra (P2/P4, 5 seeds).

This is the cheap part of the A2b+log+OD plan.  It uses the existing legacy
chart table (chart-level mean) and only adds chart-level extra signals:
    L0  msd_overall, sunny_star                 (legacy linear_extra control)
    L1  ... + log_n                             (free, crop-independent)
    L2  ... + od                                (osu!mania OverallDifficulty)
    L3  ... + log_n + od
No window recode and no canon features are needed.  L4 (legacy A2b) is
deliberately not here: it needs the legacy window store (see the plan).

Stages:
    train       per (protocol, seed, arm): step2_cm3p_rank.py linear_extra
    eval        per (protocol, seed): merge arms + fixed, rank eval + paired delta
    summarize   NLL / Pearson / Spearman tables and paired deltas

Examples:
    python scripts/run_od_legacy.py --stage train --protocols p2 --seeds 1 --arms l0,l3
    python scripts/run_od_legacy.py --stage eval --protocols p2 --seeds 1 --force-eval
    python scripts/run_od_legacy.py --stage summarize
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

CONTENT = ROOT / "data/processed/chart_content_cm3p_local.parquet"
SIGNALS = ROOT / "data/processed/rank_signals_randommerge_od.parquet"
SPLIT_P2 = ROOT / "data/processed/step1_randommerge_c_p2"
SPLIT_P4 = ROOT / "data/processed/step1_randommerge_c_p4"
OUT = ROOT / "results/randommerge_od_legacy"
CACHE = ROOT / "data/processed/rank_eval_cache"

ARMS = {
    "l0": ["msd_overall", "sunny_star"],
    "l1": ["msd_overall", "sunny_star", "log_n"],
    "l2": ["msd_overall", "sunny_star", "od"],
    "l3": ["msd_overall", "sunny_star", "log_n", "od"],
}
ARM_ORDER = ["l0", "l1", "l2", "l3"]
FIXED = ["sunny_star", "daniel_star", "star"]
PAIRS = [("l1", "l0"), ("l2", "l0"), ("l3", "l1"), ("l3", "l2"), ("l3", "l0")]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(c) for c in cmd]
    with log_path.open("w") as fh:
        fh.write(" ".join(cmd) + "\n\n")
        fh.flush()
        p = subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
                           text=True)
    if p.returncode != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-40:])
        raise SystemExit(f"command failed ({p.returncode}): {' '.join(cmd)}\n{tail}")


def _resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def _split(protocol: str, seed: int, p2: Path, p4: Path) -> Path:
    return Path(f"{p2 if protocol == 'p2' else p4}_s{seed}.parquet")


def _run_dir(protocol: str, seed: int, out: Path) -> Path:
    return out / protocol / f"s{seed}"


def stage_train(args) -> None:
    content = _resolve(args.content_path)
    signals = _resolve(args.signal_table)
    p2, p4 = _resolve(args.split_p2), _resolve(args.split_p4)
    out = _resolve(args.out_dir)
    for protocol in args.protocols:
        for seed in args.seeds:
            split = _split(protocol, seed, p2, p4)
            if not split.exists():
                raise SystemExit(f"missing split: {split}")
            run_dir = _run_dir(protocol, seed, out)
            for arm in args.arms:
                sig = run_dir / "signals" / f"od_{arm}.parquet"
                js = run_dir / "signals" / f"od_{arm}.json"
                if sig.exists() and js.exists() and not args.force:
                    log(f"{protocol} s{seed} {arm}: exists, skip")
                    continue
                cmd = [PY, "scripts/step2_cm3p_rank.py",
                       "--in", split, "--content-path", content,
                       "--content-col", args.content_col,
                       "--signal-table", signals,
                       "--extra-signals", ",".join(ARMS[arm]),
                       "--val-frac", "0", "--seed", "0",
                       "--threads", str(args.threads),
                       "--model", "linear_extra", "--lam", "1e-3",
                       "--name", f"od_{arm}",
                       "--out-signals", sig, "--out-json", js]
                if protocol == "p2":
                    cmd += ["--chart-only", "--chart-folds", "5",
                            "--chart-seed", "0"]
                log(f"{protocol} s{seed} {arm}: train linear_extra "
                    f"({','.join(ARMS[arm])})")
                run(cmd, out / "logs" / f"train_{protocol}_s{seed}_{arm}.log")


def stage_eval(args) -> None:
    signals = _resolve(args.signal_table)
    p2, p4 = _resolve(args.split_p2), _resolve(args.split_p4)
    out = _resolve(args.out_dir)
    cache = _resolve(args.cache_dir)
    for protocol in args.protocols:
        for seed in args.seeds:
            split = _split(protocol, seed, p2, p4)
            run_dir = _run_dir(protocol, seed, out)
            sig_path = run_dir / "signals_pool.parquet"
            ev_path = run_dir / "rank_eval.json"
            pd_path = run_dir / "paired_delta.json"
            if not args.force_eval and ev_path.exists() and pd_path.exists():
                log(f"{protocol} s{seed}: eval exists, skip")
                continue
            frames = []
            for arm in args.arms:
                p = run_dir / "signals" / f"od_{arm}.parquet"
                if not p.exists():
                    raise SystemExit(f"missing {p}: run --stage train first")
                col = f"od_{arm}"
                d = pd.read_parquet(p)
                if col not in d:
                    raise SystemExit(f"{p}: missing column {col}")
                frames.append(d[["beatmap_id", col]].set_index("beatmap_id"))
            base = pd.concat(frames, axis=1)
            fixed = pd.read_parquet(
                signals, columns=["beatmap_id"] + FIXED).set_index("beatmap_id")
            out_df = base.join(fixed, how="inner")
            n_charts = int(pd.read_parquet(
                split, columns=["beatmap_id"])["beatmap_id"].nunique())
            if len(out_df) != n_charts:
                raise SystemExit(f"{protocol} s{seed}: assembled {len(out_df)} "
                                 f"charts, expected {n_charts}")
            if out_df.isna().any().any():
                raise SystemExit(f"{protocol} s{seed}: NaN in assembled table")
            out_df.reset_index().to_parquet(sig_path, index=False)
            log(f"{protocol} s{seed}: {sig_path.name} "
                f"{len(out_df)} x {len(out_df.columns)}")
            names = [f"od_{a}" for a in args.arms] + FIXED
            if args.force_eval or not ev_path.exists():
                run([PY, "scripts/step2_rank_eval.py",
                     "--in", split, "--signal-table", sig_path,
                     "--signals", ",".join(names), "--split", "test",
                     "--ceiling-fit", "all", "--common-maps", "--strict-pairs",
                     "--folds", "2", "--bootstrap", "0", "--seed", "0",
                     "--min-l", str(args.min_l), "--cache-dir", cache,
                     "--out", ev_path],
                    out / "logs" / f"eval_{protocol}_s{seed}.log")
            if args.force_eval or not pd_path.exists():
                pairs = ",".join(f"od_{a}:od_{b}" for a, b in PAIRS)
                run([PY, "scripts/paired_delta_nll.py",
                     "--in", split, "--signal-table", sig_path,
                     "--signals", ",".join(names), "--pairs", pairs,
                     "--split", "test", "--bootstrap", str(args.bootstrap),
                     "--seed", "0", "--out", pd_path],
                    out / "logs" / f"paired_{protocol}_s{seed}.log")


def _mean_std(vals):
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), (float(a.std(ddof=1)) if a.size > 1 else 0.0)


def stage_summarize(args) -> None:
    out = _resolve(args.out_dir)
    rows = []
    for protocol in args.protocols:
        for seed in args.seeds:
            p = _run_dir(protocol, seed, out) / "rank_eval.json"
            if not p.exists():
                continue
            j = json.loads(p.read_text())
            for arm in args.arms:
                s = j.get("signals", {}).get(f"od_{arm}")
                if s is None:
                    continue
                rows.append({"protocol": protocol, "seed": seed, "arm": arm,
                             "nll": s.get("nll_per_instance"),
                             "pearson": s.get("pearson_bhat"),
                             "spearman": s.get("rho_bhat")})
    if not rows:
        raise SystemExit("no rank_eval.json found")
    long = pd.DataFrame(rows).sort_values(["protocol", "arm", "seed"])
    long.to_csv(out / "summary.csv", index=False)

    means = []
    for (protocol, arm), sub in long.groupby(["protocol", "arm"], sort=False):
        rec = {"protocol": protocol, "arm": arm,
               "n_seeds": int(sub.seed.nunique())}
        for col in ("nll", "pearson", "spearman"):
            mu, sd = _mean_std(sub[col].tolist())
            rec[f"{col}_mean"], rec[f"{col}_std"] = mu, sd
        means.append(rec)
    mean = pd.DataFrame(means)
    mean.to_csv(out / "summary_mean_std.csv", index=False)

    pair_rows = []
    for protocol in args.protocols:
        for a, b in PAIRS:
            sa = long[(long.protocol == protocol) & (long.arm == a)]
            sb = long[(long.protocol == protocol) & (long.arm == b)]
            if sa.empty or sb.empty:
                continue
            m = sa.set_index("seed")[["nll", "pearson", "spearman"]].join(
                sb.set_index("seed")[["nll", "pearson", "spearman"]],
                how="inner", lsuffix="_a", rsuffix="_b")
            dn = (m["nll_a"] - m["nll_b"]).dropna()
            dp = (m["pearson_a"] - m["pearson_b"]).dropna()
            ds = (m["spearman_a"] - m["spearman_b"]).dropna()
            mu, sd = _mean_std(dn.tolist())
            pmu, psd = _mean_std(dp.tolist())
            smu, ssd = _mean_std(ds.tolist())
            pair_rows.append({
                "protocol": protocol, "pair": f"{a}:{b}",
                "n_seeds": int(dn.size),
                "delta_nll_mean": mu, "delta_nll_sd": sd,
                "seeds_nll_better": int((dn < 0).sum()),
                "delta_pearson_mean": pmu, "delta_pearson_sd": psd,
                "seeds_pearson_better": int((dp > 0).sum()),
                "delta_spearman_mean": smu, "delta_spearman_sd": ssd,
                "seeds_spearman_better": int((ds > 0).sum()),
                "per_seed_nll": ",".join(f"{x:+.5f}" for x in dn),
                "per_seed_pearson": ",".join(f"{x:+.4f}" for x in dp),
                "per_seed_spearman": ",".join(f"{x:+.4f}" for x in ds),
            })
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(out / "summary_pairs.csv", index=False)
    (out / "summary.json").write_text(json.dumps(
        {"arms": means, "pairs": pair_rows}, indent=2, ensure_ascii=False))

    lines = ["# Legacy L0-L3: A0 + log_n / OD on linear_extra", "",
             "L0 = legacy A0 + msd_overall,sunny_star (control); "
             "L1 = +log_n; L2 = +od; L3 = +log_n+od.", ""]
    for protocol, title in (("p2", "P2 (new charts)"), ("p4", "P4 (cold start)")):
        sub = mean[mean.protocol == protocol]
        if sub.empty:
            continue
        lines += [f"## {title}", "",
                  "| arm | NLL | Pearson | Spearman | n |",
                  "|---|---:|---:|---:|---:|"]
        for _, r in sub.iterrows():
            lines.append(f"| {r.arm} | {r.nll_mean:.5f} ± {r.nll_std:.5f} | "
                         f"{r.pearson_mean:.4f} ± {r.pearson_std:.4f} | "
                         f"{r.spearman_mean:.4f} ± {r.spearman_std:.4f} | "
                         f"{int(r.n_seeds)} |")
        pr = pairs[pairs.protocol == protocol]
        if not pr.empty:
            lines += ["", "| pair | ΔNLL | NLL seeds | ΔPearson | P seeds | "
                      "ΔSpearman | S seeds |",
                      "|---|---:|---:|---:|---:|---:|---:|"]
            for _, r in pr.iterrows():
                lines.append(
                    f"| {r.pair} | {r.delta_nll_mean:+.5f} ± {r.delta_nll_sd:.5f} | "
                    f"{int(r.seeds_nll_better)}/{int(r.n_seeds)} | "
                    f"{r.delta_pearson_mean:+.4f} | "
                    f"{int(r.seeds_pearson_better)}/{int(r.n_seeds)} | "
                    f"{r.delta_spearman_mean:+.4f} | "
                    f"{int(r.seeds_spearman_better)}/{int(r.n_seeds)} |")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    log(f"summarize -> {out/'summary.md'} / summary.csv / summary_pairs.csv")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all",
                    choices=["all", "train", "eval", "summarize"])
    ap.add_argument("--protocols", default="p2,p4")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--arms", default=",".join(ARM_ORDER))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--force-eval", action="store_true")
    ap.add_argument("--content-path", default=str(CONTENT.relative_to(ROOT)))
    ap.add_argument("--content-col", default="embedding")
    ap.add_argument("--signal-table", default=str(SIGNALS.relative_to(ROOT)))
    ap.add_argument("--split-p2", default=str(SPLIT_P2.relative_to(ROOT)))
    ap.add_argument("--split-p4", default=str(SPLIT_P4.relative_to(ROOT)))
    ap.add_argument("--out-dir", default=str(OUT.relative_to(ROOT)))
    ap.add_argument("--cache-dir", default=str(CACHE.relative_to(ROOT)))
    ap.add_argument("--min-l", type=int, default=10)
    ap.add_argument("--bootstrap", type=int, default=400)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.protocols = [s.strip() for s in args.protocols.split(",") if s.strip()]
    args.seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    args.arms = [s.strip() for s in args.arms.split(",") if s.strip()]
    for a in args.arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a!r}; choose from {list(ARMS)}")
    for p in (_resolve(args.content_path), _resolve(args.signal_table)):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    out = _resolve(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log(f"arms={args.arms} protocols={args.protocols} seeds={args.seeds} "
        f"stage={args.stage}")
    if args.stage in ("all", "train"):
        stage_train(args)
    if args.stage in ("all", "eval"):
        stage_eval(args)
    if args.stage in ("all", "summarize"):
        stage_summarize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
