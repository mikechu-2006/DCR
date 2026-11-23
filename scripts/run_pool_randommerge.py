#!/usr/bin/env python
"""Part B driver: per-seed A0 heads + CM3P pooling features -> linear_extra.

The features are NOT all per-chart invariants: A0/A0L/A1 are label-free, but
A2/A2b/A2c/A3 use z_t = w . (W_t - mu) / sd, where w is a head fit on rank
labels.  This driver therefore fits one head per split seed (P2: the same
5 chart folds as the downstream model; P4: the P4 train rows only) and rebuilds
the z features for that seed.  All downstream models are linear_extra.
MLP is intentionally not used (random10k shows no gain over linear).

Stages (one slurm task per stage item; run shared first):
    shared     A0/A0L/A1 features (label-free; once)
    features   per seed: A0 head on the split's train rows, then A2/A2b/A2c/A3
    train      per (protocol, seed, variant): step2_cm3p_rank.py linear_extra
    eval       per seed: merge variant signals + fixed, rank eval + paired delta
    summarize  all rank_eval / paired JSON -> summary.csv / summary.md

Examples
    python scripts/run_pool_randommerge.py --stage shared
    python scripts/run_pool_randommerge.py --stage features --protocols p2 --seeds 1
    python scripts/run_pool_randommerge.py --stage train --protocols p2 --seeds 1 --variants a2b
    python scripts/run_pool_randommerge.py --stage eval --protocols p2 --seeds 1
    python scripts/run_pool_randommerge.py --stage summarize
    python scripts/run_pool_randommerge.py --stage all --protocols p2 --seeds 1
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
sys.path.insert(0, str(ROOT))
from dcr import cm3p_pool as pool  # noqa: E402

PY = sys.executable
WIN_INDEX = ROOT / ".cache/cm3p/index.parquet"
WIN_DIR = ROOT / ".cache/cm3p/win"
SIGNALS = ROOT / "data/processed/rank_signals_1k.parquet"
CACHE = ROOT / "data/processed/rank_eval_cache"
FEAT = ROOT / "data/processed/chart_pool_features_randommerge"
OUT = ROOT / "results/randommerge_pool"
SPLIT_P2 = ROOT / "data/processed/step1_randommerge_c_p2"
SPLIT_P4 = ROOT / "data/processed/step1_randommerge_c_p4"

VARIANTS = ["a0", "a0l", "a1", "a2", "a2b", "a2c", "a3"]
LABEL_FREE = ["a0", "a0l", "a1"]
Z_VARIANTS = ["a2", "a2b", "a2c", "a3"]
PAIRS = [
    "pool_a2b:pool_a0",
    "pool_a2b:pool_a0l",
    "pool_a1:pool_a0",
    "pool_a2:pool_a0l",
    "pool_a2b:pool_a2",
    "pool_a2c:pool_a0l",
    "pool_a2c:pool_a2b",
    "pool_a3:pool_a0l",
    "pool_a3:pool_a2",
]


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


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def _split(protocol: str, seed: int, p2: Path, p4: Path) -> Path:
    prefix = p2 if protocol == "p2" else p4
    return Path(f"{prefix}_s{seed}.parquet")


def _head(protocol: str, seed: int) -> Path:
    return OUT / "heads" / f"a0_heads_{protocol}_s{seed}.npz"


def _feat_dir(protocol: str, seed: int) -> Path:
    return FEAT / protocol / f"s{seed}"


def _run_dir(protocol: str, seed: int) -> Path:
    return OUT / protocol / f"s{seed}"


def _have(paths: list[Path], force: bool) -> bool:
    if not force and paths and all(p.exists() for p in paths):
        return True
    return False


# ----------------------------------------------------------------- shared
def stage_shared(args) -> None:
    todo = [] if args.force else [
        v for v in LABEL_FREE if not (FEAT / "shared" / f"{v}.parquet").exists()]
    if not todo:
        log("shared: exists, skip")
        return
    log(f"shared: build {todo}")
    cmd = [PY, "scripts/cm3p_pool.py",
           "--variants", ",".join(todo),
           "--index", args.window_index, "--win-dir", args.window_dir,
           "--out-dir", FEAT / "shared", "--threads", str(args.threads)]
    run(cmd, OUT / "logs" / "shared.log")
    for v in todo:
        p = FEAT / "shared" / f"{v}.parquet"
        if not p.exists():
            raise SystemExit(f"shared: {p} missing after build")


# ----------------------------------------------------------------- features
def stage_features(args) -> None:
    if not (FEAT / "shared" / "a0.parquet").exists():
        raise SystemExit(f"{FEAT/'shared'/'a0.parquet'} missing: run --stage shared")
    p2, p4 = _resolve(args.split_p2), _resolve(args.split_p4)
    for protocol in args.protocols:
        for seed in args.seeds:
            split = _split(protocol, seed, p2, p4)
            if not split.exists():
                raise SystemExit(f"missing split: {split}")
            head = _head(protocol, seed)
            if args.force or not head.exists():
                cmd = [PY, "scripts/step2_cm3p_rank.py",
                       "--in", split,
                       "--content-path", FEAT / "shared" / "a0.parquet",
                       "--content-col", "features",
                       "--signal-table", args.signal_table,
                       "--extra-signals", "msd_overall,sunny_star",
                       "--val-frac", "0", "--seed", "0",
                       "--threads", str(args.threads),
                       "--model", "linear", "--chart-only",
                       "--chart-folds", str(args.head_folds), "--chart-seed", "0",
                       "--lam", str(args.lam),
                       "--name", f"a0_heads_{protocol}_s{seed}",
                       "--dump-heads", head,
                       "--out-signals", OUT / "heads"
                       / f"a0_signal_{protocol}_s{seed}.parquet",
                       "--out-json", OUT / "heads"
                       / f"a0_signal_{protocol}_s{seed}.json"]
                run(cmd, OUT / "logs" / f"head_{protocol}_s{seed}.log")
            _audit_head(head, split, protocol, seed, args.head_folds)

            feat_dir = _feat_dir(protocol, seed)
            todo = [] if args.force else [
                v for v in Z_VARIANTS if not (feat_dir / f"{v}.parquet").exists()]
            if todo:
                mode = "fold" if protocol == "p2" else "full"
                cmd = [PY, "scripts/cm3p_pool.py",
                       "--variants", ",".join(todo),
                       "--heads", head, "--head-mode", mode,
                       "--index", args.window_index, "--win-dir", args.window_dir,
                       "--out-dir", feat_dir, "--threads", str(args.threads)]
                run(cmd, OUT / "logs" / f"features_{protocol}_s{seed}.log")
            for v in Z_VARIANTS:
                p = feat_dir / f"{v}.parquet"
                if not p.exists():
                    raise SystemExit(f"{p} missing after build")
                d = pd.read_parquet(p, columns=["features"])
                width = int(d["features"].iloc[0].size)
                want = int(pool.WIDTH[v])
                if width != want:
                    raise SystemExit(f"{p}: width {width} != {want}")
                log(f"{protocol} s{seed} {v}: {len(d):,} x {width}")


def _audit_head(head: Path, split: Path, protocol: str, seed: int,
                head_folds: int) -> None:
    z = np.load(head, allow_pickle=False)
    meta = {k: z[k].item() for k in z.files
            if k in ("inp", "model", "chart_seed", "lam", "val_frac")}
    log(f"{protocol} s{seed} head {head.name}: {meta}")
    if int(np.asarray(z["chart_folds"]).item()) != int(head_folds):
        raise SystemExit(f"{head}: chart_folds != {head_folds}")
    inp = str(meta.get("inp", ""))
    if Path(inp).name != Path(split).name:
        raise SystemExit(f"{head}: fit on {inp}, expected split {split}")


# -------------------------------------------------------------------- train
def stage_train(args) -> None:
    p2, p4 = _resolve(args.split_p2), _resolve(args.split_p4)
    for protocol in args.protocols:
        for seed in args.seeds:
            split = _split(protocol, seed, p2, p4)
            if not split.exists():
                raise SystemExit(f"missing split: {split}")
            run_dir = _run_dir(protocol, seed)
            for v in args.variants:
                sig = run_dir / "signals" / f"pool_{v}.parquet"
                js = run_dir / "signals" / f"pool_{v}.json"
                if sig.exists() and js.exists() and not args.force:
                    log(f"{protocol} s{seed} {v}: exists, skip")
                    continue
                if v in LABEL_FREE:
                    content = FEAT / "shared" / f"{v}.parquet"
                else:
                    content = _feat_dir(protocol, seed) / f"{v}.parquet"
                if not content.exists():
                    raise SystemExit(f"missing content {content}: "
                                     f"run --stage features first")
                cmd = [PY, "scripts/step2_cm3p_rank.py",
                       "--in", split,
                       "--content-path", content,
                       "--content-col", "features",
                       "--signal-table", args.signal_table,
                       "--extra-signals", "msd_overall,sunny_star",
                       "--val-frac", "0", "--seed", "0",
                       "--threads", str(args.threads),
                       "--model", "linear_extra", "--lam", "1e-3",
                       "--name", f"pool_{v}",
                       "--out-signals", sig, "--out-json", js]
                if protocol == "p2":
                    cmd += ["--chart-only", "--chart-folds", "5",
                            "--chart-seed", "0"]
                log(f"{protocol} s{seed} {v}: train linear_extra")
                run(cmd, OUT / "logs" / f"train_{protocol}_s{seed}_{v}.log")


# --------------------------------------------------------------------- eval
def stage_eval(args) -> None:
    p2, p4 = _resolve(args.split_p2), _resolve(args.split_p4)
    for protocol in args.protocols:
        for seed in args.seeds:
            split = _split(protocol, seed, p2, p4)
            run_dir = _run_dir(protocol, seed)
            sig_path = run_dir / "signals_pool.parquet"
            ev_path = run_dir / "rank_eval.json"
            pd_path = run_dir / "paired_delta.json"
            if not args.force_eval and ev_path.exists() and pd_path.exists():
                log(f"{protocol} s{seed}: eval exists, skip")
                continue
            frames = []
            for v in args.variants:
                p = run_dir / "signals" / f"pool_{v}.parquet"
                if not p.exists():
                    raise SystemExit(f"missing {p}: run --stage train first")
                d = pd.read_parquet(p)
                col = f"pool_{v}"
                if col not in d:
                    raise SystemExit(f"{p}: missing column {col}")
                frames.append(d[["beatmap_id", col]].set_index("beatmap_id"))
            base = pd.concat(frames, axis=1)
            fixed = pd.read_parquet(
                args.signal_table,
                columns=["beatmap_id", "sunny_star", "daniel_star", "star"]
            ).set_index("beatmap_id")
            out = base.join(fixed, how="inner")
            n_charts = int(pd.read_parquet(
                split, columns=["beatmap_id"])["beatmap_id"].nunique())
            if len(out) != n_charts:
                raise SystemExit(f"{protocol} s{seed}: assembled {len(out)} charts, "
                                 f"expected {n_charts}")
            if out.isna().any().any():
                raise SystemExit(f"{protocol} s{seed}: NaN in assembled signal table")
            out.reset_index().to_parquet(sig_path, index=False)
            log(f"{protocol} s{seed}: {sig_path.name} "
                f"{len(out)} x {len(out.columns)}")
            names = [f"pool_{v}" for v in args.variants] + [
                "sunny_star", "daniel_star", "star"]
            if args.force_eval or not ev_path.exists():
                run([PY, "scripts/step2_rank_eval.py",
                     "--in", split, "--signal-table", sig_path,
                     "--signals", ",".join(names), "--split", "test",
                     "--ceiling-fit", "all", "--common-maps", "--strict-pairs",
                     "--folds", "2", "--bootstrap", "0", "--seed", "0",
                     "--min-l", str(args.min_l), "--cache-dir", args.cache_dir,
                     "--out", ev_path],
                    OUT / "logs" / f"eval_{protocol}_s{seed}.log")
            if args.force_eval or not pd_path.exists():
                run([PY, "scripts/paired_delta_nll.py",
                     "--in", split, "--signal-table", sig_path,
                     "--signals", ",".join(names),
                     "--pairs", ",".join(PAIRS),
                     "--split", "test", "--bootstrap", str(args.bootstrap),
                     "--seed", "0", "--out", pd_path],
                    OUT / "logs" / f"paired_{protocol}_s{seed}.log")


# ---------------------------------------------------------------- summarize
def _mean_std(vals):
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), (float(a.std(ddof=1)) if a.size > 1 else 0.0)


def stage_summarize(args) -> None:
    rows = []
    for protocol in args.protocols:
        for seed in args.seeds:
            p = _run_dir(protocol, seed) / "rank_eval.json"
            if not p.exists():
                continue
            j = json.loads(p.read_text())
            for v in args.variants:
                s = j.get("signals", {}).get(f"pool_{v}")
                if s is None:
                    continue
                rows.append({"protocol": protocol, "seed": seed, "variant": v,
                             "nll": s.get("nll_per_instance"),
                             "pearson": s.get("pearson_bhat"),
                             "spearman": s.get("rho_bhat")})
    if not rows:
        raise SystemExit("no rank_eval.json found for the requested seeds")
    long = pd.DataFrame(rows).sort_values(["protocol", "variant", "seed"])
    long.to_csv(OUT / "summary.csv", index=False)

    means = []
    for (protocol, v), sub in long.groupby(["protocol", "variant"], sort=False):
        rec = {"protocol": protocol, "variant": v, "n_seeds": int(sub.seed.nunique())}
        for col in ("nll", "pearson", "spearman"):
            mu, sd = _mean_std(sub[col].tolist())
            rec[f"{col}_mean"], rec[f"{col}_std"] = mu, sd
        means.append(rec)
    mean = pd.DataFrame(means)
    mean.to_csv(OUT / "summary_mean_std.csv", index=False)

    pair_rows = []
    for protocol in args.protocols:
        for a, b in [(x.split(":")[0], x.split(":")[1]) for x in PAIRS]:
            va, vb = a.replace("pool_", ""), b.replace("pool_", "")
            sa = long[(long.protocol == protocol) & (long.variant == va)]
            sb = long[(long.protocol == protocol) & (long.variant == vb)]
            if sa.empty or sb.empty:
                continue
            m = sa.set_index("seed")["nll"].to_frame("a").join(
                sb.set_index("seed")["nll"].to_frame("b"), how="inner")
            d = (m["a"] - m["b"]).dropna()
            mu, sd = _mean_std(d.tolist())
            pair_rows.append({"protocol": protocol, "pair": f"{a}:{b}",
                              "n_seeds": int(d.size), "delta_mean": mu,
                              "delta_sd": sd,
                              "seeds_better": int((d < 0).sum()),
                              "per_seed": ",".join(f"{x:+.5f}" for x in d)})
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(OUT / "summary_pairs.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({
        "variants": means, "pairs": pair_rows}, indent=2, ensure_ascii=False))

    lines = ["# Random merge: canon pooling (linear_extra)", "",
             "Per-seed heads; P2 5 chart folds, P4 train-only full head.",
             "Delta < 0 means the first variant has lower NLL.", ""]
    for protocol, title in (("p2", "P2 (new charts)"), ("p4", "P4 (cold start)")):
        sub = mean[mean.protocol == protocol]
        if sub.empty:
            continue
        lines += [f"## {title}", "", "| variant | NLL mean | sd | n |",
                  "|---|---:|---:|---:|"]
        for _, r in sub.iterrows():
            lines.append(f"| {r.variant} | {r.nll_mean:.5f} | {r.nll_std:.5f} | "
                         f"{int(r.n_seeds)} |")
        pr = pairs[pairs.protocol == protocol]
        if not pr.empty:
            lines += ["", "| pair | mean Δ | sd | seeds better | per seed |",
                      "|---|---:|---:|---:|---|"]
            for _, r in pr.iterrows():
                lines.append(f"| {r.pair} | {r.delta_mean:+.5f} | {r.delta_sd:.5f} | "
                             f"{int(r.seeds_better)}/{int(r.n_seeds)} | {r.per_seed} |")
        lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    log(f"summarize -> {OUT/'summary.csv'} / summary_mean_std.csv / "
        f"summary_pairs.csv / summary.md")


# ---------------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all",
                    choices=["all", "shared", "features", "train", "eval",
                             "summarize"])
    ap.add_argument("--protocols", default="p2,p4")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--force", action="store_true",
                    help="rebuild features / retrain")
    ap.add_argument("--force-eval", action="store_true")
    ap.add_argument("--window-index", default=str(WIN_INDEX.relative_to(ROOT)))
    ap.add_argument("--window-dir", default=str(WIN_DIR.relative_to(ROOT)))
    ap.add_argument("--signal-table", default=str(SIGNALS.relative_to(ROOT)))
    ap.add_argument("--cache-dir", default=str(CACHE.relative_to(ROOT)))
    ap.add_argument("--split-p2", default=str(SPLIT_P2.relative_to(ROOT)))
    ap.add_argument("--split-p4", default=str(SPLIT_P4.relative_to(ROOT)))
    ap.add_argument("--feature-dir", default=str(FEAT.relative_to(ROOT)))
    ap.add_argument("--out-dir", default=str(OUT.relative_to(ROOT)))
    ap.add_argument("--head-folds", type=int, default=5)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--min-l", type=int, default=10)
    ap.add_argument("--bootstrap", type=int, default=400)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    global WIN_INDEX, WIN_DIR, SIGNALS, CACHE, FEAT, OUT
    args = parse_args(argv)
    args.protocols = [s.strip() for s in args.protocols.split(",") if s.strip()]
    args.seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    args.variants = [s.strip() for s in args.variants.split(",") if s.strip()]
    for v in args.variants:
        if v not in VARIANTS:
            raise SystemExit(f"variant {v!r} not supported by this driver "
                             f"(choose from {VARIANTS})")
    WIN_INDEX = _resolve(args.window_index)
    WIN_DIR = _resolve(args.window_dir)
    SIGNALS = _resolve(args.signal_table)
    CACHE = _resolve(args.cache_dir)
    FEAT = _resolve(args.feature_dir)
    OUT = _resolve(args.out_dir)
    args.window_index = str(WIN_INDEX)
    args.window_dir = str(WIN_DIR)
    args.signal_table = str(SIGNALS)
    args.cache_dir = str(CACHE)
    args.feature_dir = str(FEAT)
    args.out_dir = str(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    if not WIN_INDEX.exists():
        raise SystemExit(f"missing window index {WIN_INDEX}")
    if not WIN_DIR.exists():
        raise SystemExit(f"missing window dir {WIN_DIR}")
    if not SIGNALS.exists():
        raise SystemExit(f"missing signal table {SIGNALS}")
    log(f"variants={args.variants} protocols={args.protocols} seeds={args.seeds} "
        f"stage={args.stage}")
    if args.stage in ("all", "shared"):
        stage_shared(args)
    if args.stage in ("all", "features"):
        stage_features(args)
    if args.stage in ("all", "train"):
        stage_train(args)
    if args.stage in ("all", "eval"):
        stage_eval(args)
    if args.stage in ("all", "summarize"):
        stage_summarize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
