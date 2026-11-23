#!/usr/bin/env python
"""T3: build per-chart features from the CM3P window store (or the A6 signal).

Examples
    # A0/A0L/A1 need no heads; A2/A2b/A2c/A3 do.
    # P2: --head-mode fold (each chart's own fold head)
    # P4: --head-mode full (w_full fit on the P4 train rows only)
    python scripts/cm3p_pool.py --variants a0,a0l,a1,a2,a2b,a2c,a3 \
        --heads results/randommerge_pool/heads/a0_heads_p2_s1.npz \
        --head-mode fold --out-dir data/processed/chart_pool_features_randommerge/p2/s1

    # the plan's named artifact (A2)
    python scripts/cm3p_pool.py --variants a2 --heads ... \
        --out data/processed/chart_pool_features_1k.parquet

    # A6: a chart-disjoint TCN signal (needs step1 + the A0 fold heads)
    python scripts/cm3p_pool.py --variants a6 --heads ...a0_heads.npz \
        --step1 data/processed/step1_1k_pairbest.parquet \
        --out-signals results/cm3p_pool_1k/a6_signal.parquet

All variants are built from the same canon T1 window store.  A0 is the T3
control; it is not required to reproduce the legacy chart table.

A2/A3/A4 need the A0 linear head bundle written by
    scripts/step2_cm3p_rank.py --model linear --chart-folds 5 --chart-only \
        --dump-heads .../a0_heads.npz
The bundle carries the per-fold chart-disjoint weight and the train-chart
standardisation, so z_t for a held-out chart never uses that chart's labels.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dcr import cm3p_pool as pool  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default="a0",
                    help="comma-separated: " + ",".join(pool.VARIANTS))
    ap.add_argument("--index", default=".cache/cm3p/index.parquet")
    ap.add_argument("--win-dir", default=".cache/cm3p/win")
    ap.add_argument("--heads", default=None,
                    help="A0 head bundle (.npz) from step2_cm3p_rank --dump-heads")
    ap.add_argument("--out", default=None,
                    help="single output parquet (beatmap_id + features)")
    ap.add_argument("--out-dir", default=None,
                    help="one <variant>.parquet per variant (used when >1 variant)")
    ap.add_argument("--a4-beta", type=float, default=5.0)
    ap.add_argument("--a4-mode", default="weighted", choices=["weighted", "lse"])
    ap.add_argument("--head-mode", default="fold", choices=["fold", "full"],
                    help="fold = each chart's own fold head (P2); "
                         "full = w_full fit on all train rows (P4)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--exclude-truncated", action="store_true",
                    help="drop charts with n_core_tokens_dropped > 0 from the features")
    ap.add_argument("--step1", default="data/processed/step1_1k_pairbest.parquet",
                    help="A6 only")
    ap.add_argument("--out-signals", default=None, help="A6 only: signal parquet")
    ap.add_argument("--epochs", type=int, default=80, help="A6 only")
    ap.add_argument("--hid", type=int, default=128, help="A6 only")
    ap.add_argument("--lr", type=float, default=1e-3, help="A6 only")
    ap.add_argument("--val-frac", type=float, default=0.2, help="A6 only")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main() -> int:
    args = parse_args()
    variants = [v.strip().lower() for v in args.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in pool.VARIANTS]
    if bad:
        raise SystemExit(f"unknown variants {bad}; choose from {pool.VARIANTS}")
    heads = pool.HeadBundle.load(args.heads) if args.heads else None
    if any(v in ("a2", "a2b", "a2c", "a3", "a4") for v in variants) and heads is None:
        raise SystemExit("a2/a2b/a2c/a3/a4 need --heads (the A0 linear head bundle)")
    if "a6" in variants:
        if len(variants) > 1:
            raise SystemExit("run a6 on its own: it trains a signal, not a fixed "
                             "feature matrix")
        return run_a6(args, heads)

    index = pool.load_index(args.index)
    if args.exclude_truncated and "n_core_tokens_dropped" in index:
        n0 = len(index)
        index = index[index["n_core_tokens_dropped"].fillna(0).astype(int) == 0]
        log(f"excluded {n0 - len(index):,} chart(s) with truncated core tokens")
    if args.limit:
        index = index.head(args.limit)
    log(f"window index: {len(index):,} charts; variants={variants}")
    if heads is not None:
        log(f"heads: {heads.w.shape[0]} fold head(s) x {heads.w.shape[1]} dims")

    if args.out:
        if len(variants) > 1:
            raise SystemExit("--out is for a single variant; use --out-dir")
        outs = {variants[0]: Path(args.out)}
    elif args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        outs = {v: out_dir / f"{v}.parquet" for v in variants}
    elif len(variants) > 1:
        out_dir = Path("data/processed/chart_pool_features_1k")
        out_dir.mkdir(parents=True, exist_ok=True)
        outs = {v: out_dir / f"{v}.parquet" for v in variants}
    else:
        outs = {variants[0]: Path("data/processed/chart_pool_features_1k.parquet")}

    meta = {"variants": variants, "index": str(args.index), "win_dir": str(args.win_dir),
            "heads": str(args.heads) if args.heads else None,
            "head_mode": args.head_mode,
            "a4_beta": args.a4_beta, "a4_mode": args.a4_mode,
            "n_charts": int(len(index)), "widths": {}, "seconds": {}}
    for v in variants:
        t0 = time.time()
        df = pool.build_feature_table(index, args.win_dir, v, heads,
                                      beta=args.a4_beta, a4_mode=args.a4_mode,
                                      head_mode=args.head_mode, log=log)
        out = outs[v]
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".part")
        df.to_parquet(tmp, index=False)
        tmp.replace(out)
        width = int(df["features"].iloc[0].size) if len(df) else 0
        meta["widths"][v] = width
        meta["seconds"][v] = round(time.time() - t0, 1)
        log(f"{v}: {len(df):,} charts x {width} dims -> {out} "
            f"({meta['seconds'][v]:.0f}s)")
        if width != pool.WIDTH.get(v):
            log(f"WARNING: {v} width {width} != expected {pool.WIDTH.get(v)}")
    if len(variants) > 1 or args.out_dir:
        meta_dir = Path(args.out_dir or "data/processed/chart_pool_features_1k")
        (meta_dir / "meta.json").write_text(json.dumps(meta, indent=2,
                                                       ensure_ascii=False))
    return 0


def run_a6(args, heads) -> int:
    if heads is None:
        raise SystemExit("a6 needs --heads (for the chart-disjoint fold assignment)")
    if not args.out_signals:
        raise SystemExit("a6 needs --out-signals")
    from dcr import cm3p_pool as pool
    step1 = pd.read_parquet(args.step1)
    if "player_id_base" not in step1:
        step1["player_id_base"] = step1["player_id"]
    maps = np.asarray(heads.maps, dtype=np.int64)
    windows = {}
    t0 = time.time()
    for i, b in enumerate(maps):
        p = Path(args.win_dir) / f"{int(b)}.npy"
        if not p.exists():
            raise SystemExit(f"a6: {p} missing")
        windows[int(b)] = np.load(p).astype(np.float32)
        if (i + 1) % 2000 == 0:
            log(f"loaded {i + 1:,}/{len(maps):,} window files")
    log(f"loaded {len(windows):,} charts in {time.time() - t0:.0f}s")
    df = pool.train_a6_signal(step1, windows, heads, epochs=args.epochs, hid=args.hid,
                              lr=args.lr, val_frac=args.val_frac,
                              threads=args.threads, seed=args.seed, log=log)
    out = Path(args.out_signals)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    df.rename(columns={"signal": "a6"}).to_parquet(tmp, index=False)
    tmp.replace(out)
    log(f"a6: {len(df):,} charts -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
