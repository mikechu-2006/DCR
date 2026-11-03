#!/usr/bin/env python
"""Paired cluster bootstrap for Delta NLL between per-chart signals.

The runner reports independent 95% CIs.  For "is A better than B" the right
check is a paired Delta on the same clusters/pairs.  This script builds the
within-player pair table once, computes each signal's per-cluster NLL at its
own fitted link, and resamples player_id_base clusters.

    python scripts/paired_delta_nll.py \
        --signal-table results/cm3p_rank_1k/signals_1k.parquet \
        --signals cm3p_linear_extra,cm3p_linear,ext_blend,sunny_star \
        --pairs cm3p_linear_extra:cm3p_linear,cm3p_linear_extra:ext_blend \
        --split test --bootstrap 400 \
        --out results/cm3p_rank_1k/paired_delta.json

Delta is NLL(A) - NLL(B) in nats per comparison; negative means A is better.
All signals must be non-null on the same charts (the intersection is used), so
the pair set is identical across signals.
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
from dcr import rank_metrics as rm  # noqa: E402


def _encode(df, map_index):
    item = df["beatmap_id"].map(map_index).to_numpy(np.int32)
    perf = (-df["loss"].astype(np.float64)).to_numpy()
    players = np.sort(df["player_id"].unique())
    pidx = {int(p): i for i, p in enumerate(players)}
    pi = df["player_id"].map(pidx).to_numpy(np.int32)
    tmp = pd.DataFrame({"pi": pi, "base": df["player_id_base"].to_numpy()})
    idx = tmp.groupby("pi").indices
    groups = [np.asarray(idx[i]) for i in range(len(players))]
    clusters = (tmp.groupby("pi")["base"].first()
                .reindex(range(len(players))).to_numpy())
    cidx, cluster_ids = pd.factorize(np.asarray(clusters, dtype=np.int64),
                                     sort=True)
    return item, perf, groups, cidx, cluster_ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",
                    default=str(ROOT / "data/processed/step1_1k_pairbest.parquet"))
    ap.add_argument("--signal-table", required=True)
    ap.add_argument("--signals", required=True)
    ap.add_argument("--pairs", required=True,
                    help="comma-separated A:B comparisons")
    ap.add_argument("--split", default="test", choices=["all", "train", "test"])
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()

    step1 = pd.read_parquet(args.inp)
    if "player_id_base" not in step1.columns:
        step1["player_id_base"] = step1["player_id"]
    if args.split != "all":
        step1 = step1[step1["split"] == args.split].reset_index(drop=True)
    sig = pd.read_parquet(args.signal_table)
    signals = [s.strip() for s in args.signals.split(",") if s.strip()]
    missing = [s for s in signals if s not in sig.columns]
    if missing:
        raise SystemExit(f"missing signal columns: {missing}")

    # same chart set for every signal, so every Delta is on the same pairs
    common = None
    for s in signals:
        ids = pd.Index(sig.loc[sig[s].notna(), "beatmap_id"].unique())
        common = ids if common is None else common.intersection(ids)
    if common is None or len(common) == 0:
        raise SystemExit("no chart has every requested signal")
    step1 = step1[step1["beatmap_id"].isin(common)].reset_index(drop=True)
    maps = np.sort(step1["beatmap_id"].unique())
    map_idx = {int(m): i for i, m in enumerate(maps)}
    item, perf, groups, cidx, cluster_ids = _encode(step1, map_idx)
    tab = rm.build_pair_table(groups, item, perf, len(maps))
    print(f"[data] split={args.split} rows={len(step1):,} charts={len(maps):,} "
          f"pairs={tab.n_pairs:,} comparisons={tab.n_instances:,} "
          f"clusters={len(cluster_ids)}", flush=True)

    per = {}
    for s in signals:
        d = (sig.drop_duplicates("beatmap_id").set_index("beatmap_id")[s]
             .reindex(maps).to_numpy(np.float64))
        link = rm.fit_link(tab, d, kind="linear")
        g = link.transform(d)
        nll, n = rm.nll_by_cluster(groups, item, perf, g, cidx,
                                   len(cluster_ids))
        per[s] = {"nll": nll, "n": n, "beta": float(link.beta),
                  "rate": float(nll.sum() / n.sum())}
        print(f"  {s:24s} nll={per[s]['rate']:.5f} beta={per[s]['beta']:.3f}",
              flush=True)

    rng = np.random.default_rng(args.seed)
    out = {"meta": {"in": str(args.inp), "signal_table": str(args.signal_table),
                    "split": args.split, "n_rows": int(len(step1)),
                    "n_maps": int(len(maps)), "n_pairs": int(tab.n_pairs),
                    "n_instances": int(tab.n_instances),
                    "n_clusters": int(len(cluster_ids)),
                    "bootstrap": int(args.bootstrap), "seed": int(args.seed)},
           "signals": {s: {k: v for k, v in per[s].items()
                           if k in ("beta", "rate")} for s in signals},
           "pairs": {}}
    for spec in [p.strip() for p in args.pairs.split(",") if p.strip()]:
        a, b = [x.strip() for x in spec.split(":")]
        if a not in per or b not in per:
            raise SystemExit(f"pair {spec}: signal not in --signals")
        point = float(per[a]["rate"] - per[b]["rate"])
        deltas, ca, cb = [], [], []
        for _ in range(int(args.bootstrap)):
            draw = rng.integers(0, len(cluster_ids), len(cluster_ids))
            mult = np.bincount(draw, minlength=len(cluster_ids)).astype(np.float64)
            na, nb = float((per[a]["n"] * mult).sum()),                 float((per[b]["n"] * mult).sum())
            if na <= 0 or nb <= 0:
                continue
            ra = float((per[a]["nll"] * mult).sum()) / na
            rb = float((per[b]["nll"] * mult).sum()) / nb
            deltas.append(ra - rb)
            ca.append(ra)
            cb.append(rb)
        d = np.asarray(deltas)
        out["pairs"][spec] = {
            "delta": point,
            "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_delta_lt_0": float((d < 0).mean()) if d.size else None,
            "nll_a": float(per[a]["rate"]), "nll_b": float(per[b]["rate"]),
        }
        print(f"[pair] {spec:48s} delta={point:+.5f} "
              f"CI=[{d.min():+.5f},{d.max():+.5f}] "
              f"P(delta<0)={out['pairs'][spec]['p_delta_lt_0']:.3f}",
              flush=True)

    out["meta"]["seconds"] = round(time.time() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {args.out} ({out['meta']['seconds']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
