#!/usr/bin/env python
"""Paired Delta-NLL inside strata of a per-chart variable (plan section 9).

Why: removing hitsounds can change the NLL through a confound rather than
through information loss -- hitsounds correlate with "this chart was authored
carefully", and the labels come from top-1000 players.  The plan compares
legacy vs canon *within* deciles of beatmap_meta.count_total.  This script is
that comparison, generically: any per-chart variable can be the stratum.

    python scripts/stratified_delta_nll.py \
        --signal-table results/cm3p_pool_1k/signals_pool.parquet \
        --signals pool_a0,cm3p_linear_cd_l1e3 \
        --pairs pool_a0:cm3p_linear_cd_l1e3 \
        --strata-table data/processed/rank_signals_1k.parquet \
        --stratum-col count_total --strata 10 \
        --split test --bootstrap 400 \
        --out results/cm3p_pool_1k/stratified_delta.json

Delta is NLL(A)-NLL(B); negative means A is better.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dcr import rank_metrics as rm  # noqa: E402


def _encode(df: pd.DataFrame, map_index: dict):
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
    cidx, cluster_ids = pd.factorize(np.asarray(clusters, dtype=np.int64), sort=True)
    return item, perf, groups, cidx, cluster_ids


def _pair_stats(step1: pd.DataFrame, sig: pd.DataFrame, a: str, b: str,
                split: str, bootstrap: int, seed: int):
    maps = np.sort(step1["beatmap_id"].unique())
    if maps.size < 2 or len(step1) == 0:
        return None
    map_idx = {int(m): i for i, m in enumerate(maps)}
    item, perf, groups, cidx, cluster_ids = _encode(step1, map_idx)
    tab = rm.build_pair_table(groups, item, perf, len(maps))
    if tab.n_pairs == 0:
        return None
    per = {}
    for name in (a, b):
        d = (sig.drop_duplicates("beatmap_id").set_index("beatmap_id")[name]
             .reindex(maps).to_numpy(np.float64))
        if not np.isfinite(d).all():
            return None
        link = rm.fit_link(tab, d, kind="linear")
        g = link.transform(d)
        nll, n = rm.nll_by_cluster(groups, item, perf, g, cidx, len(cluster_ids))
        per[name] = {"nll": nll, "n": n, "beta": float(link.beta),
                     "rate": float(nll.sum() / max(n.sum(), 1))}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(int(bootstrap)):
        draw = rng.integers(0, len(cluster_ids), len(cluster_ids))
        mult = np.bincount(draw, minlength=len(cluster_ids)).astype(np.float64)
        na = float((per[a]["n"] * mult).sum())
        nb = float((per[b]["n"] * mult).sum())
        if na <= 0 or nb <= 0:
            continue
        deltas.append(float((per[a]["nll"] * mult).sum()) / na
                      - float((per[b]["nll"] * mult).sum()) / nb)
    d = np.asarray(deltas)
    return {"n_rows": int(len(step1)), "n_maps": int(len(maps)),
            "n_pairs": int(tab.n_pairs), "n_instances": int(tab.n_instances),
            "n_clusters": int(len(cluster_ids)),
            "nll_a": per[a]["rate"], "nll_b": per[b]["rate"],
            "beta_a": per[a]["beta"], "beta_b": per[b]["beta"],
            "delta": per[a]["rate"] - per[b]["rate"],
            "ci": ([float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
                   if d.size else None),
            "p_delta_lt_0": float((d < 0).mean()) if d.size else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default="data/processed/step1_1k_pairbest.parquet")
    ap.add_argument("--signal-table", required=True)
    ap.add_argument("--signals", required=True, help="A,B")
    ap.add_argument("--pairs", required=True, help="A:B (one comparison)")
    ap.add_argument("--strata-table", default="data/processed/rank_signals_1k.parquet")
    ap.add_argument("--stratum-col", default="count_total")
    ap.add_argument("--strata", type=int, default=10)
    ap.add_argument("--split", default="test", choices=["all", "train", "test"])
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--min-instances", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    names = [x.strip() for x in args.signals.split(",") if x.strip()]
    if len(names) != 2:
        raise SystemExit("--signals needs exactly two columns A,B")
    a, b = names
    if [x.strip() for x in args.pairs.split(":")] != [a, b]:
        raise SystemExit(f"--pairs must be {a}:{b}")
    step1 = pd.read_parquet(args.inp)
    if "player_id_base" not in step1:
        step1["player_id_base"] = step1["player_id"]
    if args.split != "all":
        step1 = step1[step1["split"] == args.split]
    sig = pd.read_parquet(args.signal_table)
    for n in names:
        if n not in sig.columns:
            raise SystemExit(f"{args.signal_table}: missing {n}")
    # shared chart set: both signals + the stratum variable
    common = pd.Index(sig.loc[sig[a].notna() & sig[b].notna(), "beatmap_id"].unique())
    strata = pd.read_parquet(args.strata_table)
    if args.stratum_col not in strata.columns:
        raise SystemExit(f"{args.strata_table}: missing {args.stratum_col}")
    s = strata.set_index("beatmap_id")[args.stratum_col]
    common = common.intersection(s.dropna().index)
    step1 = step1[step1["beatmap_id"].isin(common).to_numpy()].reset_index(drop=True)
    s = s.reindex(np.sort(step1["beatmap_id"].unique()))
    bins = pd.qcut(s, args.strata, labels=False, duplicates="drop")
    print(f"[data] split={args.split} rows={len(step1):,} charts={len(s):,} "
          f"strata={int(bins.nunique())} by {args.stratum_col}", flush=True)

    out = {"meta": {"in": args.inp, "signal_table": args.signal_table,
                    "a": a, "b": b, "stratum_col": args.stratum_col,
                    "strata": int(bins.nunique()), "split": args.split,
                    "bootstrap": int(args.bootstrap), "seed": int(args.seed)},
           "overall": _pair_stats(step1, sig, a, b, args.split, args.bootstrap,
                                  args.seed),
           "strata": []}
    for k in range(int(bins.nunique())):
        ids = set(bins.index[bins == k].tolist())
        rows = step1[step1["beatmap_id"].isin(ids)]
        r = _pair_stats(rows, sig, a, b, args.split, args.bootstrap, args.seed + k + 1)
        if r is None or r["n_instances"] < args.min_instances:
            out["strata"].append({"stratum": k, "n_charts": len(ids),
                                  "n_instances": 0 if r is None else r["n_instances"],
                                  "skipped": True})
            print(f"  stratum {k}: {len(ids):,} charts -- skipped "
                  f"({0 if r is None else r['n_instances']:,} comparisons)", flush=True)
            continue
        r.update({"stratum": k, "n_charts": len(ids), "skipped": False})
        out["strata"].append(r)
        print(f"  stratum {k}: charts={len(ids):,} pairs={r['n_pairs']:,} "
              f"delta={r['delta']:+.5f} CI=[{r['ci'][0]:+.5f},{r['ci'][1]:+.5f}] "
              f"P(d<0)={r['p_delta_lt_0']:.3f}", flush=True)
    if out["overall"]:
        r = out["overall"]
        print(f"[overall] delta={r['delta']:+.5f} "
              f"CI=[{r['ci'][0]:+.5f},{r['ci'][1]:+.5f}] "
              f"P(d<0)={r['p_delta_lt_0']:.3f}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
