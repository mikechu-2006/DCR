#!/usr/bin/env python3
"""rank_paired_delta -- paired rank-NLL difference on identical test pairs.

For two per-chart difficulty tables A and B, this script fits each signal's
linear link on the same within-player test pairs, computes the per-pair NLL of
both, and reports

    delta = NLL(A) - NLL(B)          (negative = A better)

The default is the same pair set as `step2_rank_eval --strict-pairs`: only
within-player pairs whose two charts were seen by the cross-fitting fold's BT
ceiling.  Without `--strict-pairs` it falls back to the historical all-pairs
behaviour for comparison.

Player uncertainty is a cluster bootstrap over `player_id_base`; duplicate
draws are kept as multiplicities.  Chart uncertainty is a two-way cluster
approximation with weight `mult[a] * mult[b]`; the strict mask is applied
before weighting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dcr.rank_metrics import build_pair_table, fit_link  # noqa: E402


def load_signal(path: Path, col: str) -> dict[int, float]:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "beatmap_id" not in df.columns or col not in df.columns:
        raise SystemExit(f"{path}: need beatmap_id and {col}")
    if df["beatmap_id"].duplicated().any():
        raise SystemExit(f"{path}: duplicate beatmap_id rows")
    return {int(b): float(v) for b, v in zip(df.beatmap_id, df[col])
            if np.isfinite(v)}


def load_ceiling(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    if "seen_fold_0" not in z.files:
        raise SystemExit(
            f"{path}: no seen_fold_* masks; rerun step2_rank_eval with "
            "--no-cache (cache v2)")
    if "cluster_ids" not in z.files or "map_ids" not in z.files:
        raise SystemExit(
            f"{path}: cache lacks cluster_ids/map_ids; rerun with cache v2")
    folds = sorted(int(k.split("_")[-1]) for k in z.files
                   if k.startswith("b_fold_"))
    seen = [z[f"seen_fold_{i}"] for i in folds]
    if len(seen) != len(folds):
        raise SystemExit(f"{path}: incomplete seen-fold masks")
    return {
        "map_ids": z["map_ids"].astype(np.int64),
        "cluster_ids": z["cluster_ids"].astype(np.int64),
        "fold_of_cluster": z["fold_of_cluster"].astype(np.int64),
        "seen_folds": seen,
        "nll": float(z["nll"]),
        "n": int(z["n"]),
    }


def pair_nll(y: np.ndarray, g: np.ndarray,
             xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
    u = g[xb] - g[xa]
    q = np.clip(expit(u), 1e-15, 1.0 - 1e-15)
    return -(y * np.log(q) + (1.0 - y) * np.log1p(-q))


def pair_hash(xa: np.ndarray, xb: np.ndarray, clusters: np.ndarray) -> str:
    h = hashlib.sha256()
    for arr in (xa, xb, clusters):
        h.update(np.asarray(arr, dtype=np.int64).tobytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--a", required=True, type=Path)
    ap.add_argument("--b", required=True, type=Path)
    ap.add_argument("--a-col", default="d")
    ap.add_argument("--b-col", default="d")
    ap.add_argument("--split", default="test")
    ap.add_argument("--split-col", default="split")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cluster-col", default="player_id_base")
    ap.add_argument("--strict-pairs", action="store_true")
    ap.add_argument("--ceiling-cache", type=Path, default=None)
    ap.add_argument("--allow-subset", action="store_true",
                    help="permit common charts != ceiling map_ids; delta is not "
                         "the main-table delta in that case")
    ap.add_argument("--expected-n-strict", type=int, default=None)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    df = pd.read_parquet(args.inp)
    if args.split != "all":
        df = df[df[args.split_col] == args.split].copy()
    if args.cluster_col not in df.columns:
        if args.cluster_col != "player_id_base":
            raise SystemExit(f"{args.inp}: no {args.cluster_col}")
        print(f"[warn] {args.cluster_col} missing; falling back to player_id",
              flush=True)
        df[args.cluster_col] = df["player_id"]
    dup = int(df.duplicated(["player_id", "beatmap_id"]).sum())
    if dup:
        raise SystemExit(f"{args.inp}: {dup} duplicate (player_id, beatmap_id) rows")

    a = load_signal(args.a, args.a_col)
    b = load_signal(args.b, args.b_col)
    if set(a) != set(b):
        if not args.allow_subset:
            raise SystemExit(
                "signal tables cover different chart sets; use --allow-subset only "
                "for a diagnostic, not a main-table delta")
        print(f"[warn] signal chart sets differ ({len(a)} vs {len(b)}); "
              "using their intersection", flush=True)
    common = sorted(set(df.beatmap_id.astype(int)) & set(a) & set(b))
    if len(common) < 2:
        raise SystemExit(f"only {len(common)} common charts")

    ceiling = None
    ca = np.arange(len(common), dtype=np.int64)
    if args.strict_pairs:
        if args.ceiling_cache is None:
            raise SystemExit("--strict-pairs requires --ceiling-cache")
        ceiling = load_ceiling(args.ceiling_cache)
        c_ids = ceiling["map_ids"]
        if set(common) != set(int(x) for x in c_ids):
            if not args.allow_subset:
                raise SystemExit(
                    "common charts differ from ceiling map_ids; rerun with a "
                    "matching signal/ceiling or pass --allow-subset")
            print("[warn] subset of ceiling map_ids; this is not the main-table "
                  "delta", flush=True)
        chart_index = {int(c): i for i, c in enumerate(c_ids)}
        ca = np.array([chart_index[int(c)] for c in common], np.int64)

    d_a = np.array([a[c] for c in common], np.float64)
    d_b = np.array([b[c] for c in common], np.float64)
    cmap = {c: i for i, c in enumerate(common)}
    groups, item, perf, group_cluster = [], [], [], []
    for _, g in df.groupby("player_id", sort=False):
        if len(g) < 2:
            continue
        idx = np.arange(len(g)) + len(item)
        groups.append(idx)
        item.extend([cmap[int(x)] for x in g.beatmap_id])
        perf.extend([-float(x) for x in g.loss])
        bases = np.unique(g[args.cluster_col].to_numpy())
        if bases.size != 1:
            raise SystemExit(
                f"player_id {g['player_id'].iloc[0]} maps to multiple "
                f"{args.cluster_col} values: {bases.tolist()}")
        group_cluster.append(int(bases[0]))
    item = np.asarray(item, np.int64)
    perf = np.asarray(perf, np.float64)
    if not groups:
        raise SystemExit("no player has >=2 test charts")

    table = build_pair_table(groups, item, perf, len(common), tie="half")
    fit_a = fit_link(table, d_a, kind="linear")
    fit_b = fit_link(table, d_b, kind="linear")
    ga = fit_a.transform(d_a)
    gb = fit_b.transform(d_b)

    # Cluster index per player segment.
    if ceiling is not None:
        cpos = {int(c): i for i, c in enumerate(ceiling["cluster_ids"])}
        base_to_ci = {}
        for base in group_cluster:
            if int(base) not in cpos:
                raise SystemExit(f"cluster {base} not in ceiling cache")
            base_to_ci[int(base)] = cpos[int(base)]
        n_clusters = len(ceiling["cluster_ids"])
    else:
        ids = sorted(set(group_cluster))
        cpos = {c: i for i, c in enumerate(ids)}
        base_to_ci = cpos
        n_clusters = len(ids)

    # Every within-player pair, plus the strict mask when requested.  The link
    # is fit on all pairs (matching the main table); only the metric is masked.
    all_ll_a, all_ll_b, all_xa, all_xb = [], [], [], []
    st_ll_a, st_ll_b, st_xa, st_xb, st_cluster = [], [], [], [], []
    for p, idx in enumerate(groups):
        ii, jj = np.triu_indices(len(idx), 1)
        x = item[idx][ii]
        z = item[idx][jj]
        y = (perf[idx][ii] > perf[idx][jj]).astype(np.float64)
        y = np.where(perf[idx][ii] == perf[idx][jj], 0.5, y)
        lla = pair_nll(y, ga, x, z)
        llb = pair_nll(y, gb, x, z)
        all_ll_a.append(lla); all_ll_b.append(llb)
        all_xa.append(x); all_xb.append(z)
        ci = base_to_ci[int(group_cluster[p])]
        if ceiling is None:
            st_ll_a.append(lla); st_ll_b.append(llb)
            st_xa.append(x); st_xb.append(z)
            st_cluster.append(np.full(x.size, ci, np.int64))
        else:
            fold = int(ceiling["fold_of_cluster"][ci])
            seen = ceiling["seen_folds"][fold]
            ok = seen[ca[x]] & seen[ca[z]]
            if not np.any(ok):
                continue
            st_ll_a.append(lla[ok]); st_ll_b.append(llb[ok])
            st_xa.append(x[ok]); st_xb.append(z[ok])
            st_cluster.append(np.full(int(ok.sum()), ci, np.int64))

    def cat(parts, dtype=np.float64):
        return np.concatenate(parts) if parts else np.zeros(0, dtype=dtype)

    all_ll_a = cat(all_ll_a); all_ll_b = cat(all_ll_b)
    all_xa = cat(all_xa, np.int64); all_xb = cat(all_xb, np.int64)
    st_ll_a = cat(st_ll_a); st_ll_b = cat(st_ll_b)
    st_xa = cat(st_xa, np.int64); st_xb = cat(st_xb, np.int64)
    st_cluster = cat(st_cluster, np.int64)
    if st_ll_a.size == 0:
        raise SystemExit("strict pair set is empty")
    if ceiling is not None and not args.allow_subset:
        if st_ll_a.size != ceiling["n"]:
            raise SystemExit(
                f"strict pair count {st_ll_a.size} != ceiling n {ceiling['n']}")
    if args.expected_n_strict is not None and st_ll_a.size != args.expected_n_strict:
        raise SystemExit(
            f"strict pair count {st_ll_a.size} != expected {args.expected_n_strict}")

    nll_a = float(st_ll_a.mean())
    nll_b = float(st_ll_b.mean())
    delta = nll_a - nll_b

    rng = np.random.default_rng(args.seed)
    ci_p = []
    sum_a = np.bincount(st_cluster, weights=st_ll_a, minlength=n_clusters)
    sum_b = np.bincount(st_cluster, weights=st_ll_b, minlength=n_clusters)
    cnt = np.bincount(st_cluster, weights=np.ones_like(st_ll_a),
                      minlength=n_clusters)
    for _ in range(args.bootstrap):
        draw = rng.integers(0, n_clusters, n_clusters)
        m = np.bincount(draw, minlength=n_clusters).astype(np.float64)
        den = float((m * cnt).sum())
        if den <= 0:
            continue
        ci_p.append(float((m * sum_a).sum() / den - (m * sum_b).sum() / den))

    n_charts = len(common)
    ci_c = []
    for _ in range(args.bootstrap):
        draw = rng.integers(0, n_charts, n_charts)
        mult = np.bincount(draw, minlength=n_charts).astype(np.float64)
        w = mult[st_xa] * mult[st_xb]
        if w.sum() <= 0:
            continue
        ci_c.append(float((w * st_ll_a).sum() / w.sum()
                          - (w * st_ll_b).sum() / w.sum()))

    out = {
        "in": str(args.inp), "a": str(args.a), "b": str(args.b),
        "split": args.split,
        "strict_pairs": bool(args.strict_pairs),
        "n_charts": int(n_charts),
        "n_players": int(len(groups)),
        "n_player_clusters": int(n_clusters),
        "n_pairs_all": int(all_ll_a.size),
        "n_pairs_strict": int(st_ll_a.size),
        "nll_a": nll_a, "nll_b": nll_b, "delta": delta,
        "nll_a_all": float(all_ll_a.mean()) if all_ll_a.size else None,
        "nll_b_all": float(all_ll_b.mean()) if all_ll_b.size else None,
        "delta_all": (float(all_ll_a.mean() - all_ll_b.mean())
                      if all_ll_a.size else None),
        "beta_a": float(fit_a.beta), "beta_b": float(fit_b.beta),
        "ci_player": [float(np.percentile(ci_p, 2.5)),
                       float(np.percentile(ci_p, 97.5))] if ci_p else None,
        "ci_chart": [float(np.percentile(ci_c, 2.5)),
                      float(np.percentile(ci_c, 97.5))] if ci_c else None,
        "bootstrap": int(args.bootstrap),
        "cluster_col": args.cluster_col,
        "ceiling_cache": str(args.ceiling_cache) if args.ceiling_cache else None,
        "ceiling_n": int(ceiling["n"]) if ceiling is not None else None,
        "pair_set_sha256": pair_hash(st_xa, st_xb, st_cluster),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=1, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
