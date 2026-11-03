
#!/usr/bin/env python
"""step2_rank_eval -- pairwise difficulty-ranking NLL for a per-chart signal.

The metric definition lives in dcr/rank_metrics.py and docs/metrics.md.
This CLI evaluates one or more per-chart difficulty columns against a step1
table (default: step1/pairbest).  For every signal it fits

    q_ij = sigmoid( g(d_j) - g(d_i) )

to all within-player comparisons, reports the pairwise NLL and the fraction of
the cross-fitted Bradley-Terry ceiling it captures, plus the companion
per-player Spearman.

Examples
--------
python scripts/step2_rank_eval.py \
    --in data/processed/step1_1k_pairbest.parquet \
    --signal-table data/processed/rank_signals_1k.parquet \
    --signals msd_overall,sunnywin_ln_star,sunny_star \
    --out results/rank_eval_1k/all_signals.json

The ceiling is cached by the exact held-out map set + fold seed in
--cache-dir (default data/processed/rank_eval_cache); reruns are cheap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dcr import rank_metrics as rm  # noqa: E402


def _cluster_index(clusters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = np.unique(clusters)
    pos = {int(c): i for i, c in enumerate(ids)}
    return np.array([pos[int(c)] for c in clusters], dtype=np.int64), ids


def _cache_key(map_ids: np.ndarray, seed: int, folds: int, min_l: int,
               n_rows: int, split: str, extra: str = "") -> str:
    h = hashlib.md5()
    h.update(np.asarray(map_ids, dtype=np.int64).tobytes())
    for x in (seed, folds, min_l, n_rows, split):
        h.update(str(x).encode())
    if extra:
        h.update(extra.encode())
    return h.hexdigest()[:16]


def _encode_rows(df: pd.DataFrame, map_index: dict):
    """(item, perf, groups, clusters, players) for one play table and vocab.

    groups are row-index arrays into item/perf, one per player segment;
    clusters are player_id_base ids aligned with groups.  The caller owns the
    chart vocabulary, so the same table can be encoded against a compact
    signal-only vocab or the full --in vocab.
    """
    item = df["beatmap_id"].map(map_index).to_numpy(np.int32)
    perf = (-df["loss"].astype(np.float64)).to_numpy()
    players = np.sort(df["player_id"].unique())
    p_idx = {int(p): i for i, p in enumerate(players)}
    pi = df["player_id"].map(p_idx).to_numpy(np.int32)
    tmp = pd.DataFrame({"pi": pi, "base": df["player_id_base"].to_numpy()})
    idx_map = tmp.groupby("pi").indices
    groups = [np.asarray(idx_map[i]) for i in range(len(players))]
    clusters = (tmp.groupby("pi")["base"].first()
                .reindex(range(len(players))).to_numpy())
    return item, perf, groups, np.asarray(clusters, dtype=np.int64), players


def _save_ceiling(path: Path, c: rm.Ceiling, map_ids: np.ndarray) -> None:
    d = dict(nll=c.nll, n=c.n, coverage=c.coverage, n_clusters=c.n_clusters,
             b_full=c.b_full, fold_of_cluster=c.fold_of_cluster,
             nll_by_cluster=c.nll_by_cluster, n_by_cluster=c.n_by_cluster,
             map_ids=map_ids,
             rho_lw=c.rho["lw"], rho_macro=c.rho["macro"],
             rho_fisher=c.rho["fisher_l3"], rho_median=c.rho["median"],
             rho_values=c.rho["values"], rho_ls=c.rho["ls"],
             rho_players=c.rho.get("players", np.zeros(0, int)))
    for i, b in enumerate(c.b_folds):
        d[f"b_fold_{i}"] = b
    if getattr(c, "seen_folds", None):
        for i, s in enumerate(c.seen_folds):
            d[f"seen_fold_{i}"] = np.asarray(s, dtype=bool)
    np.savez_compressed(path, **d)


def _load_ceiling(path: Path) -> rm.Ceiling:
    z = np.load(path, allow_pickle=False)
    folds = sorted(int(k.split("_")[-1]) for k in z.files if k.startswith("b_fold_"))
    rho = dict(lw=float(z["rho_lw"]), macro=float(z["rho_macro"]),
               fisher_l3=float(z["rho_fisher"]), median=float(z["rho_median"]),
               values=z["rho_values"], ls=z["rho_ls"],
               players=z["rho_players"], n=int(z["rho_values"].size))
    seen = [z[f"seen_fold_{i}"] for i in range(len(folds))
            if f"seen_fold_{i}" in z.files]
    seen_folds = seen if len(seen) == len(folds) else None
    return rm.Ceiling(nll=float(z["nll"]), n=int(z["n"]),
                      coverage=float(z["coverage"]), b_full=z["b_full"],
                      b_folds=[z[f"b_fold_{i}"] for i in folds], rho=rho,
                      nll_by_cluster=z["nll_by_cluster"],
                      n_by_cluster=z["n_by_cluster"],
                      n_clusters=int(z["n_clusters"]),
                      fold_of_cluster=z["fold_of_cluster"],
                      seen_folds=seen_folds)


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="step1 table")
    ap.add_argument("--signal-table", required=True,
                    help="parquet/csv with beatmap_id + signal columns")
    ap.add_argument("--signals", required=True, help="comma-separated columns")
    ap.add_argument("--split", default="all", choices=["all", "train", "test", "dropped"],
                    help="'all' is right for content-only signals (no train/test needed); "
                         "use 'test' only for model-derived difficulty (bm) to avoid leakage")
    ap.add_argument("--ceiling-fit", default="all", choices=["all", "split"],
                    help="fit the BT ceiling b on all rows (all) or only on the rows kept "
                         "by --split (split; legacy behaviour).  'all' is required for "
                         "--split test: fitting b on test rows alone leaves it badly "
                         "underfit, and captured can exceed 1")
    ap.add_argument("--common-maps", action="store_true",
                    help="restrict every signal to the intersection of charts with a "
                         "non-null value, so all signals share one pair set and ceiling")
    ap.add_argument("--strict-pairs", action="store_true",
                    help="measure the signal NLL on exactly the pairs the ceiling "
                         "covers, the same pair set as captured's denominator")
    ap.add_argument("--min-l", type=int, default=10)
    ap.add_argument("--link", default="linear", choices=["linear", "rank", "spline"])
    ap.add_argument("--knots", type=int, default=8)
    ap.add_argument("--folds", type=int, default=2,
                    help="ceiling folds by player cluster; 1 = in-sample ceiling "
                         "(optimistic; only compare within the same setting)")
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default="data/processed/rank_eval_cache")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-player-out", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    t0 = time.time()
    step1_full = pd.read_parquet(args.inp)
    n_rows_total = len(step1_full)
    n_players_total = step1_full["player_id"].nunique()
    n_maps_total = step1_full["beatmap_id"].nunique()
    if "player_id_base" not in step1_full.columns:
        step1_full["player_id_base"] = step1_full["player_id"]
    if args.split == "all":
        step1 = step1_full
    else:
        step1 = step1_full[step1_full["split"] == args.split].reset_index(drop=True)
    if step1.empty:
        raise SystemExit(f"--split {args.split}: no rows left")
    sig = pd.read_csv(args.signal_table) if args.signal_table.endswith(".csv") \
        else pd.read_parquet(args.signal_table)
    if "beatmap_id" not in sig.columns:
        raise SystemExit("signal table needs a beatmap_id column")
    sig["beatmap_id"] = sig["beatmap_id"].astype(step1["beatmap_id"].dtype, copy=False)
    signals = [s.strip() for s in args.signals.split(",") if s.strip()]
    missing = [s for s in signals if s not in sig.columns]
    if missing:
        raise SystemExit("missing signal columns: " + ", ".join(missing))
    if args.common_maps and len(signals) > 1:
        common = None
        for s in signals:
            ids = pd.Index(sig.loc[sig[s].notna(), "beatmap_id"].unique())
            common = ids if common is None else common.intersection(ids)
        if common is None or len(common) == 0:
            raise SystemExit("--common-maps: no chart has a value for every signal")
        n_sig_before = len(sig)
        sig = sig[sig["beatmap_id"].isin(common)].reset_index(drop=True)
        print(f"[common-maps] {len(common):,} charts shared by {len(signals)} signals "
              f"(signal table {n_sig_before:,} -> {len(sig):,} rows)", flush=True)
    cache_dir = Path(args.cache_dir)
    if not args.no_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)

    # The ceiling can be fitted on every row while the signal is evaluated on
    # the --split rows.  Encode the full table once against its own vocabulary;
    # each signal maps its compact chart ids into that vocabulary.
    fit_map_idx = None
    fit_item = fit_perf = fit_groups = fit_clusters = None
    fit_n_items = None
    if args.ceiling_fit == "all":
        fit_map_ids = np.sort(step1_full["beatmap_id"].unique())
        fit_map_idx = {int(m): i for i, m in enumerate(fit_map_ids)}
        fit_item, fit_perf, fit_groups, fit_clusters, _ = _encode_rows(
            step1_full, fit_map_idx)
        fit_n_items = int(fit_map_ids.size)
        print(f"[ceiling] fit b on all {n_rows_total:,} rows / {fit_n_items:,} charts; "
              f"evaluate on {args.split} rows ({len(step1):,} rows)", flush=True)

    results = {}
    per_player_rows = []
    for name in signals:
        t = time.time()
        sub = step1.merge(sig[["beatmap_id", name]], on="beatmap_id", how="inner")
        sub = sub[sub[name].notna()].reset_index(drop=True)
        if sub.empty:
            results[name] = {"error": "no rows after signal merge"}
            continue
        # The metric is defined on step1/pairbest: one row per (player, beatmap).
        # A duplicate would silently double-count that player's pair and change N.
        dup = int(sub.duplicated(["player_id", "beatmap_id"]).sum())
        if dup:
            raise SystemExit(
                f"{name}: {dup} duplicate (player_id, beatmap_id) rows; this metric "
                "expects the step1/pairbest unit (one row per pair)")
        # item vocabulary: compact over the signal's charts.  The fit table
        # may use a larger vocabulary; fit_item_of_eval bridges the two.
        map_ids = np.sort(sub["beatmap_id"].unique())
        map_idx = {int(m): i for i, m in enumerate(map_ids)}
        n_items = len(map_ids)
        if fit_map_idx is not None and not set(map_idx).issubset(fit_map_idx):
            raise SystemExit(f"{name}: charts missing from --in after merge")
        fit_item_of_eval = (np.array([fit_map_idx[int(m)] for m in map_ids],
                                     dtype=np.int64)
                            if fit_map_idx is not None else None)
        item, perf, groups, clusters, players = _encode_rows(sub, map_idx)
        cidx, cluster_ids = _cluster_index(clusters)
        # item-level signal (the table is one row per beatmap; take first)
        d = (sub.groupby("beatmap_id")[name].first().reindex(map_ids)
             .to_numpy(np.float64))
        if not np.all(np.isfinite(d)):
            bad = int((~np.isfinite(d)).sum())
            raise SystemExit(f"{name}: {bad} non-finite item values after groupby")
        # per-player Spearman
        pp = rm.per_player_spearman(groups, item, perf, d, min_l=args.min_l,
                                    clusters=cidx)
        # random baseline (map-level random vector, same groups)
        rng = np.random.default_rng(args.seed + 1)
        rand = []
        for _ in range(5):
            r = rm.per_player_spearman(groups, item, perf,
                                       rng.normal(size=n_items),
                                       min_l=args.min_l, clusters=cidx)
            if np.isfinite(r["lw"]):
                rand.append(r["lw"])
        random_lw = float(np.mean(rand)) if rand else float("nan")
        # ceiling (cross-fitted BT), cached by the held-out map set.  The
        # "fitall" extra marks ceilings fitted on all rows; the legacy
        # split-only ceiling keeps its old key, so existing caches still load.
        fit_all = args.ceiling_fit == "all"
        full_coverage = fit_all and n_items == n_maps_total
        extra = ("" if (not fit_all or (args.split == "all" and full_coverage))
                 else f"fitall:{n_rows_total}:{fit_n_items}:{n_maps_total}")
        key = _cache_key(map_ids, args.seed, args.folds, args.min_l, len(sub),
                         args.split, extra)
        cpath = cache_dir / f"ceiling_{key}.npz"
        if cpath.exists() and not args.no_cache:
            ceil = _load_ceiling(cpath)
        else:
            if fit_all:
                ceil = rm.cross_fit_bt(
                    groups, item, perf, n_items, clusters,
                    folds=args.folds, seed=args.seed, min_l=args.min_l,
                    fit_groups=fit_groups, fit_item=fit_item,
                    fit_perf=fit_perf, fit_clusters=fit_clusters,
                    fit_n_items=fit_n_items,
                    fit_item_of_eval=fit_item_of_eval)
            else:
                ceil = rm.cross_fit_bt(groups, item, perf, n_items, clusters,
                                       folds=args.folds, seed=args.seed,
                                       min_l=args.min_l)
            if not args.no_cache:
                _save_ceiling(cpath, ceil, map_ids)
        if int(ceil.n_clusters) != int(cluster_ids.size):
            raise SystemExit(
                f"{name}: ceiling has {ceil.n_clusters} clusters, eval has "
                f"{cluster_ids.size}; clear the cache or change --ceiling-fit")
        # link fit on all pairs
        table = rm.build_pair_table(groups, item, perf, n_items)
        link = rm.fit_link(table, d, kind=args.link, knots=args.knots)
        nll = link.nll                      # signal NLL on every eval pair
        n = table.n_instances
        ceil_nll = ceil.nll
        g = link.transform(d)
        sig_nll, sig_n = rm.nll_by_cluster(groups, item, perf, g, cidx,
                                           int(cluster_ids.size))
        # The denominator of captured is a ceiling on the covered pairs; by
        # default the numerator is every pair.  --strict-pairs measures both on
        # the covered pairs (requires the cache to carry seen-fold masks).
        strict = None
        if args.strict_pairs:
            if ceil.seen_folds is None:
                raise SystemExit(
                    f"{name}: this ceiling cache has no seen-fold masks; rerun "
                    "with --no-cache (or delete the cache) to use --strict-pairs")
            st_nll, st_n = rm.nll_on_ceiling_pairs(groups, item, perf, g, cidx,
                                                   ceil)
            st_total = float(st_n.sum())
            if st_total <= 0:
                raise SystemExit(f"{name}: --strict-pairs left no pairs")
            if abs(st_total - float(ceil.n)) > 1e-6:
                raise SystemExit(
                    f"{name}: strict pair count {st_total:.0f} != ceiling {ceil.n}")
            strict = {"nll": float(st_nll.sum()), "n": st_total,
                      "nll_by_cluster": st_nll, "n_by_cluster": st_n}
        nll_used = strict["nll"] if strict else nll
        n_used = strict["n"] if strict else float(n)
        bits = 1.0 - (nll_used / n_used) / rm.LOG2 if n_used else float("nan")
        captured = ((rm.LOG2 - nll_used / n_used) / (rm.LOG2 - ceil_nll)
                    if n_used and np.isfinite(ceil_nll) and ceil_nll < rm.LOG2
                    else float("nan"))
        bits_all = 1.0 - (nll / n) / rm.LOG2 if n else float("nan")
        captured_all = ((rm.LOG2 - nll / n) / (rm.LOG2 - ceil_nll)
                        if n and np.isfinite(ceil_nll) and ceil_nll < rm.LOG2
                        else float("nan"))
        # bootstrap on the pair set that defines captured
        boot_nll, boot_n = ((strict["nll_by_cluster"], strict["n_by_cluster"])
                            if strict else (sig_nll, sig_n))
        boot = rm.bootstrap_metric(pp, boot_nll, boot_n, ceil.nll_by_cluster,
                                   ceil.n_by_cluster, int(cluster_ids.size),
                                   n_boot=args.bootstrap, seed=args.seed)
        # correlation with the BT scale
        ok = np.isfinite(ceil.b_full) & np.isfinite(d)
        rho_b = float(spearmanr(d[ok], ceil.b_full[ok]).statistic) if ok.sum() > 2 else float("nan")
        tau_b = float(kendalltau(d[ok], ceil.b_full[ok]).statistic) if ok.sum() > 2 else float("nan")
        # Pearson is not invariant to a monotone nonlinear rescaling of d, so it
        # measures the linear agreement with the BT scale, not just the order.
        pear_b = float(pearsonr(d[ok], ceil.b_full[ok]).statistic) if ok.sum() > 2 else float("nan")
        denom = ceil.rho["lw"] - random_lw
        results[name] = {
            "n_rows": int(len(sub)), "n_players": int(len(players)),
            "n_maps": int(n_items), "n_pairs": int(table.n_pairs),
            "n_instances": int(n), "n_instances_used": int(round(n_used)),
            "n_ties": int(table.n_ties),
            "coverage_rows": float(len(sub) / n_rows_total),
            "coverage_maps": float(n_items / n_maps_total),
            "nll": float(nll_used),
            "nll_per_instance": float(nll_used / n_used),
            "bits_saved": float(bits), "captured": float(captured),
            "nll_all": float(nll), "nll_all_per_instance": float(nll / n),
            "bits_saved_all": float(bits_all),
            "captured_all": float(captured_all),
            "strict_pairs": bool(args.strict_pairs),
            "nll_strict": None if strict is None else float(strict["nll"]),
            "nll_strict_per_instance": (None if strict is None else
                                        float(strict["nll"] / strict["n"])),
            "bits_saved_strict": None if strict is None else float(bits),
            "captured_strict": None if strict is None else float(captured),
            "n_strict": None if strict is None else int(round(strict["n"])),
            "ceiling_nll": float(ceil_nll), "ceiling_n": int(ceil.n),
            "ceiling_coverage": float(ceil.coverage),
            "ceiling_fit": args.ceiling_fit,
            "ceiling_fit_rows": int(n_rows_total if fit_all else len(sub)),
            "ceiling_fit_maps": int(fit_n_items if fit_all else n_items),
            "ceiling_eval_rows": int(len(sub)),
            "link": {"kind": link.kind, "beta": float(link.beta),
                     **{k: _jsonable(v) for k, v in link.extra.items()}},
            "per_player": {k: pp[k] for k in ("lw", "macro", "fisher_l3",
                                              "median", "n")},
            "ceiling_per_player": {k: ceil.rho[k] for k in ("lw", "macro",
                                                            "fisher_l3",
                                                            "median", "n")},
            "random_lw": random_lw,
            "norm_rho_lw": (float((pp["lw"] - random_lw) / denom)
                            if np.isfinite(pp["lw"]) and abs(denom) > 1e-12 else float("nan")),
            "norm_rho_macro": (float((pp["macro"] - random_lw)
                                     / (ceil.rho["macro"] - random_lw))
                               if np.isfinite(pp["macro"])
                               and abs(ceil.rho["macro"] - random_lw) > 1e-12 else float("nan")),
            "rho_bhat": rho_b, "tau_bhat": tau_b, "pearson_bhat": pear_b,
            "bootstrap": boot,
        }
        # per-player rows
        for k in range(pp["values"].size):
            per_player_rows.append(dict(signal=name, kind="signal",
                                        player_id=int(players[pp["players"][k]]),
                                        cluster=int(cluster_ids[pp["clusters"][k]]),
                                        l=int(pp["ls"][k]), rho=float(pp["values"][k])))
        if "players" in ceil.rho and ceil.rho["players"].size:
            for k, p in enumerate(ceil.rho["players"]):
                per_player_rows.append(dict(signal=name, kind="ceiling",
                                            player_id=int(players[int(p)]),
                                            cluster=-1,
                                            l=int(ceil.rho["ls"][k]),
                                            rho=float(ceil.rho["values"][k])))
        print(f"[{time.time()-t0:7.1f}s] {name:24s} nll={nll_used/n_used:.5f} "
              f"ceil={ceil_nll:.5f} captured={captured:.4f} "
              f"n={int(n_used):,}/{int(n):,} "
              f"lw={pp['lw']:+.4f} macro={pp['macro']:+.4f} "
              f"beta={link.beta:.4g}", flush=True)

    if len(results) > 1:
        sig_sets = {(r.get("n_maps"), r.get("n_instances"),
                     round(float(r.get("ceiling_nll", float("nan"))), 12))
                    for r in results.values() if "error" not in r}
        if len(sig_sets) > 1:
            print("[warn] signals do not share the same chart/pair set or ceiling; "
                  "captured is not directly comparable across them.  Use "
                  "--common-maps or a pre-filtered signal table.", flush=True)

    out = {"meta": {
        "in": str(args.inp), "split": args.split, "n_rows_total": n_rows_total,
        "n_players_total": int(n_players_total), "n_maps_total": int(n_maps_total),
        "ceiling_fit": args.ceiling_fit,
        "ceiling_fit_rows": int(n_rows_total if args.ceiling_fit == "all"
                                else len(step1)),
        "ceiling_fit_maps": int(fit_n_items if args.ceiling_fit == "all"
                                else n_maps_total),
        "common_maps": bool(args.common_maps),
        "strict_pairs": bool(args.strict_pairs),
        "signal_table": str(args.signal_table), "signals": signals,
        "min_l": args.min_l, "link": args.link, "knots": args.knots,
        "folds": args.folds, "bootstrap": args.bootstrap, "seed": args.seed,
        "tag": args.tag, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metric": "pairwise Bradley-Terry NLL, cross-fitted ceiling",
    }, "signals": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(_jsonable(out), f, indent=1, sort_keys=True)
    if args.per_player_out:
        pd.DataFrame(per_player_rows).to_csv(args.per_player_out, index=False)
    print(f"wrote {args.out} ({time.time()-t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
