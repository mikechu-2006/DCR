#!/usr/bin/env python
"""Experiment 3: regularized D2V residual on top of a legacy strong baseline.

For each (protocol, baseline model, seed):
  * baseline scores are cross-fitted inside the training side:
      - P2-holdout: 5-fold chart-disjoint OOF scores for every chart;
      - P4-holdout: 5-fold chart-disjoint OOF inside the 6x6 block;
        test charts use the full 6x6 baseline.
  * fit  d = d_base + w' z  by BT NLL on training pairs, z = standardized D2V;
  * evaluate on the historical-best test pairs: overall, near-baseline pairs
    (bottom 25% of |Delta z| on training), and LN charts;
  * controls: random 512d, shuffled D2V, D2V-only.

This is a probe, not a new production model.  See docs/first2... style report
for the companion experiment; this script only writes the residual results.

Usage
-----
python scripts/fit_d2v_residual.py \
  --protocol p2 --model linear_extra --seed 0 \
  --split data/processed/step1_1k_pairbest_d2v_holdout.parquet \
  --baseline-oof results/d2v_residual/baseline/p2/linear_extra_oof.parquet \
  --baseline-col base_linear_extra \
  --d2v-path data/processed/chart_content_d2v_holdout.parquet \
  --signal-table data/processed/rank_signals_1k.parquet \
  --out-signals results/d2v_residual/p2/linear_extra/signals.parquet \
  --out-json results/d2v_residual/p2/linear_extra/result.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from dcr import rank_metrics as rm  # noqa: E402


def _read_split(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "player_id_base" not in df.columns:
        df["player_id_base"] = df["player_id"]
    if "split" not in df.columns:
        raise SystemExit(f"{path}: needs split")
    return df


def _load_signal(path: Path, col: str, maps: np.ndarray) -> np.ndarray:
    df = pd.read_parquet(path).set_index("beatmap_id")
    if col not in df.columns:
        # accept the common aliases written by step2_cm3p_rank.py
        for c in ("cm3p_rank", "signal", "cm3p_linear_extra", "cm3p_mlp_extra"):
            if c in df.columns:
                col = c
                break
    if col not in df.columns:
        raise SystemExit(f"{path}: no column {col}; have {list(df.columns)}")
    d = df[col].reindex(maps).to_numpy(np.float64)
    if not np.isfinite(d).all():
        raise SystemExit(f"{path}: non-finite values")
    return d


def _load_d2v(path: Path, maps: np.ndarray) -> np.ndarray:
    df = pd.read_parquet(path)
    col = "embedding" if "embedding" in df.columns else "features"
    if col not in df.columns:
        raise SystemExit(f"{path}: no embedding column")
    Z = np.stack(df[col].to_numpy()).astype(np.float64)
    ids = df["beatmap_id"].to_numpy()
    pos = pd.Index(ids).get_indexer(maps)
    if (pos < 0).any():
        raise SystemExit(f"{path}: {int((pos < 0).sum())} charts missing")
    Z = Z[pos]
    if not np.isfinite(Z).all():
        raise SystemExit(f"{path}: non-finite vectors")
    return Z


def _encode(df: pd.DataFrame, map_index: dict):
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
    cidx, cluster_ids = pd.factorize(clusters.astype(np.int64), sort=True)
    return item, perf, groups, cidx, cluster_ids


def _pair_table(groups, item, perf, n_items, players=None):
    g = groups if players is None else [groups[p] for p in players]
    return rm.build_pair_table(g, item, perf, n_items)


def _sub_table(tab: rm.PairTable, mask: np.ndarray) -> rm.PairTable:
    keep = mask[tab.ua, tab.ub]
    ua = tab.ua[keep]
    ub = tab.ub[keep]
    nab = tab.n_ab[keep]
    nba = tab.n_ba[keep]
    n = int(round(float(nab.sum() + nba.sum())))
    return rm.PairTable(ua, ub, nab, nba, tab.n_items, n, 0)


def _val_players(clusters: np.ndarray, frac: float, seed: int):
    ids = np.unique(clusters)
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(frac * ids.size))) if frac > 0 else 0
    val_ids = set(rng.choice(ids, n_val, replace=False).tolist()) if n_val else set()
    fit = np.array([p for p, c in enumerate(clusters) if int(c) not in val_ids],
                   dtype=np.int64)
    val = np.array([p for p, c in enumerate(clusters) if int(c) in val_ids],
                   dtype=np.int64)
    return fit, val


def _fit_offset(Z: np.ndarray, d_base: np.ndarray | None, tab: rm.PairTable,
                lam: float, maxiter: int = 400) -> np.ndarray:
    M, D = Z.shape
    if d_base is None:
        d_base = np.zeros(M, dtype=np.float64)
    d_base = np.asarray(d_base, dtype=np.float64)
    ua, ub = tab.ua, tab.ub
    nab, nba = tab.n_ab, tab.n_ba
    N = float(tab.n_instances)

    def f(w):
        d = d_base + Z @ w
        u = d[ub] - d[ua]
        s = expit(u)
        nll = -np.sum(nab * np.log(np.clip(s, 1e-15, 1.0))
                      + nba * np.log(np.clip(1.0 - s, 1e-15, 1.0)))
        du = -nab * (1.0 - s) + nba * s
        gd = (np.bincount(ua, weights=-du, minlength=M)
              + np.bincount(ub, weights=du, minlength=M))
        return float(nll / N + lam * float(w @ w)), Z.T @ gd / N + 2.0 * lam * w

    r = minimize(f, np.zeros(D), jac=True, method="L-BFGS-B",
                 options=dict(maxiter=maxiter, gtol=1e-8, ftol=1e-12))
    if not r.success:
        print(f"[warn] L-BFGS-B: {r.message}", flush=True)
    return r.x


def _nll(tab: rm.PairTable, d: np.ndarray) -> tuple[float, float]:
    link = rm.fit_link(tab, d, kind="linear")
    return float(link.nll / tab.n_instances), float(link.beta)


def _nll_by_cluster_subset(groups, item, perf, g, cidx, n_clusters,
                           mask: np.ndarray):
    nll = np.zeros(n_clusters, dtype=np.float64)
    cnt = np.zeros(n_clusters, dtype=np.float64)
    for p, idx in enumerate(groups):
        l = int(len(idx))
        if l < 2:
            continue
        ii, jj = np.triu_indices(l, 1)
        a = item[idx][ii]
        b = item[idx][jj]
        swap = a > b
        ua = np.where(swap, b, a)
        ub = np.where(swap, a, b)
        keep = mask[ua, ub]
        if not keep.any():
            continue
        pa = perf[idx][ii][keep]
        pb = perf[idx][jj][keep]
        y = (pa > pb) ^ swap[keep]
        tie = pa == pb
        w = np.where(tie, 0.5, y.astype(np.float64))
        s = expit(g[ub[keep]] - g[ua[keep]])
        ll = w * np.log(np.clip(s, 1e-15, 1.0)) \
            + (1.0 - w) * np.log(np.clip(1.0 - s, 1e-15, 1.0))
        nll[cidx[p]] += -float(ll.sum())
        cnt[cidx[p]] += float(ll.size)
    return nll, cnt


def _eval_subset(tab, groups, item, perf, cidx, n_clusters, mask,
                 d_base, d_corr, bootstrap, rng):
    sub = _sub_table(tab, mask)
    out = {"n_pairs": int(sub.n_pairs), "n_instances": int(sub.n_instances)}
    if sub.n_pairs == 0 or sub.n_instances == 0:
        return {**out, "nll_base": None, "nll_corr": None, "delta": None,
                "ci": None, "p_delta_lt_0": None}
    arms = {}
    for name, d in (("base", d_base), ("corr", d_corr)):
        link = rm.fit_link(sub, d, kind="linear")
        g = link.transform(d)
        nll, cnt = _nll_by_cluster_subset(groups, item, perf, g, cidx,
                                          n_clusters, mask)
        arms[name] = {"nll": float(link.nll / sub.n_instances),
                      "nll_by_cluster": nll, "n_by_cluster": cnt,
                      "rate": float(nll.sum() / cnt.sum())}
    out["nll_base"] = arms["base"]["rate"]
    out["nll_corr"] = arms["corr"]["rate"]
    out["delta"] = float(out["nll_corr"] - out["nll_base"])
    deltas = []
    for _ in range(int(bootstrap)):
        draw = rng.integers(0, n_clusters, n_clusters)
        mult = np.bincount(draw, minlength=n_clusters).astype(np.float64)
        na = float((arms["base"]["nll_by_cluster"] * mult).sum())
        nb = float((arms["corr"]["nll_by_cluster"] * mult).sum())
        ca = float((arms["base"]["n_by_cluster"] * mult).sum())
        cb = float((arms["corr"]["n_by_cluster"] * mult).sum())
        if ca <= 0 or cb <= 0:
            continue
        deltas.append(nb / cb - na / ca)
    d = np.asarray(deltas)
    out["ci"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] \
        if d.size else None
    out["p_delta_lt_0"] = float((d < 0).mean()) if d.size else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", choices=["p2", "p4"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", choices=["linear_extra", "mlp_extra"], required=True)
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--baseline-oof", required=True, type=Path)
    ap.add_argument("--baseline-full", type=Path, default=None,
                    help="P4 only: full 6x6 scores for test charts")
    ap.add_argument("--baseline-col", default=None)
    ap.add_argument("--d2v-path", default="data/processed/chart_content_d2v_holdout.parquet")
    ap.add_argument("--signal-table", default="data/processed/rank_signals_1k.parquet")
    ap.add_argument("--lam-grid", default="0,0.001,0.01,0.1,1,10")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--base-standardize", action="store_true", default=True,
                    help="z-score d_base within the train/test chart sets (default)")
    ap.add_argument("--no-base-standardize", dest="base_standardize",
                    action="store_false")
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--out-signals", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    args = ap.parse_args()
    t0 = time.time()

    df = _read_split(args.split)
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    if train.empty or test.empty:
        raise SystemExit(f"train={len(train)} test={len(test)}")
    maps = np.sort(df["beatmap_id"].unique())
    map_idx = {int(m): i for i, m in enumerate(maps)}
    M = int(maps.size)
    Z_raw = _load_d2v(Path(args.d2v_path), maps)
    train_chart_idx = np.unique(train["beatmap_id"].map(map_idx).to_numpy(np.int32))
    mu = Z_raw[train_chart_idx].mean(0)
    sd = Z_raw[train_chart_idx].std(0) + 1e-8
    Z = (Z_raw - mu) / sd

    oof = _load_signal(Path(args.baseline_oof), args.baseline_col or "", maps)
    if args.protocol == "p4":
        if args.baseline_full is None:
            raise SystemExit("P4 needs --baseline-full")
        full = _load_signal(Path(args.baseline_full), args.baseline_col or "", maps)
        tr_charts = set(train["beatmap_id"].astype(np.int64).tolist())
        te_charts = set(test["beatmap_id"].astype(np.int64).tolist())
        overlap = tr_charts & te_charts
        if overlap:
            raise SystemExit(f"P4 train/test charts overlap: {len(overlap)}")
        d_base = np.where(np.isin(maps, list(tr_charts)), oof, full)
        if args.base_standardize:
            tr_idx = np.array([map_idx[int(m)] for m in tr_charts], dtype=np.int64)
            te_idx = np.array([map_idx[int(m)] for m in te_charts], dtype=np.int64)
            d_base = d_base.copy()
            d_base[tr_idx] = (d_base[tr_idx] - d_base[tr_idx].mean()) / (
                d_base[tr_idx].std() + 1e-8)
            d_base[te_idx] = (d_base[te_idx] - d_base[te_idx].mean()) / (
                d_base[te_idx].std() + 1e-8)
    else:
        d_base = oof
        if args.base_standardize:
            tr_idx = np.unique(train["beatmap_id"].map(map_idx).to_numpy(np.int32))
            d_base = (d_base - d_base[tr_idx].mean()) / (d_base[tr_idx].std() + 1e-8)
    if not np.isfinite(d_base).all():
        raise SystemExit("non-finite baseline scores")

    item_tr, perf_tr, groups_tr, clusters_tr, players_tr = _encode(train, map_idx)
    tab_tr = _pair_table(groups_tr, item_tr, perf_tr, M)
    item_te, perf_te, groups_te, clusters_te, players_te = _encode(test, map_idx)
    tab_te = _pair_table(groups_te, item_te, perf_te, M)
    n_clusters = int(clusters_te.size)

    grid = [float(x) for x in args.lam_grid.split(",") if x.strip()]
    best_lam, best_val, curve = grid[0], None, []
    if args.val_frac > 0 and len(grid) > 1:
        # Chart-disjoint validation: the residual is a chart-level correction;
        # a player-cluster split would let the same charts appear on both sides
        # and would select lambda=0 by overfitting chart noise.
        train_charts = np.unique(train["beatmap_id"].to_numpy())
        rng = np.random.default_rng(args.seed + 5)
        n_val = max(1, int(round(args.val_frac * train_charts.size)))
        val_charts = set(rng.choice(train_charts, n_val, replace=False).tolist())
        fit_rows = train[~train["beatmap_id"].isin(val_charts)].reset_index(drop=True)
        val_rows = train[train["beatmap_id"].isin(val_charts)].reset_index(drop=True)
        item_f, perf_f, groups_f, _, _ = _encode(fit_rows, map_idx)
        tab_fit = _pair_table(groups_f, item_f, perf_f, M)
        item_v, perf_v, groups_v, _, _ = _encode(val_rows, map_idx)
        tab_val = _pair_table(groups_v, item_v, perf_v, M)
        for lam in grid:
            w = _fit_offset(Z, d_base, tab_fit, lam, args.maxiter)
            vl, _ = _nll(tab_val, d_base + Z @ w)
            curve.append({"lam": lam, "val_nll": vl})
            print(f"[val] lam={lam:g} val_nll={vl:.5f}", flush=True)
            if best_val is None or vl < best_val:
                best_val, best_lam = vl, lam
    w = _fit_offset(Z, d_base, tab_tr, best_lam, args.maxiter)
    d_corr = d_base + Z @ w

    # controls (same selected lambda)
    rng = np.random.default_rng(args.seed + 17)
    Z_rand = rng.normal(size=Z.shape)
    Z_rand = (Z_rand - Z_rand[train_chart_idx].mean(0)) / (
        Z_rand[train_chart_idx].std(0) + 1e-8)
    Z_shuf = Z[rng.permutation(M)]
    w_rand = _fit_offset(Z_rand, d_base, tab_tr, best_lam, args.maxiter)
    w_shuf = _fit_offset(Z_shuf, d_base, tab_tr, best_lam, args.maxiter)
    w_only = _fit_offset(Z, None, tab_tr, best_lam, args.maxiter)

    # test correlations against an in-sample BT scale (diagnostic only)
    b = rm.fit_bt(tab_te)[0]
    ok = np.isfinite(b)
    corr = {
        "base_pearson": float(pearsonr(d_base[ok], b[ok]).statistic) if ok.sum() > 2 else None,
        "base_spearman": float(spearmanr(d_base[ok], b[ok]).statistic) if ok.sum() > 2 else None,
        "corr_pearson": float(pearsonr(d_corr[ok], b[ok]).statistic) if ok.sum() > 2 else None,
        "corr_spearman": float(spearmanr(d_corr[ok], b[ok]).statistic) if ok.sum() > 2 else None,
    }

    # near-baseline threshold on training chart pairs
    d_base_tr = d_base
    sd_base = float(np.std(d_base[train_chart_idx]) + 1e-8)
    pair_dz = (d_base_tr[tab_tr.ub] - d_base_tr[tab_tr.ua]) / sd_base
    tau = float(np.quantile(np.abs(pair_dz), 0.25))
    # LN flags
    sig = pd.read_parquet(args.signal_table).set_index("beatmap_id")
    ln_ratio = sig["ln_ratio"].reindex(maps).to_numpy(np.float64)
    ln = np.nan_to_num(ln_ratio, nan=0.0) >= 0.15
    masks = {
        "all": np.ones((M, M), dtype=bool),
        "near": np.abs((d_base[:, None] - d_base[None, :]) / sd_base) <= tau,
        "ln_any": ln[:, None] | ln[None, :],
        "ln_both": ln[:, None] & ln[None, :],
    }
    rng_eval = np.random.default_rng(args.seed + 23)
    evals = {}
    for name, mask in masks.items():
        evals[name] = _eval_subset(tab_te, groups_te, item_te, perf_te,
                                   clusters_te, n_clusters, mask, d_base,
                                   d_corr, args.bootstrap, rng_eval)

    # controls: overall NLL and delta
    controls = {}
    for name, dc in (("random", d_base + Z_rand @ w_rand),
                     ("shuffled", d_base + Z_shuf @ w_shuf),
                     ("d2v_only", Z @ w_only)):
        nll_c, _ = _nll(tab_te, dc)
        r = _eval_subset(tab_te, groups_te, item_te, perf_te, clusters_te,
                         n_clusters, masks["all"], d_base, dc,
                         args.bootstrap, rng_eval)
        controls[name] = {"test_nll": nll_c, "delta": r["delta"],
                          "ci": r["ci"], "p_delta_lt_0": r["p_delta_lt_0"]}

    train_nll_base, _ = _nll(tab_tr, d_base)
    train_nll_corr, _ = _nll(tab_tr, d_corr)
    test_nll_base, _ = _nll(tab_te, d_base)
    test_nll_corr, _ = _nll(tab_te, d_corr)

    report = {
        "protocol": args.protocol, "seed": args.seed, "model": args.model,
        "split": str(args.split), "baseline_oof": str(args.baseline_oof),
        "baseline_full": str(args.baseline_full) if args.baseline_full else None,
        "n_train_rows": int(len(train)), "n_test_rows": int(len(test)),
        "n_charts": int(M), "n_train_charts": int(train_chart_idx.size),
        "train_pairs": int(tab_tr.n_pairs), "test_pairs": int(tab_te.n_pairs),
        "train_instances": int(tab_tr.n_instances),
        "test_instances": int(tab_te.n_instances),
        "lam_grid": grid, "best_lam": float(best_lam), "val_nll": best_val,
        "val_curve": curve,
        "train_nll_base": train_nll_base, "train_nll_corr": train_nll_corr,
        "test_nll_base": test_nll_base, "test_nll_corr": test_nll_corr,
        "overall_delta": evals["all"]["delta"], "overall_ci": evals["all"]["ci"],
        "near_tau": tau, "near": evals["near"],
        "ln_any": evals["ln_any"], "ln_both": evals["ln_both"],
        "controls": controls, "corr": corr,
        "seconds": round(time.time() - t0, 1),
    }
    args.out_signals.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"beatmap_id": maps, "d_base": d_base, "d_corr": d_corr}).to_parquet(
        args.out_signals, index=False)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(f"[done] {args.protocol} s{args.seed} {args.model} lam={best_lam:g} "
          f"test {test_nll_base:.5f}->{test_nll_corr:.5f} "
          f"delta={evals['all']['delta']:+.5f} "
          f"near={evals['near']['delta']:+.5f} "
          f"ln={evals['ln_any']['delta']:+.5f} ({report['seconds']}s)", flush=True)
    print(json.dumps({k: report[k] for k in
                      ("train_nll_base", "test_nll_base", "test_nll_corr",
                       "overall_delta", "overall_ci", "near_tau", "controls",
                       "seconds")}, indent=1, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
