#!/usr/bin/env python
"""Shared-difficulty linear model with a first-two-record auxiliary likelihood.

Experiment 1's minimal model:

    d_m = w' C_m + v' S_m
    L(w) = L_hist(w) + λ_aux * L_first2(w) + λ_ridge ||w||^2

Both likelihoods use the same per-chart difficulty d; only the training labels
change.  The test label / test pair set / ceiling are untouched: the primary
readout is the historical-best rank-NLL from scripts/step2_rank_eval.py or
scripts/paired_delta_nll.py.

This script is deliberately linear_extra-only.  It is the pilot / confirmatory
workhorse; the MLP version is not part of Experiment 1's minimal test.

Examples
--------
# A0 baseline (λ_aux = 0), randommerge P4 s1
python scripts/step2_cm3p_rank_aux.py \
  --in data/processed/step1_randommerge_c_p4_s1.parquet \
  --first2 data/processed/step1_randommerge_first2.parquet \
  --content-path data/processed/chart_content_cm3p_local.parquet \
  --signal-table data/processed/rank_signals_randommerge_od.parquet \
  --extra-signals msd_overall,sunny_star,log_n \
  --aux-weight-grid 0 --val-frac 0 \
  --name first2_a0 --out-signals results/first2_aux/pilot/a0.parquet \
  --out-json results/first2_aux/pilot/a0.json

# A1 first2 auxiliary (λ selected on 20% player clusters)
python scripts/step2_cm3p_rank_aux.py ... \
  --aux-weight-grid 0,0.05,0.1,0.2,0.5,1.0 --val-frac 0.2 \
  --name first2_a1 --out-signals results/first2_aux/pilot/a1.parquet \
  --out-json results/first2_aux/pilot/a1.json
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from dcr import rank_metrics as rm  # noqa: E402


def _read_table(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    if "player_id_base" not in df.columns:
        df["player_id_base"] = df["player_id"]
    if "split" not in df.columns:
        raise SystemExit(f"{path}: needs a split column")
    return df


def _load_content(path: Path, col: str | None) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(path)
    if col is None:
        for c in ("embedding", "element", "vector"):
            if c in df.columns:
                col = c
                break
    if col is None or col not in df.columns:
        raise SystemExit(f"{path}: no embedding column (tried embedding/element/vector)")
    ids = df["beatmap_id"].to_numpy()
    X = np.stack(df[col].to_numpy()).astype(np.float32)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise SystemExit(f"{path}: bad content matrix")
    return ids, X


def _load_extra(path: Path, names: list[str], maps: np.ndarray) -> np.ndarray | None:
    if not names:
        return None
    df = pd.read_parquet(path).set_index("beatmap_id")
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing extra signals {missing}")
    X = df[names].reindex(maps).to_numpy(np.float64)
    if not np.isfinite(X).all():
        raise SystemExit(f"{path}: non-finite extra signal values")
    return X


def _encode_rows(df: pd.DataFrame, map_index: dict):
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


def _pair_table(groups, item, perf, n_items, players=None):
    g = groups if players is None else [groups[p] for p in players]
    return rm.build_pair_table(g, item, perf, n_items)


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


def fit_linear_aux(X: np.ndarray, tab_h: rm.PairTable, tab_a: rm.PairTable | None,
                   lam_aux: float, lam_ridge: float,
                   maxiter: int = 400) -> tuple[np.ndarray, float]:
    """L2 logistic regression on aggregated pair counts with a second likelihood."""
    if tab_h.n_pairs == 0:
        raise SystemExit("empty historical pair table")
    if lam_aux > 0 and (tab_a is None or tab_a.n_pairs == 0):
        raise SystemExit("λ_aux > 0 but the auxiliary pair table is empty")
    X = np.asarray(X, dtype=np.float64)
    M, D = X.shape
    uh_a, uh_b = tab_h.ua, tab_h.ub
    nh_ab, nh_ba = tab_h.n_ab, tab_h.n_ba
    Nh = float(tab_h.n_instances)
    if tab_a is not None and tab_a.n_pairs:
        ua_a, ua_b = tab_a.ua, tab_a.ub
        na_ab, na_ba = tab_a.n_ab, tab_a.n_ba
        Na = float(tab_a.n_instances)
    else:
        ua_a = ua_b = None
        na_ab = na_ba = None
        Na = 1.0

    def f(w):
        d = X @ w

        u = d[uh_b] - d[uh_a]
        s = expit(u)
        nll_h = -np.sum(nh_ab * np.log(np.clip(s, 1e-15, 1.0))
                        + nh_ba * np.log(np.clip(1.0 - s, 1e-15, 1.0)))
        du = -nh_ab * (1.0 - s) + nh_ba * s
        gd = (np.bincount(uh_a, weights=-du, minlength=M)
              + np.bincount(uh_b, weights=du, minlength=M))
        val = nll_h / Nh
        grad = gd / Nh

        if lam_aux > 0 and tab_a is not None and tab_a.n_pairs:
            ua = d[ua_b] - d[ua_a]
            sa = expit(ua)
            nll_a = -np.sum(na_ab * np.log(np.clip(sa, 1e-15, 1.0))
                            + na_ba * np.log(np.clip(1.0 - sa, 1e-15, 1.0)))
            dua = -na_ab * (1.0 - sa) + na_ba * sa
            gda = (np.bincount(ua_a, weights=-dua, minlength=M)
                   + np.bincount(ua_b, weights=dua, minlength=M))
            val += lam_aux * nll_a / Na
            grad += lam_aux * gda / Na

        val += lam_ridge * float(w @ w)
        g_w = X.T @ grad + 2.0 * lam_ridge * w
        return float(val), g_w

    r = minimize(f, np.zeros(D), jac=True, method="L-BFGS-B",
                 options=dict(maxiter=maxiter, gtol=1e-8, ftol=1e-12))
    if not r.success:
        print(f"[warn] L-BFGS-B: {r.message}", flush=True)
    return r.x, float(r.fun)


def _nll(tab: rm.PairTable, d: np.ndarray) -> tuple[float, float]:
    link = rm.fit_link(tab, d, kind="linear")
    return float(link.nll / tab.n_instances), float(link.beta)


def _scope_mask(df: pd.DataFrame, scope: str, aux_col: str) -> np.ndarray:
    if scope == "all":
        return np.ones(len(df), dtype=bool)
    if scope == "n_plays_gt2":
        return df["n_plays"].to_numpy(np.int64) > 2
    if scope == "support":
        acc = 1.0 - np.exp(df[aux_col].to_numpy(np.float64))
        return (acc > 0.90) & (acc <= 0.998)
    raise SystemExit(f"unknown --aux-scope {scope!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, type=Path,
                    help="step1 split table (train/test; player_id is the segment)")
    ap.add_argument("--first2", required=True, type=Path,
                    help="first2 label table from step1_first2_labels.py; merged on "
                         "(player_id, beatmap_id)")
    ap.add_argument("--aux-label-col", default="loss_first2")
    ap.add_argument("--aux-scope", default="all",
                    choices=["all", "n_plays_gt2", "support"])
    ap.add_argument("--aux-weight-grid", default="0,0.05,0.1,0.2,0.5,1.0")
    ap.add_argument("--lam-ridge", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--val-seed", type=int, default=0)
    ap.add_argument("--content-path", default="data/processed/chart_content_cm3p_local.parquet")
    ap.add_argument("--content-col", default=None)
    ap.add_argument("--signal-table", default="data/processed/rank_signals_randommerge_od.parquet")
    ap.add_argument("--extra-signals", default="msd_overall,sunny_star,log_n")
    ap.add_argument("--name", default="first2_aux")
    ap.add_argument("--out-signals", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    df = _read_table(args.inp)
    lab = pd.read_parquet(args.first2)
    need = ["player_id", "beatmap_id", args.aux_label_col]
    missing_cols = [c for c in need if c not in lab.columns]
    if missing_cols:
        raise SystemExit(f"{args.first2}: missing {missing_cols}")
    lab = lab[need].copy()
    df = df.merge(lab, on=["player_id", "beatmap_id"], how="left")
    if df[args.aux_label_col].isna().any():
        raise SystemExit(f"{int(df[args.aux_label_col].isna().sum())} rows missing "
                         f"{args.aux_label_col} after merge")
    df["aux_loss"] = df[args.aux_label_col].astype(np.float64)

    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    if train.empty or test.empty:
        raise SystemExit(f"train={len(train)} test={len(test)}")
    print(f"[data] {args.inp.name}: train={len(train):,} test={len(test):,} "
          f"scope={args.aux_scope}", flush=True)

    cids, C_raw = _load_content(args.content_path, args.content_col)
    step1_maps = np.sort(df["beatmap_id"].unique())
    pos = pd.Index(cids).get_indexer(step1_maps)
    keep = pos >= 0
    maps = step1_maps[keep]
    C_raw = C_raw[pos[keep]]
    M = int(maps.size)
    map_idx = {int(m): i for i, m in enumerate(maps)}
    train = train[train["beatmap_id"].isin(maps)].reset_index(drop=True)
    test = test[test["beatmap_id"].isin(maps)].reset_index(drop=True)
    if train.empty or test.empty:
        raise SystemExit(f"after content coverage: train={len(train)} test={len(test)}")

    train_chart_idx = np.unique(train["beatmap_id"].map(map_idx).to_numpy(np.int32))
    mu = C_raw[train_chart_idx].mean(0)
    sd = C_raw[train_chart_idx].std(0) + 1e-8
    Cs = (C_raw - mu) / sd
    extra_names = [s.strip() for s in args.extra_signals.split(",") if s.strip()]
    extra = _load_extra(args.signal_table, extra_names, maps) if extra_names else None
    if extra is not None:
        em = extra[train_chart_idx].mean(0)
        es = extra[train_chart_idx].std(0) + 1e-8
        extra = (extra - em) / es
        X = np.hstack([Cs, extra]).astype(np.float64)
    else:
        X = Cs.astype(np.float64)

    item_tr, perf_tr, groups_tr, clusters_tr, players_tr = _encode_rows(train, map_idx)
    tab_h_tr = _pair_table(groups_tr, item_tr, perf_tr, M)
    item_te, perf_te, groups_te, clusters_te, players_te = _encode_rows(test, map_idx)
    tab_h_te = _pair_table(groups_te, item_te, perf_te, M)

    # auxiliary table on all train rows in scope
    scope_tr = _scope_mask(train, args.aux_scope, args.aux_label_col)
    aux_train = train.loc[scope_tr].copy()
    aux_train["loss"] = aux_train["aux_loss"]
    item_a, perf_a, groups_a, _, _ = _encode_rows(aux_train, map_idx)
    tab_a_tr = _pair_table(groups_a, item_a, perf_a, M)
    print(f"[aux] scope rows {int(scope_tr.sum()):,}/{len(train):,}; "
          f"pairs {tab_a_tr.n_pairs:,}; comparisons {tab_a_tr.n_instances:,}",
          flush=True)

    grid = [float(x) for x in args.aux_weight_grid.split(",") if x.strip()]
    best_lam, best_val = (grid[0], None), None
    val_curve = []
    w = None
    if args.val_frac > 0 and len(grid) > 1:
        fit_players, val_players = _val_players(clusters_tr, args.val_frac, args.val_seed)
        tab_h_fit = _pair_table(groups_tr, item_tr, perf_tr, M, players=fit_players)
        tab_h_val = _pair_table(groups_tr, item_tr, perf_tr, M, players=val_players)
        fit_ids = set(int(players_tr[p]) for p in fit_players)
        fit_scope = scope_tr & train["player_id"].isin(fit_ids).to_numpy()
        aux_fit = train.loc[fit_scope].copy()
        aux_fit["loss"] = aux_fit["aux_loss"]
        item_af, perf_af, groups_af, _, _ = _encode_rows(aux_fit, map_idx)
        tab_a_fit = _pair_table(groups_af, item_af, perf_af, M)
        for lam in grid:
            wl, _ = fit_linear_aux(X, tab_h_fit, tab_a_fit, lam, args.lam_ridge,
                                   args.maxiter)
            vl, _ = _nll(tab_h_val, X @ wl)
            val_curve.append({"lam_aux": lam, "val_nll": vl})
            print(f"[val] lam_aux={lam:g} val_nll={vl:.5f}", flush=True)
            if best_val is None or vl < best_val:
                best_val, best_lam = vl, lam
    else:
        best_lam = grid[0]

    w, train_obj = fit_linear_aux(X, tab_h_tr, tab_a_tr, best_lam, args.lam_ridge,
                                  args.maxiter)
    d = X @ w
    train_nll, train_beta = _nll(tab_h_tr, d)
    test_nll, test_beta = _nll(tab_h_te, d)
    test_first2_nll = None
    scope_te = _scope_mask(test, args.aux_scope, args.aux_label_col)
    if scope_te.any():
        aux_test = test.loc[scope_te].copy()
        aux_test["loss"] = aux_test["aux_loss"]
        item_at, perf_at, groups_at, _, _ = _encode_rows(aux_test, map_idx)
        tab_a_te = _pair_table(groups_at, item_at, perf_at, M)
        if tab_a_te.n_pairs:
            test_first2_nll, _ = _nll(tab_a_te, d)

    report = {
        "in": str(args.inp), "first2": str(args.first2),
        "aux_label_col": args.aux_label_col, "aux_scope": args.aux_scope,
        "n_rows": {"train": int(len(train)), "test": int(len(test)),
                   "aux_train": int(scope_tr.sum()), "aux_test": int(scope_te.sum())},
        "n_charts": int(M),
        "pairs": {"train": int(tab_h_tr.n_pairs),
                  "train_instances": int(tab_h_tr.n_instances),
                  "test": int(tab_h_te.n_pairs),
                  "test_instances": int(tab_h_te.n_instances),
                  "aux_train": int(tab_a_tr.n_pairs),
                  "aux_train_instances": int(tab_a_tr.n_instances)},
        "aux_weight_grid": grid, "best_lam_aux": float(best_lam),
        "val_nll": best_val, "val_curve": val_curve,
        "train_nll_hist": train_nll, "train_beta": train_beta,
        "test_nll_hist": test_nll, "test_beta": test_beta,
        "test_nll_first2": test_first2_nll,
        "train_obj": train_obj, "n_params": int(X.shape[1]),
        "seconds": round(time.time() - t0, 1),
    }
    args.out_signals.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"beatmap_id": maps, args.name: d}).to_parquet(
        args.out_signals, index=False)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(f"[done] lam_aux={best_lam:g} train_nll={train_nll:.5f} "
          f"test_nll={test_nll:.5f} test_first2={test_first2_nll} "
          f"({report['seconds']}s)", flush=True)
    print(json.dumps({k: report[k] for k in
                      ("pairs", "best_lam_aux", "val_nll", "test_nll_hist",
                       "test_nll_first2", "seconds")}, indent=1, ensure_ascii=False),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
