#!/usr/bin/env python
"""step2_cm3p_rank -- per-chart difficulty from frozen CM3P content, fitted for rank-NLL.

The evaluated object is a single difficulty score per chart, d_m = f_theta(C_m).
The objective is the EXACT marginal Bradley-Terry NLL of docs/metrics.md, not a
per-(player, chart) regression:

    q_ab = sigmoid(tau * (z_b - z_a)),   z = (d - mean(d)) / sd(d)
    L    = - sum_{a<b} [ n_ab log q_ab + n_ba log(1 - q_ab) ] / N

n_ab / n_ba are the within-player pair counts built from training rows (one row
per player-chart, ties 0.5/0.5); they are the sufficient statistics of the
metric.  There is no per-chart free parameter: d is a function of the frozen
CM3P vector only.  A free b_m would be the BT ceiling.

Models (M0-M2):
    linear        d = w'C                     (M0)
    linear_extra  d = w'C + v'S               (M2)
    mlp           d = w'C + MLP(C)            (M1, residual on the linear probe)
    mlp_extra     d = w'[C,S] + MLP([C,S])    (M2)
    ext_blend     d = v'S                     (M2 baseline, external signals only)

S is a matrix of optional per-chart external signals (MSD / star / ...).  The
evaluation re-fits the link's beta on the evaluation pairs, so a global affine
transform of d is irrelevant; its spacing is not (linear link).

Split protocols:
    default            P1: train rows -> d, evaluate on test rows (new pairs,
                        same charts).  The test rows are never used in fitting.
    --chart-folds K    P2: K-fold chart-disjoint.  Each chart is scored by a
                        model trained on rows whose charts are all outside that
                        chart's fold; the K fold scores are composed into one
                        signal column.  This is the "new chart" protocol.
    --content-ceiling  diagnostic: fit the same model on the evaluation pairs
                        themselves (in-sample, optimistic) to bound the content
                        family.  Label it as an oracle, not a result.

Outputs:
    {out-signals}: parquet with beatmap_id + one signal column
    {out-json}:    train/val/test NLL, hyperparameters and coverage

The official captured / strict-pair evaluation is scripts/step2_rank_eval.py.
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

try:  # the linear models do not need torch
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from dcr import rank_metrics as rm  # noqa: E402

LOG2 = float(np.log(2.0))


# --------------------------------------------------------------------------- io
def _read_step1(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "player_id_base" not in df.columns:
        df["player_id_base"] = df["player_id"]
    if "split" not in df.columns:
        raise SystemExit("step1 table needs a split column")
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
    if X.ndim != 2:
        raise SystemExit(f"{path}: {col} is not a 2-D matrix")
    if not np.isfinite(X).all():
        raise SystemExit(f"{path}: non-finite embedding values")
    return ids, X


def _encode_rows(df: pd.DataFrame, map_index: dict):
    """item, perf, groups, clusters, players for one play table."""
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


# ----------------------------------------------------------------------- linear
def fit_linear(X: np.ndarray, tab: rm.PairTable, lam: float,
               maxiter: int = 400) -> tuple[np.ndarray, float]:
    """L2-regularised logistic regression on the aggregated pair counts."""
    if tab.n_pairs == 0:
        raise SystemExit("empty pair table")
    X = np.asarray(X, dtype=np.float64)
    M, D = X.shape
    ua, ub = tab.ua, tab.ub
    nab, nba = tab.n_ab, tab.n_ba
    N = float(tab.n_instances)

    def f(w):
        d = X @ w
        u = d[ub] - d[ua]
        s = expit(u)
        nll = -np.sum(nab * np.log(np.clip(s, 1e-15, 1.0))
                      + nba * np.log(np.clip(1.0 - s, 1e-15, 1.0)))
        du = -nab * (1.0 - s) + nba * s
        gd = (np.bincount(ua, weights=-du, minlength=M)
              + np.bincount(ub, weights=du, minlength=M))
        return nll / N + lam * float(w @ w), X.T @ gd / N + 2.0 * lam * w

    r = minimize(f, np.zeros(D), jac=True, method="L-BFGS-B",
                 options=dict(maxiter=maxiter, gtol=1e-8, ftol=1e-12))
    w = r.x
    return w, float(rm.nll_counts(tab, X @ w) / tab.n_instances)


def eval_linear(X: np.ndarray, tab: rm.PairTable, w: np.ndarray) -> dict:
    d = np.asarray(X, dtype=np.float64) @ w
    link = rm.fit_link(tab, d, kind="linear")
    return {"nll_per_instance": float(link.nll / tab.n_instances),
            "beta": float(link.beta), "d": d}


# -------------------------------------------------------------------------- mlp
class Scorer:
    """d = w'x + MLP(x), a residual nonlinear correction to the linear probe."""

    def __init__(self, d_in: int, hid: int, depth: int, dropout: float,
                 lin_init: np.ndarray | None = None):
        if torch is None:
            raise SystemExit("MLP needs torch")
        self.model = nn.Module()
        lin = nn.Linear(d_in, 1, bias=False)
        if lin_init is not None:
            with torch.no_grad():
                lin.weight.copy_(torch.as_tensor(
                    np.asarray(lin_init, dtype=np.float32)).view(1, -1))
        norm = nn.LayerNorm(d_in)
        layers, prev = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(prev, hid), nn.GELU()]
            if dropout:
                layers.append(nn.Dropout(dropout))
            prev = hid
        mlp = nn.Sequential(*layers)
        out = nn.Linear(prev, 1)
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        self.model.lin, self.model.norm = lin, norm
        self.model.mlp, self.model.out = mlp, out

    def params(self):
        lin = [self.model.lin.weight]
        rest = (list(self.model.norm.parameters())
                + list(self.model.mlp.parameters())
                + list(self.model.out.parameters()))
        return lin, rest

    def n_params(self) -> int:
        a, b = self.params()
        return sum(p.numel() for p in a + b)

    def forward(self, x):
        m = self.model
        return (m.lin(x).squeeze(-1)
                + m.out(m.mlp(m.norm(x))).squeeze(-1))


def _mlp_loss(model, X, ua, ub, nab, nba, N):
    d = model.forward(X)
    z = (d - d.mean()) / (d.std() + 1e-8)
    u = z[ub] - z[ua]
    return -(nab * F.logsigmoid(u) + nba * F.logsigmoid(-u)).sum() / N


def fit_mlp(X: np.ndarray, tab_fit: rm.PairTable, tab_val: rm.PairTable | None,
            lin_init: np.ndarray, args, seed: int):
    if torch is None:
        raise SystemExit("MLP needs torch")
    torch.manual_seed(seed)
    torch.set_num_threads(int(args.threads) if args.threads else 6)
    Xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
    ua = torch.as_tensor(tab_fit.ua, dtype=torch.long)
    ub = torch.as_tensor(tab_fit.ub, dtype=torch.long)
    nab = torch.as_tensor(tab_fit.n_ab, dtype=torch.float32)
    nba = torch.as_tensor(tab_fit.n_ba, dtype=torch.float32)
    N = float(tab_fit.n_instances)
    if tab_val is not None and tab_val.n_pairs:
        vua = torch.as_tensor(tab_val.ua, dtype=torch.long)
        vub = torch.as_tensor(tab_val.ub, dtype=torch.long)
        vnab = torch.as_tensor(tab_val.n_ab, dtype=torch.float32)
        vnba = torch.as_tensor(tab_val.n_ba, dtype=torch.float32)
        vN = float(tab_val.n_instances)
    else:
        vua = vub = vnab = vnba = None
        vN = 0.0

    scorer = Scorer(X.shape[1], args.hid, args.depth, args.dropout, lin_init)
    lin_p, rest_p = scorer.params()
    opt = torch.optim.Adam([
        {"params": lin_p, "weight_decay": float(args.lin_wd)},
        {"params": rest_p, "weight_decay": float(args.wd)},
    ], lr=float(args.lr))

    best = (float("inf"), 0, None)
    hist = []
    patience = int(args.patience)
    bad = 0
    for ep in range(1, int(args.epochs) + 1):
        scorer.model.train()
        opt.zero_grad(set_to_none=True)
        loss = _mlp_loss(scorer, Xt, ua, ub, nab, nba, N)
        loss.backward()
        opt.step()
        tr = float(loss.detach())
        if vua is not None:
            scorer.model.eval()
            with torch.no_grad():
                vl = float(_mlp_loss(scorer, Xt, vua, vub, vnab, vnba, vN))
            if vl < best[0] - 1e-7:
                state = {k: v.detach().clone()
                         for k, v in scorer.model.state_dict().items()}
                best = (vl, ep, state)
                bad = 0
            else:
                bad += 1
            if ep % max(1, int(args.log_every)) == 0 or ep == 1:
                print(f"  [mlp ep {ep:3d}] train={tr:.5f} val={vl:.5f} "
                      f"best=({best[0]:.5f}@{best[1]})", flush=True)
            if bad >= patience:
                break
        else:
            if ep % max(1, int(args.log_every)) == 0 or ep == 1:
                print(f"  [mlp ep {ep:3d}] train={tr:.5f}", flush=True)
            best = (tr, ep, None)
        hist.append({"epoch": ep, "train": tr,
                     "val": None if vua is None else vl})
    if best[2] is not None:
        scorer.model.load_state_dict(best[2])
    return scorer, int(best[1]), (None if vua is None else float(best[0])), hist


def predict_mlp(scorer, X: np.ndarray) -> np.ndarray:
    scorer.model.eval()
    with torch.no_grad():
        d = scorer.forward(torch.as_tensor(np.asarray(X, dtype=np.float32)))
        d = d.detach().cpu().numpy().astype(np.float64)
    return (d - d.mean()) / (d.std() + 1e-8)


def refit_mlp(X, tab, lin_init, args, seed, epochs):
    """Train on all train rows for a fixed number of epochs (no validation)."""
    a = argparse.Namespace(**vars(args))
    a.epochs = int(epochs)
    a.log_every = 10 ** 9
    scorer, _, _, _ = fit_mlp(X, tab, None, lin_init, a, seed)
    return scorer


# ------------------------------------------------------------------- data / cv
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


def _chart_folds(train: pd.DataFrame, maps: np.ndarray, k: int, seed: int):
    """Balance chart folds by the number of train rows (greedy min-load)."""
    counts = (train.groupby("beatmap_id").size()
              .reindex(maps).fillna(0).to_numpy(np.float64))
    rng = np.random.default_rng(seed)
    order = np.argsort(-counts, kind="stable")
    # random tie-break so equal-sized charts do not always go to the same fold
    counts = counts[order] + rng.random(order.size) * 1e-6
    order = order[np.argsort(-counts, kind="stable")]
    fold = np.empty(len(maps), dtype=np.int64)
    load = np.zeros(k, dtype=np.float64)
    for idx in order:
        f = int(np.argmin(load))
        fold[idx] = f
        load[f] += float(counts[idx])
    return fold


def _load_extra(path: Path, names: list[str], maps: np.ndarray) -> np.ndarray | None:
    if not names:
        return None
    df = pd.read_parquet(path).set_index("beatmap_id")
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing extra signals {missing}")
    X = df[names].reindex(maps).to_numpy(np.float64)
    if not np.isfinite(X).all():
        bad = int((~np.isfinite(X)).sum())
        raise SystemExit(f"{path}: {bad} non-finite extra signal values on step1 maps")
    return X


# ------------------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default="data/processed/step1_1k_pairbest.parquet")
    ap.add_argument("--content-path", default="data/processed/chart_content_cm3p_local.parquet")
    ap.add_argument("--content-col", default=None)
    ap.add_argument("--signal-table", default="data/processed/rank_signals_1k.parquet")
    ap.add_argument("--extra-signals", default="msd_overall,sunny_star")
    ap.add_argument("--model", default="linear",
                    choices=["linear", "linear_extra", "mlp", "mlp_extra", "ext_blend"])
    ap.add_argument("--name", default="cm3p_rank")
    ap.add_argument("--lam", type=float, default=1e-2,
                    help="L2 for the linear probe (selected by validation when --val-frac>0)")
    ap.add_argument("--lam-grid", default="1e-3,1e-2,1e-1")
    ap.add_argument("--hid", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2,
                    help="number of hidden layers in the MLP (1 or 2)")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4,
                    help="weight decay on the MLP tower")
    ap.add_argument("--lin-wd", type=float, default=1e-4,
                    help="weight decay on the linear residual path")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="fraction of player clusters used for validation; 0 = fixed epochs")
    ap.add_argument("--val-seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=1,
                    help="number of final refits averaged after z-scoring")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chart-folds", type=int, default=0,
                    help="K-fold chart-disjoint protocol; 0 = P1 (new pairs, same charts)")
    ap.add_argument("--chart-seed", type=int, default=0)
    ap.add_argument("--chart-only", action="store_true",
                    help="skip the P1 fit and only run the chart-disjoint composition")
    ap.add_argument("--content-ceiling", action="store_true",
                    help="also fit the model on the evaluation pairs themselves "
                         "(in-sample oracle; not a generalising result)")
    ap.add_argument("--out-signals", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--log-every", type=int, default=10)
    return ap.parse_args(argv)


def main() -> int:
    t0 = time.time()
    args = parse_args()
    out_sig = Path(args.out_signals)
    out_json = Path(args.out_json)
    out_sig.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    df_all = _read_step1(Path(args.inp))
    cids, C_raw = _load_content(Path(args.content_path), args.content_col)
    step1_maps = np.sort(df_all["beatmap_id"].unique())
    pos = pd.Index(cids).get_indexer(step1_maps)
    keep = pos >= 0
    maps = step1_maps[keep]
    C_raw = C_raw[pos[keep]]
    M = int(maps.size)
    map_idx = {int(m): i for i, m in enumerate(maps)}
    df = df_all[df_all["beatmap_id"].isin(maps)].reset_index(drop=True)
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    if train.empty or test.empty:
        raise SystemExit(f"train={len(train)} test={len(test)} after content filter")
    print(f"[data] maps={M:,} (content coverage {keep.mean():.1%} of step1 maps) "
          f"train_rows={len(train):,} test_rows={len(test):,}", flush=True)

    item_tr, perf_tr, groups_tr, clusters_tr, players_tr = _encode_rows(train, map_idx)
    item_te, perf_te, groups_te, clusters_te, players_te = _encode_rows(test, map_idx)
    tab_tr = _pair_table(groups_tr, item_tr, perf_tr, M)
    tab_te = _pair_table(groups_te, item_te, perf_te, M)
    print(f"[pairs] train {tab_tr.n_pairs:,} ({tab_tr.n_instances:,} comparisons) | "
          f"test {tab_te.n_pairs:,} ({tab_te.n_instances:,})", flush=True)

    # content standardisation on training charts only (content-only; no labels)
    train_chart_idx = np.unique(item_tr)
    mu = C_raw[train_chart_idx].mean(0)
    sd = C_raw[train_chart_idx].std(0) + 1e-8
    Cs = (C_raw - mu) / sd
    X = Cs.astype(np.float64)
    extra_names = [s.strip() for s in args.extra_signals.split(",") if s.strip()]
    extra = _load_extra(Path(args.signal_table), extra_names, maps) if extra_names else None
    if args.model in ("linear_extra", "mlp_extra", "ext_blend") and extra is None:
        raise SystemExit(f"--model {args.model} needs --extra-signals")
    if extra is not None:
        em = extra[train_chart_idx].mean(0)
        es = extra[train_chart_idx].std(0) + 1e-8
        extra = (extra - em) / es
    if args.model in ("linear_extra", "mlp_extra"):
        X = np.hstack([Cs, extra]).astype(np.float64)
    elif args.model == "ext_blend":
        X = extra.astype(np.float64)
    else:
        X = Cs.astype(np.float64)

    # validation split by player cluster (test rows are untouched)
    fit_players, val_players = _val_players(clusters_tr, float(args.val_frac),
                                            int(args.val_seed))
    tab_fit = _pair_table(groups_tr, item_tr, perf_tr, M, players=fit_players)
    tab_val = (_pair_table(groups_tr, item_tr, perf_tr, M, players=val_players)
               if val_players.size else None)
    print(f"[val] {fit_players.size} fit players / {val_players.size} val players; "
          f"fit pairs {tab_fit.n_pairs:,}"
          + (f", val pairs {tab_val.n_pairs:,}" if tab_val is not None else ""),
          flush=True)

    # ---------------------------------------------------------------- fit signal
    signal = np.full(M, np.nan, dtype=np.float64)
    best_lam = float(args.lam)
    best_epoch = None
    val_nll = None
    fold_info = []
    epochs_for_final = int(args.epochs)
    if args.chart_only:
        pass
    elif args.model in ("linear", "linear_extra", "ext_blend"):
        grid = ([float(x) for x in args.lam_grid.split(",") if x.strip()]
                if args.val_frac > 0 else [float(args.lam)])
        best = (float("inf"), None, None)
        for lam in grid:
            w, tr_nll = fit_linear(X, tab_fit, lam)
            if tab_val is None or not tab_val.n_pairs:
                best = (tr_nll, lam, w)
                break
            vl = float(rm.fit_link(tab_val, X @ w).nll / tab_val.n_instances)
            print(f"[linear] lam={lam:g} fit_nll={tr_nll:.5f} val_nll={vl:.5f}",
                  flush=True)
            if np.isfinite(vl) and vl < best[0]:
                best = (vl, lam, w)
        if best[2] is None:
            best = (float("inf"), float(args.lam),
                    fit_linear(X, tab_fit, float(args.lam))[0])
        best_lam = float(best[1])
        # keep the validation NLL of the model fitted on the 80% fit players;
        # the refit on all train rows must not be evaluated on those same rows
        val_nll = (float(best[0]) if tab_val is not None and tab_val.n_pairs
                   else None)
        w, _ = fit_linear(X, tab_tr, best_lam)
        signal = X @ w
    else:
        # MLP: linear probe on the fit rows is the residual path's init
        lam = float(args.lam)
        if tab_val is not None and tab_val.n_pairs:
            w0, _ = fit_linear(X, tab_fit, lam)
            scorer, best_epoch, val_nll, hist = fit_mlp(
                X, tab_fit, tab_val, w0, args, int(args.seed))
            print(f"[mlp] best epoch {best_epoch}, val_nll={val_nll:.5f}", flush=True)
            epochs_for_final = max(1, best_epoch)
            w0_all, _ = fit_linear(X, tab_tr, lam)
            ds = []
            for s in range(int(args.seeds)):
                sc = refit_mlp(X, tab_tr, w0_all, args, int(args.seed) + s,
                               epochs_for_final)
                ds.append(predict_mlp(sc, X))
        else:
            # no validation: train each seed directly on every train row for
            # --epochs (used for the seed ensemble and chart-disjoint folds)
            w0_all, _ = fit_linear(X, tab_tr, lam)
            ds = []
            for s in range(int(args.seeds)):
                sc, _, _, _ = fit_mlp(X, tab_tr, None, w0_all, args,
                                      int(args.seed) + s)
                ds.append(predict_mlp(sc, X))
            best_epoch = int(args.epochs)
            val_nll = None
        signal = np.mean(ds, axis=0)
        best_lam = lam

    # ------------------------------------------------------ chart-disjoint (P2)
    if args.chart_folds and int(args.chart_folds) > 1:
        K = int(args.chart_folds)
        fold = _chart_folds(train, maps, K, int(args.chart_seed))
        signal = np.full(M, np.nan, dtype=np.float64)
        fold_rows = []
        for k in range(K):
            train_charts = np.flatnonzero(fold != k)
            rows = train[train["beatmap_id"].isin(maps[train_charts])]
            if rows.empty:
                raise SystemExit(f"chart fold {k}: no training rows")
            it, pf, gr, cl, pl = _encode_rows(rows, map_idx)
            tab_k = _pair_table(gr, it, pf, M)
            fp, vp = _val_players(cl, float(args.val_frac), int(args.val_seed) + k)
            tab_kf = _pair_table(gr, it, pf, M, players=fp)
            tab_kv = (_pair_table(gr, it, pf, M, players=vp)
                      if vp.size else None)
            if args.model in ("linear", "linear_extra", "ext_blend"):
                wk, tr_nll = fit_linear(X, tab_kf, best_lam)
                dk = X @ wk
            else:
                w0k, _ = fit_linear(X, tab_kf, float(args.lam))
                if tab_kv is not None and tab_kv.n_pairs:
                    sc, epk, _, _ = fit_mlp(X, tab_kf, tab_kv, w0k, args,
                                            int(args.seed) + k)
                    epk = max(1, epk)
                    w0k_all, _ = fit_linear(X, tab_k, float(args.lam))
                    sc = refit_mlp(X, tab_k, w0k_all, args,
                                   int(args.seed) + k, epk)
                else:
                    # no validation: train the fixed --epochs budget directly
                    # on every training row of this chart fold
                    sc, epk, _, _ = fit_mlp(X, tab_k, None, w0k, args,
                                            int(args.seed) + k)
                dk = predict_mlp(sc, X)
            dz = (dk - dk.mean()) / (dk.std() + 1e-8)
            signal[fold == k] = dz[fold == k]
            fold_rows.append({"fold": k, "n_train_rows": int(len(rows)),
                              "n_train_pairs": int(tab_k.n_pairs),
                              "n_val_players": int(vp.size),
                              "n_heldout_charts": int((fold == k).sum()),
                              "train_nll": float(rm.nll_counts(tab_k, dk)
                                                 / tab_k.n_instances)})
            print(f"[chart-fold {k}] charts={int((fold == k).sum())} "
                  f"rows={len(rows):,} pairs={tab_k.n_pairs:,} "
                  f"train_nll={fold_rows[-1]['train_nll']:.5f}", flush=True)
        fold_info = fold_rows
        signal = np.asarray(signal)
        val_nll = None
        best_epoch = None

    if not np.isfinite(signal).all():
        raise SystemExit(f"{signal[~np.isfinite(signal)].size} charts have no signal")

    # ---------------------------------------------------- content ceiling (M4)
    ceiling_nll = None
    if args.content_ceiling:
        if args.model in ("linear", "linear_extra", "ext_blend"):
            wc, _ = fit_linear(X, tab_te, best_lam)
            dc = X @ wc
        else:
            w0c, _ = fit_linear(X, tab_te, float(args.lam))
            a = argparse.Namespace(**vars(args))
            a.epochs = min(120, int(args.epochs))
            a.log_every = 10 ** 9
            sc, _, _, _ = fit_mlp(X, tab_te, None, w0c, a, int(args.seed))
            dc = predict_mlp(sc, X)
        ceiling_nll = float(rm.fit_link(tab_te, dc).nll / tab_te.n_instances)
        print(f"[content-ceiling] in-sample test nll={ceiling_nll:.5f} "
              f"(oracle; not a generalising result)", flush=True)

    # ---------------------------------------------------------------- evaluate
    link = rm.fit_link(tab_te, signal)
    test_nll = float(link.nll / tab_te.n_instances)
    train_link = rm.fit_link(tab_tr, signal)
    out = {
        "name": args.name,
        "model": args.model,
        "content_path": str(args.content_path),
        "content_dim": int(Cs.shape[1]),
        "content_coverage_maps": float(keep.mean()),
        "n_maps": M,
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "n_train_pairs": int(tab_tr.n_pairs),
        "n_train_comparisons": int(tab_tr.n_instances),
        "n_test_pairs": int(tab_te.n_pairs),
        "n_test_comparisons": int(tab_te.n_instances),
        "train_nll_per_instance": float(train_link.nll / tab_tr.n_instances),
        "val_nll_per_instance": val_nll,
        "test_nll_per_instance": test_nll,
        "test_bits_saved": float(1.0 - test_nll / LOG2),
        "test_link_beta": float(link.beta),
        "best_lam": best_lam,
        "best_epoch": best_epoch,
        "chart_folds": int(args.chart_folds),
        "chart_fold_info": fold_info,
        "content_ceiling_test_nll": ceiling_nll,
        "seed": int(args.seed),
        "seeds": int(args.seeds),
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
        "seconds": round(time.time() - t0, 1),
    }
    if args.model.startswith("mlp"):
        out["mlp_params"] = int(scorer.n_params()) if "scorer" in locals() else None

    sig_df = pd.DataFrame({"beatmap_id": maps, args.name: signal})
    if args.chart_folds and int(args.chart_folds) > 1:
        sig_df["chart_fold"] = fold
    sig_df.to_parquet(out_sig, index=False)
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[done] test_nll={test_nll:.5f} bits={out['test_bits_saved']:.4f} "
          f"beta={link.beta:.3f} -> {out_sig.name} / {out_json.name} "
          f"({out['seconds']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
