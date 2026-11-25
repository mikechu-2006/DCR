
"""Pairwise difficulty-ranking metrics.

The metric answers one question: **how well does a per-chart difficulty score
predict which of two charts a player performs better on?**

For player p with played set S_p and performance s_pm (higher = better, i.e.
s = -loss), every unordered pair {i,j} in S_p is one Bernoulli observation

    y_pij = 1[ s_pi > s_pj ]        (i beat j)
    q_ij  = sigmoid( g(d_j) - g(d_i) )

where d is the system difficulty (higher = harder) and g is a link: linear
(beta*d), rank, or a hat-function spline.  The loss is the negative
log-likelihood of the observed comparisons:

    L(d) = - sum_{i<j} [ n_ij log q_ij + n_ji log(1 - q_ij) ]

with n_ij the number of players for whom i beat j.  The counts are the
sufficient statistics of the Bradley-Terry model; do **not** first turn them
into an empirical probability and then take a cross-entropy (that is a noisy
plug-in target, and 58% of the map pairs in 1k occur exactly once).

The unconstrained Bradley-Terry fit (b_m free) is the ceiling for any global
ordering.  It is cross-fitted by player cluster, and the reported metric is:

    bits_saved = 1 - L(d)/N / log(2)
    captured   = (log(2) - L(d)/N) / (log(2) - L*/N)

captured in [0,1] is the fraction of the achievable pairwise log-loss
reduction that the system's ordering captures.  It is the primary metric.

The companion per-player Spearman is a descriptive statistic (not a proper
scoring rule): rho_p = Spearman( d|S_p, -s_p|S_p ), summarised with Fisher-z /
l-weighted / macro averages and a cluster bootstrap.

All functions are numpy-only; the CLI is scripts/step2_rank_eval.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit
from scipy.stats import rankdata

LOG2 = float(np.log(2.0))


# --------------------------------------------------------------------------- pairs
@dataclass
class PairTable:
    """Aggregated unordered-pair counts.

    ua/ub are item indices with ua < ub; n_ab is the (possibly fractional)
    count of comparisons where a beat b, n_ba the count where b beat a.
    Ties are split 0.5/0.5 by default, so n_ab + n_ba == the number of
    comparisons of that pair.
    """

    ua: np.ndarray
    ub: np.ndarray
    n_ab: np.ndarray
    n_ba: np.ndarray
    n_items: int
    n_instances: int
    n_ties: int = 0

    @property
    def n_pairs(self) -> int:
        return int(self.ua.size)


def build_pair_table(groups: Iterable[np.ndarray], item: np.ndarray,
                     perf: np.ndarray, n_items: int,
                     tie: str = "half") -> PairTable:
    """Aggregate every within-player unordered pair into a PairTable.

    groups : iterable of row-index arrays, one per player (segment).
    item   : row-level item index (int32/int64, 0 <= item < n_items).
    perf   : row-level performance (higher is better).
    """
    if tie not in ("half", "drop"):
        raise ValueError("tie must be 'half' or 'drop'")
    keys, ys = [], []
    n_ties = 0
    for idx in groups:
        l = int(len(idx))
        if l < 2:
            continue
        ii, jj = np.triu_indices(l, 1)
        a = item[idx][ii]
        b = item[idx][jj]
        swap = a > b
        aa = np.where(swap, b, a).astype(np.int64)
        bb = np.where(swap, a, b).astype(np.int64)
        pa = perf[idx][ii]
        pb = perf[idx][jj]
        y = (pa > pb) ^ swap
        tie_mask = pa == pb
        n_ties += int(tie_mask.sum())
        if tie == "drop":
            keep = ~tie_mask
            aa, bb, y = aa[keep], bb[keep], y[keep]
            if aa.size == 0:
                continue
            w = y.astype(np.float64)
        else:
            w = np.where(tie_mask, 0.5, y.astype(np.float64))
        keys.append(aa * n_items + bb)
        ys.append(w)
    if not keys:
        return PairTable(np.zeros(0, np.int32), np.zeros(0, np.int32),
                         np.zeros(0), np.zeros(0), n_items, 0, n_ties)
    key = np.concatenate(keys)
    y = np.concatenate(ys)
    order = np.argsort(key, kind="stable")
    key_s = key[order]
    y_s = y[order]
    u, start, cnt = np.unique(key_s, return_index=True, return_counts=True)
    wins = np.add.reduceat(y_s, start)
    ua = (u // n_items).astype(np.int32)
    ub = (u % n_items).astype(np.int32)
    return PairTable(ua, ub, wins.astype(np.float64),
                     (cnt - wins).astype(np.float64), n_items,
                     int(cnt.sum()), n_ties)


# ----------------------------------------------------------------------- nll / bt
def nll_counts(table: PairTable, b: np.ndarray) -> float:
    """Negative pairwise log-likelihood of item scores b (higher = harder)."""
    u = b[table.ub] - b[table.ua]
    s = np.clip(expit(u), 1e-15, 1.0 - 1e-15)
    return float(-np.sum(table.n_ab * np.log(s) + table.n_ba * np.log(1.0 - s)))


def fit_bt(table: PairTable, maxiter: int = 300, gtol: float = 1e-6,
           lam: float = 0.0) -> tuple[np.ndarray, float, dict]:
    """Unconstrained Bradley-Terry MLE for item scores b (centred)."""
    n = table.n_items
    if table.n_pairs == 0:
        return np.zeros(n), 0.0, {"ok": False, "nit": 0, "nll": 0.0}
    ua, ub, nab, nba = table.ua, table.ub, table.n_ab, table.n_ba

    def f(x: np.ndarray):
        b = x - x.mean()
        u = b[ub] - b[ua]
        s = np.clip(expit(u), 1e-15, 1.0 - 1e-15)
        nll = -np.sum(nab * np.log(s) + nba * np.log(1.0 - s))
        du = -nab * (1.0 - s) + nba * s
        g = (np.bincount(ua, weights=-du, minlength=n)
             + np.bincount(ub, weights=du, minlength=n))
        g = g - g.mean()
        if lam:
            nll = nll + lam * float(np.sum(b * b))
            g = g + 2.0 * lam * b
        return float(nll), g

    r = minimize(f, np.zeros(n), jac=True, method="L-BFGS-B",
                 options=dict(maxiter=maxiter, gtol=gtol, ftol=1e-13))
    b = r.x - r.x.mean()
    return b, float(r.fun), {"ok": bool(r.success), "nit": int(r.nit),
                             "nll": float(r.fun), "message": str(r.message)}


# -------------------------------------------------------------------------- links
def _scaled(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=np.float64)
    sd = float(d.std())
    return (d - d.mean()) / (sd if sd > 0 else 1.0)


def hat_basis(d: np.ndarray, knots: int = 8) -> np.ndarray:
    """Piecewise-linear hat functions at interior quantiles of d."""
    d = _scaled(d)
    if knots < 1:
        return d[:, None]
    qs = np.quantile(d, np.linspace(0.0, 1.0, knots + 2)[1:-1])
    qs = np.unique(qs)
    if qs.size == 0:
        return d[:, None]
    h = float(np.median(np.diff(qs))) if qs.size > 1 else 1.0
    h = max(h, 1e-6)
    return np.maximum(0.0, 1.0 - np.abs(d[:, None] - qs[None, :]) / h)


@dataclass
class LinkFit:
    kind: str
    params: np.ndarray
    nll: float
    transform: Callable[[np.ndarray], np.ndarray]
    extra: dict = field(default_factory=dict)

    @property
    def beta(self) -> float:
        return float(self.params[0]) if self.params.size else float("nan")


def _fit_beta_1d(table: PairTable, z: np.ndarray) -> float:
    """MLE of beta for q = sigmoid(beta * (z_j - z_i)), beta >= 0.

    The 1-D negative log-likelihood is convex, so its derivative is monotone;
    a bisection on the derivative is exact and much more robust than a bounded
    scalar search over a wide interval (which can stop at the upper bound when
    the signal is nearly separable).
    """
    from scipy.optimize import brentq
    dz = z[table.ub] - z[table.ua]
    nab, nba = table.n_ab, table.n_ba

    def deriv(beta: float) -> float:
        s = expit(beta * dz)
        du = -nab * (1.0 - s) + nba * s
        return float(np.sum(du * dz))

    if deriv(0.0) >= 0.0:
        return 0.0
    hi = 1.0
    while hi < 1e3 and deriv(hi) <= 0.0:
        hi *= 2.0
    if deriv(hi) <= 0.0:
        return 1e3
    try:
        return float(brentq(deriv, 1e-9, hi, xtol=1e-8, rtol=1e-10,
                            maxiter=300))
    except Exception:
        return float(hi)


def fit_link(table: PairTable, d: np.ndarray, kind: str = "linear",
             knots: int = 8, lam: float = 1e-4) -> LinkFit:
    """Fit the link g(d) by MLE of the pairwise log-likelihood.

    linear : g = beta * z(d),  beta >= 0 (claimed direction: harder -> lower performance)
    rank   : g = beta * rank(z(d))/M, beta >= 0 (monotone, scale-free)
    spline : g = sum_k beta_k hat_k(z(d)), unconstrained (upper bound; may be non-monotone)
    """
    d = np.asarray(d, dtype=np.float64)
    if kind == "linear":
        z = _scaled(d)
        beta = _fit_beta_1d(table, z)
        return LinkFit("linear", np.array([beta]), nll_counts(table, beta * z),
                       lambda x, beta=beta: beta * _scaled(x), {"beta": beta})
    if kind == "rank":
        z = rankdata(d) / max(len(d), 1)
        beta = _fit_beta_1d(table, z)
        return LinkFit("rank", np.array([beta]), nll_counts(table, beta * z),
                       lambda x, beta=beta: beta * (rankdata(x) / max(len(x), 1)),
                       {"beta": beta})
    if kind == "spline":
        X = hat_basis(d, knots=knots)
        n = X.shape[1]

        def f(beta):
            b = X @ beta
            u = b[table.ub] - b[table.ua]
            s = np.clip(expit(u), 1e-15, 1.0 - 1e-15)
            nll = -np.sum(table.n_ab * np.log(s) + table.n_ba * np.log(1.0 - s))
            du = -table.n_ab * (1.0 - s) + table.n_ba * s
            g = (np.bincount(table.ua, weights=-du, minlength=table.n_items)
                 + np.bincount(table.ub, weights=du, minlength=table.n_items))
            nll = nll + lam * table.n_instances * float(np.sum(beta * beta))
            return float(nll), X.T @ g + 2.0 * lam * table.n_instances * beta

        r = minimize(f, np.zeros(n), jac=True, method="L-BFGS-B",
                     options=dict(maxiter=200, gtol=1e-6))
        beta = r.x
        return LinkFit("spline", beta, float(r.fun),
                       lambda x, beta=beta, knots=knots: hat_basis(x, knots) @ beta,
                       {"knots": int(X.shape[1]), "ok": bool(r.success)})
    raise ValueError("unknown link kind: " + kind)


# ------------------------------------------------------------------- per-player rho
def _rankcorr(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata(x)
    ry = rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt(float(np.dot(rx, rx)) * float(np.dot(ry, ry)))
    return float(np.dot(rx, ry) / den) if den > 0 else float("nan")


def per_player_spearman(groups: Sequence[np.ndarray], item: np.ndarray,
                        perf: np.ndarray, d: np.ndarray, min_l: int = 3,
                        clusters: np.ndarray | None = None) -> dict:
    """rho_p = Spearman(d|S_p, -perf|S_p), one value per player with l >= min_l."""
    vals, ls, cls, ps = [], [], [], []
    for p, idx in enumerate(groups):
        l = int(len(idx))
        if l < min_l:
            continue
        r = _rankcorr(-d[item[idx]], perf[idx])
        if np.isfinite(r):
            vals.append(r)
            ls.append(l)
            ps.append(p)
            if clusters is not None:
                cls.append(int(clusters[p]))
    if not vals:
        return {"values": np.zeros(0), "ls": np.zeros(0, int),
                "clusters": np.zeros(0, int), "players": np.zeros(0, int),
                "lw": float("nan"), "macro": float("nan"),
                "fisher_l3": float("nan"), "median": float("nan"), "n": 0}
    v = np.asarray(vals, dtype=np.float64)
    l = np.asarray(ls, dtype=np.float64)
    c = np.asarray(cls, dtype=np.int64) if clusters is not None else np.zeros(0, int)
    z = np.arctanh(np.clip(v, -0.999999, 0.999999))
    w = np.maximum(l - 3.0, 1e-9)
    return {"values": v, "ls": l, "clusters": c,
            "players": np.asarray(ps, dtype=np.int64),
            "lw": float(np.sum(v * l) / np.sum(l)),
            "macro": float(np.mean(v)),
            "fisher_l3": float(np.tanh(np.sum(z * w) / np.sum(w))),
            "median": float(np.median(v)), "n": int(v.size)}


# -------------------------------------------------------------------- cross-fit bt
@dataclass
class Ceiling:
    nll: float
    n: int
    coverage: float
    b_full: np.ndarray
    b_folds: list
    rho: dict
    nll_by_cluster: np.ndarray
    n_by_cluster: np.ndarray
    n_clusters: int
    fold_of_cluster: np.ndarray
    seen_folds: list | None = None
    cluster_ids: np.ndarray | None = None
    map_ids: np.ndarray | None = None


def cross_fit_bt(groups: Sequence[np.ndarray], item: np.ndarray, perf: np.ndarray,
                 n_items: int, clusters: np.ndarray, folds: int = 2,
                 seed: int = 0, min_l: int = 3, lam: float = 0.0,
                 fit_groups: Sequence[np.ndarray] | None = None,
                 fit_item: np.ndarray | None = None,
                 fit_perf: np.ndarray | None = None,
                 fit_clusters: np.ndarray | None = None,
                 fit_n_items: int | None = None,
                 fit_item_of_eval: np.ndarray | None = None) -> Ceiling:
    """Cross-fitted Bradley-Terry ceiling, split by player cluster.

    groups/item/perf/clusters describe the rows that are *evaluated* (for
    example the --split test rows).  fit_* describe the rows used to estimate
    the item scores b: every fold-train player contributes all of their rows
    from that table, while fold-test players are only scored.  The defaults
    (all fit_* None) fit and evaluate on the same rows, i.e. the historical
    behaviour.

    Why this split matters: with --split test the historical call fitted b on
    the test rows alone.  The test half has ~1/3 of the comparisons and ~1/2
    of the charts, so a free per-chart b is badly underfit; the resulting
    "ceiling" can be worse than a fixed signal and captured can exceed 1.
    Passing the full play table as fit_* gives the correct reference: b is
    estimated from every row of the fold-train players, but only evaluated on
    the held-out rows of the fold-test players.

    fit_item_of_eval maps evaluator item ids (0..n_items-1) into the fit
    vocabulary (0..fit_n_items-1), so the fit and evaluation tables may have
    different chart vocabularies.
    """
    if folds < 1:
        raise ValueError("folds must be >= 1")
    groups = list(groups)
    fit_groups = list(groups) if fit_groups is None else list(fit_groups)
    item = np.asarray(item)
    perf = np.asarray(perf)
    fit_item = np.asarray(item if fit_item is None else fit_item)
    fit_perf = np.asarray(perf if fit_perf is None else fit_perf)
    clusters = np.asarray(clusters, dtype=np.int64)
    fit_clusters = np.asarray(clusters if fit_clusters is None else fit_clusters,
                              dtype=np.int64)
    n_items = int(n_items)
    fit_n_items = int(n_items if fit_n_items is None else fit_n_items)
    if len(groups) != clusters.size:
        raise ValueError("len(groups) != len(clusters)")
    if len(fit_groups) != fit_clusters.size:
        raise ValueError("len(fit_groups) != len(fit_clusters)")
    if item.size and (item.min() < 0 or item.max() >= n_items):
        raise ValueError("item values out of range for n_items")
    if fit_item.size and (fit_item.min() < 0 or fit_item.max() >= fit_n_items):
        raise ValueError("fit_item values out of range for fit_n_items")
    if fit_item_of_eval is None:
        if fit_n_items < n_items:
            raise ValueError("fit_n_items < n_items requires fit_item_of_eval")
        fit_item_of_eval = np.arange(n_items, dtype=np.int64)
    fit_item_of_eval = np.asarray(fit_item_of_eval, dtype=np.int64)
    if fit_item_of_eval.size != n_items:
        raise ValueError("fit_item_of_eval must have length n_items")
    if fit_item_of_eval.size and (fit_item_of_eval.min() < 0
                                  or fit_item_of_eval.max() >= fit_n_items):
        raise ValueError("fit_item_of_eval out of range for fit_n_items")

    # One fold map over the union of fit and eval clusters.  A cluster must be
    # on the same side of the cross-fit in both tables; assigning folds
    # separately would put the same player in a different fold on each side.
    cluster_ids = np.unique(np.concatenate([clusters, fit_clusters]))
    cluster_to_idx = {int(c): i for i, c in enumerate(cluster_ids)}
    if folds == 1:
        # in-sample ceiling: one fold, all players.  The signal is fixed, but the
        # ceiling is still estimated from the same pairs (optimistic; see docs).
        fold_union = np.zeros(cluster_ids.size, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(cluster_ids.size)
        fold_union = np.empty(cluster_ids.size, dtype=np.int64)
        fold_union[perm] = np.arange(cluster_ids.size) % folds

    eval_cluster_ids = np.unique(clusters)
    n_clusters = int(eval_cluster_ids.size)
    eval_cluster_to_idx = {int(c): i for i, c in enumerate(eval_cluster_ids)}
    eval_cluster_union_idx = np.array(
        [cluster_to_idx[int(c)] for c in eval_cluster_ids], dtype=np.int64)
    eval_cluster_idx = np.array(
        [eval_cluster_to_idx[int(c)] for c in clusters], dtype=np.int64)
    # Folds are looked up in the union map; per-cluster NLL is indexed by the
    # compact eval-cluster id (0..n_clusters-1).
    player_fold = fold_union[np.array(
        [cluster_to_idx[int(c)] for c in clusters], dtype=np.int64)]
    fit_player_fold = fold_union[np.array(
        [cluster_to_idx[int(c)] for c in fit_clusters], dtype=np.int64)]
    fold_of_cluster = fold_union[eval_cluster_union_idx]

    full_fit = build_pair_table(fit_groups, fit_item, fit_perf, fit_n_items)
    b_full_fit, _, _ = fit_bt(full_fit, lam=lam)
    b_full = b_full_fit[fit_item_of_eval]

    nll_sum = 0.0
    n_used = 0
    n_held = 0
    rho_vals, rho_ls, rho_players = [], [], []
    nll_by_cluster = np.zeros(n_clusters)
    n_by_cluster = np.zeros(n_clusters)
    b_folds, seen_folds = [], []
    for f in range(folds):
        if folds == 1:
            # in-sample: fit and evaluate on the same players, but the eval rows
            # may still be a subset of the fit rows.
            b = b_full
            seen_fit = np.zeros(fit_n_items, dtype=bool)
            if full_fit.n_pairs:
                seen_fit[np.unique(np.concatenate([full_fit.ua, full_fit.ub]))] = True
            seen_eval = seen_fit[fit_item_of_eval]
            test_groups = list(range(len(groups)))
        else:
            train_groups = [g for p, g in enumerate(fit_groups)
                            if fit_player_fold[p] != f]
            test_groups = [p for p in range(len(groups)) if player_fold[p] == f]
            if not train_groups or not test_groups:
                b_folds.append(np.zeros(n_items))
                seen_folds.append(np.zeros(n_items, dtype=bool))
                continue
            tr = build_pair_table(train_groups, fit_item, fit_perf, fit_n_items)
            b_fit, _, _ = fit_bt(tr, lam=lam)
            b = b_fit[fit_item_of_eval]
            seen_fit = np.zeros(fit_n_items, dtype=bool)
            if tr.n_pairs:
                seen_fit[np.unique(np.concatenate([tr.ua, tr.ub]))] = True
            seen_eval = seen_fit[fit_item_of_eval]
        b_folds.append(b)
        seen_folds.append(seen_eval)
        for p in test_groups:
            idx = groups[p]
            l = int(len(idx))
            if l < 2:
                continue
            ii, jj = np.triu_indices(l, 1)
            a = item[idx][ii]
            b_ = item[idx][jj]
            ok = seen_eval[a] & seen_eval[b_]
            n_held += int(a.size)
            if not np.any(ok):
                continue
            a, b_ = a[ok], b_[ok]
            pa = perf[idx][ii][ok]
            pb = perf[idx][jj][ok]
            y = (pa > pb).astype(np.float64)
            y = np.where(pa == pb, 0.5, y)
            u = b[b_] - b[a]
            s = np.clip(expit(u), 1e-15, 1 - 1e-15)
            ll = -(y * np.log(s) + (1 - y) * np.log(1 - s))
            nll_sum += float(ll.sum())
            n_used += int(ll.size)
            nll_by_cluster[eval_cluster_idx[p]] += float(ll.sum())
            n_by_cluster[eval_cluster_idx[p]] += float(ll.size)
            if l >= min_l:
                sel = seen_eval[item[idx]]
                if int(sel.sum()) >= min_l:
                    r = _rankcorr(-b[item[idx][sel]], perf[idx][sel])
                    if np.isfinite(r):
                        rho_vals.append(r)
                        rho_ls.append(int(sel.sum()))
                        rho_players.append(int(p))
    rho_vals = np.asarray(rho_vals, dtype=np.float64)
    rho_ls = np.asarray(rho_ls, dtype=np.float64)
    if rho_vals.size:
        z = np.arctanh(np.clip(rho_vals, -0.999999, 0.999999))
        w = np.maximum(rho_ls - 3.0, 1e-9)
        rho = {"lw": float(np.sum(rho_vals * rho_ls) / np.sum(rho_ls)),
               "macro": float(np.mean(rho_vals)),
               "fisher_l3": float(np.tanh(np.sum(z * w) / np.sum(w))),
               "median": float(np.median(rho_vals)), "n": int(rho_vals.size),
               "values": rho_vals, "ls": rho_ls,
               "players": np.asarray(rho_players, dtype=np.int64)}
    else:
        rho = {"lw": float("nan"), "macro": float("nan"),
               "fisher_l3": float("nan"), "median": float("nan"),
               "n": 0, "values": rho_vals, "ls": rho_ls}
    return Ceiling(nll=float(nll_sum / n_used) if n_used else float("nan"),
                   n=int(n_used),
                   coverage=float(n_used / n_held) if n_held else float("nan"),
                   b_full=b_full, b_folds=b_folds, rho=rho,
                   nll_by_cluster=nll_by_cluster, n_by_cluster=n_by_cluster,
                   n_clusters=n_clusters,
                   fold_of_cluster=fold_of_cluster,
                   seen_folds=seen_folds,
                   cluster_ids=eval_cluster_ids)


# --------------------------------------------------------------------- evaluation
def nll_by_cluster(groups: Sequence[np.ndarray], item: np.ndarray,
                   perf: np.ndarray, g: np.ndarray, clusters: np.ndarray,
                   n_clusters: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-cluster pairwise NLL and count, at a fixed item-level score g."""
    nll = np.zeros(n_clusters)
    cnt = np.zeros(n_clusters)
    for p, idx in enumerate(groups):
        l = int(len(idx))
        if l < 2:
            continue
        ii, jj = np.triu_indices(l, 1)
        a = item[idx][ii]
        b = item[idx][jj]
        pa = perf[idx][ii]
        pb = perf[idx][jj]
        y = (pa > pb).astype(np.float64)
        y = np.where(pa == pb, 0.5, y)
        u = g[b] - g[a]
        s = np.clip(expit(u), 1e-15, 1 - 1e-15)
        ll = -(y * np.log(s) + (1 - y) * np.log(1 - s))
        nll[clusters[p]] += float(ll.sum())
        cnt[clusters[p]] += float(ll.size)
    return nll, cnt


def nll_on_ceiling_pairs(groups: Sequence[np.ndarray], item: np.ndarray,
                         perf: np.ndarray, g: np.ndarray, clusters: np.ndarray,
                         ceiling: Ceiling) -> tuple[np.ndarray, np.ndarray]:
    """Signal NLL on exactly the pairs the ceiling actually covers.

    clusters are the compact eval-cluster ids (0..ceiling.n_clusters-1), as
    returned by the runner's _cluster_index.  This is the strict counterpart of
    nll_by_cluster: the denominator of captured is a ceiling on the covered
    pairs, so the numerator should be measured on the same pairs.  Requires
    ceiling.seen_folds (caches written before that field was added do not have
    it).
    """
    if ceiling.seen_folds is None:
        raise ValueError("ceiling has no seen-fold masks; recompute it")
    nll = np.zeros(ceiling.n_clusters)
    cnt = np.zeros(ceiling.n_clusters)
    for p, idx in enumerate(groups):
        l = int(len(idx))
        if l < 2:
            continue
        fold = int(ceiling.fold_of_cluster[clusters[p]])
        seen = ceiling.seen_folds[fold]
        ii, jj = np.triu_indices(l, 1)
        a = item[idx][ii]
        b = item[idx][jj]
        ok = seen[a] & seen[b]
        if not np.any(ok):
            continue
        a, b = a[ok], b[ok]
        pa = perf[idx][ii][ok]
        pb = perf[idx][jj][ok]
        y = (pa > pb).astype(np.float64)
        y = np.where(pa == pb, 0.5, y)
        u = g[b] - g[a]
        s = np.clip(expit(u), 1e-15, 1 - 1e-15)
        ll = -(y * np.log(s) + (1 - y) * np.log(1 - s))
        nll[clusters[p]] += float(ll.sum())
        cnt[clusters[p]] += float(ll.size)
    return nll, cnt


def bootstrap_metric(per_player: dict, sig_nll: np.ndarray, sig_n: np.ndarray,
                     ceil_nll: np.ndarray, ceil_n: np.ndarray,
                     n_clusters: int, n_boot: int = 400,
                     seed: int = 0) -> dict:
    """Cluster bootstrap CIs for NLL, captured and per-player Spearman."""
    rng = np.random.default_rng(seed)
    v = per_player["values"]
    ls = per_player["ls"]
    cls = per_player["clusters"]
    nl, cap, lw, mac = [], [], [], []
    for _ in range(n_boot):
        draw = rng.integers(0, n_clusters, n_clusters)
        mult = np.bincount(draw, minlength=n_clusters).astype(np.float64)
        sn = float((sig_nll * mult).sum())
        snn = float((sig_n * mult).sum())
        cn = float((ceil_nll * mult).sum())
        cnn = float((ceil_n * mult).sum())
        if snn > 0:
            nll = sn / snn
            nl.append(nll)
            if cnn > 0:
                nll_c = cn / cnn
                if nll_c < LOG2:
                    cap.append((LOG2 - nll) / (LOG2 - nll_c))
        if cls.size:
            w = mult[cls]
            if w.sum() > 0:
                lw.append(float(np.sum(v * ls * w) / np.sum(ls * w)))
                mac.append(float(np.sum(v * w) / np.sum(w)))
    out = {}
    for key, arr in (("nll", nl), ("captured", cap), ("lw", lw), ("macro", mac)):
        a = np.asarray(arr, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size:
            out[key] = [float(np.percentile(a, 2.5)),
                        float(np.percentile(a, 97.5))]
    return out


# ------------------------------------------------------------------- training loss
def pairwise_rank_loss(pred, entity, target, beta: float = 1.0,
                       eps: float = 1e-9, ties: str = "skip"):
    """Torch pairwise Bradley-Terry loss on a mini-batch.

    pred   : (B,) predicted loss (lower = better)
    entity : (B,) player index (int)
    target : (B,) observed loss (lower = better)
    beta   : logistic scale (fixed; learn a temperature outside if needed)
    ties   : "skip" (historical behaviour; ties and singleton entities are
             dropped) or "half" (a tied pair has target 0.5, matching
             build_pair_table / the rank-NLL metric).  Use "half" when this
             loss is meant to approximate the rank-NLL objective.

    Pairs are formed within each entity by linking each row to the next row of
    the same entity (cyclic).  Every row contributes at most one pair, so the
    batch has O(B) pairs; use player-grouped batches for a better estimate.
    The loss is invariant to a per-player additive constant and to monotone
    transforms of pred, so keep an MSE term if the model also has to predict
    the absolute loss level.
    """
    import torch

    if ties not in ("skip", "half"):
        raise ValueError("ties must be 'skip' or 'half'")
    if pred.numel() < 2:
        return pred.sum() * 0.0
    order = torch.argsort(entity)
    e = entity[order]
    p = pred[order]
    t = target[order]
    n = e.numel()
    change = torch.ones(n, dtype=torch.bool, device=e.device)
    change[1:] = e[1:] != e[:-1]
    gid = torch.cumsum(change.long(), 0) - 1
    counts = torch.bincount(gid)
    starts = torch.cumsum(counts, 0) - counts
    pos = torch.arange(n, device=e.device) - starts[gid]
    partner = starts[gid] + torch.remainder(pos + 1, counts[gid])
    valid = counts[gid] >= 2
    if ties == "skip":
        valid = valid & (t != t[partner])
    if not bool(valid.any()):
        return pred.sum() * 0.0
    pa, pb = p[valid], p[partner[valid]]
    ta, tb = t[valid], t[partner[valid]]
    y = (ta < tb).to(torch.float32)            # a performed better (lower loss)
    if ties == "half":
        y = torch.where(ta == tb, torch.full_like(y, 0.5), y)
    # BCEWithLogits is stable for large |logits|; sigmoid+clamp+log is not.
    logits = (beta * (pb - pa)).float()
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits, y, reduction="mean")
