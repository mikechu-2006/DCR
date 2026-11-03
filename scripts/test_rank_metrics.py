
#!/usr/bin/env python
"""Synthetic self-checks for dcr/rank_metrics.py (no data/ needed).

Run:  python scripts/test_rank_metrics.py
Prints one PASS/FAIL line per check; exit code 1 if any check fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dcr import rank_metrics as rm  # noqa: E402

CHECKS = []


def check(name: str, cond: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def make_synth(n_items=8, n_players=240, p_play=0.75, noise=0.6, seed=0):
    """Perf = -b_true + logistic noise; each player plays a random subset."""
    rng = np.random.default_rng(seed)
    b_true = np.linspace(-1.5, 1.5, n_items)
    item, perf = [], []
    groups = []
    for _ in range(n_players):
        mask = rng.random(n_items) < p_play
        if mask.sum() < 3:
            mask[rng.choice(n_items, 3, replace=False)] = True
        idx = np.flatnonzero(mask)
        start = len(item)
        item.extend(idx.tolist())
        perf.extend((-b_true[idx] + rng.logistic(0.0, noise, size=idx.size)).tolist())
        groups.append(np.arange(start, start + idx.size))
    return b_true, np.asarray(item, np.int32), np.asarray(perf, np.float64), groups


def main() -> int:
    b_true, item, perf, groups = make_synth()
    n_items = int(item.max()) + 1
    table = rm.build_pair_table(groups, item, perf, n_items)
    check("pair counts sum to instances",
          int(table.n_ab.sum() + table.n_ba.sum()) == table.n_instances,
          f"{table.n_instances} comparisons, {table.n_pairs} unique pairs")
    check("nll at b=0 equals N*log2",
          abs(rm.nll_counts(table, np.zeros(n_items)) / table.n_instances - rm.LOG2) < 1e-9,
          f"{rm.nll_counts(table, np.zeros(n_items))/table.n_instances:.6f}")
    b, nll, _ = rm.fit_bt(table)
    rho = spearmanr(b, b_true).statistic
    check("BT recovers the true item order", rho > 0.9, f"rho={rho:.3f}")
    check("BT NLL <= uniform NLL", nll <= table.n_instances * rm.LOG2 + 1e-9,
          f"{nll/table.n_instances:.6f} <= {rm.LOG2:.6f}")

    link = rm.fit_link(table, b_true, kind="linear")
    check("linear link has beta > 0", link.beta > 0, f"beta={link.beta:.3f}")
    check("linear link NLL < uniform",
          link.nll / table.n_instances < rm.LOG2 - 0.05,
          f"{link.nll/table.n_instances:.6f}")
    ranklink = rm.fit_link(table, b_true, kind="rank")
    check("rank link beta > 0", ranklink.beta > 0, f"beta={ranklink.beta:.3f}")

    pp = rm.per_player_spearman(groups, item, perf, b_true, min_l=3)
    check("per-player Spearman positive on the true order", pp["lw"] > 0.3,
          f"lw={pp['lw']:.3f} n={pp['n']}")
    pp_rev = rm.per_player_spearman(groups, item, perf, -b_true, min_l=3)
    check("reversed order gives negative rho", pp_rev["lw"] < -0.3,
          f"lw={pp_rev['lw']:.3f}")

    clusters = np.arange(len(groups))
    ceil1 = rm.cross_fit_bt(groups, item, perf, n_items, clusters, folds=1,
                            seed=0, min_l=3)
    check("folds=1 ceiling covers every pair", abs(ceil1.coverage - 1.0) < 1e-12,
          f"coverage={ceil1.coverage:.3f}")
    check("folds=1 ceiling NLL <= uniform", ceil1.nll <= rm.LOG2 + 1e-9,
          f"{ceil1.nll:.4f}")
    ceil2 = rm.cross_fit_bt(groups, item, perf, n_items, clusters, folds=2,
                            seed=0, min_l=3)
    check("folds=2 ceiling NLL <= uniform", ceil2.nll <= rm.LOG2 + 1e-9,
          f"{ceil2.nll:.4f} coverage={ceil2.coverage:.3f}")
    check("ceiling stores per-fold seen masks",
          ceil2.seen_folds is not None and len(ceil2.seen_folds) == 2
          and all(np.asarray(s).size == n_items for s in ceil2.seen_folds),
          f"{None if ceil2.seen_folds is None else len(ceil2.seen_folds)} folds")
    st_nll, st_n = rm.nll_on_ceiling_pairs(groups, item, perf, b_true,
                                           clusters, ceil2)
    check("strict pair set matches the ceiling pair count",
          abs(float(st_n.sum()) - float(ceil2.n)) < 1e-9,
          f"{int(st_n.sum())} vs {ceil2.n}")
    check("strict pair NLL is finite",
          np.isfinite(st_nll.sum()) and st_n.sum() > 0,
          f"nll/inst={st_nll.sum() / st_n.sum():.4f}")

    # --- fitting on all rows while evaluating on a subset of players.  This is
    #     the --split test + --ceiling-fit all path: b sees every row of the
    #     fold-train players, but only the subset's rows are scored.
    eval_players = np.arange(0, len(groups), 2)
    eval_groups = [groups[p] for p in eval_players]
    eval_clusters = clusters[eval_players]
    ceil_split = rm.cross_fit_bt(
        eval_groups, item, perf, n_items, eval_clusters, folds=2, seed=0, min_l=3,
        fit_groups=groups, fit_item=item, fit_perf=perf, fit_clusters=clusters)
    check("fit/eval split gives a finite ceiling",
          np.isfinite(ceil_split.nll) and ceil_split.n > 0,
          f"nll={ceil_split.nll:.4f} n={ceil_split.n} "
          f"cov={ceil_split.coverage:.3f}")
    check("fit/eval split keeps the eval cluster count",
          ceil_split.n_clusters == int(np.unique(eval_clusters).size),
          f"{ceil_split.n_clusters} clusters")
    check("fit/eval split b_folds use the eval vocabulary",
          all(np.asarray(b).size == n_items for b in ceil_split.b_folds),
          f"{len(ceil_split.b_folds)} folds")

    # --- a shifted fit vocabulary must give the same answer as the default
    shift = 3
    ceil_shift = rm.cross_fit_bt(
        groups, item, perf, n_items, clusters, folds=2, seed=0, min_l=3,
        fit_groups=groups, fit_item=item + shift, fit_perf=perf,
        fit_clusters=clusters, fit_n_items=n_items + shift,
        fit_item_of_eval=np.arange(n_items) + shift)
    check("fit vocabulary remap matches the default",
          abs(ceil_shift.nll - ceil2.nll) < 1e-12
          and np.allclose(ceil_shift.b_full, ceil2.b_full),
          f"{ceil_shift.nll:.6f} vs {ceil2.nll:.6f}")

    # --- explicit fit=eval arguments must reproduce the historical behaviour
    ceil_same = rm.cross_fit_bt(
        groups, item, perf, n_items, clusters, folds=2, seed=0, min_l=3,
        fit_groups=groups, fit_item=item, fit_perf=perf, fit_clusters=clusters)
    check("explicit fit=eval matches the default",
          abs(ceil_same.nll - ceil2.nll) < 1e-12
          and np.allclose(ceil_same.b_full, ceil2.b_full),
          f"{ceil_same.nll:.6f} vs {ceil2.nll:.6f}")

    import torch
    ent = torch.tensor([0, 0, 0, 1, 1])
    tgt = torch.tensor([1.0, 2.0, 3.0, 3.0, 1.0])
    good = float(rm.pairwise_rank_loss(tgt, ent, tgt, beta=1.0))
    bad = float(rm.pairwise_rank_loss(-tgt, ent, tgt, beta=1.0))
    check("torch rank loss prefers the correct order", good < bad,
          f"good={good:.3f} bad={bad:.3f}")
    shifted = tgt + torch.tensor([10.0, 10.0, 10.0, -5.0, -5.0])
    check("torch rank loss is invariant to a per-player constant",
          abs(float(rm.pairwise_rank_loss(shifted, ent, tgt, beta=1.0)) - good) < 1e-5)
    tie_target = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0])
    tied = float(rm.pairwise_rank_loss(tgt, ent, tie_target, beta=1.0))
    check("torch rank loss skips ties by default", abs(tied) < 1e-9,
          f"{tied:.3g}")
    # A 0.5 target is minimised at q=0.5, so use constant predictions; with
    # non-constant predictions the loss is above log 2 by design.
    tied_half = float(rm.pairwise_rank_loss(torch.zeros(5), ent, tie_target,
                                            beta=1.0, ties="half"))
    check("torch rank loss half-ties equals log 2 at a constant score",
          abs(tied_half - float(np.log(2.0))) < 1e-6, f"{tied_half:.6f}")

    ok = all(v for _, v in CHECKS)
    print(f"\n{sum(v for _, v in CHECKS)}/{len(CHECKS)} checks passed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
