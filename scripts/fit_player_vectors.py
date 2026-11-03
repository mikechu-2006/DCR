#!/usr/bin/env python3
"""Fit one MinaCalc skill vector per player and report loss-space RMSE.

The solver is intentionally per-player: `fit_player_vector` in
`minacalc/player_fit.py` only sees one player's chart indices and losses.
Players can still be evaluated in parallel because they are independent.

Examples
--------
# 100 players sampled from the 10k pairbest table; fit on train, score on test
python scripts/fit_player_vectors.py --dataset 10k --players 100 --seed 20261023

# external player 17598247: 5-fold cross-validation over his played charts
python scripts/fit_player_vectors.py --dataset player --player-id 17598247

Outputs go to `results/player_vector_fit/` by default.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minacalc.data import Charts  # noqa: E402
from minacalc.player_fit import (  # noqa: E402
    DEFAULT_SCORE_BOUNDS,
    fit_player_vector,
    rmse_of,
)
from minacalc.torch_model import Theta, TorchScorer  # noqa: E402

DIMS = ("stream", "js", "hs", "jackspeed", "cj", "tech")
PLAYER_MODS = {"NM", "CL", "PF+CL", "MR+CL", "SD+CL"}


def chart_lookup(charts: Charts) -> dict[int, int]:
    return {int(b): i for i, b in enumerate(charts.beatmap)}


def task_from_frame(sub: pd.DataFrame, player_id, pos, *, split: str,
                    max_rows: int | None = None, seed: int = 0):
    part = sub[sub["split"] == split]
    if max_rows is not None and len(part) > max_rows:
        part = part.sample(int(max_rows), random_state=seed)
    cidx = np.asarray([pos[int(b)] for b in part["beatmap_id"]], dtype=np.int64)
    loss = part["loss"].to_numpy(dtype=np.float32)
    return cidx, loss


def build_10k_tasks(args, charts: Charts):
    """Sample players and materialize one independent task per player."""
    pos = chart_lookup(charts)
    bm = set(pos)
    df = pd.read_parquet(args.plays_10k,
                         columns=["player_id", "beatmap_id", "split", "loss"])
    df = df[df["beatmap_id"].isin(bm)].copy()
    counts = (df.groupby(["player_id", "split"]).size()
                .unstack(fill_value=0))
    cand = counts[(counts.get("train", 0) >= args.min_train)
                  & (counts.get("test", 0) >= args.min_test)].index.to_numpy()
    if len(cand) == 0:
        raise SystemExit("no player satisfies the min train/test requirements")
    rng = np.random.default_rng(args.seed)
    n_take = min(int(args.players), len(cand))
    meta = counts.loc[cand, ["train", "test"]].reset_index()
    meta = meta.rename(columns={"train": "n_train", "test": "n_test"})
    if args.strata > 1 and n_take > 1:
        n_bins = min(int(args.strata), len(meta), n_take)
        meta["stratum"] = pd.qcut(meta["n_train"], q=n_bins,
                                  labels=False, duplicates="drop")
        bins = sorted(meta["stratum"].dropna().unique().tolist())
        base, rem = divmod(n_take, len(bins))
        chosen = []
        for i, b in enumerate(bins):
            want = base + (1 if i < rem else 0)
            pool = meta.loc[meta["stratum"] == b, "player_id"].to_numpy()
            take = min(want, len(pool))
            if take:
                chosen.extend(rng.choice(pool, size=take, replace=False).tolist())
        if len(chosen) < n_take:
            rest = np.setdiff1d(cand, np.asarray(chosen, dtype=np.int64))
            chosen.extend(rng.choice(rest, size=n_take - len(chosen),
                                     replace=False).tolist())
        players = np.asarray(chosen, dtype=np.int64)
        stratum_of = dict(zip(meta["player_id"].to_numpy(), meta["stratum"]))
        print("[10k] strata by train-play quantiles: "
              + ", ".join(f"{int(b)}:{int((meta['stratum'] == b).sum())}"
                          for b in bins), flush=True)
    else:
        players = (rng.choice(cand, size=n_take, replace=False)
                   if n_take < len(cand) else cand)
        stratum_of = {int(p): 0 for p in players}
    tasks = []
    for p in np.sort(players):
        sub = df[df["player_id"] == p]
        tr_cidx, tr_loss = task_from_frame(
            sub, int(p), pos, split="train",
            max_rows=args.max_train, seed=args.seed)
        te_cidx, te_loss = task_from_frame(
            sub, int(p), pos, split="test",
            max_rows=args.max_test, seed=args.seed + 1)
        if len(tr_cidx) < args.min_train or len(te_cidx) < args.min_test:
            continue
        tasks.append({
            "player_id": int(p),
            "stratum": int(stratum_of.get(int(p), -1)),
            "fold": -1,
            "tr_cidx": tr_cidx,
            "tr_loss": tr_loss,
            "te_cidx": te_cidx,
            "te_loss": te_loss,
        })
    print(f"[10k] sampled {len(tasks)} players from {len(cand):,} candidates; "
          f"train rows={sum(len(t['tr_cidx']) for t in tasks):,} "
          f"test rows={sum(len(t['te_cidx']) for t in tasks):,}", flush=True)
    return tasks


def load_external_player(args, charts: Charts):
    """Build observations for player 17598247 from local scores.db/realm dumps."""
    pos = chart_lookup(charts)
    hits = pd.read_csv(args.hits_csv)
    plays = pd.read_csv(args.plays_csv)
    keep = (plays["mode"].eq("mania")
            & plays["mods"].isin(PLAYER_MODS)
            & plays["beatmap_id"].isin(hits["beatmap_id"])
            & plays["accuracy"].notna())
    m = plays[keep].copy()
    m["loss"] = np.log(np.maximum(1.0 - m["accuracy"].to_numpy(dtype=float), 1e-12))
    # One observation per chart, matching step1/pairbest's "best" rule.
    best = (m.sort_values("loss")
              .groupby("beatmap_id", as_index=False)
              .first())
    best = best[(best["loss"] >= args.loss_support[0])
                & (best["loss"] < args.loss_support[1])].copy()
    best["cidx"] = [pos[int(b)] for b in best["beatmap_id"]]
    best = best.sort_values("played_at").reset_index(drop=True)
    print(f"[player {args.player_id}] charts={len(best)} "
          f"loss=[{best.loss.min():.3f}, {best.loss.max():.3f}] "
          f"mods={sorted(m['mods'].unique().tolist())}", flush=True)
    return best


def fit_one(sc, th, task, args):
    t0 = time.time()
    x, info = fit_player_vector(
        sc, task["tr_cidx"], task["tr_loss"], th, sc.device,
        score_bounds=args.score_bounds,
        huber_delta=args.huber,
        lam=args.lam,
        iters=args.iters,
    )
    te_rmse, pred_te = rmse_of(
        sc, x, task["te_cidx"], task["te_loss"], th,
        score_bounds=args.score_bounds)
    base_rmse, base_pred = rmse_of(
        sc, torch.full((6,), 20.0, device=sc.device),
        task["te_cidx"], task["te_loss"], th,
        score_bounds=args.score_bounds)
    y = np.asarray(task["te_loss"], dtype=np.float64)
    p = pred_te.detach().cpu().numpy().astype(np.float64)
    bp = base_pred.detach().cpu().numpy().astype(np.float64)
    rho = float(spearmanr(p, y).statistic) if len(y) > 1 else float("nan")
    rho_base = float(spearmanr(bp, y).statistic) if len(y) > 1 else float("nan")
    return {
        "player_id": task["player_id"],
        "stratum": int(task.get("stratum", -1)),
        "fold": int(task["fold"]),
        "n_train": int(len(task["tr_loss"])),
        "n_test": int(len(task["te_loss"])),
        "rmse_train": float(info["rmse"]),
        "rmse_test": float(te_rmse),
        "spearman_test": rho,
        "baseline_test": float(base_rmse),
        "baseline_spearman": rho_base,
        "sse_test": float(np.sum((p - y) ** 2)),
        "baseline_sse": float(np.sum((bp - y) ** 2)),
        "x": x.detach().cpu().numpy().astype(float).tolist(),
        "active": info["active"],
        "pred_test": p.tolist(),
        "baseline_pred_test": bp.tolist(),
        "test_loss": y.tolist(),
        "seconds": time.time() - t0,
    }


# Forked workers inherit these globals, so the 500 MB chart tensor is shared
# copy-on-write instead of being loaded once per worker.
_POOL_SC = None
_POOL_TH = None
_POOL_ARGS = None


def _fit_one_global(task):
    return fit_one(_POOL_SC, _POOL_TH, task, _POOL_ARGS)


def _run_sequential(sc, th, tasks, args):
    results = []
    for i, task in enumerate(tasks, 1):
        results.append(fit_one(sc, th, task, args))
        if i % 10 == 0 or i == len(tasks):
            print(f"  fitted {i}/{len(tasks)}", flush=True)
    return results


def run_tasks(sc, th, tasks, args):
    global _POOL_SC, _POOL_TH, _POOL_ARGS
    if args.workers <= 1:
        return _run_sequential(sc, th, tasks, args)

    # Python-level work inside the torch forward (the stamina scan and the
    # aggregate solver) holds the GIL, so ThreadPoolExecutor stays ~1 core.
    # Forked processes each inherit the read-only chart arrays via COW.
    # Some sandboxes deny POSIX semaphores; fall back to the in-process loop.
    _POOL_SC, _POOL_TH, _POOL_ARGS = sc, th, args
    try:
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            futs = {ex.submit(_fit_one_global, t): t for t in tasks}
            results = []
            done = 0
            for fut in as_completed(futs):
                results.append(fut.result())
                done += 1
                if done % 10 == 0 or done == len(tasks):
                    print(f"  fitted {done}/{len(tasks)}", flush=True)
        return results
    except (PermissionError, OSError) as exc:
        print(f"[parallel] process pool unavailable ({exc}); "
              f"falling back to --workers 1", flush=True)
        return _run_sequential(sc, th, tasks, args)


def summarize(results):
    n = sum(r["n_test"] for r in results)
    sse = sum(r["sse_test"] for r in results)
    base_sse = sum(r["baseline_sse"] for r in results)
    rmses = np.asarray([r["rmse_test"] for r in results], dtype=float)
    train = np.asarray([r["rmse_train"] for r in results], dtype=float)
    rhos = np.asarray([r["spearman_test"] for r in results], dtype=float)
    base_rhos = np.asarray([r["baseline_spearman"] for r in results], dtype=float)
    preds = np.concatenate([np.asarray(r["pred_test"]) for r in results])
    ys = np.concatenate([np.asarray(r["test_loss"]) for r in results])
    base_preds = np.concatenate([np.asarray(r["baseline_pred_test"]) for r in results])
    pooled_rho = (float(spearmanr(preds, ys).statistic)
                  if len(ys) > 1 else float("nan"))
    base_pooled_rho = (float(spearmanr(base_preds, ys).statistic)
                       if len(ys) > 1 else float("nan"))
    return {
        "n_players": len(results),
        "n_test": int(n),
        "pooled_rmse": float(math.sqrt(sse / max(n, 1))),
        "pooled_spearman": pooled_rho,
        "mean_player_rmse": float(rmses.mean()),
        "median_player_rmse": float(np.median(rmses)),
        "min_player_rmse": float(rmses.min()),
        "max_player_rmse": float(rmses.max()),
        "mean_player_spearman": float(np.nanmean(rhos)),
        "median_player_spearman": float(np.nanmedian(rhos)),
        "mean_train_rmse": float(train.mean()),
        "baseline_pooled_rmse": float(math.sqrt(base_sse / max(n, 1))),
        "baseline_pooled_spearman": base_pooled_rho,
        "baseline_mean_player_spearman": float(np.nanmean(base_rhos)),
    }


def write_10k(args, results, out_dir: Path):
    summary = summarize(results)
    rows = []
    for r in results:
        row = {"player_id": r["player_id"],
               "stratum": r.get("stratum", -1),
               "n_train": r["n_train"], "n_test": r["n_test"],
               "rmse_train": r["rmse_train"],
               "rmse_test": r["rmse_test"],
               "spearman_test": r["spearman_test"],
               "baseline_test": r["baseline_test"],
               "baseline_spearman": r["baseline_spearman"],
               "seconds": r["seconds"]}
        row.update({f"x_{k}": v for k, v in zip(DIMS, r["x"])})
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "10k_players.csv", index=False,
                              encoding="utf-8")
    np.save(out_dir / "10k_X.npy",
            np.asarray([r["x"] for r in results], dtype=np.float32))
    report = {
        "dataset": "step1_10k_pairbest",
        "plays": args.plays_10k,
        "charts": args.charts,
        "seed": args.seed,
        "requested_players": args.players,
        "max_train": args.max_train,
        "max_test": args.max_test,
        "strata": args.strata,
        "iters": args.iters,
        "lam": args.lam,
        "huber": args.huber,
        "score_bounds": None if args.score_bounds is None else list(args.score_bounds),
        "summary": summary,
        "players": results,
    }
    (out_dir / "10k_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[10k] " + json.dumps(summary, ensure_ascii=False), flush=True)


def run_10k(args, charts, sc, th):
    tasks = build_10k_tasks(args, charts)
    results = run_tasks(sc, th, tasks, args)
    out_dir = Path(args.out) / "10k"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_10k(args, results, out_dir)
    return results


def run_player(args, charts, sc, th):
    best = load_external_player(args, charts)
    n = len(best)
    if n < args.folds * 2:
        raise SystemExit(f"not enough observations for {args.folds} folds: {n}")
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n)
    folds = np.empty(n, dtype=np.int64)
    for k in range(args.folds):
        folds[order[k::args.folds]] = k

    cidx = best["cidx"].to_numpy(dtype=np.int64)
    loss = best["loss"].to_numpy(dtype=np.float32)
    tasks = []
    for k in range(args.folds):
        te = folds == k
        tr = ~te
        tasks.append({
            "player_id": int(args.player_id),
            "fold": int(k),
            "tr_cidx": cidx[tr],
            "tr_loss": loss[tr],
            "te_cidx": cidx[te],
            "te_loss": loss[te],
        })
    print(f"[player {args.player_id}] {args.folds}-fold CV: "
          f"{n} charts, train~{n - n // args.folds}, test~{n // args.folds}",
          flush=True)
    fold_results = run_tasks(sc, th, tasks, args)

    # Out-of-fold pooled RMSE.
    preds = np.concatenate([np.asarray(r["pred_test"]) for r in fold_results])
    base_preds = np.concatenate([
        np.asarray(r["baseline_pred_test"]) for r in fold_results])
    ys = np.concatenate([np.asarray(r["test_loss"]) for r in fold_results])
    oof = {
        "n": int(len(ys)),
        "pooled_rmse": float(np.sqrt(np.mean((preds - ys) ** 2))),
        "pooled_spearman": float(spearmanr(preds, ys).statistic),
        "mean_fold_rmse": float(np.mean([r["rmse_test"] for r in fold_results])),
        "median_fold_rmse": float(np.median([r["rmse_test"] for r in fold_results])),
        "mean_fold_spearman": float(np.nanmean(
            [r["spearman_test"] for r in fold_results])),
        "median_fold_spearman": float(np.nanmedian(
            [r["spearman_test"] for r in fold_results])),
        "mean_train_rmse": float(np.mean([r["rmse_train"] for r in fold_results])),
        "baseline_pooled_rmse": float(np.sqrt(np.mean((base_preds - ys) ** 2))),
        "baseline_pooled_spearman": float(spearmanr(base_preds, ys).statistic),
    }

    # Final vector on all observations.
    t0 = time.time()
    x_full, info_full = fit_player_vector(
        sc, cidx, loss, th, sc.device,
        score_bounds=args.score_bounds, huber_delta=args.huber,
        lam=args.lam, iters=args.iters)
    full_rmse, _ = rmse_of(sc, x_full, cidx, loss, th,
                           score_bounds=args.score_bounds)
    full = {
        "n": int(n),
        "rmse_train": float(full_rmse),
        "x": x_full.detach().cpu().numpy().astype(float).tolist(),
        "active": info_full["active"],
        "seconds": time.time() - t0,
    }

    out_dir = Path(args.out) / f"player_{args.player_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in fold_results:
        row = {"fold": r["fold"], "n_train": r["n_train"],
               "n_test": r["n_test"], "rmse_train": r["rmse_train"],
               "rmse_test": r["rmse_test"], "spearman_test": r["spearman_test"]}
        row.update({f"x_{k}": v for k, v in zip(DIMS, r["x"])})
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "folds.csv", index=False,
                              encoding="utf-8")
    np.save(out_dir / "x_full.npy",
            x_full.detach().cpu().numpy().astype(np.float32))
    report = {
        "dataset": "player",
        "player_id": int(args.player_id),
        "plays_csv": args.plays_csv,
        "hits_csv": args.hits_csv,
        "charts": args.charts,
        "seed": args.seed,
        "folds": args.folds,
        "iters": args.iters,
        "lam": args.lam,
        "huber": args.huber,
        "score_bounds": None if args.score_bounds is None else list(args.score_bounds),
        "observations": int(n),
        "cv": oof,
        "full_fit": full,
        "fold_details": fold_results,
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[player {args.player_id}] CV pooled RMSE={oof['pooled_rmse']:.4f} "
          f"Spearman={oof['pooled_spearman']:.4f} "
          f"(n={oof['n']}); full-fit train RMSE={full['rmse_train']:.4f}; "
          f"x={np.round(full['x'], 3).tolist()}", flush=True)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=("10k", "player"), required=True)
    ap.add_argument("--charts", default=str(ROOT / "data/processed/charts_1k.npz"))
    ap.add_argument("--plays-10k",
                    default=str(ROOT / "data/processed/step1_10k_pairbest.parquet"))
    ap.add_argument("--plays-csv",
                    default=str(ROOT / "data/processed/player_17598247/plays.csv"))
    ap.add_argument("--hits-csv",
                    default=str(ROOT / "data/processed/player_17598247/whitelist_hits.csv"))
    ap.add_argument("--player-id", type=int, default=17598247)
    ap.add_argument("--players", type=int, default=100)
    ap.add_argument("--strata", type=int, default=5,
                    help="quantile strata by train-play count; 1 = simple random")
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--min-train", type=int, default=30)
    ap.add_argument("--min-test", type=int, default=10)
    ap.add_argument("--max-train", type=int, default=60)
    ap.add_argument("--max-test", type=int, default=120)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--huber", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=1,
                    help="independent player processes (fork); 1 = in-process")
    ap.add_argument("--threads", type=int, default=4,
                    help="torch intra-op threads for the in-process solver")
    ap.add_argument("--score-lo", type=float, default=DEFAULT_SCORE_BOUNDS[0],
                    help="lower clip bound on minacalc score/ACC (default 0.90)")
    ap.add_argument("--score-hi", type=float, default=DEFAULT_SCORE_BOUNDS[1],
                    help="upper clip bound on minacalc score/ACC (default 0.998)")
    ap.add_argument("--no-clip", action="store_true",
                    help="disable the [0.90, 0.998] score clip")
    ap.add_argument("--out", default=str(ROOT / "results/player_vector_fit"))
    args = ap.parse_args(argv)

    args.score_bounds = (None if args.no_clip
                         else (float(args.score_lo), float(args.score_hi)))
    if args.score_bounds is not None and args.score_lo >= args.score_hi:
        raise SystemExit("--score-lo must be smaller than --score-hi")
    # Observation support used for the external player's best-per-chart filter.
    args.loss_support = (math.log(1.0 - float(args.score_hi)),
                         math.log(1.0 - float(args.score_lo)))

    torch.set_num_threads(max(1, int(args.threads)))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    charts = Charts(args.charts)
    sc = TorchScorer(charts, device=dev)
    th = Theta().to(dev)
    print(f"[setup] charts={charts.n_charts:,} device={dev} "
          f"load={time.time() - t0:.1f}s", flush=True)

    if args.dataset == "10k":
        results = run_10k(args, charts, sc, th)
        print("[done] " + json.dumps(summarize(results), ensure_ascii=False),
              flush=True)
    else:
        run_player(args, charts, sc, th)
    print(f"[done] {time.time() - t0:.1f}s -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
