"""Materialise the top-n x top-n (player, chart) matrix with log-loss labels.

Chart = (beatmap_id, rate bucket); a DT chart counts as a different chart than its NM version.

Usage: python scripts/make_matrix_n.py [--n 1000] [--cells data/processed/charts_4k.parquet] [--tag chart]
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

INTERIM = Path("data/interim")
PROC = Path("data/processed")
RATE_NAME = {0: "NM", 1: "DT", 2: "HT"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--cells", default="data/processed/charts_4k.parquet")
    ap.add_argument("--tag", default="chart")
    ap.add_argument("--no-primary-filter", action="store_true")
    args = ap.parse_args()

    stats = pd.read_csv(INTERIM / "user_stats_mania.csv")
    stats["rank_score_index"] = pd.to_numeric(stats.rank_score_index, errors="coerce")
    players = stats.sort_values("rank_score_index").head(args.n).user_id.tolist()

    cells = pd.read_parquet(args.cells)
    if not args.no_primary_filter and "primary" in cells.columns:
        cells = cells[cells.primary]
    cells = cells[cells.user_id.isin(players)]
    pop = cells.groupby("chart_id").user_id.nunique().sort_values(ascending=False)
    charts = pop.head(args.n).index
    mat = cells[cells.chart_id.isin(charts)].copy()
    mat = mat.merge(pd.read_csv(INTERIM / "beatmap_difficulty_4k.csv").pipe(
        lambda d: (d.assign(star_mod=pd.to_numeric(d.star, errors="coerce"),
                            rate=np.where(pd.to_numeric(d.mods, errors="coerce").fillna(0).astype("int64") & (64 | 512) != 0, 1,
                                          np.where(pd.to_numeric(d.mods, errors="coerce").fillna(0).astype("int64") & 256 != 0, 2, 0)))
                     .query("star_mod > 0").sort_values("star_mod").groupby(["beatmap_id", "rate"], as_index=False).star_mod.max()
                     [["beatmap_id", "rate", "star_mod"]])),
        on=["beatmap_id", "rate"], how="left")

    report = {
        "n": args.n, "cells_file": args.cells, "tag": args.tag,
        "n_players_requested": len(players), "n_players_present": int(mat.user_id.nunique()),
        "n_charts": int(mat.chart_id.nunique()),
        "rate_mix": mat.rate.value_counts().to_dict(),
        "cells": int(len(mat)), "density": round(float(len(mat) / (len(players) * len(charts))), 4),
        "plays_per_cell": {str(k): int(v) for k, v in mat.n_scores.value_counts().sort_index().head(6).items()},
        "single_play_cells": int((mat.n_scores == 1).sum()),
        "median_cell_acc_v2": float(mat.acc_v2_med.median()),
        "celldiff": {c: {k: round(float(v), 4) for k, v in mat[c].describe()[["mean", "std", "min", "50%", "max"]].items()}
                     for c in ["loss_v2_med", "loss_v2_max", "loss_flat_med"]},
    }
    dec = pd.qcut(mat.star_mod, 10, labels=False, duplicates="drop")
    rep = mat.groupby(dec).agg(star=("star_mod", "mean"), loss_med=("loss_v2_med", "mean"),
                               acc_med=("acc_v2_med", "mean"), n=("star_mod", "size"))
    report["loss_by_star_decile"] = rep.round(4).to_dict(orient="index")

    out = PROC / f"matrix_n{args.n}_{args.tag}.parquet"
    mat.to_parquet(out, index=False)
    (PROC / f"matrix_n{args.n}_{args.tag}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "loss_by_star_decile"}, indent=2))
    print("\nby difficulty-star decile:")
    print(rep.round(3).to_string())
    print("\nwrote", out)


if __name__ == "__main__":
    main()
