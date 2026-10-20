"""How much does the cell-aggregation choice (median vs max vs mean) matter downstream?

Ranks charts by average cell loss under each aggregation and compares the rankings
with each other and with osu!'s per-chart star rating.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROC = Path("data/processed")
INTERIM = Path("data/interim")


def main():
    cells = pd.read_parquet(PROC / "charts_4k.parquet")
    cells = cells[cells.primary]
    pop = cells.groupby("chart_id").user_id.nunique()
    cells = cells[cells.chart_id.isin(pop[pop >= 20].index)]

    per_chart = cells.groupby(["chart_id", "beatmap_id", "rate"]).agg(
        n=("n_scores", "size"),
        med=("loss_v2_med", "mean"), mx=("loss_v2_max", "mean"),
        mean_acc=("acc_v2_med", "mean"),
    ).reset_index()
    ss = cells.groupby("chart_id").n_scores.apply(lambda s: float((s == 1).mean())).rename("single_share")
    per_chart = per_chart.merge(ss, on="chart_id")

    d = pd.read_csv(INTERIM / "beatmap_difficulty_4k.csv")
    d["star"] = pd.to_numeric(d.star, errors="coerce")
    mods = pd.to_numeric(d.mods, errors="coerce").fillna(0).astype("int64")
    d["rate"] = np.where(mods & (64 | 512) != 0, 1, np.where(mods & 256 != 0, 2, 0))
    star = d[d.star > 0].sort_values("star").groupby(["beatmap_id", "rate"], as_index=False).star.max()
    per_chart = per_chart.merge(star, on=["beatmap_id", "rate"], how="left")

    out = {"n_charts": int(len(per_chart))}
    print(f"charts with >=20 players: {len(per_chart)}, median cells/chart: {per_chart.n.median():.0f}")
    print(f"share of single-play cells per chart: mean {per_chart.single_share.mean():.3f}")
    for a, b in [("med", "mx"), ("med", "mean_acc")]:
        rho = spearmanr(per_chart[a], per_chart[b]).statistic
        top = set(per_chart.nlargest(100, a).chart_id) & set(per_chart.nlargest(100, b).chart_id)
        out[f"spearman_{a}_vs_{b}"] = float(rho)
        out[f"top100_overlap_{a}_vs_{b}"] = len(top)
        print(f"spearman({a} vs {b}) = {rho:.4f}   top-100 overlap = {len(top)}/100")
    for col in ["med", "mx"]:
        rho = spearmanr(per_chart[col], per_chart.star).statistic
        out[f"spearman_{col}_vs_star"] = float(rho)
        print(f"spearman({col} vs osu star) = {rho:.4f}")
    (PROC / "aggregation_sensitivity.json").write_text(json.dumps(out, indent=2))
    print("saved", PROC / "aggregation_sensitivity.json")


if __name__ == "__main__":
    main()
