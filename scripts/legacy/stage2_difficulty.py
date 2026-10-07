"""Stage-2: use the learned chart vectors as features for difficulty regression.

Difficulty label: osu!'s own per-mod star rating (osu_beatmap_difficulty), so DT charts are
compared against their DT star rating rather than the NM one.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

INTERIM = Path("data/interim")
PROC = Path("data/processed")
RATE_NAME = {0: "NM", 1: "DT", 2: "HT"}


def chart_star() -> pd.DataFrame:
    """star rating per (beatmap_id, rate bucket) from osu_beatmap_difficulty."""
    d = pd.read_csv(INTERIM / "beatmap_difficulty_4k.csv")
    d["star"] = pd.to_numeric(d.star, errors="coerce")
    mods = pd.to_numeric(d.mods, errors="coerce").fillna(0).astype("int64")
    d["rate"] = np.where(mods & (64 | 512) != 0, 1, np.where(mods & 256 != 0, 2, 0))
    d = d[d.star > 0]
    out = d.sort_values("star").groupby(["beatmap_id", "rate"], as_index=False).star.max()
    return out


def evaluate(X, y, name, model="ridge", folds=5):
    kf = KFold(n_splits=folds, shuffle=True, random_state=0)
    if model == "ridge":
        est = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    else:
        est = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=3000,
                                                           random_state=0, early_stopping=True))
    pred = cross_val_predict(est, X, y, cv=kf)
    rmse = float(np.sqrt(((pred - y) ** 2).mean()))
    r2 = float(1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    rho = float(spearmanr(pred, y).statistic)
    print(f"{name:30s} n_feat={X.shape[1]:3d} rmse={rmse:7.4f} r2={r2:6.3f} spearman={rho:.4f}", flush=True)
    return {"name": name, "n_feat": int(X.shape[1]), "rmse": rmse, "r2": r2, "spearman": rho}


def main(emb_path: str):
    emb = np.load(emb_path, allow_pickle=True)
    C, D, bm = emb["C"], emb["D"], emb["bm"]
    beatmap_ids = emb["beatmap_ids"].astype("int64")
    rates = emb["rates"].astype("int64")

    bmaps = pd.read_csv(INTERIM / "beatmaps.csv").set_index("beatmap_id")
    base = bmaps.reindex(beatmap_ids).reset_index()
    charts = pd.DataFrame({"beatmap_id": beatmap_ids, "rate": rates})
    cs = chart_star()
    charts = charts.merge(cs, on=["beatmap_id", "rate"], how="left")
    star = charts.star.values
    ok = np.isfinite(star) & (star > 0)
    print(f"embeddings {emb_path}: {len(beatmap_ids)} charts; with per-rate star: {ok.sum()} "
          f"({pd.Series(rates[ok]).map(RATE_NAME).value_counts().to_dict()})")
    print(f"star: mean={star[ok].mean():.3f} std={star[ok].std():.3f} range=({star[ok].min():.3f},{star[ok].max():.3f})")

    C, D, bm, rate = C[ok], D[ok], bm[ok], rates[ok]
    b = base[ok].reset_index(drop=True)
    star = star[ok]
    meta = np.nan_to_num(np.column_stack([
        np.log1p(pd.to_numeric(b.max_combo, errors="coerce").values),
        np.log1p(pd.to_numeric(b.countTotal, errors="coerce").values),
        np.log1p(pd.to_numeric(b.playcount, errors="coerce").values),
        np.log1p(pd.to_numeric(b.passcount, errors="coerce").values),
        pd.to_numeric(b.passcount, errors="coerce").values / np.maximum(pd.to_numeric(b.playcount, errors="coerce").values, 1),
        (b.approved.values >= 1).astype(float),
        (rate == 1).astype(float), (rate == 2).astype(float),
    ]), nan=0.0, posinf=0.0, neginf=0.0)

    res = []
    res.append(evaluate(meta, star, "map metadata (baseline)"))
    res.append(evaluate(bm.reshape(-1, 1), star, "chart bias b_m (1 dim)"))
    res.append(evaluate(C, star, "C difficulty vector"))
    res.append(evaluate(D, star, "D discrimination vector"))
    res.append(evaluate(np.column_stack([C, D]), star, "C + D"))
    res.append(evaluate(np.column_stack([C, meta]), star, "C + metadata"))
    res.append(evaluate(C, star, "C (MLP head)", model="mlp"))

    # per-rate breakdown for C + D
    for r, name in RATE_NAME.items():
        m = rate == r
        if m.sum() > 50:
            res.append(evaluate(np.column_stack([C, D])[m], star[m], f"C + D [{name} only]"))

    out = {"embeddings": emb_path, "n_charts": int(ok.sum()), "results": res}
    p = PROC / f"stage2_{Path(emb_path).stem}.json"
    p.write_text(json.dumps(out, indent=2))
    print("saved", p)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(PROC / "stage1_mirt_loss_v2_med.npz"))
