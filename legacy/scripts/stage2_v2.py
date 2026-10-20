"""Stage-2 (v2): difficulty regression from the learned chart vectors.

The difficulty label is osu!'s per-(beatmap, mod) star rating, so a DT chart is compared against
its DT star.  Also reports the gauge-invariant chart difficulty  s_c = b_m,c + (Pbar - C_c) . D_c,
because with a free per-chart D the vector C_c itself has (dim-1) gauge degrees of freedom.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

INTERIM = Path("data/interim")
PROC = Path("data/processed")
RATE_NAME = {0: "NM", 1: "DT", 2: "HT"}


def chart_star():
    d = pd.read_csv(INTERIM / "beatmap_difficulty_4k.csv")
    d["star"] = pd.to_numeric(d.star, errors="coerce")
    mods = pd.to_numeric(d.mods, errors="coerce").fillna(0).astype("int64")
    d["rate"] = np.where(mods & (64 | 512) != 0, 1, np.where(mods & 256 != 0, 2, 0))
    return d[d.star > 0].sort_values("star").groupby(["beatmap_id", "rate"], as_index=False).star.max()


def evaluate(X, y, name, model="ridge"):
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    est = (make_pipeline(StandardScaler(), Ridge(alpha=1.0)) if model == "ridge"
           else make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=3000,
                                                             random_state=0, early_stopping=True)))
    pred = cross_val_predict(est, X, y, cv=kf)
    rmse = float(np.sqrt(((pred - y) ** 2).mean()))
    r2 = float(1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    rho = float(spearmanr(pred, y).statistic)
    print(f"{name:34s} n_feat={X.shape[1]:3d} rmse={rmse:7.4f} r2={r2:6.3f} spearman={rho:.4f}", flush=True)
    return {"name": name, "n_feat": int(X.shape[1]), "rmse": rmse, "r2": r2, "spearman": rho}


def main(path):
    z = np.load(path, allow_pickle=True)
    chart_ids = z["chart_ids"].astype("int64")
    C, D, bm = z["C"], z["D"], z["bm"]
    P = z["P"]
    beatmap_id, rate = chart_ids // 4, chart_ids % 4
    star = chart_star()
    df = pd.DataFrame({"beatmap_id": beatmap_id, "rate": rate, "bm": bm})
    df = df.merge(star, on=["beatmap_id", "rate"], how="left")
    ok = df.star.notna().values & (df.star.values > 0)
    print(f"{Path(path).name}: charts={len(df)} with per-rate star={ok.sum()} "
          f"({pd.Series(rate[ok]).map(RATE_NAME).value_counts().to_dict()})", flush=True)

    C, D, bm, df = C[ok], D[ok], bm[ok], df[ok].reset_index(drop=True)
    Pbar = P.mean(0)
    s = bm + ((Pbar[None, :] - C) * D).sum(1)
    bmaps = pd.read_csv(INTERIM / "beatmaps.csv").set_index("beatmap_id").reindex(df.beatmap_id).reset_index()
    meta = np.nan_to_num(np.column_stack([
        np.log1p(pd.to_numeric(bmaps.max_combo, errors="coerce").values),
        np.log1p(pd.to_numeric(bmaps.countTotal, errors="coerce").values),
        np.log1p(pd.to_numeric(bmaps.playcount, errors="coerce").values),
        np.log1p(pd.to_numeric(bmaps.passcount, errors="coerce").values),
        pd.to_numeric(bmaps.passcount, errors="coerce").values / np.maximum(pd.to_numeric(bmaps.playcount, errors="coerce").values, 1),
        (bmaps.approved.values >= 1).astype(float),
        (df.rate.values == 1).astype(float), (df.rate.values == 2).astype(float),
    ]), nan=0.0, posinf=0.0, neginf=0.0)
    y = df.star.values
    print(f"star: mean={y.mean():.3f} std={y.std():.3f}")

    res = [evaluate(meta, y, "metadata (baseline)"),
           evaluate(s.reshape(-1, 1), y, "gauge-invariant s = b_m+(Pb-C).D"),
           evaluate(C, y, "C (level vector)"),
           evaluate(D, y, "D (discrimination)"),
           evaluate(np.column_stack([C, D]), y, "C + D"),
           evaluate(np.column_stack([C, meta]), y, "C + metadata"),
           evaluate(np.column_stack([np.column_stack([C, D]), meta]), y, "C + D + metadata"),
           evaluate(C, y, "C (MLP head)", model="mlp")]
    for r, name in RATE_NAME.items():
        m = df.rate.values == r
        if m.sum() > 100:
            res.append(evaluate(np.column_stack([C, D])[m], y[m], f"C+D [{name} only]"))
    out = {"embeddings": str(path), "n_charts": int(ok.sum()), "results": res}
    (PROC / f"stage2v2_{Path(path).stem}.json").write_text(json.dumps(out, indent=2))
    print("saved", PROC / f"stage2v2_{Path(path).stem}.json")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
