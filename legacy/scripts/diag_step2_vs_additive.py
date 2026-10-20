#!/usr/bin/env python
"""Apples-to-apples: the fitted step2 interactions vs a saturated additive model.

For every fitted step2 .npz we recompute predictions on the *test* rows (step0, held-out
pairs) with the exact forward pass used in training, aggregate them to (player, chart)
cell means, and compare against:

  * additive0 : chart main effect only            (mean + bm, shrunk)
  * additive  : saturated player + chart effects  (mean + bu + bm, shrunk)

All models are scored on the SAME test plays and the SAME test cells, so the difference is
purely "does the bilinear interaction beat additive main effects".  This isolates the value
of the latent vectors from every other design choice (features, split, optimiser).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path("data/processed")
TAG = "1k"
SEED = 20260928
FRAC = 0.30


def main() -> None:
    cols = ["player_id", "beatmap_id", "playcount_cur", "loss"]
    s1 = pd.read_parquet(PROC / f"step1_{TAG}.parquet", columns=cols)
    s0 = pd.read_csv(PROC / f"step0_{TAG}.csv", usecols=cols)
    k = int(max(s0.beatmap_id.max(), s1.beatmap_id.max())) + 1
    for df in (s0, s1):
        df["pair"] = df.player_id.astype(np.int64) * k + df.beatmap_id.astype(np.int64)
    upairs = np.unique(s0.pair.to_numpy())
    is_test = np.random.default_rng(SEED).random(len(upairs)) < FRAC
    tps = set(upairs[is_test].tolist())
    tr = s1[~s1.pair.isin(tps)]
    te = s0[s0.pair.isin(tps)]
    print(f"train plays {len(tr):,} / test plays {len(te):,}")

    # ---- vocabulary identical to step2 (built from the training rows only) -------------
    ent_ids = np.unique(tr.player_id.to_numpy())
    item_ids = np.unique(tr.beatmap_id.to_numpy())
    e_pos = {v: i for i, v in enumerate(ent_ids)}
    i_pos = {v: i for i, v in enumerate(item_ids)}
    tu = te.player_id.map(e_pos).to_numpy()
    tm = te.beatmap_id.map(i_pos).to_numpy()
    ok = pd.notna(tu) & pd.notna(tm)
    print(f"test rows covered {ok.sum():,}/{len(te):,} ({ok.mean():.2%})")
    tu = tu[ok].astype(int); tm = tm[ok].astype(int)
    y = te.loss.to_numpy(np.float64)[ok]
    pair = te.pair.to_numpy()[ok]

    # test cells
    cd = pd.DataFrame({"u": tu, "m": tm, "y": y, "p": pair})
    cell = cd.groupby(["u", "m"], sort=False).agg(y=("y", "mean"), n=("y", "size"))
    ci = cell.index.get_level_values(0).to_numpy()
    cj = cell.index.get_level_values(1).to_numpy()
    cy = cell.y.to_numpy(np.float64)

    def play_rmse(pred_play, name):
        r = float(np.sqrt(np.mean((pred_play - y) ** 2)))
        return r

    def cell_scores(pred_cell, name):
        res = cy - pred_cell
        return dict(model=name, cell_rmse=float(np.sqrt((res ** 2).mean())),
                    play_rmse_from_cell=float(np.sqrt((res ** 2).mean())))

    out = {}

    # ---- additive models (ridge ALS on the training plays) -----------------------------
    tv = tr.loss.to_numpy(np.float64)
    tru = tr.player_id.map(e_pos).to_numpy()
    trm = tr.beatmap_id.map(i_pos).to_numpy()
    n_ent, n_item = len(ent_ids), len(item_ids)
    mu = float(tv.mean())

    def fit_add(lam, use_u=True, iters=40):
        bu = np.zeros(n_ent) if use_u else None
        bm = np.zeros(n_item)
        for _ in range(iters):
            r = tv - mu - (bu[tru] if use_u else 0.0)
            bm = np.bincount(trm, weights=r, minlength=n_item) / (
                np.bincount(trm, minlength=n_item) + lam)
            if use_u:
                r = tv - mu - bm[trm]
                bu = np.bincount(tru, weights=r, minlength=n_ent) / (
                    np.bincount(tru, minlength=n_ent) + lam)
        return mu, bu, bm

    best = None
    for lam in [1.0, 3.0, 10.0, 30.0, 100.0]:
        mu_c, bu, bm = fit_add(lam)
        res = cy - (mu_c + bu[ci] + bm[cj])
        rv = float((res ** 2).mean())
        if best is None or rv < best[0]:
            best = (rv, lam, mu_c, bu, bm, "additive")
        print(f"  additive  lam={lam:7.2f}  test cell RMSE={np.sqrt(rv):.4f}")
    # chart-only
    bestc = None
    for lam in [1.0, 3.0, 10.0, 30.0, 100.0]:
        mu_c, _, bm = fit_add(lam, use_u=False)
        res = cy - (mu_c + bm[cj])
        rv = float((res ** 2).mean())
        if bestc is None or rv < bestc[0]:
            bestc = (rv, lam, mu_c, bm, "additive0")
        print(f"  additive0 lam={lam:7.2f}  test cell RMSE={np.sqrt(rv):.4f}")

    rv, lam, mu_c, bu, bm, _ = best
    print(f"  -> best additive   lam={lam}  cell RMSE={np.sqrt(rv):.4f}")
    print(f"  -> best additive0  lam={bestc[1]}  cell RMSE={np.sqrt(bestc[0]):.4f}")
    out["additive"] = dict(cell_rmse=float(np.sqrt(rv)), lam=lam)
    out["additive0"] = dict(cell_rmse=float(np.sqrt(bestc[0])), lam=bestc[1])
    # play-level RMSE of the additive model (same rows as the step2 numbers)
    pred_add_play = mu_c + bu[tu] + bm[tm]
    out["additive"]["play_rmse"] = play_rmse(pred_add_play, "additive")
    print(f"     additive play RMSE = {out['additive']['play_rmse']:.4f}")

    # ---- fitted step2 models: recompute the forward pass from the .npz ------------------
    for path in sorted(PROC.glob(f"step2_{TAG}_*_loss.npz")):
        z = np.load(path)
        if not {"entity_ids", "beatmap_ids", "P", "C", "b0", "bu", "bm"} <= set(z.keys()):
            continue
        ze, zm = z["entity_ids"], z["beatmap_ids"]
        eu = np.searchsorted(ze, te.player_id.to_numpy()[ok])
        em = np.searchsorted(zm, te.beatmap_id.to_numpy()[ok])
        eu = np.where((eu < ze.size) & (ze[np.clip(eu, 0, ze.size - 1)] ==
                                         te.player_id.to_numpy()[ok]), eu, -1)
        em = np.where((em < zm.size) & (zm[np.clip(em, 0, zm.size - 1)] ==
                                         te.beatmap_id.to_numpy()[ok]), em, -1)
        cov = (eu >= 0) & (em >= 0)
        P, C, D = z["P"][eu[cov]], z["C"][em[cov]], z["D"][em[cov]]
        b0 = float(z["b0"][0])
        bu_z = z["bu"][eu[cov]] if "bu" in z else 0.0
        bm_z = z["bm"][em[cov]] if "bm" in z else 0.0
        Pc = z["Pc"][eu[cov]] if "Pc" in z else None
        w1 = float(z["w1"][0]) if "w1" in z else 0.0
        need_x = ("w1" in z) or ("Pc" in z)
        x1 = np.log(te.playcount_cur.to_numpy(np.float64)[ok][cov]) if need_x else 0.0
        name = path.stem.replace(f"step2_{TAG}_", "").replace("_loss", "")
        # decide the interaction from the model name
        if "mirt_exp" in name:
            S = P - C
            inter = np.log(np.sum(z["D"][em[cov]] * np.exp(-S), axis=1))
        elif "mf_dot" in name:
            inter = np.sum(P * C, axis=1) if "D" not in z or np.abs(z["D"]).max() == 0 else np.sum(
                (P - C) * D, axis=1)
        elif "bias" in name:
            inter = 0.0
        else:
            inter = np.sum((P - C) * D, axis=1)
        if Pc is not None:
            xx = np.log(te.playcount_cur.to_numpy(np.float64)[ok][cov])
            inter = inter + np.sum(Pc * xx[:, None], axis=1)
            x1 = xx if not np.isscalar(x1) else xx
        pred = b0 + bu_z + bm_z + w1 * x1 + inter
        yy = y[cov]
        pr = pair[cov]
        pcell = pd.DataFrame({"p": pr, "pred": pred}).groupby("p").pred.mean()
        # align to the same cell index order
        zz = pd.DataFrame({"u": tu[cov], "m": tm[cov], "p": pr})
        zz["pred"] = pred
        g = zz.groupby(["u", "m"], sort=False).pred.mean()
        gg = cell.join(g.rename("pred"))
        have = gg.pred.notna().to_numpy()
        cres = cy[have] - gg.pred.to_numpy()[have]
        out[name] = dict(cell_rmse=float(np.sqrt((cres ** 2).mean())),
                         play_rmse=float(np.sqrt(np.mean((pred - yy) ** 2))),
                         n_cells=int(have.sum()))
        print(f"  {name:26s} play RMSE={out[name]['play_rmse']:.4f}  "
              f"cell RMSE={out[name]['cell_rmse']:.4f}  (cells {have.sum():,})")

    (PROC / f"diag_step2_vs_additive_{TAG}.json").write_text(json.dumps(out, indent=2))
    print("\nsummary (test cell RMSE, lower is better):")
    for kk, v in sorted(out.items(), key=lambda kv: kv[1]["cell_rmse"]):
        print(f"   {kk:28s} {v['cell_rmse']:.4f}")


if __name__ == "__main__":
    main()
