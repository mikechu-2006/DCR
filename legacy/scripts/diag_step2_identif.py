#!/usr/bin/env python
"""Is the seed-to-seed disagreement non-convergence, or genuine non-identifiability?

The previous run found corr(D_seed1, D_seed2) ~ 0 while the test RMSE stayed ~2.13 -- but the
same-seed run found 0.9993.  Two competing explanations:
   (i)  the optimiser simply has not converged in 30 passes (float32 + Adam needs more), so the
        parameters are still near their random starting point;
   (ii) the data genuinely does not pin down the chart vectors.

Test: run many more passes and a stronger ridge.  If (i), seed-to-seed agreement should climb
towards 1 as iterations grow.  If (ii), it stays near 0 no matter how long we optimise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from diag_step2_feasible import rank_k_fit, predict

PROC = Path("data/processed")


def main() -> None:
    cols = ["player_id", "beatmap_id", "loss"]
    s1 = pd.read_parquet(PROC / "step1_1k.parquet", columns=cols)
    s0 = pd.read_csv(PROC / "step0_1k.csv", usecols=cols)
    k = int(max(s0.beatmap_id.max(), s1.beatmap_id.max())) + 1
    for df in (s0, s1):
        df["pair"] = df.player_id.astype(np.int64) * k + df.beatmap_id.astype(np.int64)
    up = np.unique(s0.pair.to_numpy())
    is_test = np.random.default_rng(20260928).random(len(up)) < 0.30
    tps = set(up[is_test].tolist())
    tr = s1[~s1.pair.isin(tps)]
    te = s0[s0.pair.isin(tps)]
    ent = np.unique(tr.player_id.to_numpy()); itm = np.unique(tr.beatmap_id.to_numpy())
    ep = {v: i for i, v in enumerate(ent)}; ip = {v: i for i, v in enumerate(itm)}
    n_ent, n_item = len(ent), len(itm)
    tu = te.player_id.map(ep).to_numpy(); tm = te.beatmap_id.map(ip).to_numpy()
    ok = pd.notna(tu) & pd.notna(tm)
    tu = tu[ok].astype(int); tm = tm[ok].astype(int)
    y = te.loss.to_numpy(np.float64)[ok]
    u = tr.player_id.map(ep).to_numpy().astype(np.int64)
    m = tr.beatmap_id.map(ip).to_numpy().astype(np.int64)
    yt = tr.loss.to_numpy(np.float64)

    K = 4
    for iters, lam in [(30, 1e-3), (200, 1e-3), (600, 1e-3), (600, 1e-1)]:
        fs = [rank_k_fit(u, m, yt, n_ent, n_item, K, lam, iters=iters, seed=s)
              for s in (0, 1, 2)]
        pv = [predict(f, tu, tm) for f in fs]
        rmse = [float(np.sqrt(np.mean((p - y) ** 2))) for p in pv]
        cors = [float(np.corrcoef(pv[0], pv[i])[0, 1]) for i in (1, 2)]
        # residual scale of the fit on TRAIN
        ptr = [predict(f, u, m) for f in fs]
        tr_rmse = [float(np.sqrt(np.mean((p - yt) ** 2))) for p in ptr]
        # D agreement, unaligned
        D0 = fs[0]["D"]
        dcor = [float(np.corrcoef(D0.ravel(), f["D"].ravel())[0, 1]) for f in fs[1:]]
        print(f"iters={iters:4d} lam={lam:g}: test RMSE {np.round(rmse,4)}  "
              f"train RMSE {np.round(tr_rmse,4)}  corr(pred, seed0) {np.round(cors,4)}  "
              f"corr(D, seed0) {np.round(dcor,4)}", flush=True)


if __name__ == "__main__":
    main()
