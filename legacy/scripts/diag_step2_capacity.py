#!/usr/bin/env python
"""Capacity sweep at the PLAY level, with a fast enough optimiser that the main effects are
actually fitted.

Why the previous attempt was invalid: Adam takes one step size for every parameter, but a
chart with 1 play and a chart with 2267 plays have Hessian curvatures differing by 3 orders of
magnitude.  Consequently the earlier runs never even matched the additive fit (train RMSE got
*worse* as K grew, which is impossible since K=0 is nested in K>0).  A capacity conclusion
drawn from that would have been worthless.

Fixes here:
  * per-group learning rates: 0.05 for the interaction factors, 0.5 for bu/bm/mu;
  * bu/bm initialised from an exact ridge ALS solution (so training starts at the additive
    optimum and *improves* from there -- a free sanity check on the whole run);
  * 240 epochs with cosine decay, inner-validation checkpoints every 30 epochs;
  * target standardised, weights on the *play* level (the actual prediction task).
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROC = Path("data/processed")


def als_init(u, m, y, n_ent, n_item, lam, iters=40):
    bu = np.zeros(n_ent); bm = np.zeros(n_item); mu = float(y.mean())
    for _ in range(iters):
        bm = np.bincount(m, weights=y - mu - bu[u], minlength=n_item) / (
            np.bincount(m, minlength=n_item) + lam)
        bu = np.bincount(u, weights=y - mu - bm[m], minlength=n_ent) / (
            np.bincount(u, minlength=n_ent) + lam)
    return mu, bu, bm


def main() -> None:
    cols = ["player_id", "beatmap_id", "playcount_cur", "loss"]
    s1 = pd.read_parquet(PROC / "step1_1k.parquet", columns=cols)
    s0 = pd.read_csv(PROC / "step0_1k.csv", usecols=cols)
    k = int(max(s0.beatmap_id.max(), s1.beatmap_id.max())) + 1
    for df in (s0, s1):
        df["pair"] = df.player_id.astype(np.int64) * k + df.beatmap_id.astype(np.int64)
    up = np.unique(s0.pair.to_numpy())
    is_test = np.random.default_rng(20260928).random(len(up)) < 0.30
    tps = set(up[is_test].tolist())
    tr = s1[~s1.pair.isin(tps)]; te = s0[s0.pair.isin(tps)]
    del s0, s1
    gc.collect()

    trp = np.unique(tr.pair.to_numpy())
    iv = np.random.default_rng(999).random(len(trp)) < 0.15
    ivp = set(trp[iv].tolist())
    is_iv = tr.pair.isin(ivp).to_numpy()
    fit_df = tr[~is_iv]; val_df = tr[is_iv]

    ent = np.unique(tr.player_id.to_numpy()); itm = np.unique(tr.beatmap_id.to_numpy())
    ep = {v: i for i, v in enumerate(ent)}; ip = {v: i for i, v in enumerate(itm)}
    n_ent, n_item = len(ent), len(itm)
    del tr
    gc.collect()

    def enc(df):
        return (torch.as_tensor(df.player_id.map(ep).to_numpy(np.int64)),
                torch.as_tensor(df.beatmap_id.map(ip).to_numpy(np.int64)),
                df.loss.to_numpy(np.float64))

    uf, mf, yf = enc(fit_df); uv, mv, yv = enc(val_df)
    ymu, ysd = yf.mean(), yf.std()
    Yf = torch.as_tensor((yf - ymu) / ysd)
    Yv = torch.as_tensor((yv - ymu) / ysd)
    # inner-val cells
    vc = pd.DataFrame({"u": uv.numpy(), "m": mv.numpy(), "y": (yv - ymu) / ysd}).groupby(
        ["u", "m"], sort=False).y.mean()
    vi = torch.as_tensor(vc.index.get_level_values(0).to_numpy())
    vj = torch.as_tensor(vc.index.get_level_values(1).to_numpy())
    vy = torch.as_tensor(vc.to_numpy(np.float64))

    tu = te.player_id.map(ep).to_numpy(); tm = te.beatmap_id.map(ip).to_numpy()
    ok = pd.notna(tu) & pd.notna(tm)
    tu = tu[ok].astype(np.int64); tm = tm[ok].astype(np.int64)
    yte = (te.loss.to_numpy(np.float64)[ok] - ymu) / ysd
    tpair = te.pair.to_numpy()[ok]
    del te
    gc.collect()
    TU = torch.as_tensor(tu); TM = torch.as_tensor(tm); YT = torch.as_tensor(yte)
    tc = pd.DataFrame({"u": tu, "m": tm, "y": yte, "p": tpair}).groupby(
        ["u", "m"], sort=False).y.mean()
    ti = torch.as_tensor(tc.index.get_level_values(0).to_numpy())
    tj = torch.as_tensor(tc.index.get_level_values(1).to_numpy())
    ty = torch.as_tensor(tc.to_numpy(np.float64))
    print(f"fit {len(uf):,} / inner-val {len(uv):,} / test plays {len(TU):,} / "
          f"test cells {len(ty):,}", flush=True)

    import os
    klist = [int(x) for x in os.environ.get("K_LIST", "0,1,2,4,8,16,32,64").split(",")]
    res = {}
    for K in klist:
        torch.manual_seed(0)
        mu0, bu0, bm0 = als_init(uf.numpy(), mf.numpy(), (yf - ymu) / ysd, n_ent, n_item, 10.0)
        mu = torch.nn.Parameter(torch.tensor(float(mu0)))
        bu = torch.nn.Parameter(torch.as_tensor(bu0))
        bm = torch.nn.Parameter(torch.as_tensor(bm0))
        P = torch.nn.Parameter(torch.randn(n_ent, K) * 0.05 if K else torch.zeros(n_ent, 0))
        D = torch.nn.Parameter(torch.randn(n_item, K) * 0.05 if K else torch.zeros(n_item, 0))
        groups = [{"params": [mu, bu, bm], "lr": 0.5}]
        if K:
            groups.append({"params": [P, D], "lr": 0.05})
        opt = torch.optim.Adam(groups)
        EPOCHS, BS = 120, 16384
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        best = (1e9, -1)
        rng = np.random.default_rng(0)
        n = len(uf); bs = BS
        for ep in range(EPOCHS):
            perm = torch.as_tensor(rng.permutation(n))
            for s in range(0, n, bs):
                b = perm[s:s + bs]
                pred = mu + bu[uf[b]] + bm[mf[b]]
                if K:
                    pred = pred + (P[uf[b]] * D[mf[b]]).sum(1)
                ((pred - Yf[b]) ** 2).mean().backward()
                opt.step(); opt.zero_grad(set_to_none=True)
            sched.step()
            if (ep + 1) % 20 == 0:
                with torch.no_grad():
                    pv = mu + bu[vi] + bm[vj]
                    if K:
                        pv = pv + (P[vi] * D[vj]).sum(1)
                    r = float(torch.sqrt(((pv - vy) ** 2).mean()))
                if r < best[0]:
                    best = (r, ep + 1, {kk: v.detach().clone() for kk, v in
                                        dict(mu=mu, bu=bu, bm=bm, P=P, D=D).items()})
        with torch.no_grad():
            st = best[2]
            def pr(i, j):
                out = st["mu"] + st["bu"][i] + st["bm"][j]
                if K:
                    out = out + (st["P"][i] * st["D"][j]).sum(1)
                return out
            tr_rmse = float(torch.sqrt(((pr(uf, mf) - Yf) ** 2).mean()))
            te_cell = float(torch.sqrt(((pr(ti, tj) - ty) ** 2).mean()))
            te_play = float(torch.sqrt(((pr(TU, TM) - YT) ** 2).mean()))
        res[K] = dict(inner_val=best[0], best_epoch=best[1], train=tr_rmse,
                      test_cell=te_cell, test_play=te_play, params=K * (n_ent + n_item))
        print(f"K={K:3d} params={res[K]['params']:>9,}  inner-val={best[0]:.4f}"
              f"(ep{best[1]})  train={tr_rmse:.4f}  TEST cell={te_cell:.4f}  "
              f"TEST play={te_play:.4f}", flush=True)
        del P, D, bu, bm, mu, opt, best
        gc.collect()
    Path("data/processed/diag_step2_capacity_1k.json").write_text(json.dumps(res, indent=2))
    print("wrote data/processed/diag_step2_capacity_1k.json")


if __name__ == "__main__":
    main()
