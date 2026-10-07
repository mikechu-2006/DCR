#!/usr/bin/env python
"""Capacity sweep on the CHEAP-CHART-FILTERED universe.

Universe (built on step0, as requested):
    keep charts whose number of DISTINCT players in step0 is > 20.
    Charts that fail are removed from BOTH train and eval before the split.

Then the usual step2 protocol: 30% of ordered pairs -> test, train = step1 rows of the other
pairs, test = step0 rows of the held-out pairs.  Everything else identical to
scripts/diag_step2_capacity.py (per-group learning rates, ALS warm start, inner-val early stop).
"""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROC = Path("data/processed")
MIN_PLAYERS = int(os.environ.get("MIN_PLAYERS", "20"))


def als_init(u, m, y, n_ent, n_item, lam, iters=40):
    bu = np.zeros(n_ent); bm = np.zeros(n_item); mu = float(y.mean())
    for _ in range(iters):
        bm = np.bincount(m, weights=y - mu - bu[u], minlength=n_item) / (
            np.bincount(m, minlength=n_item) + lam)
        bu = np.bincount(u, weights=y - mu - bm[m], minlength=n_ent) / (
            np.bincount(u, minlength=n_ent) + lam)
    return mu, bu, bm


def main() -> None:
    cols = ["player_id", "beatmap_id", "loss"]
    s0 = pd.read_csv(PROC / "step0_1k.csv", usecols=cols)
    s1 = pd.read_parquet(PROC / "step1_1k.parquet", columns=cols)
    k = int(max(s0.beatmap_id.max(), s1.beatmap_id.max())) + 1
    for df in (s0, s1):
        df["pair"] = df.player_id.astype(np.int64) * k + df.beatmap_id.astype(np.int64)

    pc = s0.groupby("beatmap_id").player_id.nunique()
    keep = set(pc[pc > MIN_PLAYERS].index.tolist())
    n_all = len(pc)
    s0 = s0[s0.beatmap_id.isin(keep)].copy()
    s1 = s1[s1.beatmap_id.isin(keep)].copy()
    print(f"filter: distinct players > {MIN_PLAYERS}  ->  charts {n_all:,} -> {len(keep):,}",
          flush=True)

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
        return (df.player_id.map(ep).to_numpy(), df.beatmap_id.map(ip).to_numpy(),
                df.loss.to_numpy(np.float64))

    uf_np, mf_np, yf = enc(fit_df)
    uv_np, mv_np, yv = enc(val_df)
    okf = pd.notna(uf_np) & pd.notna(mf_np)
    uf_np = uf_np[okf].astype(int); mf_np = mf_np[okf].astype(int); yf = yf[okf]
    okv = pd.notna(uv_np) & pd.notna(mv_np)
    uv_np = uv_np[okv].astype(int); mv_np = mv_np[okv].astype(int); yv = yv[okv]
    ymu, ysd = yf.mean(), yf.std()
    Yf = torch.as_tensor((yf - ymu) / ysd)
    uf = torch.as_tensor(uf_np); mf = torch.as_tensor(mf_np)
    vc = pd.DataFrame({"u": uv_np, "m": mv_np, "y": (yv - ymu) / ysd}).groupby(
        ["u", "m"], sort=False).y.mean()
    vi = torch.as_tensor(vc.index.get_level_values(0).to_numpy())
    vj = torch.as_tensor(vc.index.get_level_values(1).to_numpy())
    vy = torch.as_tensor(vc.to_numpy(np.float64))

    tu_np = te.player_id.map(ep).to_numpy(); tm_np = te.beatmap_id.map(ip).to_numpy()
    okt = pd.notna(tu_np) & pd.notna(tm_np)
    tu_np = tu_np[okt].astype(int); tm_np = tm_np[okt].astype(int)
    yte = (te.loss.to_numpy(np.float64)[okt] - ymu) / ysd
    tpair = te.pair.to_numpy()[okt]
    del te
    gc.collect()
    TU = torch.as_tensor(tu_np); TM = torch.as_tensor(tm_np); YT = torch.as_tensor(yte)
    tc = pd.DataFrame({"u": tu_np, "m": tm_np, "y": yte, "p": tpair}).groupby(
        ["u", "m"], sort=False).y.mean()
    ti = torch.as_tensor(tc.index.get_level_values(0).to_numpy())
    tj = torch.as_tensor(tc.index.get_level_values(1).to_numpy())
    ty = torch.as_tensor(tc.to_numpy(np.float64))
    print(f"charts {n_item:,} / players {n_ent:,} | fit {len(uf):,} / inner-val {len(uv_np):,} "
          f"/ test plays {len(TU):,} / test cells {len(ty):,}", flush=True)

    npl = pd.Series(mf_np).value_counts()
    print(f"train plays per chart: mean {npl.mean():.1f} median {npl.median():.0f} "
          f"min {npl.min()}", flush=True)

    klist = [int(x) for x in os.environ.get("K_LIST", "0,1,2,3,4,8").split(",")]
    seeds = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
    res = {}
    for K in klist:
      for sd in seeds:
        torch.manual_seed(sd)
        mu0, bu0, bm0 = als_init(uf_np, mf_np, (yf - ymu) / ysd, n_ent, n_item, 10.0)
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
        best = (1e9, -1, None)
        rng = np.random.default_rng(0); n = len(uf)
        for ep in range(EPOCHS):
            perm = torch.as_tensor(rng.permutation(n))
            for s in range(0, n, BS):
                b = perm[s:s + BS]
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
            tr_r = float(torch.sqrt(((pr(uf, mf) - Yf) ** 2).mean()))
            te_c = float(torch.sqrt(((pr(ti, tj) - ty) ** 2).mean()))
            te_p = float(torch.sqrt(((pr(TU, TM) - YT) ** 2).mean()))
        res.setdefault(K, {})[sd] = dict(inner_val=best[0], best_epoch=best[1], train=tr_r,
                                         test_cell=te_c, test_play=te_p,
                                         test_play_loss=te_p * ysd,
                                         params=K * (n_ent + n_item))
        print(f"K={K:3d} sd={sd}  inner={best[0]:.4f}(ep{best[1]})  train={tr_r:.4f}  "
              f"TEST cell={te_c:.4f}  TEST play={te_p:.4f} ({te_p*ysd:.4f} loss)", flush=True)
        del P, D, bu, bm, mu, opt, best
        gc.collect()
    out = PROC / f"diag_step2_capacity_filtered{MIN_PLAYERS}_1k.json"
    out.write_text(json.dumps(res, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
