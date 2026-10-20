#!/usr/bin/env python
"""Content-anchored interaction (factorization machine) vs free latent bilinear factors.

Both models predict the same 418,856 test plays on the same split and are trained with the
same per-group-rate Adam, so the only difference is HOW THE INTERACTION IS PARAMETERISED.

  free latents  : interaction = <P_u, D_m>
                  params = K*(n_ent + n_item)   -- ~20k for K=1, ~330k for K=16
  anchored (FM) : interaction = sum_f  <P_u, W[:, f]> * x_mf
                  = player-specific slope on each chart-content feature
                  params = dim*(n_ent + n_feat) + dim*n_feat  -- ~21k, INDEPENDENT of the
                  number of charts.  New charts get a vector for free.

The second form is the direct answer to "our chart vectors are not identified": here the
interaction DIRECTION is supplied by observed content, so nothing has to be learned per chart
and there is no rotation gauge.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROC = Path("data/processed")
import os
_ALL = ["star", "bpm", "max_combo", "count_total", "diff_overall", "diff_drain", "hit_length"]
_keep = os.environ.get("FEATS", "")
FEATS = [f for f in _ALL if (not _keep) or (f in _keep.split(","))]


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
    tps = set(up[np.random.default_rng(20260928).random(len(up)) < 0.30].tolist())
    tr = s1[~s1.pair.isin(tps)]; te = s0[s0.pair.isin(tps)]
    del s0, s1
    gc.collect()
    # (tr / te are consumed after the entity/item vocabularies are built below)
    ent = np.unique(tr.player_id.to_numpy()); itm = np.unique(tr.beatmap_id.to_numpy())
    ep = {v: i for i, v in enumerate(ent)}; ip = {v: i for i, v in enumerate(itm)}
    n_ent, n_item = len(ent), len(itm)

    meta = pd.read_csv(PROC / "beatmap_meta_1k.csv", usecols=["beatmap_id"] + FEATS)
    meta = meta.set_index("beatmap_id").reindex(itm)
    meta = meta.fillna(meta.median(numeric_only=True))
    RAW = meta[FEATS].to_numpy(np.float64)
    F = (RAW - RAW.mean(0)) / (RAW.std(0) + 1e-12)

    def load(df):
        u = df.player_id.map(ep).to_numpy(); m = df.beatmap_id.map(ip).to_numpy()
        ok = pd.notna(u) & pd.notna(m)
        return (torch.as_tensor(u[ok].astype(np.int64)), torch.as_tensor(m[ok].astype(np.int64)),
                df.loss.to_numpy(np.float64)[ok])
    tu, tm, ty = load(te)
    uf, mf, yf = load(tr)
    del tr, te
    gc.collect()
    ymu, ysd = yf.mean(), yf.std()
    Yf = torch.as_tensor((yf - ymu) / ysd); Yt = torch.as_tensor((ty - ymu) / ysd)
    FT = torch.as_tensor(F, dtype=torch.float32)
    print(f"train {len(uf):,} / test {len(tu):,} / charts {n_item:,} / feats {F.shape[1]}",
          flush=True)

    def als_bias():
        mu0, bu0, bm0 = als_init(uf.numpy(), mf.numpy(), (yf - ymu) / ysd, n_ent, n_item, 10.0)
        return (torch.nn.Parameter(torch.tensor(float(mu0))),
                torch.nn.Parameter(torch.as_tensor(bu0)),
                torch.nn.Parameter(torch.as_tensor(bm0)))

    def train(params, kinds, use_bm, EPOCHS=120, BS=16384, dim=16, seed=0):
        torch.manual_seed(seed)
        groups = [{"params": [p for p, kd in zip(params, kinds) if kd in ("bias", "bu", "bm")],
                   "lr": 0.5},
                  {"params": [p for p, kd in zip(params, kinds) if kd not in ("bias", "bu", "bm")],
                   "lr": 0.03}]
        groups = [g for g in groups if g["params"]]
        opt = torch.optim.Adam(groups)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        n = len(uf); rng = np.random.default_rng(0)
        for ep in range(EPOCHS):
            perm = torch.as_tensor(rng.permutation(n))
            for s in range(0, n, BS):
                b = perm[s:s + BS]
                uu, mm = uf[b], mf[b]
                pred = params[0] + params[1][uu]
                if use_bm:
                    pred = pred + params[2][mm]
                off = 3 if use_bm else 2
                for p, kd in zip(params[off:], kinds[off:]):
                    if kd == "P":
                        pred = pred + (p[uu] * params[off + 1][mm]).sum(1)
                    elif kd == "W":
                        z = (p[uu].unsqueeze(1) * params[off + 1].unsqueeze(0)).sum(2)  # (B,F)
                        pred = pred + (z * FT[mm]).sum(1)
                ((pred - Yf[b]) ** 2).mean().backward()
                opt.step(); opt.zero_grad(set_to_none=True)
            sched.step()
        with torch.no_grad():
            def pr(uu, mm):
                pred = params[0] + params[1][uu]
                if use_bm:
                    pred = pred + params[2][mm]
                off = 3 if use_bm else 2
                for p, kd in zip(params[off:], kinds[off:]):
                    if kd == "P":
                        pred = pred + (p[uu] * params[off + 1][mm]).sum(1)
                    elif kd == "W":
                        z = (p[uu].unsqueeze(1) * params[off + 1].unsqueeze(0)).sum(2)
                        pred = pred + (z * FT[mm]).sum(1)
                return pred
            return (float(torch.sqrt(((pr(uf, mf) - Yf) ** 2).mean())),
                    float(torch.sqrt(((pr(tu, tm) - Yt) ** 2).mean())))

    res = {}
    print("\n--- A. 自由低秩交互 <P_u, D_m> ---", flush=True)
    for K in [1, 4, 16]:
        mu, bu, bm = als_bias()
        P = torch.nn.Parameter(torch.randn(n_ent, K) * 0.05)
        D = torch.nn.Parameter(torch.randn(n_item, K) * 0.05)
        tr_r, te_r = train([mu, bu, bm, P, D], ["bias", "bu", "bm", "P", "D"], True)
        npar = K * (n_ent + n_item)
        res[f"free_K{K}"] = dict(train=tr_r, test=te_r, params=npar)
        print(f"  K={K:2d}  params={npar:>8,}  train={tr_r:.4f}  TEST play={te_r:.4f}", flush=True)
        del mu, bu, bm, P, D
        gc.collect()

    print("\n--- B. 内容锚定交互 (FM) ---", flush=True)
    for dim in [4, 16]:
        mu, bu, bm = als_bias()
        W = torch.nn.Parameter(torch.randn(n_ent, dim) * 0.05)
        V = torch.nn.Parameter(torch.randn(F.shape[1], dim) * 0.05)
        tr_r, te_r = train([mu, bu, bm, W, V], ["bias", "bu", "bm", "W", "V"], True, dim=dim)
        npar = dim * (n_ent + F.shape[1])
        res[f"fm_dim{dim}"] = dict(train=tr_r, test=te_r, params=npar)
        print(f"  dim={dim:2d} params={npar:>8,}  train={tr_r:.4f}  TEST play={te_r:.4f}",
              flush=True)
        del mu, bu, bm, W, V
        gc.collect()

    tag = os.environ.get("FEATS", "all").replace(",", "+")
    Path(f"data/processed/diag_step2_fm_1k_{tag}.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote data/processed/diag_step2_fm_1k_{tag}.json")


if __name__ == "__main__":
    main()
