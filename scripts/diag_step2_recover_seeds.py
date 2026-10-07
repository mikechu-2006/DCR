#!/usr/bin/env python
"""Same recoverability test, but with DIFFERENT random inits (the previous run used seed=0
for both halves, so 'no alignment needed' could have been an artefact of shared init).

Reported for each seed pair:
  - prediction agreement on identical test cells
  - raw (unaligned) agreement of the chart vectors
  - Procrustes-aligned agreement (the best any rotation could do)
  - rotation-invariant agreement: ||D_m|| and pairwise cos(D_m, D_m')
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from diag_step2_feasible import rank_k_fit, predict

PROC = Path("data/processed")
K, LAM = 4, 1e-3


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
    td = pd.DataFrame({"u": tu, "m": tm, "y": y})
    tc = td.groupby(["u", "m"], sort=False).y.mean()
    ci = tc.index.get_level_values(0).to_numpy()
    cj = tc.index.get_level_values(1).to_numpy()
    cy = tc.to_numpy(np.float64)

    u = tr.player_id.map(ep).to_numpy().astype(np.int64)
    m = tr.beatmap_id.map(ip).to_numpy().astype(np.int64)
    yt = tr.loss.to_numpy(np.float64)

    fits = {s: rank_k_fit(u, m, yt, n_ent, n_item, K, LAM, iters=30, seed=s)
            for s in (0, 1, 2, 3, 4)}
    ref = fits[0]
    idx = np.random.default_rng(0).choice(n_item, 1500, replace=False)
    nref = np.linalg.norm(ref["D"], axis=1)
    Cref = ref["D"][idx] @ ref["D"][idx].T / (np.outer(nref[idx], nref[idx]) + 1e-12)
    iu = np.triu_indices(len(idx), 1)

    print(f"{'seed':>4} {'pred corr':>10} {'pred RMSd':>10} | {'raw corr':>9} {'raw relF':>9} "
          f"| {'proc corr':>9} {'relF':>7} | {'||D|| corr':>10} {'cos corr':>9}")
    for s, f in fits.items():
        if s == 0:
            continue
        p0 = predict(ref, ci, cj); p1 = predict(f, ci, cj)
        pc = float(np.corrcoef(p0, p1)[0, 1]); pr = float(np.sqrt(np.mean((p0 - p1) ** 2)))
        D0, D1 = ref["D"], f["D"]
        raw_c = float(np.corrcoef(D0.ravel(), D1.ravel())[0, 1])
        raw_f = float(np.linalg.norm(D0 - D1) / np.linalg.norm(D0))
        U, _, Vt = np.linalg.svd(D0.T @ D1)
        R = U @ Vt
        D1a = D1 @ R
        pr_c = float(np.corrcoef(D0.ravel(), D1a.ravel())[0, 1])
        pr_f = float(np.linalg.norm(D0 - D1a) / np.linalg.norm(D0))
        n1 = np.linalg.norm(D1, axis=1)
        n_c = float(np.corrcoef(nref, n1)[0, 1])
        C1 = D1[idx] @ D1[idx].T / (np.outer(n1[idx], n1[idx]) + 1e-12)
        cs = float(np.corrcoef(Cref[iu], C1[iu])[0, 1])
        print(f"{s:>4} {pc:>10.4f} {pr:>10.4f} | {raw_c:>9.4f} {raw_f:>9.4f} "
              f"| {pr_c:>9.4f} {pr_f:>7.4f} | {n_c:>10.4f} {cs:>9.4f}")


if __name__ == "__main__":
    main()
