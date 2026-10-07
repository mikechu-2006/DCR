#!/usr/bin/env python
"""Can the chart vector itself be recovered, or only the predictions?

Experiment 1 (gauge).  Take the fitted (A, B) pair, apply a random rotation R to the latent
space:  P -> P R,  D -> D R.  Predictions <P_u,D_m> are provably unchanged, yet every
coordinate of every chart vector changes.  This shows the coordinates are not identifiable.

Experiment 2 (what IS recoverable).  Between the two independent half-sample fits, compare:
   a) the raw chart vectors, aligned by Procrustes (best possible rotation)
   b) the rotation-invariant summaries: ||D_m||, and pairwise cos(D_m, D_m')
   c) the raw vectors WITHOUT alignment, after ridge has fixed the gauge to minimum norm
If (c) already agrees well, the minimum-norm representative is a reproducible object and the
per-chart vector is usable output.  If only Procrustes-aligned (a) agrees, then only the
geometry is real and the coordinates must never be interpreted individually.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from diag_step2_feasible import rank_k_fit, predict

PROC = Path("data/processed")
K = 4
LAM = 1e-3


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

    trp = np.unique(tr.pair.to_numpy())
    perm = np.random.default_rng(12345).permutation(len(trp))
    half = len(trp) // 2
    isA = np.isin(tr.pair.to_numpy(), trp[perm[:half]])
    parts = {"A": tr[isA], "B": tr[~isA]}
    fits = {}
    for nm, p in parts.items():
        u = p.player_id.map(ep).to_numpy().astype(np.int64)
        m = p.beatmap_id.map(ip).to_numpy().astype(np.int64)
        fits[nm] = rank_k_fit(u, m, p.loss.to_numpy(), n_ent, n_item, K, LAM, iters=25)
        print(f"fit {nm}: ||D|| mean = {np.linalg.norm(fits[nm]['D'],axis=1).mean():.4f}", flush=True)

    # ---------------- experiment 1: gauge rotation ---------------------------------
    f = fits["A"]
    rng = np.random.default_rng(7)
    Q, _ = np.linalg.qr(rng.normal(size=(K, K)))
    pv0 = predict(f, tu, tm)
    f2 = dict(f)
    f2["P"] = f["P"] @ Q
    f2["D"] = f["D"] @ Q
    pv1 = predict(f2, tu, tm)
    print(f"\n[1] rotate P,D by a random orthogonal Q:")
    print(f"    max |pred_rotated - pred_original| = {np.abs(pv1-pv0).max():.3e}")
    print(f"    but  ||D_rot - D_orig||_F / ||D_orig||_F = "
          f"{np.linalg.norm(f2['D']-f['D'])/np.linalg.norm(f['D']):.4f}")
    print(f"    corr(||D_rot||, ||D_orig||) = "
          f"{np.corrcoef(np.linalg.norm(f2['D'],axis=1), np.linalg.norm(f['D'],axis=1))[0,1]:+.4f}")

    # ---------------- experiment 2: what is recoverable ---------------------------
    DA, DB = fits["A"]["D"], fits["B"]["D"]
    print("\n[2] comparing the two independent half-sample fits")
    print(f"    (a) raw, NO alignment:  corr(D_A, D_B) elementwise = "
          f"{np.corrcoef(DA.ravel(), DB.ravel())[0,1]:+.4f}")
    print(f"        relative Frobenius distance = {np.linalg.norm(DA-DB)/np.linalg.norm(DA):.4f}")
    # Procrustes: best rotation aligning DA onto DB
    U, s, Vt = np.linalg.svd(DA.T @ DB)
    R = U @ Vt
    DAa = DA @ R
    print(f"    (b) Procrustes-aligned: corr = {np.corrcoef(DAa.ravel(), DB.ravel())[0,1]:+.4f}"
          f"   rel.F = {np.linalg.norm(DAa-DB)/np.linalg.norm(DAa):.4f}")
    nA = np.linalg.norm(DA, axis=1); nB = np.linalg.norm(DB, axis=1)
    print(f"    (c) rotation-invariant: corr(||D_A||, ||D_B||) = {np.corrcoef(nA,nB)[0,1]:+.4f}")
    # pairwise cosine similarity agreement on a subsample
    idx = np.random.default_rng(0).choice(n_item, 1500, replace=False)
    CA = DA[idx] @ DA[idx].T / (np.outer(nA[idx], nA[idx]) + 1e-12)
    CB = DB[idx] @ DB[idx].T / (np.outer(nB[idx], nB[idx]) + 1e-12)
    iu = np.triu_indices(len(idx), 1)
    print(f"        corr(pairwise cos(D_m,D_m') A vs B) = {np.corrcoef(CA[iu], CB[iu])[0,1]:+.4f}")
    # per-coordinate agreement after Procrustes
    cc = [np.corrcoef(DAa[:, j], DB[:, j])[0, 1] for j in range(K)]
    print(f"        per-coordinate corr after alignment = {np.round(cc,4)}")
    cc0 = [np.corrcoef(DA[:, j], DB[:, j])[0, 1] for j in range(K)]
    print(f"        per-coordinate corr WITHOUT alignment = {np.round(cc0,4)}")

    # ---------------- which charts have enough data ------------------------------
    npl = tr.groupby("beatmap_id").size()
    print(f"\n[3] charts with fewer than K={K} training plays: "
          f"{int((npl<K).sum()):,}/{len(npl):,} ({(npl<K).mean():.1%}) -- "
          f"their K-dim vector is not determined by their own plays alone")


if __name__ == "__main__":
    main()
