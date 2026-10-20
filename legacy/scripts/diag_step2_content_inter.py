#!/usr/bin/env python
"""Is the interaction WELL-PARAMETERISED, or just under-powered?

Same train/test split as everything else in this session.  All models are linear in their
parameters and fitted by exact ridge (no optimiser, no local minima), so any difference is
attributable to the *parameterisation of the interaction*, not to training.

  M0  b_u + b_m                                  (additive; the reference)
  M1  b_u + <global slope, content_m>            (content affects everyone the same)
  M2  b_u + <slope_u, content_m>                 (PER-PLAYER content slopes = interaction)
  M3  M2 + b_m                                   (+ free per-chart intercept)

If M2/M3 beat M0 by much more than the rank-16 bilinear model did, the interaction is real
and the free-latent-factor parameterisation is what is failing us.  If they barely help, the
interaction signal is genuinely small at this sample size.

'content' = standardised beatmap metadata (star, bpm, max_combo, count_total, diff_overall,
diff_drain, hit_length) plus 1 (intercept).  Reported with and without star, because star is
a 2026 snapshot and therefore anachronistic for the early plays.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

PROC = Path("data/processed")
FEATS = ["star", "bpm", "max_combo", "count_total", "diff_overall", "diff_drain", "hit_length"]


def build(ent_idx, item_idx, F1, n_ent):
    """rows: [one-hot player | one-hot(player) * feat_1 ... feat_p]"""
    n = len(ent_idx)
    f = F1.shape[1]
    cols = [ent_idx]
    for j in range(f):
        cols.append(ent_idx + n_ent * (j + 1))
    data = [np.ones(n)] + [F1[:, j] for j in range(f)]
    idx = np.concatenate([c[:, None] for c in cols], axis=1)
    return sp.csr_matrix((np.concatenate(data), idx.ravel() + 0,
                          np.arange(0, n * (f + 1) + 1, f + 1)), shape=(n, n_ent * (f + 1)))


def build_global(item_idx, F1, n_ent):
    """rows: [one-hot player | feat_1 ... feat_p]  (slopes shared by all players)"""
    n = len(item_idx)
    f = F1.shape[1]
    col = np.concatenate([item_idx[:, None], n_ent + np.arange(f)[None, :].repeat(n, 0)], axis=1)
    data = np.concatenate([np.ones(n), F1.ravel()])
    return sp.csr_matrix((data, col.ravel(), np.arange(0, n * (f + 1) + 1, f + 1)),
                         shape=(n, n_ent + f))


def solve_ridge(X, y, lam_vec):
    XtX = (X.T @ X).tocsc()
    XtX.setdiag(XtX.diagonal() + lam_vec)
    return spsolve(XtX, X.T @ y)


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

    ent = np.unique(tr.player_id.to_numpy()); itm = np.unique(tr.beatmap_id.to_numpy())
    ep = {v: i for i, v in enumerate(ent)}; ip = {v: i for i, v in enumerate(itm)}
    n_ent, n_item = len(ent), len(itm)
    u = tr.player_id.map(ep).to_numpy(np.int64); m = tr.beatmap_id.map(ip).to_numpy(np.int64)
    y = tr.loss.to_numpy(np.float64)
    ymu, ysd = y.mean(), y.std()
    ys = (y - ymu) / ysd
    del tr
    gc.collect()

    meta = pd.read_csv(PROC / "beatmap_meta_1k.csv", usecols=["beatmap_id"] + FEATS)
    meta = meta.set_index("beatmap_id").reindex(itm)
    meta = meta.fillna(meta.median(numeric_only=True))
    fallback = meta.median(numeric_only=True)
    RAW = meta[FEATS].to_numpy(np.float64)

    tu = te.player_id.map(ep).to_numpy(); tm = te.beatmap_id.map(ip).to_numpy()
    ok = pd.notna(tu) & pd.notna(tm)
    tu = tu[ok].astype(np.int64); tm = tm[ok].astype(np.int64)
    yte = (te.loss.to_numpy(np.float64)[ok] - ymu) / ysd
    tpair = te.pair.to_numpy()[ok]
    del te
    gc.collect()
    tc = pd.DataFrame({"u": tu, "m": tm, "y": yte, "p": tpair}).groupby(
        ["u", "m"], sort=False).y.mean()
    ci = tc.index.get_level_values(0).to_numpy(); cj = tc.index.get_level_values(1).to_numpy()
    cy = tc.to_numpy(np.float64)
    print(f"train {len(u):,} / test plays {len(yte):,} / test cells {len(cy):,}", flush=True)

    def run(tag, cols_used):
        raw = RAW[:, [FEATS.index(c) for c in cols_used]]
        F = (raw - raw.mean(0)) / (raw.std(0) + 1e-12)
        F = np.hstack([F, np.ones((len(F), 1))])          # + intercept column
        Ftr = F[m]; Fte = F[tm]
        out = {}
        # M0 additive: player + chart intercepts
        X0 = sp.hstack([sp.csr_matrix((np.ones(len(u)), (np.arange(len(u)), u)),
                                      shape=(len(u), n_ent)),
                        sp.csr_matrix((np.ones(len(u)), (np.arange(len(u)), m)),
                                      shape=(len(u), n_item))], format="csr")
        for lam in (1.0, 10.0, 100.0):
            w = solve_ridge(X0, ys, np.full(X0.shape[1], lam))
            p_cell = w[ci] + w[n_ent + cj]
            p_play = w[tu] + w[n_ent + tm]
            out[f"M0_lam{lam:g}"] = (float(np.sqrt(np.mean((cy - p_cell) ** 2))),
                                     float(np.sqrt(np.mean((yte - p_play) ** 2))))
        # M1 global content slopes
        X1 = sp.hstack([sp.csr_matrix((np.ones(len(u)), (np.arange(len(u)), u)),
                                      shape=(len(u), n_ent)),
                        sp.csr_matrix(Ftr)], format="csr")
        for lam in (1.0, 10.0):
            w = solve_ridge(X1, ys, np.r_[np.full(n_ent, lam), np.full(F.shape[1], lam)])
            p_cell = w[ci] + F[list(cj)] @ w[n_ent:]
            p_play = w[tu] + Fte @ w[n_ent:]
            out[f"M1_lam{lam:g}"] = (float(np.sqrt(np.mean((cy - p_cell) ** 2))),
                                     float(np.sqrt(np.mean((yte - p_play) ** 2))))
        # M2 per-player content slopes
        X2 = build(u, m, Ftr, n_ent)
        for lam_s in (1.0, 10.0, 100.0):
            lv = np.r_[np.full(n_ent, 1.0), np.full(n_ent * F.shape[1], lam_s)]
            w = solve_ridge(X2, ys, lv)
            W = w.reshape(F.shape[1] + 1, n_ent).T          # (n_ent, f+1)
            p_cell = W[ci] @ F[list(cj)].T
            p_play = np.einsum("ik,ik->i", W[tu], Fte)
            out[f"M2_lam{lam_s:g}"] = (float(np.sqrt(np.mean((cy - p_cell) ** 2))),
                                       float(np.sqrt(np.mean((yte - p_play) ** 2))))
        # M3 = M2 + free chart intercept
        X3 = sp.hstack([X2, sp.csr_matrix((np.ones(len(u)), (np.arange(len(u)), m)),
                                          shape=(len(u), n_item))], format="csr")
        for lam_s in (10.0, 100.0):
            lv = np.r_[np.full(n_ent, 1.0), np.full(n_ent * F.shape[1], lam_s),
                       np.full(n_item, 10.0)]
            w = solve_ridge(X3, ys, lv)
            W = w[:n_ent * (F.shape[1] + 1)].reshape(F.shape[1] + 1, n_ent).T
            bm = w[n_ent * (F.shape[1] + 1):]
            p_cell = np.einsum("ik,ik->i", W[ci], F[list(cj)]) + bm[cj]
            p_play = np.einsum("ik,ik->i", W[tu], Fte) + bm[tm]
            out[f"M3_lam{lam_s:g}"] = (float(np.sqrt(np.mean((cy - p_cell) ** 2))),
                                       float(np.sqrt(np.mean((yte - p_play) ** 2))))
        print(f"\n=== content features: {tag} ===", flush=True)
        for kk, (c, p) in sorted(out.items(), key=lambda kv: kv[1][1]):
            print(f"   {kk:<14} cell RMSE={c:.4f}   play RMSE={p:.4f}", flush=True)
        return out

    res = {}
    res["with_star"] = run("全部 7 个（含 star）", FEATS)
    res["no_star"] = run("去掉 star", [f for f in FEATS if f != "star"])
    Path("data/processed/diag_step2_content_inter_1k.json").write_text(json.dumps(res, indent=2))
    print("\nwrote data/processed/diag_step2_content_inter_1k.json")


if __name__ == "__main__":
    main()
