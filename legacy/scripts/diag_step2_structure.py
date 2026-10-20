#!/usr/bin/env python
"""Structural diagnostic for step2 -- how much *interaction* structure is really there?

Motivation (user question).  mirt writes the interaction as

    <P_u - C_m, D_m> = <P_u, D_m> - <C_m, D_m>

and the second term depends on m only, so it is collinear with the item intercept b_m.
This project's own ablation already found that bm collapses to exactly 0.0 in the fitted
mirt model; the .npz shows C stays at its initialisation scale while D grows.  So with a
free D, mirt and mf_dot are the *same* rank-k bilinear family, and the per-chart vector can
only carry a direction (its norm is absorbed by b_m).  Everything the model can express
beyond the additive main effects is therefore a rank-k matrix factorization.

This script measures, on the real data and on the exact step2 split:
  1. the irreducible per-play noise  sigma^2  = pooled within-(player,chart) variance,
  2. the residual cell-mean variance after an additive player+chart fit -- i.e. how much
     per-pair structure the additive model does NOT capture,
  3. the SVD spectrum of that residual matrix (effective rank),
  4. a cross-validated rank sweep k = 0..32: does more rank actually predict held-out
     pairs better, and where does it saturate?
  5. invariance/identifiability diagnostics of the fitted .npz embeddings
     (C norm vs initialisation, bm collapse, rotation-invariant chart functionals),
  6. correlation of the rotation-invariant chart functionals with beatmap metadata.

Reported numbers are *debiased* for the measurement noise of the per-cell mean, so the
"interaction variance" is a lower bound on real structure (not an artefact of n=1.6).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path("data/processed")
META_COLS = ["beatmap_id", "star", "bpm", "max_combo", "count_total", "diff_overall",
             "diff_drain", "hit_length", "playcount", "passcount"]


# ------------------------------------------------------------------ helpers
def within_var(y: np.ndarray, pair: np.ndarray) -> tuple[float, float, int, int]:
    """Pooled within-pair variance, unbiased.  Also returns the number of plays that live
    in pairs with >= 2 plays, so that E[sigma^2 / n] over those pairs can be formed."""
    df = pd.DataFrame({"p": pair, "y": y})
    g = df.groupby("p", sort=False).y
    cnt = g.size()
    sd = g.std(ddof=1)
    multi = cnt >= 2
    dof = (cnt[multi] - 1).sum()
    ss = ((sd[multi] ** 2) * (cnt[multi] - 1)).sum()
    sig2 = float(ss / dof)
    # E[sigma^2/n] over plays inside multi-play pairs -- the noise left in a cell mean
    nn = cnt[multi].to_numpy(np.float64)
    e_sig_over_n = float((sig2 / nn).sum() / nn.sum())
    return sig2, e_sig_over_n, int(dof), int(multi.sum())


def weighted_als(R, W, k, lam=1.0, iters=12, seed=0):
    """Weighted ridge ALS for R ~= A B^T on the observed mask (W > 0)."""
    rng = np.random.default_rng(seed)
    na, nb = R.shape
    # Scale-aware init: the update for A is driven by B^T R, so if B starts at 0.05 the
    # player loadings have to be ~20x too large before the gradient makes sense.  Start A
    # at the residual scale (in player units) and B at O(1).
    a_sd = float(np.sqrt(((R ** 2)[W > 0]).mean() / max(k, 1)))
    A = rng.normal(0, a_sd, (na, k))
    B = rng.normal(0, 1.0, (nb, k))
    I = lam * np.eye(k)
    obs = W > 0

    def solve_side(F, Rm, Wm, Om):
        """One ALS half-step.

        F  : (no, k)   the *known* factor, indexed by the shared axis
        Rm : (nl, no)  target, transposed
        Wm : (nl, no)  weights, transposed
        Om : (nl, no)  observation mask, transposed
        returns (nl, k) for the other factor.  Both the target and the mask are passed in
        the *same* orientation as F's shared axis, which is where the earlier version of
        this function went wrong.
        """
        nl = Rm.shape[0]
        out = np.zeros((nl, k))
        for j in range(nl):
            o = np.where(Om[j])[0]
            if o.size == 0:
                continue
            w = Wm[j, o]
            Fo = F[o]
            G = Fo * w[:, None]
            out[j] = np.linalg.solve(Fo.T @ G + I, G.T @ Rm[j, o])
        return out

    for _ in range(iters):
        B = solve_side(A, R.T, W.T, obs.T)      # A is (na, k) -> B is (nb, k)
        A = solve_side(B, R, W, obs)            # B is (nb, k) -> A is (na, k)
    return A, B


def ranked_map(values: np.ndarray) -> dict:
    return {v: i for i, v in enumerate(np.unique(values))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="1k")
    ap.add_argument("--split-seed", type=int, default=20260928)
    ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--k-list", default="0,1,2,3,4,6,8,12,16,24,32")
    ap.add_argument("--min-cell-n", type=int, default=1,
                    help="only use train cells with at least this many plays")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cols = ["player_id", "beatmap_id", "playcount_cur", "loss"]
    s1 = pd.read_parquet(PROC / f"step1_{a.tag}.parquet", columns=cols)
    pq = PROC / f"step0_{a.tag}.parquet"
    s0 = (pd.read_parquet(pq, columns=cols) if pq.exists()
          else pd.read_csv(PROC / f"step0_{a.tag}.csv", usecols=cols))
    print(f"step0 {len(s0):,} rows / step1 {len(s1):,} rows", flush=True)

    kkey = int(max(s0.beatmap_id.max(), s1.beatmap_id.max())) + 1
    for df in (s0, s1):
        df["pair"] = df.player_id.astype(np.int64) * kkey + df.beatmap_id.astype(np.int64)
    upairs = np.unique(s0.pair.to_numpy())
    is_test = np.random.default_rng(a.split_seed).random(len(upairs)) < a.test_frac
    test_pairs = upairs[is_test]
    print(f"pairs {len(upairs):,} -> test {len(test_pairs):,}", flush=True)

    tr = s1[~s1.pair.isin(test_pairs)]
    te = s0[s0.pair.isin(test_pairs)]
    print(f"train plays {len(tr):,} / test plays {len(te):,}", flush=True)

    # ---- 1. noise floor ---------------------------------------------------------------
    sig2_tr, esn_tr, dof_tr, nc_tr = within_var(tr.loss.to_numpy(np.float64),
                                                tr.pair.to_numpy())
    sig2_te, esn_te, dof_te, nc_te = within_var(te.loss.to_numpy(np.float64),
                                                te.pair.to_numpy())
    print(f"\n[1] irreducible play noise sigma^2 = {sig2_tr:.4f} (sigma={np.sqrt(sig2_tr):.4f})"
          f"  [train step1, dof {dof_tr:,}, {nc_tr:,} multi-play cells]")
    print(f"    test  step0 sigma^2 = {sig2_te:.4f} (sigma={np.sqrt(sig2_te):.4f});"
          f"  E[sigma^2/n] inside multi-play cells = {esn_te:.4f}")

    # ---- 2/3. additive fit and per-pair residual --------------------------------------
    print("\n[2] additive player+chart model (ridge ALS on train plays)", flush=True)
    tv = tr.loss.to_numpy(np.float64)
    pmap, bmap = ranked_map(tr.player_id.to_numpy()), ranked_map(tr.beatmap_id.to_numpy())
    u = tr.player_id.map(pmap).to_numpy()
    m = tr.beatmap_id.map(bmap).to_numpy()
    n_ent, n_item = len(pmap), len(bmap)

    def fit_additive(lam: float, iters: int = 25):
        bu = np.zeros(n_ent)
        bm = np.zeros(n_item)
        for _ in range(iters):
            r = tv - bu[u]
            bm = np.bincount(m, weights=r, minlength=n_item) / (
                np.bincount(m, minlength=n_item) + lam)
            r = tv - bm[m]
            bu = np.bincount(u, weights=r, minlength=n_ent) / (
                np.bincount(u, minlength=n_ent) + lam)
        return bu, bm

    # choose lam on held-out *pairs* by cell-mean RMSE
    # ids the training rows never showed map to NaN -> clamp them to index 0 so that
    # rv_ok below can mark them out; the additive model then uses that row's value, which
    # is the same fallback the step2 script makes (-1 -> not scored).
    tu = pd.to_numeric(te.player_id.map(pmap), errors="coerce").to_numpy(np.float64)
    tm = pd.to_numeric(te.beatmap_id.map(bmap), errors="coerce").to_numpy(np.float64)
    rv_ok = ~(np.isnan(tu) | np.isnan(tm))
    tu = np.clip(np.nan_to_num(tu, nan=0.0), 0, n_ent - 1).astype(int)
    tm = np.clip(np.nan_to_num(tm, nan=0.0), 0, n_item - 1).astype(int)
    ty_play = te.loss.to_numpy(np.float64)[rv_ok]
    tpair = te.pair.to_numpy()[rv_ok]
    tc = pd.DataFrame({"u": tu[rv_ok], "m": tm[rv_ok], "y": ty_play,
                       "p": tpair}).groupby(
        ["u", "m"], sort=False).agg(y=("y", "mean"), n=("y", "size"))
    ti = tc.index.get_level_values(0).to_numpy()
    tj = tc.index.get_level_values(1).to_numpy()
    ty = tc.y.to_numpy(np.float64)
    tn = tc.n.to_numpy(np.float64)
    print(f"    test cells (held-out pairs, covered): {len(tc):,}; "
          f"covered test plays {int(rv_ok.sum()):,}/{len(rv_ok):,}", flush=True)

    best = None
    for lam in [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
        bu, bm = fit_additive(lam)
        res = ty - bu[ti] - bm[tj]          # ty is the *raw* cell mean here
        rv = float((res ** 2).mean())
        print(f"    lam={lam:7.2f}  test cell RMSE={np.sqrt(rv):.4f}  "
              f"cell resid var={rv:.4f}  (noise in a cell mean ~ {esn_te:.4f})", flush=True)
        if best is None or rv < best["cell_resid_var"]:
            best = dict(lam=lam, cell_resid_var=rv, cell_rmse=float(np.sqrt(rv)))
            best_pack = (bu, bm)
    bu, bm = best_pack
    ty_res = ty - bu[ti] - bm[tj]           # what a rank-k term still has to explain
    print(f"    -> best additive lam={best['lam']}  cell-resid var={best['cell_resid_var']:.4f}"
          f"  | test cell-mean var = {ty.var(ddof=0):.4f}", flush=True)
    print(f"    => real (debiased) interaction variance at cell level = "
          f"{max(ty.var(ddof=0) - best['cell_resid_var'], 0):.4f}"
          f" ; shrunken additive pred var = {bu[ti].var(ddof=0)+bm[tj].var(ddof=0):.4f}")

    # train-side residual matrix
    rtr = tv - bu[u] - bm[m]
    tdf = pd.DataFrame({"u": u, "m": m, "r": rtr})
    cell = tdf.groupby(["u", "m"], sort=False).r.agg(["mean", "size"])
    mcn = a.min_cell_n
    sub = cell[cell["size"] >= mcn]
    print(f"\n[3] train residual matrix from cells with n>={mcn}: {len(sub):,} cells")
    for nn in (1, 2, 3, 5):
        s2 = cell[cell["size"] >= nn]
        print(f"      n>={nn}: {len(s2):,} cells  raw resid-cell-mean var="
              f"{s2['mean'].var(ddof=0):.4f}  (noise-inflated by ~"
              f"{(sig2_tr/s2['size'].to_numpy()).sum()/len(s2):.4f})", flush=True)

    R = np.zeros((n_ent, n_item))
    W = np.zeros_like(R)
    ii = sub.index.get_level_values(0).to_numpy()
    jj = sub.index.get_level_values(1).to_numpy()
    R[ii, jj] = sub["mean"].to_numpy(np.float64)
    W[ii, jj] = sub["size"].to_numpy(np.float64)
    Rc = R - R.mean(0, keepdims=True)
    Rc = Rc - Rc.mean(1, keepdims=True)
    sv = np.linalg.svd(Rc, compute_uv=False)
    sv2 = sv ** 2
    tot = sv2.sum()
    print("    SVD spectrum of the doubly-centred residual matrix:")
    energy = {}
    for kk in [1, 2, 3, 4, 5, 8, 12, 16, 24, 32, 48, 64]:
        energy[kk] = float(sv2[:kk].sum() / tot)
        print(f"      k={kk:3d}: cumulative |.|_F^2 energy = {energy[kk]*100:6.2f}%", flush=True)

    # ---- 4. cross-validated rank sweep ------------------------------------------------
    print("\n[4] cross-validated rank sweep (fit on train cells, score on test cells)",
          flush=True)
    sweep = {}
    for kk in [int(x) for x in a.k_list.split(",")]:
        if kk == 0:
            pres = np.zeros(len(ty))
        else:
            A, B = weighted_als(R, W, kk, lam=1.0, iters=12)
            pres = np.einsum("ik,ik->i", A[ti], B[tj])
        res = ty_res - pres                 # residual after additive + rank-k
        rv = float((res ** 2).mean())
        deb = float(rv - esn_te)          # remove the measurement noise of the cell mean
        sweep[kk] = dict(cell_resid_var=rv, cell_rmse=float(np.sqrt(rv)),
                         debiased_var=deb,
                         debiased_rmse=float(np.sqrt(deb)) if deb > 0 else 0.0,
                         play_rmse_vs_additive=float(np.sqrt(max(deb, 0) + sig2_te)))
        print(f"    k={kk:3d}  cell RMSE={np.sqrt(rv):.4f}   debiased cell RMSE="
              f"{sweep[kk]['debiased_rmse']:.4f}   => play-level RMSE vs additive "
              f"{sweep[kk]['play_rmse_vs_additive']:.4f}   (additive k=0: "
              f"{sweep[0]['play_rmse_vs_additive']:.4f})", flush=True)

    # ---- 5. invariance diagnostics of the fitted embeddings ---------------------------
    print("\n[5] fitted-.npz diagnostics (rotation-invariant quantities)", flush=True)
    npz_path = PROC / f"step2_{a.tag}_mirt_F0_loss.npz"
    if npz_path.exists():
        z = np.load(npz_path)
        P, C, D, bmz = (z["P"].astype(np.float64), z["C"].astype(np.float64),
                        z["D"].astype(np.float64), z["bm"].astype(np.float64))
        init = 0.05
        print(f"    ||C||_2 mean = {np.linalg.norm(C,axis=1).mean():.4f} "
              f"(init expectation ~ {init*np.sqrt(C.shape[1]):.4f});  "
              f"||D||_2 mean = {np.linalg.norm(D,axis=1).mean():.4f}")
        print(f"    ||bm||_inf = {np.abs(bmz).max():.3e}  (item intercept absorbed by <-C,D>)")
        # rotation-invariant chart functional: the effective chart vector in PLAYER space
        b_eff = z["bu"].astype(np.float64) - D @ P.T          # (n_item, n_ent)
        b_eff_c = b_eff - b_eff.mean(1, keepdims=True)
        svb = np.linalg.svd(b_eff_c - b_eff_c.mean(0, keepdims=True), compute_uv=False)
        eb = svb ** 2
        print(f"    effective chart x player matrix: {b_eff.shape}, "
              f"top-1 singular energy {eb[0]/eb.sum()*100:.1f}%, "
              f"top-3 {eb[:3].sum()/eb.sum()*100:.1f}%, top-8 {eb[:8].sum()/eb.sum()*100:.1f}%")
        # how much of the interaction can be captured by 1/2/3 player directions?
        uu = np.unique(tu)
        d1 = b_eff_c[:, uu]
        print(f"    on the {len(uu)} seen test players: rank-1 captures "
              f"{(np.linalg.svd(d1,compute_uv=False)[0]**2/ (d1**2).sum())*100:.1f}% of energy")
        # correlation of the chart's mean shift with metadata
        meta = pd.read_csv(PROC / f"beatmap_meta_{a.tag}.csv", usecols=META_COLS)
        chart_ids = z["beatmap_ids"]
        dfd = pd.DataFrame({"beatmap_id": chart_ids, "shift": b_eff.mean(1),
                            "spread": b_eff_c.std(1), "dnorm": np.linalg.norm(D, axis=1),
                            "cmeandot": (C * D).sum(1)})
        mg = dfd.merge(meta, on="beatmap_id", how="left")
        print(f"    metadata join coverage: "
              f"{mg['star'].notna().mean()*100:.1f}% of {len(mg):,} charts")
        for col in ["star", "bpm", "max_combo", "count_total", "diff_overall",
                    "diff_drain", "hit_length"]:
            for val in ["shift", "spread", "c_norm" if False else "dnorm"]:
                x = mg[[col, val]].dropna()
                if len(x) > 10:
                    r = float(np.corrcoef(x[col], x[val])[0, 1])
                    print(f"      corr({col:12s}, {val:7s}) = {r:+.4f}  (n={len(x):,})")
    else:
        print(f"    (no {npz_path})")

    res = dict(tag=a.tag, split_seed=a.split_seed, test_frac=a.test_frac,
               n_pairs=int(len(upairs)), n_test_pairs=int(len(test_pairs)),
               n_train_plays=int(len(tr)), n_test_plays=int(len(te)),
               sigma2_train=sig2_tr, sigma2_test=sig2_te,
               e_sigma2_over_n_test=esn_te,
               additive=best, test_cell_mean_var=float(ty.var(ddof=0)),
               debiased_interaction_var=float(max(ty.var(ddof=0) - best["cell_resid_var"], 0)),
               rank_sweep={str(kk): v for kk, v in sweep.items()},
               sv_energy={str(kk): v for kk, v in energy.items()})
    path = Path(a.out) if a.out else PROC / f"diag_step2_structure_{a.tag}.json"
    path.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
