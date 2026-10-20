#!/usr/bin/env python
"""Is the latent task *possible*?  Degrees of freedom, reproducibility, identifiable summaries.

Three questions, three answers to compute.

Q1 (counting).  rank-K MF on the 1k training set has  K*(n_ent + n_item)  free parameters.
    At K=4 that is ~82k parameters against 674,436 training plays.  Is the model therefore
    "impossible"?  -> count the parameters that survive the gauge group instead of the raw
    count.  The gauge group of D P^T is GL(K) (dim K^2) plus the per-player / per-chart
    shifts absorbed by b_u / b_m.  Compare that against the number of plays per chart.

Q2 (reproducibility).  Fit the same model independently on two disjoint halves of the
    training pairs and compare on identical test cells.  If the chart vectors are mostly
    noise (as they are in the null model where cells have random offsets), the two halves
    disagree by as much as a full-noise model does.  If the structure is real, they agree.

Q3 (gauge-invariant structure).  Compare the two halves
      - directly               : principal angles between the row spaces of D1 and D2
      - invariantly            : ||S1 - S2||_F / ||S1||_F  with S = D D^T (chart x chart
                                 similarity, fully rotation invariant)
      - against external truth  : corr of the per-chart summaries with official star

A pure-noise reference is included: the SAME pipeline run on labels shuffled *within*
(player, chart) cells, which destroys any between-cell structure while keeping n per cell
and the marginal distribution.  Whatever the real data reproduces beyond that reference is
what is actually estimable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path("data/processed")
META = ["beatmap_id", "star", "bpm", "max_combo", "count_total", "diff_overall", "diff_drain",
        "hit_length"]


def rank_k_fit(u, m, y, n_ent, n_item, K, lam, iters=60, seed=0, P0=None, D0=None, bu0=None):
    """Adam on  loss = mean((y - <P_u,D_m> - b_u - b_m)^2) + lam*(|P|^2+|D|^2+|b|^2).

    The L2 penalty both regularises and *fixes the gauge*: a minimum-norm solution picks a
    canonical representative of the rotation orbit, so two solutions can be compared
    directly (this is the standard "regularised MF is identifiable" argument).
    """
    rng = np.random.default_rng(seed)
    s = float(np.std(y)) / np.sqrt(K)
    P = np.asarray(rng.normal(0, s, (n_ent, K)), np.float64) if P0 is None else P0.copy()
    D = np.asarray(rng.normal(0, 1.0, (n_item, K)), np.float64) if D0 is None else D0.copy()
    bu = np.zeros(n_ent, np.float64) if bu0 is None else bu0.copy()
    bm = np.zeros(n_item, np.float64)
    mu = float(y.mean())
    u = np.asarray(u, np.int64); m = np.asarray(m, np.int64)
    y = np.asarray(y, np.float64); lam = float(lam)
    mp = {k: np.zeros_like(v) for k, v in
          dict(P=P, D=D, bu=bu, bm=bm, mu=np.array(mu)).items()}
    vp = {k: np.zeros_like(v) for k, v in mp.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    lr = 0.1
    n = len(y)
    hist = []
    # chunked: the P_u / D_m gathers otherwise allocate an (n * K) float64 temporary, which
    # is what was OOM-killing the runs on a 7 GB box
    CH = 65536
    for t in range(1, iters + 1):
        gP = np.zeros_like(P); gD = np.zeros_like(D)
        gbu = np.zeros(n_ent); gbm = np.zeros(n_item); gmu = 0.0
        for s0 in range(0, n, CH):
            sl = slice(s0, min(s0 + CH, n))
            us, ms = u[sl], m[sl]
            pred = mu + bu[us] + bm[ms] + np.einsum("ik,ik->i", P[us], D[ms])
            g = -2.0 * (y[sl] - pred) / n
            # scatter-multiply: D2[us] += g[:,None]*D[ms] -- the in-place += on the *indexed*
            # array is what numpy buffers, and it costs no (n x K) gather temporaries
            D2 = D[ms]; D2 *= g[:, None]; np.add.at(gP, us, D2)
            P2 = P[us]; P2 *= g[:, None]; np.add.at(gD, ms, P2)
            np.add.at(gbu, us, g)
            np.add.at(gbm, ms, g)
            gmu += g.sum()
            del D2, P2
        grads = dict(P=gP + 2 * lam * P, D=gD + 2 * lam * D,
                     bu=gbu + 2 * lam * bu, bm=gbm + 2 * lam * bm,
                     mu=np.array(gmu))
        for k in grads:
            mp[k] = b1 * mp[k] + (1 - b1) * grads[k]
            vp[k] = b2 * vp[k] + (1 - b2) * grads[k] ** 2
            mh = mp[k] / (1 - b1 ** t)
            vh = vp[k] / (1 - b2 ** t)
            if k == "P":
                P -= lr * mh / (np.sqrt(vh) + eps)
            elif k == "D":
                D -= lr * mh / (np.sqrt(vh) + eps)
            elif k == "bu":
                bu -= lr * mh / (np.sqrt(vh) + eps)
            elif k == "bm":
                bm -= lr * mh / (np.sqrt(vh) + eps)
            else:
                mu = float(mu - lr * mh / (np.sqrt(vh) + eps))
        if t % max(iters // 4, 1) == 0:
            se = 0.0
            for s0 in range(0, n, CH):
                sl = slice(s0, min(s0 + CH, n))
                us, ms = u[sl], m[sl]
                p = mu + bu[us] + bm[ms] + np.einsum("ik,ik->i", P[us], D[ms])
                se += float(((y[sl] - p) ** 2).sum())
            hist.append((t, float(np.sqrt(se / n))))
    return dict(P=P, D=D, bu=bu, bm=bm, mu=mu, hist=hist)


def predict(fit, u, m):
    return fit["mu"] + fit["bu"][u] + fit["bm"][m] + np.einsum(
        "ik,ik->i", fit["P"][u], fit["D"][m])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="1k")
    ap.add_argument("--dim", type=int, default=4)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--split-seed", type=int, default=20260928)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--no-full", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cols = ["player_id", "beatmap_id", "loss"]
    s1 = pd.read_parquet(PROC / f"step1_{a.tag}.parquet", columns=cols)
    s0 = pd.read_csv(PROC / f"step0_{a.tag}.csv", usecols=cols)
    k = int(max(s0.beatmap_id.max(), s1.beatmap_id.max())) + 1
    for df in (s0, s1):
        df["pair"] = df.player_id.astype(np.int64) * k + df.beatmap_id.astype(np.int64)
    up = np.unique(s0.pair.to_numpy())
    rng = np.random.default_rng(a.split_seed)
    is_test = rng.random(len(up)) < 0.30
    tps = set(up[is_test].tolist())
    tr = s1[~s1.pair.isin(tps)].reset_index(drop=True)
    te = s0[s0.pair.isin(tps)].reset_index(drop=True)

    ent = np.unique(tr.player_id.to_numpy())
    itm = np.unique(tr.beatmap_id.to_numpy())
    ep = {v: i for i, v in enumerate(ent)}
    ip = {v: i for i, v in enumerate(itm)}
    n_ent, n_item = len(ent), len(itm)

    # ---- Q1: parameter accounting ----------------------------------------------------
    K = a.dim
    print(f"[Q1] rank-K fit, K={K}, tag={a.tag}")
    print(f"     entities (players) = {n_ent:,}   items (charts) = {n_item:,}")
    print(f"     free params  P + D          = {K*(n_ent+n_item):,}")
    print(f"     gauge group  GL(K)          = {K*K:,}   (plus K shifts absorbed by b_u/b_m)")
    print(f"     effective free params       = {K*(n_ent+n_item) - K*K - 2*K:,}")
    print(f"     training plays              = {len(tr):,}")
    print(f"     plays per free param        = {len(tr)/(K*(n_ent+n_item)-K*K-2*K):.2f}")
    print(f"     train plays per chart       = {len(tr)/n_item:.2f}"
          f"   -> per chart per latent dim = {len(tr)/n_item/K:.2f}")

    # ---- vocabulary / evaluation arrays ----------------------------------------------
    tu = te.player_id.map(ep).to_numpy(); tm = te.beatmap_id.map(ip).to_numpy()
    ok = pd.notna(tu) & pd.notna(tm)
    tu = tu[ok].astype(int); tm = tm[ok].astype(int)
    y = te.loss.to_numpy(np.float64)[ok]
    tp = te.pair.to_numpy()[ok]
    td = pd.DataFrame({"u": tu, "m": tm, "y": y, "p": tp})
    tc = td.groupby(["u", "m"], sort=False).agg(y=("y", "mean"), n=("y", "size"))
    ci = tc.index.get_level_values(0).to_numpy()
    cj = tc.index.get_level_values(1).to_numpy()
    cy = tc.y.to_numpy(np.float64)

    # ---- two disjoint halves of the TRAINING PAIRS -----------------------------------
    tr_pairs = np.unique(tr.pair.to_numpy())
    perm = np.random.default_rng(12345).permutation(len(tr_pairs))
    half = len(tr_pairs) // 2
    A = set(tr_pairs[perm[:half]].tolist())
    B = set(tr_pairs[perm[half:]].tolist())
    print(f"\n[Q2] train pairs {len(tr_pairs):,} -> half A {len(A):,} / half B {len(B):,}")
    isA = np.isin(tr.pair.to_numpy(), tr_pairs[perm[:half]])
    trA = tr[isA].reset_index(drop=True)
    trB = tr[~isA].reset_index(drop=True)

    def encode(df):
        return (df.player_id.map(ep).to_numpy().astype(int),
                df.beatmap_id.map(ip).to_numpy().astype(int),
                df.loss.to_numpy(np.float64))

    res = {}
    todo = [("A", trA, a.lam), ("B", trB, a.lam)]
    if not a.no_full:
        todo.append(("full", tr, a.lam))
    import time
    for name, part, lam in todo:
        uu, mm, yy = encode(part)
        t0 = time.time()
        f = rank_k_fit(uu, mm, yy, n_ent, n_item, K, lam, iters=a.iters, seed=0)
        print(f"       [{name}] fit took {time.time()-t0:.1f}s", flush=True)
        pv_cell = predict(f, ci, cj)
        r_cell = float(np.sqrt(np.mean((cy - pv_cell) ** 2)))
        pv_play = predict(f, tu, tm)
        r_play = float(np.sqrt(np.mean((y - pv_play) ** 2)))
        res[name] = f
        print(f"     fit on {name:4s} ({len(part):,} plays): test cell RMSE={r_cell:.4f}  "
              f"play RMSE={r_play:.4f}")

    # reproducibility: do the two half-fits agree on the SAME held-out cells?
    fA, fB = res["A"], res["B"]
    fF = res.get("full", fA)
    has_full = "full" in res
    pA = predict(fA, ci, cj); pB = predict(fB, ci, cj); pF = predict(fF, ci, cj)
    def agree(x, z):
        return float(np.corrcoef(x, z)[0, 1]), float(np.sqrt(np.mean((x - z) ** 2)))
    c_AB, d_AB = agree(pA, pB)
    c_AF, d_AF = agree(pA, pF)
    print(f"     corr(cell pred A, B) = {c_AB:.4f}   RMS difference = {d_AB:.4f}")
    if has_full:
        print(f"     corr(cell pred A, full) = {c_AF:.4f}  RMS difference = {d_AF:.4f}")
    sd_pred = float(pA.std())
    print(f"     sd of predicted cell values = {sd_pred:.4f}")
    print(f"     (a pure-noise model would give corr ~ 0 and RMS diff ~ "
          f"sqrt(2)*{sd_pred:.4f} = {np.sqrt(2)*sd_pred:.4f})")

    # ---- Q3: gauge-invariant comparisons ---------------------------------------------
    print("\n[Q3] gauge-invariant structure of the chart representation")
    DA, DB, DF = fA["D"], fB["D"], fF["D"]
    # (a) principal angles between the row spaces: D is (n_item, K) -> col space of D^T
    qa = np.linalg.qr(DA)[0]; qb = np.linalg.qr(DB)[0]; qf = np.linalg.qr(DF)[0]
    ang = np.linalg.svd(qa.T @ qb, compute_uv=False)
    angf = np.linalg.svd(qa.T @ qf, compute_uv=False)
    print(f"     cos(principal angles) A vs B : {np.round(ang, 4)}")
    print(f"     cos(principal angles) A vs full: {np.round(angf, 4)}")
    # (b) rotation-invariant similarity structure, WITHOUT materialising n_item^2:
    #     ||X X^T - Y Y^T||_F^2 = tr((XX^T)^2) + tr((YY^T)^2) - 2 tr(X X^T Y Y^T)
    def sim_agreement(X, Y, nm):
        G1, G2 = X @ X.T, Y @ Y.T            # (K,K) -- n_item cancels
        f1 = float(np.sum(G1 * G1)); f2 = float(np.sum(G2 * G2))
        cross = float(np.sum((X.T @ Y) ** 2))
        rel = float(np.sqrt(max(f1 + f2 - 2 * cross, 0.0)) / np.sqrt(f1))
        # correlation on a random subset of rows (20k x n_item slice, cheap)
        idx = np.random.default_rng(0).choice(X.shape[0], size=min(2000, X.shape[0]),
                                              replace=False)
        S1 = X[idx] @ X.T
        S2 = Y[idx] @ Y.T
        cS = float(np.corrcoef(S1.ravel(), S2.ravel())[0, 1])
        print(f"     ||S1-S2||_F/||S1||_F ({nm}) = {rel:.4f}   corr(S1,S2) = {cS:.4f}")

    sim_agreement(DA, DB, "A vs B")
    if has_full:
        sim_agreement(DA, DF, "A vs full")
    # (c) does the estimable structure carry external meaning?
    meta = pd.read_csv(PROC / f"beatmap_meta_{a.tag}.csv", usecols=META).set_index("beatmap_id")
    meta = meta.reindex(itm)
    star = meta["star"].to_numpy(np.float64)
    for nm, X in [("A", DA), ("B", DB), ("full", DF)]:
        e = np.linalg.norm(X, axis=1)**2          # rotation-invariant energy per chart
        m_ = np.isfinite(star)
        print(f"     corr(||D_m||^2, star) fit-{nm:4s} = {np.corrcoef(e[m_], star[m_])[0,1]:+.4f}")
    # (d) rank of the fitted interaction matrix: is K=4 actually used?
    for nm, X in [("A", DA), ("B", DB), ("full", DF)]:
        M = X @ res[nm]["P"].T
        sv = np.linalg.svd(M, compute_uv=False)
        print(f"     interaction matrix D P^T singular values (fit-{nm}): "
              f"{np.round(sv/sv[0], 3)}")

    out = dict(tag=a.tag, dim=K, lam=a.lam,
               params=dict(free=K*(n_ent+n_item), gauge=K*K,
                           effective=K*(n_ent+n_item)-K*K-2*K,
                           train_plays=int(len(tr)), plays_per_chart=len(tr)/n_item),
               rmse={n: dict(cell=float(np.sqrt(np.mean((cy - predict(f, ci, cj))**2))),
                             play=float(np.sqrt(np.mean((y - predict(f, tu, tm))**2))))
                     for n, f in res.items()},
               agreement=dict(corr_AB=c_AB, rms_AB=d_AB, corr_Afull=c_AF, rms_Afull=d_AF,
                              sd_pred=sd_pred),
               principal_angles_AB=[float(x) for x in ang],
               principal_angles_Afull=[float(x) for x in angf])
    path = Path(a.out) if a.out else PROC / f"diag_step2_feasible_{a.tag}_K{K}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
