#!/usr/bin/env python
"""Step 2 - prediction task:  (player_id, beatmap_id) -> loss.

Design (docs/step2_prediction.md):

  * observation unit is the **play**; the label is step0's ``loss``.  No cell table, no
    per-cell medians, no aggregation of any kind.
  * **train** on ``step1_{tag}`` (the cleaned play table).
  * **test** on ``step0_{tag}`` (the raw play table) -- step1's two rules are a
    *training-side* treatment, so letting them select the test rows would be
    conditioning on the treatment (docs/cleaning_plan.md section 3.2).
  * **split**: a fixed-seed random 30% of the ``(player_id, beatmap_id)`` *ordered pairs*
    goes to the test set; the remaining pairs supply the training plays.  A pair is never
    on both sides, so every test pair is unseen in training.

Models are small bilinear forms, so this file is **numpy-only** (no torch): forward and
backward are written out explicitly and optimised with Adam.

    L_hat = b0 + bu[u] + bm[m] + interaction
      bias    : interaction = 0
      mf_dot  : interaction = <P[u], C[m]>
      mirt    : interaction = <P[u] - C[m], D[m]>

    --features F1 adds  w1 * ln(playcount_cur)     (step1's own column, used as-is)
    --features F2 / --entity player-year additionally use
        P_eff = P[e] + (tau - 0.5) * V[e]          e = (player_id, calendar year)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path("data/processed")
COLS = ["player_id", "beatmap_id", "timestamp", "playcount_cur", "loss"]


# ----------------------------------------------------------------------------- io
def load_play_table(tag: str, which: str) -> pd.DataFrame:
    """which in {'step0','step1'} -> that play table for `tag` (.parquet preferred)."""
    pq, csv = PROC / f"{which}_{tag}.parquet", PROC / f"{which}_{tag}.csv"
    if pq.exists():
        df, src = pd.read_parquet(pq, columns=COLS), pq
    elif csv.exists():
        df, src = pd.read_csv(csv, usecols=COLS), csv
    else:
        raise SystemExit(f"neither {pq} nor {csv} exists")
    print(f"[{which}] {src.name}: {len(df):,} rows, {df.player_id.nunique():,} players, "
          f"{df.beatmap_id.nunique():,} beatmaps", flush=True)
    return df


def pair_keys(player, beatmap, k: int) -> np.ndarray:
    """Ordered-pair key, int64.  k = beatmap_id.max()+1 makes the encoding injective."""
    return player.astype(np.int64) * np.int64(k) + beatmap.astype(np.int64)


# ------------------------------------------------------------------------- metrics
def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(len(x), dtype=np.float64)
    return r


def spearman(pred: np.ndarray, y: np.ndarray) -> float:
    a, b = _ranks(pred), _ranks(y)
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def report(pred: np.ndarray, y: np.ndarray, n_total: int) -> dict:
    e = pred - y
    # loss -> ACC percentage points:  |exp(loss_pred) - exp(loss_true)| * 100
    dpp = np.abs(np.exp(pred.astype(np.float64)) - np.exp(y.astype(np.float64))) * 100.0
    return {
        "n_scored": int(len(y)),
        "coverage": round(len(y) / n_total, 6) if n_total else None,
        "rmse": round(float(np.sqrt((e ** 2).mean())), 6),
        "mae": round(float(np.abs(e).mean()), 6),
        "spearman": round(spearman(pred, y), 6),
        "pp_p50": round(float(np.percentile(dpp, 50)), 6),
        "pp_p90": round(float(np.percentile(dpp, 90)), 6),
        "pp_mean": round(float(dpp.mean()), 6),
        "pp_gt1_frac": round(float((dpp > 1).mean()), 6),
        "pp_gt5_frac": round(float((dpp > 5).mean()), 6),
    }


# --------------------------------------------------------------------------- model
class Model:
    """Embedding model with explicit numpy gradients + Adam."""

    def __init__(self, kind, n_ent, n_item, dim, features, seed=0):
        self.kind, self.dim, self.features = kind, dim, features
        self.n_ent, self.n_item = n_ent, n_item
        rng = np.random.default_rng(seed)
        p = {
            "b0": np.zeros(1, np.float32),
            "bu": np.zeros(n_ent, np.float32),
            "bm": np.zeros(n_item, np.float32),
            "P": rng.normal(0, 0.05, (n_ent, dim)).astype(np.float32),
            "C": rng.normal(0, 0.05, (n_item, dim)).astype(np.float32),
        }
        if kind == "mirt":
            p["D"] = rng.normal(0, 0.05, (n_item, dim)).astype(np.float32)
        if features == "F2":
            p["V"] = np.zeros((n_ent, dim), np.float32)
        if features in ("F1", "F2"):
            p["w1"] = np.zeros(1, np.float32)
        self.p = p
        self.g = {k: np.zeros_like(v) for k, v in p.items()}   # reused gradient buffers
        self.m = {k: np.zeros_like(v) for k, v in p.items()}
        self.v = {k: np.zeros_like(v) for k, v in p.items()}
        # only the parameters this model actually uses (the rest are never touched)
        self.active = ["b0", "bu", "bm"]
        if kind != "bias":
            self.active += ["P", "C"]
        if kind == "mirt":
            self.active += ["D"]
        if features == "F2":
            self.active += ["V"]
        if features in ("F1", "F2"):
            self.active += ["w1"]
        self.t = 0
        self.x1 = None
        self.tau = None

    def forward(self, u, m, x1=None, tau=None):
        p = self.p
        P = p["P"][u]
        if self.features == "F2":
            P = P + (tau - 0.5)[:, None] * p["V"][u]
        out = p["b0"][0] + p["bu"][u] + p["bm"][m]
        if self.features in ("F1", "F2"):
            out = out + p["w1"][0] * x1
        if self.kind == "bias":
            return out, (u, m, None, None, None)
        C = p["C"][m]
        D = p["D"][m] if self.kind == "mirt" else None
        if self.kind == "mf_dot":
            out = out + (P * C).sum(1)
        else:
            out = out + ((P - C) * D).sum(1)
        return out, (u, m, P, C, D)

    def backward(self, ctx, e):
        """Exact MSE gradients, written in place into self.g.

        The embedding gradients are one *flattened* ``bincount`` per parameter (index
        ``row*dim + col``) instead of one bincount per dimension: same maths, ~16x fewer
        numpy calls, which is what makes the 10k run tractable.
        """
        u, m, P, C, D = ctx
        g = self.g
        ee = (e * (2.0 / len(e))).astype(np.float32)
        g["b0"][0] = ee.sum()
        g["bu"][:] = np.bincount(u, weights=ee, minlength=self.n_ent)
        g["bm"][:] = np.bincount(m, weights=ee, minlength=self.n_item)
        if self.features in ("F1", "F2"):
            g["w1"][0] = (ee * self.x1).sum()
        if self.kind == "bias":
            return g
        dim, ar = self.dim, np.arange(self.dim)
        fu = (u[:, None] * dim + ar[None, :]).ravel()
        fi = (m[:, None] * dim + ar[None, :]).ravel()
        if self.kind == "mirt":
            R = P - C                                   # dL/dD = R * ee,  dL/dC = -D * ee
            g["C"][:] = -np.bincount(fi, weights=(ee[:, None] * D).ravel(),
                                     minlength=self.n_item * dim).reshape(self.n_item, dim)
            g["D"][:] = np.bincount(fi, weights=(ee[:, None] * R).ravel(),
                                    minlength=self.n_item * dim).reshape(self.n_item, dim)
            Wp = (ee[:, None] * D).ravel()              # dL/dP = D * ee
        else:                                           # mf_dot
            g["C"][:] = np.bincount(fi, weights=(ee[:, None] * P).ravel(),
                                    minlength=self.n_item * dim).reshape(self.n_item, dim)
            Wp = (ee[:, None] * C).ravel()
        g["P"][:] = np.bincount(fu, weights=Wp, minlength=self.n_ent * dim).reshape(self.n_ent, dim)
        if self.features == "F2":
            dt = (self.tau - 0.5).astype(np.float32)
            g["V"][:] = np.bincount(fu, weights=(ee[:, None] * (dt[:, None] * D)).ravel(),
                                    minlength=self.n_ent * dim).reshape(self.n_ent, dim)
        return g

    def adam(self, lr, wd, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        inv1 = 1.0 / (1 - b1 ** self.t)
        s2 = np.sqrt(1 - b2 ** self.t)
        for k in self.active:
            p, g = self.p[k], self.g[k]
            if wd:
                g += (wd * p).astype(np.float32)
            mm, vv = self.m[k], self.v[k]
            mm *= b1
            mm += (1 - b1) * g
            vv *= b2
            vv += (1 - b2) * g * g
            p -= (lr * (mm * inv1) / (np.sqrt(vv) / s2 + eps)).astype(np.float32)

    def predict(self, u, m, x1=None, tau=None):
        return self.forward(u, m, x1, tau)[0]


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="1k")
    ap.add_argument("--target", default="loss")
    ap.add_argument("--models", default="bias,mf_dot,mirt")
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--features", default="F0", choices=["F0", "F1", "F2"])
    ap.add_argument("--entity", default="player", choices=["player", "player-year"])
    ap.add_argument("--pair-universe", default="step0", choices=["step0", "step1"],
                    help="which table's (player_id, beatmap_id) pairs the 30%% split is drawn from")
    ap.add_argument("--test-pair-frac", type=float, default=0.30)
    ap.add_argument("--split-seed", type=int, default=20260928)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", default="step2")
    ap.add_argument("--eval-every", type=int, default=20)
    args = ap.parse_args()

    t_start = time.time()
    tr_all = load_play_table(args.tag, "step1")
    te_all = load_play_table(args.tag, "step0")
    universe = te_all if args.pair_universe == "step0" else tr_all
    k = int(universe.beatmap_id.max()) + 1

    # ---------------------------------------------------------------- split (by pair)
    ukey = np.unique(pair_keys(universe.player_id.to_numpy(), universe.beatmap_id.to_numpy(), k))
    rng = np.random.default_rng(args.split_seed)
    is_test = rng.random(ukey.size) < args.test_pair_frac
    print(f"pair universe = {args.pair_universe}: {ukey.size:,} ordered pairs -> test "
          f"{int(is_test.sum()):,} ({is_test.mean():.2%})", flush=True)

    def test_flag(df):
        """True iff the row's pair is one of the held-out pairs.

        With --pair-universe step0 every key is present by construction.  With `step1` the
        step0 table also holds pairs that never survived cleaning; those are outside the
        universe and simply are not test rows (they are reported as `rows_outside_universe`).
        """
        keys = pair_keys(df.player_id.to_numpy(), df.beatmap_id.to_numpy(), k)
        pos = np.searchsorted(ukey, keys)
        found = ukey[pos] == keys
        return np.where(found, is_test[pos], False), int((~found).sum())

    tr_flag, tr_out = test_flag(tr_all)                  # step1 rows of a *test* pair
    te_flag, te_out = test_flag(te_all)                  # step0 rows of a *test* pair
    if te_out:
        print(f"note: {te_out:,} step0 rows are outside the '{args.pair_universe}' pair universe "
              f"and are neither train nor test", flush=True)
    train = tr_all[~tr_flag].reset_index(drop=True)      # train plays  (step1, 70% pairs)
    test = te_all[te_flag].reset_index(drop=True)        # test plays   (step0, 30% pairs)
    test_clean = tr_all[tr_flag].reset_index(drop=True)  # secondary: same pairs, step1 labels
    del tr_all, te_all
    print(f"train plays = {len(train):,} (step1)   test plays = {len(test):,} (step0)   "
          f"test pairs' step1 rows = {len(test_clean):,}", flush=True)

    # ------------------------------------------------------------- entity / item ids
    def add_entity(df):
        if args.entity == "player-year":
            y = pd.to_datetime(df.timestamp).dt.year.to_numpy()
            return df.assign(entity=df.player_id.to_numpy(np.int64) * 100 + (y - 2000))
        return df.assign(entity=df.player_id.to_numpy(np.int64))

    train, test, test_clean = add_entity(train), add_entity(test), add_entity(test_clean)
    ents = {e: i for i, e in enumerate(np.unique(train.entity.to_numpy()))}
    items = {m: i for i, m in enumerate(np.unique(train.beatmap_id.to_numpy()))}
    print(f"train vocab: {len(ents):,} entities, {len(items):,} items", flush=True)

    def encode(df):
        e = df.entity.map(ents).fillna(-1).astype(np.int64).to_numpy()
        i = df.beatmap_id.map(items).fillna(-1).astype(np.int64).to_numpy()
        x1 = np.log(df.playcount_cur.to_numpy(np.float64)).astype(np.float32) \
            if args.features in ("F1", "F2") else None
        tau = ((pd.to_datetime(df.timestamp).dt.dayofyear.to_numpy() - 1) / 365.0).astype(np.float32) \
            if args.features == "F2" else None
        return e, i, x1, tau, df[args.target].to_numpy(np.float32)

    tr_e, tr_i, tr_x, tr_tau, tr_y = encode(train)
    te_e, te_i, te_x, te_tau, te_y = encode(test)
    tc_e, tc_i, tc_x, tc_tau, tc_y = encode(test_clean)
    tr_ok, te_ok, tc_ok = (tr_e >= 0) & (tr_i >= 0), (te_e >= 0) & (te_i >= 0), (tc_e >= 0) & (tc_i >= 0)
    print(f"train rows in vocab = {int(tr_ok.sum()):,}/{len(train):,}   "
          f"test rows covered = {int(te_ok.sum()):,}/{len(test):,} ({te_ok.mean():.2%})", flush=True)

    idx_tr = np.flatnonzero(tr_ok)
    results, fitted = {}, {}
    for kind in [s for s in args.models.split(",") if s]:
        mdl = Model(kind, len(ents), len(items), args.dim, args.features, args.seed)
        rng_tr = np.random.default_rng(args.seed)
        t0 = time.time()
        for ep in range(args.epochs):
            perm = rng_tr.permutation(idx_tr.size)
            for s in range(0, perm.size, args.batch):
                b = idx_tr[perm[s:s + args.batch]]
                x1 = tr_x[b] if tr_x is not None else None
                tau = tr_tau[b] if tr_tau is not None else None
                pred, ctx = mdl.forward(tr_e[b], tr_i[b], x1, tau)
                mdl.x1, mdl.tau = x1, tau
                mdl.backward(ctx, pred - tr_y[b])
                mdl.adam(args.lr, args.wd)
            if args.eval_every and (ep + 1) % args.eval_every == 0:
                pv = mdl.predict(te_e[te_ok], te_i[te_ok],
                                 te_x[te_ok] if te_x is not None else None,
                                 te_tau[te_ok] if te_tau is not None else None)
                print(f"  [{kind}] epoch {ep+1:3d}  test RMSE(step0) = "
                      f"{np.sqrt(((pv - te_y[te_ok]) ** 2).mean()):.5f}  ({time.time()-t0:.1f}s)",
                      flush=True)
        pv = mdl.predict(te_e[te_ok], te_i[te_ok],
                         te_x[te_ok] if te_x is not None else None,
                         te_tau[te_ok] if te_tau is not None else None)
        results[f"{kind}|step0"] = report(pv, te_y[te_ok], len(test))
        if tc_ok.any():
            pc = mdl.predict(tc_e[tc_ok], tc_i[tc_ok],
                             tc_x[tc_ok] if tc_x is not None else None,
                             tc_tau[tc_ok] if tc_tau is not None else None)
            results[f"{kind}|step1"] = report(pc, tc_y[tc_ok], len(test_clean))
        fitted[kind] = mdl
        print(f"{kind:8s} step0 rmse={results[f'{kind}|step0']['rmse']:.5f} "
              f"cov={results[f'{kind}|step0']['coverage']:.3f}", flush=True)

    # ------------------------------------------------------------------------- save
    inv_e = {v: kk for kk, v in ents.items()}
    inv_i = {v: kk for kk, v in items.items()}
    variant = args.models.replace(",", "+")
    if args.features != "F0":
        variant += f"_{args.features}"
    if args.entity != "player":
        variant += f"_{args.entity.replace('player-year', 'py')}"
    if args.pair_universe != "step0":
        variant += f"_{args.pair_universe}"
    for kind, mdl in fitted.items():
        np.savez(PROC / f"{args.out_prefix}_{args.tag}_{kind}_{args.features}"
                       f"{'' if args.entity == 'player' else '_py'}"
                       f"{'' if args.pair_universe == 'step0' else '_u' + args.pair_universe}"
                       f"_{args.target}.npz",
                 entity_ids=np.array(list(inv_e.values()), dtype=np.int64),
                 beatmap_ids=np.array(list(inv_i.values()), dtype=np.int32),
                 P=mdl.p["P"], V=mdl.p.get("V", np.zeros_like(mdl.p["P"])),
                 C=mdl.p["C"], D=mdl.p.get("D", np.zeros_like(mdl.p["C"])),
                 b0=mdl.p["b0"], bu=mdl.p["bu"], bm=mdl.p["bm"])

    summary = {
        "variant": variant, "args": vars(args),
        "split": {"universe": args.pair_universe, "n_pairs": int(ukey.size),
                  "n_test_pairs": int(is_test.sum()), "test_pair_frac": float(is_test.mean()),
                  "split_seed": args.split_seed},
        "sizes": {"train_plays": int(len(train)), "test_plays": int(len(test)),
                  "test_pairs_step1_rows": int(len(test_clean)),
                  "rows_outside_universe": int(te_out),
                  "n_entities": len(ents), "n_items": len(items),
                  "train_rows_in_vocab": int(tr_ok.sum()),
                  "test_rows_covered": int(te_ok.sum())},
        "results": results,
        "seconds": round(time.time() - t_start, 1),
    }
    merged_path = PROC / f"{args.out_prefix}_{args.tag}_summary_{args.target}.json"
    merged = json.loads(merged_path.read_text()) if merged_path.exists() else {"runs": {}}
    merged["runs"][variant] = summary
    merged_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    print(f"\nwrote npz per model + {merged_path.name} ({summary['seconds']}s total)")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
