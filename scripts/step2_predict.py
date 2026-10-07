#!/usr/bin/env python
"""Step 2 - prediction task:  (player_id, beatmap_id) -> loss.   [torch engine]

Design (docs/step2_prediction.md):

  * observation unit is the **play**; the label is step0's "loss".  No cell table, no
    per-cell medians, no aggregation of any kind.
  * **train** on step1_{tag} (the cleaned play table).
  * **test** on step0_{tag} (the raw play table) -- step1's two rules are a
    *training-side* treatment, so letting them select the test rows would be
    conditioning on the treatment (docs/cleaning_plan.md section 3.2).
  * **split**: a fixed-seed random 30% of the (player_id, beatmap_id) *ordered pairs*
    goes to the test set; the remaining pairs supply the training plays.  A pair is never
    on both sides, so every test pair is unseen in training.

Engine: **PyTorch**.  Forward, backward and the optimiser are autograd + torch.optim.Adam
(the reference numpy implementation -- explicit gradients + hand written Adam -- is kept at
scripts/legacy/step2_predict_numpy.py).  Seeding still uses numpy.random.default_rng(seed),
drawing the same values in the same order as that reference, so a torch run is directly
comparable with the documented numbers.  The nine assertions of docs/step2_prediction.md
section 10 run on every invocation (the npz/summary files are only written when they pass).

    L_hat = b0 + bu[u] + bm[m] + interaction
      bias    : interaction = 0
      mf_dot  : interaction = <P[u], C[m]>
      mirt    : interaction = <P[u] - C[m], D[m]>

    --features F1 adds  w1 * ln(playcount_cur)     (step1's own column, used as-is)
    --features F2 / --entity player-year additionally use
        P_eff = P[e] + (tau - 0.5) * V[e]          e = (player_id, calendar year)

    --p-drift logpc   adds a per-player, per-dimension drift rate Pc (n_ent, dim):
        P_eff = P[u] + Pc[u] * ln(playcount_cur)   x1 = ln(playcount_cur), Pc starts at 0
    --d-constraint    constrains mirt's discrimination vector D (per chart):
        none    : D free (default; current behaviour)
        nonneg  : D = softplus(Z)                  -> D > 0, no sum constraint
        simplex : D = softmax(Z, dim=1)            -> D >= 0 and sum_i D_i = 1 exactly
      The constraint is a reparameterisation: Z are the parameters, the npz stores the
      *effective* D (plus D_raw = Z).  Adam's wd is a constant pull to Z=0, which for a
      softmax is the uniform-D symmetry sink, so a constrained run defaults to
      --d-logit-wd 0 and a spread Z init (--d-init-std).
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

PROC = Path("data/processed")
BASE_COLS = ["player_id", "beatmap_id"]
# Expected loss range of the *uncleaned* test table (step0), per tag -- checklist item 4.
# Values are step0's observed extremes; a tag without an entry is only checked for finiteness.
LOSS_RANGE = {"1k": (-12.4663, -0.2669), "10k": (-12.4663, -0.1358)}


# ----------------------------------------------------------------------------- io
def load_play_table(tag: str, which: str, cols: list) -> pd.DataFrame:
    """which in {'step0','step1'} -> that play table for tag (.parquet preferred)."""
    pq, csv = PROC / f"{which}_{tag}.parquet", PROC / f"{which}_{tag}.csv"
    if pq.exists():
        df, src = pd.read_parquet(pq, columns=cols), pq
    elif csv.exists():
        df, src = pd.read_csv(csv, usecols=cols), csv
    else:
        raise SystemExit(f"neither {pq} nor {csv} exists")
    print(f"[{which}] {src.name}: {len(df):,} rows, {df.player_id.nunique():,} players, "
          f"{df.beatmap_id.nunique():,} beatmaps", flush=True)
    return df


def pair_keys(player, beatmap, k: int) -> np.ndarray:
    """Ordered-pair key, int64.  k = beatmap_id.max()+1 makes the encoding injective."""
    return player.astype(np.int64) * np.int64(k) + beatmap.astype(np.int64)


def map_ids(values: np.ndarray, vocab_ids: np.ndarray) -> np.ndarray:
    """Index into a sorted unique vocab; -1 for ids the training rows never showed.

    Same semantics as pandas Series.map(dict).fillna(-1) in the reference implementation,
    without building a python dict over millions of keys.
    """
    v = values.astype(np.int64, copy=False)
    if vocab_ids.size == 0:
        return np.full(v.shape, -1, dtype=np.int64)
    pos = np.searchsorted(vocab_ids, v)
    pos[pos >= vocab_ids.size] = 0                      # clamp before the membership test
    ok = vocab_ids[pos] == v
    return np.where(ok, pos, np.int64(-1)).astype(np.int64)


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
class EmbeddingModel(nn.Module):
    """b0 + bu[u] + bm[m] + interaction, trained with autograd + torch.optim.Adam.

    Parameter set and initialisation mirror the numpy reference exactly: b0/bu/bm start at
    zero, P/C (and D for mirt) are drawn from default_rng(seed).normal(0, 0.05) in that
    order, V/w1 start at zero, and only the parameters a model actually uses are handed to
    the optimiser (bias never touches P/C/D).

    Kinds with a D vector: "mirt" (linear interaction <P-C, D>) and "mirt_exp"
    (exponent/"ACC-space" interaction  ln( sum_i D_mi * exp(-(P_ui - C_mi)) ) ).  mirt_exp
    needs D > 0 so that the sum -- and therefore its log -- stays positive; free D is not
    usable there (the scale of D is absorbed by b0, so no normalisation is needed).
    """

    def __init__(self, kind, n_ent, n_item, dim, features, seed=0, device=torch.device("cpu"),
                 p_drift="none", d_constraint="none", d_init_std=None, drop_bias="none"):
        super().__init__()
        self.kind, self.dim, self.features = kind, dim, features
        if kind == "mirt_exp" and d_constraint == "none":
            d_constraint = "nonneg"        # ln(sum D e^-S) needs D > 0; scale lives in b0
        self.d_constraint = d_constraint
        self.p_drift, self.d_init_std = p_drift, d_init_std
        self.n_ent, self.n_item = n_ent, n_item
        rng = np.random.default_rng(seed)

        def draw(shape, std=0.05):
            a = np.ascontiguousarray(rng.normal(0, std, shape).astype(np.float32))
            return torch.as_tensor(a, dtype=torch.float32, device=device)

        self.drop_bias = drop_bias
        self.p = {"b0": nn.Parameter(torch.zeros(1, dtype=torch.float32, device=device))}
        if drop_bias not in ("u", "both"):      # b_u: per-entity main effect
            self.p["bu"] = nn.Parameter(torch.zeros(n_ent, dtype=torch.float32, device=device))
        if drop_bias not in ("m", "both"):      # b_m: per-item main effect
            self.p["bm"] = nn.Parameter(torch.zeros(n_item, dtype=torch.float32, device=device))
        self.p["P"] = nn.Parameter(draw((n_ent, dim)))
        self.p["C"] = nn.Parameter(draw((n_item, dim)))
        if kind in ("mirt", "mirt_exp"):
            # simplex starts from a *spread* softmax (all charts identical at Z=0 would be a
            # symmetry sink: the softmax Jacobian is only ~1/dim there, so D never differentiates)
            std = d_init_std if d_init_std is not None else (0.5 if d_constraint == "simplex" else 0.05)
            self.p["D"] = nn.Parameter(draw((n_item, dim), std))
            if d_constraint == "nonneg":
                # start from D ~ 0.05 like the free model: Z = softplus_inv(0.05) + N(0, 0.05)
                with torch.no_grad():
                    self.p["D"].add_(float(np.log(np.expm1(0.05))))
        if p_drift == "logpc" and kind != "bias":
            # drift rate, same shape as P; starts at 0 so the model starts drift-free
            self.p["Pc"] = nn.Parameter(torch.zeros((n_ent, dim), dtype=torch.float32, device=device))
        if features == "F2":
            self.p["V"] = nn.Parameter(torch.zeros((n_ent, dim), dtype=torch.float32, device=device))
        if features in ("F1", "F2"):
            self.p["w1"] = nn.Parameter(torch.zeros(1, dtype=torch.float32, device=device))

        # only the parameters this model actually uses (the rest are never handed to Adam)
        self.active = [k for k in ("b0", "bu", "bm") if k in self.p]
        if kind != "bias":
            self.active += ["P", "C"]
        if kind in ("mirt", "mirt_exp"):
            self.active += ["D"]
        if p_drift == "logpc" and kind != "bias":
            self.active += ["Pc"]
        if features == "F2":
            self.active += ["V"]
        if features in ("F1", "F2"):
            self.active += ["w1"]

    def D_matrix(self, m):
        """Effective D for item ids m: the raw parameter, or its constrained transform."""
        Z = self.p["D"][m]
        if self.d_constraint == "simplex":
            return torch.softmax(Z, dim=1)          # D >= 0, sum_i D_i = 1
        if self.d_constraint == "nonneg":
            return F.softplus(Z)                    # D > 0
        return Z

    def forward(self, u, m, x1=None, tau=None):
        p = self.p
        P = p["P"][u]
        if self.p_drift == "logpc":
            P = P + p["Pc"][u] * x1.unsqueeze(1)    # per-player drift along ln(playcount_cur)
        if self.features == "F2":
            P = P + (tau - 0.5).unsqueeze(1) * p["V"][u]
        # materialise a (B,) vector: with every intercept dropped the model can be a bare scalar
        out = p["b0"][0] + torch.zeros(u.shape[0], dtype=torch.float32, device=u.device)
        if "bu" in p:
            out = out + p["bu"][u]
        if "bm" in p:
            out = out + p["bm"][m]
        if self.features in ("F1", "F2"):
            out = out + p["w1"][0] * x1
        if self.kind == "bias":
            return out
        C = p["C"][m]
        if self.kind == "mf_dot":
            return out + (P * C).sum(1)
        D = self.D_matrix(m)
        if self.kind == "mirt_exp":
            # l = b0 + ln( sum_i D_mi * exp(-(P_ui - C_mi)) );  computed as a stable logsumexp.
            # D > 0 (softplus) keeps the sum positive; exp() is the "restore to ACC space".
            return out + torch.logsumexp(torch.log(D) - (P - C), dim=1)
        return out + ((P - C) * D).sum(1)

    @torch.no_grad()
    def predict(self, u, m, x1=None, tau=None) -> np.ndarray:
        return self.forward(u, m, x1, tau).detach().to("cpu").numpy()

    def np_array(self, name, fallback=None):
        """Parameter as a numpy array; fallback (e.g. zeros_like) when the model has none."""
        if name in self.p:
            return self.p[name].detach().to("cpu").numpy()
        return fallback

    def export_D(self) -> np.ndarray:
        """Effective D (after the constraint transform) for the whole chart vocabulary."""
        Z = self.p["D"].detach()
        if self.d_constraint == "simplex":
            Z = torch.softmax(Z, dim=1)
        elif self.d_constraint == "nonneg":
            Z = F.softplus(Z)
        return Z.to("cpu").numpy()


# ------------------------------------------------------------------------ checklist
class Checklist:
    """docs/step2_prediction.md section 10 -- run on every invocation."""

    def __init__(self):
        self.items = []

    def check(self, name, ok, detail=""):
        ok = bool(ok)
        self.items.append({"name": name, "ok": ok, "detail": detail})
        print(f"  [check] {'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
        return ok

    def require(self):
        bad = [c["name"] for c in self.items if not c["ok"]]
        if bad:
            raise AssertionError(f"validation checklist failed: {', '.join(bad)}")


def determinism_selftest(device) -> bool:
    """Checklist item 8: the training path is fully seeded -- same seed -> bitwise equal."""

    def run_once():
        m = EmbeddingModel("mirt", 24, 17, 4, "F2", seed=0, device=device,
                           p_drift="logpc", d_constraint="simplex")
        opt = torch.optim.Adam([m.p[k] for k in m.active], lr=0.02, weight_decay=1e-4)
        r = np.random.default_rng(7)
        u = torch.as_tensor(r.integers(0, 24, 96), dtype=torch.long, device=device)
        i = torch.as_tensor(r.integers(0, 17, 96), dtype=torch.long, device=device)
        y = torch.as_tensor(r.normal(-3, 1, 96).astype(np.float32), device=device)
        x1 = torch.as_tensor(r.normal(1, 0.5, 96).astype(np.float32), device=device)
        tau = torch.as_tensor(r.random(96).astype(np.float32), device=device)
        rng = np.random.default_rng(3)
        for _ in range(3):
            for s in range(0, 96, 32):
                b = torch.as_tensor(rng.permutation(96)[s:s + 32], dtype=torch.long, device=device)
                opt.zero_grad(set_to_none=True)
                ((m.forward(u[b], i[b], x1[b], tau[b]) - y[b]) ** 2).mean().backward()
                opt.step()
        return {k: m.p[k].detach().clone() for k in m.p}

    a, b = run_once(), run_once()
    return all(torch.equal(a[k], b[k]) for k in a)


# ---------------------------------------------------------------------------- main
def parse_args(argv=None):
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
    ap.add_argument("--p-drift", default="none", choices=["none", "logpc"],
                    help="player-side drift: P_eff = P[u] + Pc[u] * ln(playcount_cur), Pc starts at 0")
    ap.add_argument("--d-constraint", default="none", choices=["none", "nonneg", "simplex"],
                    help="mirt's D: none (free) / nonneg (softplus, D>0) / "
                         "simplex (softmax, D>=0 and sum_i D_i = 1)")
    ap.add_argument("--d-logit-wd", type=float, default=None,
                    help="weight decay for the raw D logits.  Default: 0 when a D constraint is "
                         "active (Adam's wd is a constant pull to Z=0, i.e. uniform D -- a symmetry "
                         "sink that kills the interaction), else --wd")
    ap.add_argument("--drop-bias", default="none", choices=["none", "u", "m", "both"],
                    help="drop the main-effect intercepts: u = no b_u (per player), m = no b_m "
                         "(per chart), both = keep only the global b0")
    ap.add_argument("--d-init-std", type=float, default=None,
                    help="std of the raw D init; default: 0.05 (free/nonneg), 0.5 (simplex, to break "
                         "the uniform-D symmetry sink)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="torch device; 'auto' picks cuda when available, else cpu")
    ap.add_argument("--threads", type=int, default=0,
                    help="torch.set_num_threads(); 0 = leave the default.  Reproducibility "
                         "across machines needs the same thread count (parallel reductions)")
    return ap.parse_args(argv)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")
    return torch.device(name)


def main():
    args = parse_args()
    t_start = time.time()
    device = pick_device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    print(f"engine: torch {torch.__version__} on {device}, threads={torch.get_num_threads()}, "
          f"deterministic_algorithms={torch.are_deterministic_algorithms_enabled()}", flush=True)
    models_wanted = [s for s in args.models.split(",") if s]
    if args.p_drift != "none" or args.d_constraint != "none" or "mirt_exp" in models_wanted:
        print(f"mechanism: p_drift={args.p_drift} (P_eff = P + Pc*ln(playcount_cur))  "
              f"d_constraint={args.d_constraint}  d_logit_wd={args.d_logit_wd}", flush=True)
    if "mirt_exp" in models_wanted:
        print("note: mirt_exp = b0 + ln(sum_i D_mi*exp(-(P_ui-C_mi))); D is forced positive "
              "(softplus) because a free D can make the sum negative, and its log undefined",
              flush=True)

    need_pc = args.features in ("F1", "F2") or args.p_drift == "logpc"
    need_ts = args.features == "F2" or args.entity == "player-year"
    cols = BASE_COLS + [args.target] \
        + (["playcount_cur"] if need_pc else []) + (["timestamp"] if need_ts else [])

    tr_all = load_play_table(args.tag, "step1", cols)
    te_all = load_play_table(args.tag, "step0", cols)
    universe = te_all if args.pair_universe == "step0" else tr_all
    k = int(universe.beatmap_id.max()) + 1

    # ---------------------------------------------------------------- split (by pair)
    ukey = np.unique(pair_keys(universe.player_id.to_numpy(), universe.beatmap_id.to_numpy(), k))
    rng = np.random.default_rng(args.split_seed)
    is_test = rng.random(ukey.size) < args.test_pair_frac
    n_test_pairs = int(is_test.sum())
    print(f"pair universe = {args.pair_universe}: {ukey.size:,} ordered pairs -> test "
          f"{n_test_pairs:,} ({is_test.mean():.2%})", flush=True)

    def test_flag(df):
        """True iff the row's pair is one of the held-out pairs.

        With --pair-universe step0 every key is present by construction.  With step1 the
        step0 table also holds pairs that never survived cleaning; those are outside the
        universe and simply are not test rows (they are reported as rows_outside_universe).
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
    gc.collect()
    print(f"train plays = {len(train):,} (step1)   test plays = {len(test):,} (step0)   "
          f"test pairs' step1 rows = {len(test_clean):,}", flush=True)

    # ------------------------------------------------------------- checklist 1-3 (split)
    chk = Checklist()
    n_pairs = int(ukey.size)
    test_pairs = ukey[is_test]
    tr_pairs = np.unique(pair_keys(train.player_id.to_numpy(), train.beatmap_id.to_numpy(), k))
    shared = np.intersect1d(tr_pairs, test_pairs, assume_unique=True)
    chk.check("pair_never_crosses_split", shared.size == 0,
              f"{shared.size} ordered pairs on both sides (train {tr_pairs.size:,} / "
              f"test {test_pairs.size:,} unique pairs)")
    tr_keys = pair_keys(train.player_id.to_numpy(), train.beatmap_id.to_numpy(), k)
    te_keys = pair_keys(test.player_id.to_numpy(), test.beatmap_id.to_numpy(), k)
    tr_in_test = np.isin(tr_keys, test_pairs)
    te_in_test = np.isin(te_keys, test_pairs)
    chk.check("train_and_test_disjoint",
              (not tr_in_test.any()) and bool(te_in_test.all()),
              f"train rows on a held-out pair = {int(tr_in_test.sum()):,}; "
              f"test rows on a held-out pair = {int(te_in_test.sum()):,}/{len(test):,}")
    chk.check("test_pair_frac", abs(float(is_test.mean()) - args.test_pair_frac) <= 0.01,
              f"{is_test.mean():.4%} vs target {args.test_pair_frac:.2%} (tolerance 1%)")
    del tr_keys, te_keys, tr_in_test, te_in_test, shared
    gc.collect()

    # ------------------------------------------------------------- entity / item ids
    def add_entity(df):
        if args.entity == "player-year":
            y = pd.to_datetime(df.timestamp).dt.year.to_numpy()
            return df.assign(entity=df.player_id.to_numpy(np.int64) * 100 + (y - 2000))
        return df.assign(entity=df.player_id.to_numpy(np.int64))

    train, test, test_clean = add_entity(train), add_entity(test), add_entity(test_clean)
    ent_ids = np.unique(train.entity.to_numpy())
    item_ids = np.unique(train.beatmap_id.to_numpy())
    ents = {e: i for i, e in enumerate(ent_ids)}
    items = {m: i for i, m in enumerate(item_ids)}
    print(f"train vocab: {len(ents):,} entities, {len(items):,} items", flush=True)

    def encode(df):
        e = map_ids(df.entity.to_numpy(), ent_ids)
        i = map_ids(df.beatmap_id.to_numpy(), item_ids)
        x1 = np.log(df.playcount_cur.to_numpy(np.float64)).astype(np.float32) if need_pc else None
        tau = ((pd.to_datetime(df.timestamp).dt.dayofyear.to_numpy() - 1) / 365.0).astype(np.float32) \
            if args.features == "F2" else None
        return e, i, x1, tau, df[args.target].to_numpy(np.float32)

    tr_e, tr_i, tr_x, tr_tau, tr_y = encode(train)
    te_e, te_i, te_x, te_tau, te_y = encode(test)
    tc_e, tc_i, tc_x, tc_tau, tc_y = encode(test_clean)
    tr_ok = (tr_e >= 0) & (tr_i >= 0)
    te_ok = (te_e >= 0) & (te_i >= 0)
    tc_ok = (tc_e >= 0) & (tc_i >= 0)
    print(f"train rows in vocab = {int(tr_ok.sum()):,}/{len(train):,}   "
          f"test rows covered = {int(te_ok.sum()):,}/{len(test):,} ({te_ok.mean():.2%})", flush=True)

    # ------------------------------------------------- checklist 4, 6, 8 (data + seeds)
    if args.tag in LOSS_RANGE:
        lo, hi = LOSS_RANGE[args.tag]
        loss_ok = bool(np.isfinite(te_y).all() and te_y.min() >= lo - 1e-3 and te_y.max() <= hi + 1e-3)
        loss_detail = f"test({args.tag}) loss in [{te_y.min():.4f}, {te_y.max():.4f}], expected [{lo}, {hi}]"
    else:
        loss_ok = bool(np.isfinite(te_y).all() and (te_y <= 0).all())
        loss_detail = f"test({args.tag}) loss in [{te_y.min():.4f}, {te_y.max():.4f}] (no per-tag range registered)"
    chk.check("loss_in_range", loss_ok, loss_detail)
    chk.check("vocab_from_train_only",
              np.array_equal(ent_ids, np.unique(train.entity.to_numpy()))
              and np.array_equal(item_ids, np.unique(train.beatmap_id.to_numpy())),
              f"{ent_ids.size:,} entities / {item_ids.size:,} items == unique train ids")
    chk.check("determinism_selftest", determinism_selftest(device),
              "same seed -> bitwise identical parameters on a synthetic training path")

    # ----------------------------------------------------------------------- tensors
    def dev(a):
        return torch.as_tensor(a, device=device)

    tr_e_t, tr_i_t = dev(tr_e), dev(tr_i)
    tr_y_t = dev(tr_y)
    tr_x_t = dev(tr_x) if tr_x is not None else None
    tr_tau_t = dev(tr_tau) if tr_tau is not None else None
    te_e_t, te_i_t = dev(te_e[te_ok]), dev(te_i[te_ok])
    te_y_np = te_y[te_ok]
    te_x_t = dev(te_x[te_ok]) if te_x is not None else None
    te_tau_t = dev(te_tau[te_ok]) if te_tau is not None else None
    tc_e_t, tc_i_t = dev(tc_e[tc_ok]), dev(tc_i[tc_ok])
    tc_y_np = tc_y[tc_ok]
    tc_x_t = dev(tc_x[tc_ok]) if tc_x is not None else None
    tc_tau_t = dev(tc_tau[tc_ok]) if tc_tau is not None else None
    idx_tr = np.flatnonzero(tr_ok).astype(np.int64)
    del tr_e, tr_i, tr_x, tr_tau, tr_y, te_e, te_i, te_x, te_tau, te_y, tc_e, tc_i, tc_x, tc_tau, tc_y
    gc.collect()

    # ---------------------------------------------------------------------- training
    results, fitted, mech_by_model = {}, {}, {}
    for kind in models_wanted:
        model = EmbeddingModel(kind, len(ents), len(items), args.dim, args.features,
                               args.seed, device, args.p_drift, args.d_constraint,
                               args.d_init_std, args.drop_bias)
        if args.p_drift != "none" and kind == "bias":
            print("note: --p-drift has no effect on bias (it has no P vector)", flush=True)
        # per model: a constrained D gets no wd on its logits by default (Adam's wd pulls the
        # logits to Z=0, i.e. uniform D / D->0 -- a degenerate direction for both links)
        model_d_wd = args.d_logit_wd if args.d_logit_wd is not None else \
            (0.0 if model.d_constraint != "none" else args.wd)
        mech_by_model[kind] = {"d_constraint": model.d_constraint, "d_logit_wd": model_d_wd}
        if "D" in model.active and (model.d_constraint != "none" or args.d_logit_wd is not None):
            print(f"  [{kind}] d_constraint={model.d_constraint}  d_logit_wd={model_d_wd:g}",
                  flush=True)
        if model_d_wd == args.wd:
            opt = torch.optim.Adam([model.p[k] for k in model.active],
                                   lr=args.lr, weight_decay=args.wd)
        else:   # per-group wd: the raw D logits get their own decay (see --d-logit-wd)
            groups = [{"params": [model.p[k] for k in model.active if k != "D"],
                       "weight_decay": args.wd}]
            if "D" in model.active:
                groups.append({"params": [model.p["D"]], "weight_decay": model_d_wd})
            opt = torch.optim.Adam(groups, lr=args.lr)
        rng_tr = np.random.default_rng(args.seed)
        t0 = time.time()
        for ep in range(args.epochs):
            perm = rng_tr.permutation(idx_tr.size)
            for s in range(0, perm.size, args.batch):
                b = torch.as_tensor(idx_tr[perm[s:s + args.batch]], device=device)
                x1 = tr_x_t[b] if tr_x_t is not None else None
                tau = tr_tau_t[b] if tr_tau_t is not None else None
                opt.zero_grad(set_to_none=True)
                pred = model.forward(tr_e_t[b], tr_i_t[b], x1, tau)
                ((pred - tr_y_t[b]) ** 2).mean().backward()
                opt.step()
            if args.eval_every and (ep + 1) % args.eval_every == 0:
                pv = model.predict(te_e_t, te_i_t, te_x_t, te_tau_t)
                print(f"  [{kind}] epoch {ep+1:3d}  test RMSE(step0) = "
                      f"{np.sqrt(((pv - te_y_np) ** 2).mean()):.5f}  ({time.time()-t0:.1f}s)",
                      flush=True)
        pv = model.predict(te_e_t, te_i_t, te_x_t, te_tau_t)
        results[f"{kind}|step0"] = report(pv, te_y_np, len(test))
        if tc_ok.any():
            pc = model.predict(tc_e_t, tc_i_t, tc_x_t, tc_tau_t)
            results[f"{kind}|step1"] = report(pc, tc_y_np, len(test_clean))
        fitted[kind] = model
        print(f"{kind:8s} step0 rmse={results[f'{kind}|step0']['rmse']:.5f} "
              f"cov={results[f'{kind}|step0']['coverage']:.3f}", flush=True)

    # ----------------------------------------------- checklist 5, 7 (coverage / leak)
    chk.check("coverage_reported",
              all(("coverage" in r) and ("n_scored" in r) and r["coverage"] is not None
                  for r in results.values()),
              f"{len(results)} result blocks carry coverage + n_scored")
    leak = np.isin(pair_keys(train.player_id.to_numpy()[idx_tr],
                             train.beatmap_id.to_numpy()[idx_tr], k), test_pairs).sum()
    chk.check("no_test_leak", bool(leak == 0 and idx_tr.size == int(tr_ok.sum())),
              f"training rows used = {idx_tr.size:,}; of those on a held-out pair = {int(leak):,}; "
              f"test tensors never enter the optimiser")
    del test_pairs, tr_pairs
    gc.collect()

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
    if args.p_drift != "none":
        variant += "_pd"
    if args.d_constraint != "none":
        variant += f"_d{args.d_constraint}"
    if args.d_init_std is not None:
        variant += f"_di{args.d_init_std}"
    if args.d_logit_wd is not None:
        variant += f"_dlw{args.d_logit_wd}"
    if args.drop_bias != "none":
        variant += {"u": "_nobu", "m": "_nobm", "both": "_nobubm"}[args.drop_bias]
    if args.dim != 16:
        variant += f"_dim{args.dim}"
    entity_ids = np.array(list(inv_e.values()), dtype=np.int64)
    beatmap_ids = np.array(list(inv_i.values()), dtype=np.int32)
    npz_paths = {}
    for kind, model in fitted.items():
        path = PROC / f"{args.out_prefix}_{args.tag}_{kind}_{args.features}" \
                      f"{'' if args.dim == 16 else '_dim' + str(args.dim)}" \
                      f"{'' if args.p_drift == 'none' else '_pd'}" \
                      f"{'' if args.d_constraint == 'none' else '_d' + args.d_constraint}" \
                      f"{'' if args.entity == 'player' else '_py'}" \
                      f"{'' if args.pair_universe == 'step0' else '_u' + args.pair_universe}" \
                      f"_{args.target}.npz"
        extra = {}
        if "Pc" in model.p:
            extra["Pc"] = model.np_array("Pc")
        if model.d_constraint != "none":
            extra["D_raw"] = model.np_array("D")        # logits, so the constraint is reproducible
        np.savez(path,
                 entity_ids=entity_ids,
                 beatmap_ids=beatmap_ids,
                 P=model.np_array("P"),
                 V=model.np_array("V", np.zeros_like(model.np_array("P"))),
                 C=model.np_array("C"),
                 # effective D (constraint applied); bias/mf_dot have no D -> zeros like C
                 D=model.export_D() if "D" in model.p else np.zeros_like(model.np_array("C")),
                 b0=model.np_array("b0"),
                 bu=model.np_array("bu", np.zeros(len(ents), np.float32)),
                 bm=model.np_array("bm", np.zeros(len(items), np.float32)),
                 **extra)
        npz_paths[kind] = path

    # ------------------------------------------------------------- checklist 9 (npz)
    ok9, detail9 = True, []
    for kind, path in npz_paths.items():
        with np.load(path) as z:
            good = (len(z["entity_ids"]) == len(ents) and len(z["beatmap_ids"]) == len(items)
                    and z["P"].shape == (len(ents), args.dim)
                    and z["C"].shape == (len(items), args.dim)
                    and z["D"].shape == z["C"].shape and z["V"].shape == z["P"].shape
                    and z["b0"].shape == (1,) and len(z["bu"]) == len(ents)
                    and len(z["bm"]) == len(items)
                    and (("Pc" not in z.files) or z["Pc"].shape == z["P"].shape)
                    and (("D_raw" not in z.files) or z["D_raw"].shape == z["C"].shape))
        ok9 = ok9 and bool(good)
        detail9.append(f"{kind}:{'ok' if good else 'BAD'}")
    chk.check("npz_matches_summary", ok9, " ".join(detail9))

    okc, detailc = True, []
    for kind, path in npz_paths.items():
        if args.d_constraint == "none" or "D" not in np.load(path).files:
            continue
        with np.load(path) as z:
            D = z["D"]
            good = bool((D >= 0).all())
            if args.d_constraint == "simplex":
                good = good and bool(np.allclose(D.sum(1), 1.0, atol=1e-5))
                detailc.append(f"{kind}: D>=0, max|sum_i D_i - 1| = "
                               f"{np.abs(D.sum(1) - 1).max():.2e}")
            else:
                detailc.append(f"{kind}: D>0, min D = {D.min():.3e}")
        okc = okc and good
    if detailc:
        chk.check("d_constraint_holds", okc, " ".join(detailc))
    chk.require()

    summary = {
        "variant": variant, "args": vars(args),
        "mechanism": {"p_drift": args.p_drift, "d_constraint": args.d_constraint,
                      "d_init_std": args.d_init_std if args.d_init_std is not None
                      else (0.5 if args.d_constraint == "simplex" else 0.05),
                      "by_model": mech_by_model},
        "engine": {"name": "torch", "version": torch.__version__, "device": str(device),
                   "threads": torch.get_num_threads(),
                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
        "split": {"universe": args.pair_universe, "n_pairs": n_pairs,
                  "n_test_pairs": n_test_pairs, "test_pair_frac": float(is_test.mean()),
                  "split_seed": args.split_seed},
        "sizes": {"train_plays": int(len(train)), "test_plays": int(len(test)),
                  "test_pairs_step1_rows": int(len(test_clean)),
                  "rows_outside_universe": int(te_out),
                  "n_entities": len(ents), "n_items": len(items),
                  "train_rows_in_vocab": int(tr_ok.sum()),
                  "test_rows_covered": int(te_ok.sum())},
        "results": results,
        "checks": chk.items,
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
