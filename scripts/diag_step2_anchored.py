#!/usr/bin/env python
"""Does the interactive latent model carry anything a *content-anchored* scalar cannot?

Rationale.  The diagnostics show the fitted 16-dim interaction is effectively rank ~2-3 and
that ||D_m|| correlates 0.74 with the official star rating -- i.e. this is close to a
one-difficulty-axis x one-skill model with a mild interaction.  If so, a simple anchored
model, whose chart side is a function of *content features* instead of free per-chart
parameters, should already capture most of the predictive gain -- with ~20x fewer
parameters, and with a chart representation that (a) exists for unseen charts and (b) has
an interpretable scale (which is exactly what step 3 wants).

Models compared on the step2 split (test = step0 rows of held-out pairs):

  A. chart-bias only                      mu + b_m
  B. saturated additive                   mu + b_u + b_m            (free per-chart params)
  C. anchored additive                    mu + g(features_m) + b_u
  D. anchored + one interaction axis      mu + g_m + b_u + b_u * q_m
  E. per-chart random slope on ability    (C, but q_m free per chart = rank-1 MF)

g_m and q_m are linear in chart content features (standardised, with an intercept), and q_m
is squashed to a modest range so the interaction cannot blow up.  All models are fitted with
torch + Adam on the *training* plays only and scored on the same 418,856 test plays.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROC = Path("data/processed")
TAG = "1k"
SEED = 20260928
FRAC = 0.30
FEATS = ["star", "bpm", "max_combo", "count_total", "diff_overall", "diff_drain", "hit_length"]


class AnchoredModel(torch.nn.Module):
    def __init__(self, n_ent: int, feats: np.ndarray, mode: str, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.mode = mode
        n_ent, n_feat = n_ent, feats.shape[1]
        self.register_buffer("F", torch.as_tensor(feats, dtype=torch.float32))
        self.mu = torch.nn.Parameter(torch.tensor(float(feats.shape[0]) * 0.0 - 5.0))
        self.bu = torch.nn.Parameter(torch.zeros(n_ent))
        self.w = torch.nn.Parameter(torch.zeros(n_feat))
        self.q = torch.nn.Parameter(torch.zeros(n_feat))
        if mode == "q_free":                      # per-chart free interaction slope
            self.q_free = torch.nn.Parameter(torch.zeros(feats.shape[0]))

    def chart_terms(self, m):
        g = self.F[m] @ self.w
        if self.mode == "q_free":
            q = torch.tanh(self.q_free[m]) * 0.5
        else:
            q = torch.tanh(self.F[m] @ self.q) * 0.5
        return g, q

    def forward(self, u, m):
        g, q = self.chart_terms(m)
        out = self.mu + g
        if self.mode != "chart_only":
            out = out + self.bu[u]
        if self.mode in ("anchored_inter", "q_free"):
            out = out + self.bu[u] * q
        return out


def main() -> None:
    cols = ["player_id", "beatmap_id", "playcount_cur", "loss"]
    s1 = pd.read_parquet(PROC / f"step1_{TAG}.parquet", columns=cols)
    s0 = pd.read_csv(PROC / f"step0_{TAG}.csv", usecols=cols)
    k = int(max(s0.beatmap_id.max(), s1.beatmap_id.max())) + 1
    for df in (s0, s1):
        df["pair"] = df.player_id.astype(np.int64) * k + df.beatmap_id.astype(np.int64)
    upairs = np.unique(s0.pair.to_numpy())
    is_test = np.random.default_rng(SEED).random(len(upairs)) < FRAC
    tps = set(upairs[is_test].tolist())
    tr = s1[~s1.pair.isin(tps)]
    te = s0[s0.pair.isin(tps)]

    ent_ids = np.unique(tr.player_id.to_numpy())
    item_ids = np.unique(tr.beatmap_id.to_numpy())
    e_pos = {v: i for i, v in enumerate(ent_ids)}
    i_pos = {v: i for i, v in enumerate(item_ids)}

    meta = pd.read_csv(PROC / f"beatmap_meta_{TAG}.csv", usecols=["beatmap_id"] + FEATS)
    meta = meta.set_index("beatmap_id").reindex(item_ids)
    print(f"charts {len(item_ids):,}; metadata missing for {int(meta.isna().any(axis=1).sum()):,} "
          f"charts -> median-imputed")
    meta = meta.fillna(meta.median(numeric_only=True))
    F = meta[FEATS].to_numpy(np.float64)
    F = (F - F.mean(0)) / (F.std(0) + 1e-12)

    tu_tr = tr.player_id.map(e_pos).to_numpy()
    tm_tr = tr.beatmap_id.map(i_pos).to_numpy()
    y_tr = tr.loss.to_numpy(np.float32)
    tu_te = te.player_id.map(e_pos).to_numpy()
    tm_te = te.beatmap_id.map(i_pos).to_numpy()
    ok = pd.notna(tu_te) & pd.notna(tm_te)
    tu_te = tu_te[ok].astype(int)
    tm_te = tm_te[ok].astype(int)
    y_te = te.loss.to_numpy(np.float32)[ok]
    print(f"train {len(tr):,} / test {ok.sum():,}")

    U_tr = torch.as_tensor(tu_tr, dtype=torch.long)
    M_tr = torch.as_tensor(tm_tr, dtype=torch.long)
    Y_tr = torch.as_tensor(y_tr)
    U_te = torch.as_tensor(tu_te, dtype=torch.long)
    M_te = torch.as_tensor(tm_te, dtype=torch.long)

    out = {}
    mus = ["chart_only", "anchored", "anchored_inter", "q_free"]
    for mode in mus:
        model = AnchoredModel(len(ent_ids), F, mode, seed=0)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=0.05, weight_decay=1e-5)
        rng = np.random.default_rng(0)
        n = len(tr)
        bs = 16384
        for ep in range(60):
            perm = rng.permutation(n)
            for s in range(0, n, bs):
                b = torch.as_tensor(perm[s:s + bs], dtype=torch.long)
                opt.zero_grad(set_to_none=True)
                pred = model(U_tr[b], M_tr[b])
                ((pred - Y_tr[b]) ** 2).mean().backward()
                opt.step()
        with torch.no_grad():
            pv = model(U_te, M_te).numpy()
        rmse = float(np.sqrt(np.mean((pv - y_te) ** 2)))
        n_par = sum(p.numel() for p in params)
        out[mode] = dict(rmse=rmse, n_params=int(n_par))
        print(f"  {mode:16s} test play RMSE = {rmse:.4f}   params = {n_par:,}")
        if mode in ("anchored", "anchored_inter"):
            with torch.no_grad():
                g, q = model.chart_terms(torch.arange(len(item_ids)))
            q = q.numpy()
            meta2 = meta.copy()
            meta2["q"] = q
            print(f"      q_m range [{q.min():.3f}, {q.max():.3f}] sd={q.std():.3f}; "
                  f"corr(q, star) = {np.corrcoef(q, meta2['star'])[0,1]:+.3f}")

    (PROC / f"diag_step2_anchored_{TAG}.json").write_text(json.dumps(out, indent=2))
    print("\nwrote", PROC / f"diag_step2_anchored_{TAG}.json")


if __name__ == "__main__":
    main()
