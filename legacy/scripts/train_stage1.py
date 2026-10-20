"""Stage-1: train (player, beatmap) -> log-loss models and compare interaction structures.

Models
  bias      : b0 + b_u + b_m
  mf_dot    : bias + <P_u, C_m>                       (classic matrix factorisation)
  mirt      : bias + sum_i (P_ui - C_mi) * D_mi       (the user's formula, MIRT-style)
  mlp       : bias + MLP([P_u ; C_m])                 (deep fully-connected control)

Usage: python scripts/train_stage1.py [--target loss_v2_med] [--cells data/processed/cells_nm.parquet]
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)
DEV = "cpu"


class Model(nn.Module):
    def __init__(self, n_user, n_map, dim, kind, hidden=64, n_cf=0):
        super().__init__()
        self.kind = kind
        self.n_cf = n_cf
        if n_cf:
            self.cellf = nn.Linear(n_cf, 1)
            nn.init.zeros_(self.cellf.weight); nn.init.zeros_(self.cellf.bias)
        self.b0 = nn.Parameter(torch.zeros(1))
        self.bu = nn.Embedding(n_user, 1)
        self.bm = nn.Embedding(n_map, 1)
        nn.init.zeros_(self.bu.weight); nn.init.zeros_(self.bm.weight)
        self.P = nn.Embedding(n_user, dim)
        self.C = nn.Embedding(n_map, dim)
        nn.init.normal_(self.P.weight, std=0.05)
        nn.init.normal_(self.C.weight, std=0.05)
        if kind == "mirt":
            self.D = nn.Embedding(n_map, dim)
            nn.init.normal_(self.D.weight, std=0.05)
        elif kind == "mlp":
            self.mlp = nn.Sequential(nn.Linear(2 * dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, u, m, cf=None):
        out = self.b0 + self.bu(u).squeeze(-1) + self.bm(m).squeeze(-1)
        if self.n_cf:
            out = out + self.cellf(cf).squeeze(-1)
        if self.kind == "bias":
            return out
        p, c = self.P(u), self.C(m)
        if self.kind == "mf_dot":
            return out + (p * c).sum(-1)
        if self.kind == "mirt":
            return out + ((p - c) * self.D(m)).sum(-1)
        if self.kind == "mlp":
            return out + self.mlp(torch.cat([p, c], -1)).squeeze(-1)
        raise ValueError(self.kind)


def train_one(kind, tr, va, n_user, n_map, dim, epochs, lr, wd, loss_name, weights=None, n_cf=0):
    model = Model(n_user, n_map, dim, kind, n_cf=n_cf).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.MSELoss(reduction="none") if loss_name == "mse" else nn.HuberLoss(reduction="none", delta=1.0)
    u, m, y = tr[0], tr[1], tr[2]
    cf = tr[3] if len(tr) > 3 else None
    vu, vm, vy = va[0], va[1], va[2]
    vcf = va[3] if len(va) > 3 else None
    best = None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(u))
        for i in range(0, len(u), 8192):
            idx = perm[i:i + 8192]
            opt.zero_grad()
            pred = model(u[idx], m[idx], cf[idx] if cf is not None else None)
            l = lossf(pred, y[idx])
            if weights is not None:
                l = l * weights[idx]
            l.mean().backward()
            opt.step()
        if ep % 5 == 4 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                pv = model(vu, vm, vcf)
                rmse = float(torch.sqrt(((pv - vy) ** 2).mean()))
            best = rmse
    model.eval()
    with torch.no_grad():
        pv = model(vu, vm, vcf)
        rmse = float(torch.sqrt(((pv - vy) ** 2).mean()))
        mae = float((pv - vy).abs().mean())
        ss_res = float(((pv - vy) ** 2).sum())
        ss_tot = float(((vy - vy.mean()) ** 2).sum())
    return model, {"rmse": rmse, "mae": mae, "r2": 1 - ss_res / ss_tot, "n_params": sum(p.numel() for p in model.parameters())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="data/processed/cells_nm.parquet")
    ap.add_argument("--target", default="loss_v2_med")
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--wd", type=float, default=1e-6)
    ap.add_argument("--loss", default="mse")
    ap.add_argument("--min-map-players", type=int, default=20)
    ap.add_argument("--min-player-maps", type=int, default=20)
    ap.add_argument("--out-prefix", default="stage1")
    ap.add_argument("--item-col", default="chart_id", help="column defining the item (chart) side")
    ap.add_argument("--primary-only", action="store_true", help="keep only cells with median ACC >= 0.90")
    ap.add_argument("--tier-col", default=None,
                    help="categorical column to filter on, e.g. conv_tier (see scripts/convergence_filter.py)")
    ap.add_argument("--tiers", default="A,B,C", help="comma list of accepted values for --tier-col")
    ap.add_argument("--cell-features", action="store_true",
                    help="add log1p(n_scores), log1p(attempts), attempts/score as linear cell features")
    args = ap.parse_args()

    df = pd.read_parquet(args.cells)
    df = df.dropna(subset=[args.target])
    if args.primary_only and "primary" in df.columns:
        df = df[df.primary]
    if args.tier_col:
        keep = {v.strip() for v in args.tiers.split(",") if v.strip()}
        df = df[df[args.tier_col].isin(keep)]
        print(f"tier filter {args.tier_col} in {sorted(keep)} -> {len(df)} cells", flush=True)
    item = args.item_col
    mp = df.groupby(item).user_id.nunique()
    pu = df.groupby("user_id")[item].nunique()
    df = df[df[item].isin(mp[mp >= args.min_map_players].index) &
            df.user_id.isin(pu[pu >= args.min_player_maps].index)].copy()
    users = {u: i for i, u in enumerate(df.user_id.unique())}
    maps = {m: i for i, m in enumerate(df[item].unique())}
    df["ui"] = df.user_id.map(users)
    df["mi"] = df[item].map(maps)
    item_meta = df.groupby("mi")[[c for c in ("beatmap_id", "rate", "star") if c in df.columns]].first()
    print(f"cells={len(df)} users={len(users)} maps={len(maps)} target={args.target}", flush=True)
    print("target stats:", df[args.target].describe()[["mean", "std", "min", "max"]].round(4).to_dict(), flush=True)

    rng = np.random.default_rng(0)
    hold = rng.random(len(df)) < 0.1
    tr_df, va_df = df[~hold], df[hold]

    def cell_feats(d):
        if not args.cell_features:
            return None
        f = np.column_stack([
            np.log1p(d.n_scores.values),
            np.log1p(np.nan_to_num(d.attempts.values, nan=0.0)),
            np.nan_to_num(d.attempts_per_score.values, nan=0.0),
        ]).astype("float32")
        return torch.tensor(f)

    cf_tr, cf_va = cell_feats(tr_df), cell_feats(va_df)
    n_cf = 0 if cf_tr is None else cf_tr.shape[1]
    tr = (torch.tensor(tr_df.ui.values), torch.tensor(tr_df.mi.values),
          torch.tensor(tr_df[args.target].values, dtype=torch.float32)) + ((cf_tr,) if cf_tr is not None else ())
    va = (torch.tensor(va_df.ui.values), torch.tensor(va_df.mi.values),
          torch.tensor(va_df[args.target].values, dtype=torch.float32)) + ((cf_va,) if cf_va is not None else ())
    w_tr = torch.tensor(tr_df.n_scores.values, dtype=torch.float32)

    results, models = {}, {}
    for kind in ["bias", "mf_dot", "mirt", "mlp"]:
        t0 = time.time()
        mdl, res = train_one(kind, tr, va, len(users), len(maps), args.dim, args.epochs, args.lr, args.wd, args.loss, n_cf=n_cf)
        res["seconds"] = round(time.time() - t0, 1)
        results[kind] = res
        models[kind] = mdl
        print(f"{kind:8s} rmse={res['rmse']:.5f} mae={res['mae']:.5f} r2={res['r2']:.4f} "
              f"params={res['n_params']} {res['seconds']}s", flush=True)

    # weighted variant of mirt (cells with more plays weighted more)
    mdl_w, res_w = train_one("mirt", tr, va, len(users), len(maps), args.dim, args.epochs, args.lr, args.wd,
                             args.loss, weights=w_tr, n_cf=n_cf)
    results["mirt_weighted"] = res_w
    models["mirt_weighted"] = mdl_w
    print(f"{'mirt_w':8s} rmse={res_w['rmse']:.5f} r2={res_w['r2']:.4f}", flush=True)

    # ---------------- save embeddings + summary ----------------
    out = Path("data/processed")
    inv_users = {v: k for k, v in users.items()}
    inv_maps = {v: k for k, v in maps.items()}
    for name, mdl in models.items():
        if mdl.kind == "bias":
            continue
        np.savez(out / f"{args.out_prefix}_{name}_{args.target}.npz",
                 user_ids=np.array(list(inv_users.values())), map_ids=np.array(list(inv_maps.values())),
                 beatmap_ids=item_meta.beatmap_id.values, rates=item_meta.rate.values,
                 P=mdl.P.weight.detach().numpy(), C=mdl.C.weight.detach().numpy(),
                 D=mdl.D.weight.detach().numpy() if mdl.kind == "mirt" else np.zeros((len(maps), args.dim)),
                 b0=mdl.b0.detach().numpy(), bu=mdl.bu.weight.detach().numpy().ravel(),
                 bm=mdl.bm.weight.detach().numpy().ravel())
    summary = {"args": vars(args), "n_cells": len(df), "n_users": len(users), "n_maps": len(maps),
               "target": args.target, "results": results}
    (out / f"{args.out_prefix}_summary_{args.target}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
