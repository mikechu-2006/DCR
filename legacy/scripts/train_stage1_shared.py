"""Stage-1 with a *shared* per-map difficulty vector across rate buckets.

chart_mirt      : separate C/D per (map, rate)          -- current model
map_mirt_shared : C_m and D_m shared across NM/DT/HT, plus a rate offset delta_r
                  L = b0 + b_u + b_m + b_r + sum_i (P_ui - C_mi - delta_ri) * D_mi
map_mf_dot      : b0 + b_u + b_m + b_r + <P_u, C_m>

The point: a "level vector" should describe the *level*, not the mod; DT/HT then only shift it.
Sparse DT charts borrow strength from the NM plays of the same map.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.manual_seed(0); np.random.seed(0)
DEV = "cpu"


class Shared(nn.Module):
    def __init__(self, n_user, n_map, dim, kind, n_rate=3):
        super().__init__()
        self.kind, self.n_rate = kind, n_rate
        self.b0 = nn.Parameter(torch.zeros(1))
        self.bu = nn.Embedding(n_user, 1); nn.init.zeros_(self.bu.weight)
        self.bm = nn.Embedding(n_map, 1); nn.init.zeros_(self.bm.weight)
        self.br = nn.Embedding(n_rate, 1); nn.init.zeros_(self.br.weight)
        self.P = nn.Embedding(n_user, dim); nn.init.normal_(self.P.weight, std=0.05)
        self.C = nn.Embedding(n_map, dim); nn.init.normal_(self.C.weight, std=0.05)
        if kind in ("mirt", "mirt_shared", "mirt_linear"):
            self.D = nn.Embedding(n_map, dim); nn.init.normal_(self.D.weight, std=0.05)
        if kind == "mirt_d":
            # one *global* discrimination vector: makes the level vector C identifiable per item
            self.d = nn.Parameter(torch.randn(dim) * 0.05)
        if kind == "mirt_shared":
            self.delta = nn.Embedding(n_rate, dim); nn.init.zeros_(self.delta.weight)
        if kind == "mirt_linear":
            self.A = nn.Parameter(torch.zeros(n_rate, dim, dim))
            self.B = nn.Parameter(torch.zeros(n_rate, dim, dim))

    def forward(self, u, m, r):
        out = (self.b0 + self.bu(u).squeeze(-1) + self.bm(m).squeeze(-1) + self.br(r).squeeze(-1))
        p, c = self.P(u), self.C(m)
        if self.kind == "mf_dot":
            return out + (p * c).sum(-1)
        if self.kind == "bias":
            return out
        if self.kind == "mirt_d":
            return out + ((p - c) * self.d).sum(-1)
        d = self.D(m)
        if self.kind == "mirt_shared":
            # rate 0 (NM) is the reference: its offset is fixed at zero
            delta = self.delta(r) * (r > 0).unsqueeze(-1)
            c = c + delta
        elif self.kind == "mirt_linear":
            # per-rate linear modulation of both the level vector and its discrimination
            c = c + torch.einsum("bij,bj->bi", self.A[r], c)
            d = d + torch.einsum("bij,bj->bi", self.B[r], d)
        return out + ((p - c) * d).sum(-1)


def run(kind, tr, va, n_user, n_map, dim, epochs, lr, wd, n_rate=3):
    model = Shared(n_user, n_map, dim, kind, n_rate).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    u, m, r, y = tr
    vu, vm, vr, vy = va
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(u))
        for i in range(0, len(u), 8192):
            idx = perm[i:i + 8192]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(u[idx], m[idx], r[idx]), y[idx])
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pv = model(vu, vm, vr)
        rmse = float(torch.sqrt(((pv - vy) ** 2).mean()))
        r2 = float(1 - ((pv - vy) ** 2).sum() / ((vy - vy.mean()) ** 2).sum())
        per_rate = {}
        for rr in range(n_rate):
            mask = vr == rr
            if int(mask.sum()) > 30:
                e = pv[mask] - vy[mask]
                per_rate[int(rr)] = {"n": int(mask.sum()),
                                     "rmse": round(float(torch.sqrt((e ** 2).mean())), 5),
                                     "bias": round(float(e.mean()), 5)}
    return model, {"rmse": round(rmse, 5), "r2": round(r2, 4), "seconds": round(time.time() - t0, 1),
                   "per_rate": per_rate}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="data/processed/charts_4k.parquet")
    ap.add_argument("--target", default="loss_v2_med")
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--wd", type=float, default=1e-6)
    ap.add_argument("--min-map-players", type=int, default=10)
    ap.add_argument("--out-prefix", default="stage1_shared")
    ap.add_argument("--models", default="", help="comma separated subset, e.g. map_mirt_linear")
    args = ap.parse_args()

    need = ["user_id", "beatmap_id", "chart_id", "rate", "n_scores", "attempts", "primary", args.target]
    cols = pd.read_parquet(args.cells).columns
    df = pd.read_parquet(args.cells, columns=[c for c in need if c in cols])
    df = df[df.primary] if "primary" in df.columns else df
    df = df.dropna(subset=[args.target])
    mp = df.groupby("chart_id").user_id.nunique()
    df = df[df.chart_id.isin(mp[mp >= args.min_map_players].index)].copy()

    users = {u: i for i, u in enumerate(df.user_id.unique())}
    maps = {m: i for i, m in enumerate(df.beatmap_id.unique())}
    charts = {c: i for i, c in enumerate(df.chart_id.unique())}
    df["ui"] = df.user_id.map(users)
    df["mi"] = df.beatmap_id.map(maps)
    df["ci"] = df.chart_id.map(charts)
    print(f"cells={len(df)} users={len(users)} maps={len(maps)} charts={len(charts)}", flush=True)
    print("rate mix:", df.rate.value_counts().to_dict(), flush=True)

    rng = np.random.default_rng(0)
    hold = rng.random(len(df)) < 0.1
    tr_df, va_df = df[~hold], df[hold]

    def tensors(d, item):
        return (torch.tensor(d.ui.values).long(), torch.tensor(d[item].values).long(),
                torch.tensor(d.rate.values).long(), torch.tensor(d[args.target].values, dtype=torch.float32))

    results, models = {}, {}
    all_models = [("chart_mirt", "ci", "mirt"),
                  ("chart_mirt_d", "ci", "mirt_d"),
                  ("map_mirt_shared", "mi", "mirt_shared"),
                  ("map_mirt_linear", "mi", "mirt_linear"),
                  ("map_mf_dot", "mi", "mf_dot"),
                  ("map_bias", "mi", "bias")]
    wanted = set(args.models.split(",")) if args.models else None
    for label, item, kind in [m for m in all_models if wanted is None or m[0] in wanted]:
        n_items = len(charts) if item == "ci" else len(maps)
        mdl, res = run(kind, tensors(tr_df, item), tensors(va_df, item), len(users), n_items,
                       args.dim, args.epochs, args.lr, args.wd)
        results[label] = res
        models[label] = mdl
        print(f"{label:16s} rmse={res['rmse']:.5f} r2={res['r2']:.4f} {res['seconds']}s per_rate={res['per_rate']}",
              flush=True)

    # chart_mirt must also use the per-chart offset semantics: re-run with kind=mirt (no delta)
    out = Path("data/processed")
    key = next((k for k in ("map_mirt_linear", "map_mirt_shared", "chart_mirt_d", "chart_mirt") if k in models),
               next(iter(models)))
    mdl = models[key]
    np.savez(out / f"{args.out_prefix}_{args.target}.npz",
             map_ids=np.array(list(maps.keys())), C=mdl.C.weight.detach().numpy(),
             D=mdl.D.weight.detach().numpy() if hasattr(mdl, "D") else np.zeros((len(maps), args.dim)),
             d_global=mdl.d.detach().numpy() if hasattr(mdl, "d") else np.zeros(args.dim),
             A=mdl.A.detach().numpy() if hasattr(mdl, "A") else np.zeros((3, args.dim, args.dim)),
             B=mdl.B.detach().numpy() if hasattr(mdl, "B") else np.zeros((3, args.dim, args.dim)),
             b0=mdl.b0.detach().numpy(), bu=mdl.bu.weight.detach().numpy().ravel(),
             bm=mdl.bm.weight.detach().numpy().ravel(), br=mdl.br.weight.detach().numpy().ravel(),
             user_ids=np.array(list(users.keys())), P=mdl.P.weight.detach().numpy())
    (out / f"{args.out_prefix}_summary_{args.target}.json").write_text(
        json.dumps({"args": vars(args), "n_cells": len(df), "n_users": len(users),
                    "n_maps": len(maps), "n_charts": len(charts), "results": results}, indent=2))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_rate"} for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
