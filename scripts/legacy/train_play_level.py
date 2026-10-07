"""Play-level (non-aggregated) stage-1: every play of a (player, chart) cell is one row.

Split is by *cell* so that validation cells are entirely unseen; that makes the comparison with
the cell-median model fair (both predict unseen cells).
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.manual_seed(0); np.random.seed(0)


class MIRT(nn.Module):
    def __init__(self, n_user, n_item, dim, kind):
        super().__init__()
        self.kind = kind
        self.b0 = nn.Parameter(torch.zeros(1))
        self.bu = nn.Embedding(n_user, 1); nn.init.zeros_(self.bu.weight)
        self.bm = nn.Embedding(n_item, 1); nn.init.zeros_(self.bm.weight)
        self.P = nn.Embedding(n_user, dim); nn.init.normal_(self.P.weight, std=0.05)
        self.C = nn.Embedding(n_item, dim); nn.init.normal_(self.C.weight, std=0.05)
        if kind == "mirt":
            self.D = nn.Embedding(n_item, dim); nn.init.normal_(self.D.weight, std=0.05)
        else:
            self.mlp = nn.Sequential(nn.Linear(2 * dim, 64), nn.ReLU(), nn.Linear(64, 1))
            nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, u, m):
        out = self.b0 + self.bu(u).squeeze(-1) + self.bm(m).squeeze(-1)
        p, c = self.P(u), self.C(m)
        if self.kind == "mirt":
            return out + ((p - c) * self.D(m)).sum(-1)
        return out + self.mlp(torch.cat([p, c], -1)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plays", default="data/processed/plays_4k.parquet")
    ap.add_argument("--cells", default="data/processed/charts_4k.parquet")
    ap.add_argument("--target", default="loss_v2")
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--out-prefix", default="stage1_play")
    args = ap.parse_args()

    cells = pd.read_parquet(args.cells, columns=["user_id", "chart_id", "n_scores", "primary",
                                                 "loss_v2_med", "acc_v2_med"])
    cells = cells[cells.primary]
    plays = pd.read_parquet(args.plays, columns=["user_id", "chart_id", "clean", "loss_v2", "rate", "beatmap_id"])
    plays = plays[plays.clean].merge(cells[["user_id", "chart_id"]], on=["user_id", "chart_id"], how="inner")
    print(f"plays={len(plays)} cells={cells.chart_id.nunique()} users={plays.user_id.nunique()}", flush=True)

    # cell-level split
    key = plays.user_id.astype("int64") * 100000 + plays.chart_id.astype("int64")
    uk = key.unique()
    rng = np.random.default_rng(0)
    hold = set(uk[rng.random(len(uk)) < 0.1])
    is_va = key.isin(list(hold)).values
    tr_df, va_df = plays[~is_va], plays[is_va]
    print(f"train plays={len(tr_df)} val plays={len(va_df)} val cells={len(hold)}", flush=True)

    users = {u: i for i, u in enumerate(plays.user_id.unique())}
    items = {c: i for i, c in enumerate(plays.chart_id.unique())}
    tr = (torch.tensor(tr_df.user_id.map(users).values), torch.tensor(tr_df.chart_id.map(items).values),
          torch.tensor(tr_df[args.target].values, dtype=torch.float32))
    va = (torch.tensor(va_df.user_id.map(users).values), torch.tensor(va_df.chart_id.map(items).values),
          torch.tensor(va_df[args.target].values, dtype=torch.float32))

    # cell-median target for the same validation cells (for a like-for-like number)
    val_med = va_df.groupby(["user_id", "chart_id"])[args.target].median().rename("med").reset_index()
    vm_u = torch.tensor(val_med.user_id.map(users).values)
    vm_i = torch.tensor(val_med.chart_id.map(items).values)
    vm_y = torch.tensor(val_med.med.values, dtype=torch.float32)

    out = Path("data/processed")
    results = {}
    for kind in ["mirt", "mlp"]:
        model = MIRT(len(users), len(items), args.dim, kind)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
        t0 = time.time()
        for ep in range(args.epochs):
            model.train()
            perm = torch.randperm(len(tr[0]))
            for i in range(0, len(tr[0]), 8192):
                idx = perm[i:i + 8192]
                opt.zero_grad()
                nn.functional.mse_loss(model(tr[0][idx], tr[1][idx]), tr[2][idx]).backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(*va[:2])
            rmse_play = float(torch.sqrt(((pv - va[2]) ** 2).mean()))
            pvm = model(vm_u, vm_i)
            rmse_cell = float(torch.sqrt(((pvm - vm_y) ** 2).mean()))
            r2_cell = float(1 - ((pvm - vm_y) ** 2).sum() / ((vm_y - vm_y.mean()) ** 2).sum())
        results[kind] = {"rmse_play": round(rmse_play, 5), "rmse_cellmedian": round(rmse_cell, 5),
                         "r2_cellmedian": round(r2_cell, 4), "seconds": round(time.time() - t0, 1)}
        print(kind, results[kind], flush=True)
        np.savez(out / f"{args.out_prefix}_{kind}_{args.target}.npz",
                 item_ids=np.array(list(items.keys())), user_ids=np.array(list(users.keys())),
                 C=model.C.weight.detach().numpy(), P=model.P.weight.detach().numpy(),
                 D=model.D.weight.detach().numpy() if kind == "mirt" else np.zeros((len(items), args.dim)),
                 bm=model.bm.weight.detach().numpy().ravel(),
                 bu=model.bu.weight.detach().numpy().ravel(), b0=model.b0.detach().numpy())
    (out / f"{args.out_prefix}_summary_{args.target}.json").write_text(json.dumps(
        {"args": vars(args), "n_plays": len(plays), "n_train": len(tr_df), "n_val": len(va_df),
         "n_val_cells": len(hold), "results": results}, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
