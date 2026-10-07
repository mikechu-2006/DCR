"""Stage-1 v2: player-year entities with linear drift inside each calendar year.

    P_eff(u,y,t) = P[u,y] + (tau - 0.5) * V[u,y]        tau = day_of_year / 365
    L_hat = b0 + b_u[u,y] + b_m[chart] + (P_eff - C[chart]) . D[chart]

Comparison models: plain (no time), bias only, dot product, and the same MIRT without drift.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.manual_seed(0); np.random.seed(0)


class YearlyMIRT(nn.Module):
    def __init__(self, n_ent, n_item, dim, kind, drift=True):
        super().__init__()
        self.kind, self.drift = kind, drift
        self.b0 = nn.Parameter(torch.zeros(1))
        self.bu = nn.Embedding(n_ent, 1); nn.init.zeros_(self.bu.weight)
        self.bm = nn.Embedding(n_item, 1); nn.init.zeros_(self.bm.weight)
        self.P = nn.Embedding(n_ent, dim); nn.init.normal_(self.P.weight, std=0.05)
        if drift:
            self.V = nn.Embedding(n_ent, dim); nn.init.zeros_(self.V.weight)
        self.C = nn.Embedding(n_item, dim); nn.init.normal_(self.C.weight, std=0.05)
        if kind == "mirt":
            self.D = nn.Embedding(n_item, dim); nn.init.normal_(self.D.weight, std=0.05)

    def forward(self, ent, item, tau):
        out = self.b0 + self.bu(ent).squeeze(-1) + self.bm(item).squeeze(-1)
        p, c = self.P(ent), self.C(item)
        if self.drift:
            p = p + (tau - 0.5).unsqueeze(-1) * self.V(ent)
        if self.kind == "bias":
            return out
        if self.kind == "mf_dot":
            return out + (p * c).sum(-1)
        return out + ((p - c) * self.D(item)).sum(-1)


def run(kind, drift, tr, va, n_ent, n_item, dim, epochs, lr, wd, warm=None):
    m = YearlyMIRT(n_ent, n_item, dim, kind, drift)
    if warm:
        with torch.no_grad():
            for a, b in warm:
                m.C.weight[a] = m.C.weight[b]
                if hasattr(m, "D"):
                    m.D.weight[a] = m.D.weight[b]
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    eu, it, tau, y = tr
    veu, vit, vtau, vy = va
    t0 = time.time()
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(eu))
        for i in range(0, len(eu), 8192):
            idx = perm[i:i + 8192]
            opt.zero_grad()
            nn.functional.mse_loss(m(eu[idx], it[idx], tau[idx]), y[idx]).backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        pv = m(veu, vit, vtau)
        e = pv - vy
        rmse = float(torch.sqrt((e ** 2).mean()))
        r2 = float(1 - (e ** 2).sum() / ((vy - vy.mean()) ** 2).sum())
    return m, {"rmse": round(rmse, 5), "r2": round(r2, 4), "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plays", default="data/processed/plays_v2_1k.parquet")
    ap.add_argument("--cells", default="data/processed/charts_v2_1k.parquet")
    ap.add_argument("--target", default="loss")
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--out-prefix", default="stage1_v2")
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--models", default="", help="subset of: bias,mf_dot,mirt,mirt_nodrift")
    ap.add_argument("--warm-start", action="store_true",
                    help="init DT/HT chart rows from the NM row of the same map")
    ap.add_argument("--min-cells-per-chart", type=int, default=10)
    ap.add_argument("--tier-col", default=None,
                    help="categorical column to filter cells on, e.g. conv_tier (scripts/convergence_filter.py)")
    ap.add_argument("--tiers", default="A,B", help="comma list of accepted values for --tier-col")
    ap.add_argument("--cell-level", action="store_true", help="train on cell medians instead of plays")
    args = ap.parse_args()

    cells = pd.read_parquet(args.cells, columns=["user_id", "chart_id", "primary", "n_scores", "loss_med",
                                                 "beatmap_id", "rate"] +
                            ([args.tier_col] if args.tier_col else []))
    cells = cells[cells.primary]
    if args.tier_col:
        keep = {v.strip() for v in args.tiers.split(",") if v.strip()}
        cells = cells[cells[args.tier_col].isin(keep)]
        print(f"tier filter {args.tier_col} in {sorted(keep)} -> {len(cells)} cells", flush=True)
    cols = ["user_id", "entity_id", "chart_id", "tau", "loss"]
    if args.warm_start:
        cols += ["beatmap_id", "rate"]
    plays = pd.read_parquet(args.plays, columns=cols)
    plays = plays.merge(cells[["user_id", "chart_id"]], on=["user_id", "chart_id"], how="inner")
    print(f"plays={len(plays)} cells={len(cells)} users={plays.user_id.nunique()} "
          f"entities={plays.entity_id.nunique()} charts={plays.chart_id.nunique()}", flush=True)

    nplay = plays.groupby("chart_id").size()
    plays = plays[plays.chart_id.isin(nplay[nplay >= args.min_cells_per_chart].index)]
    print("after chart frequency filter:", len(plays), "charts", plays.chart_id.nunique(), flush=True)

    ents = {e: i for i, e in enumerate(plays.entity_id.unique())}
    items = {c: i for i, c in enumerate(plays.chart_id.unique())}
    ui = plays.entity_id.map(ents).to_numpy()
    ii = plays.chart_id.map(items).to_numpy()
    tau = plays.tau.to_numpy("float32")
    y = plays[args.target].to_numpy("float32")

    # split by cell so validation cells are entirely unseen
    key = plays.user_id.to_numpy("int64") * 1_000_000 + plays.chart_id.to_numpy("int64")
    uk = np.unique(key)
    rng = np.random.default_rng(0)
    hold = set(uk[rng.random(len(uk)) < 0.1].tolist())
    is_va = np.fromiter((k in hold for k in key), dtype=bool, count=len(key))
    tr = (torch.tensor(ui[~is_va]).long(), torch.tensor(ii[~is_va]).long(),
          torch.tensor(tau[~is_va]), torch.tensor(y[~is_va]))
    va = (torch.tensor(ui[is_va]).long(), torch.tensor(ii[is_va]).long(),
          torch.tensor(tau[is_va]), torch.tensor(y[is_va]))
    print(f"train plays={len(tr[0])} val plays={len(va[0])} val cells={len(hold)}", flush=True)

    warm = []
    if args.warm_start:
        rate_of = plays.groupby("chart_id").rate.first()
        map_of = plays.groupby("chart_id").beatmap_id.first()
        idx_of = {c: items[c] for c in items}
        by_map = {}
        for c in items:
            by_map.setdefault((map_of[c], rate_of[c]), idx_of[c])
        for (m, r), i in list(by_map.items()):
            if r in (1, 2) and (m, 0) in by_map:
                warm.append((i, by_map[(m, 0)]))
        print("warm-start pairs:", len(warm), flush=True)

    results, models = {}, {}
    combos = [("bias", True), ("mf_dot", True), ("mirt", True), ("mirt_nodrift", False)]
    if args.models:
        want = set(args.models.split(","))
        combos = [c for c in combos if (c[0] if c[1] else "mirt_nodrift") in want]
    for kind, drift in combos:
        label = kind if drift else "mirt_no_drift"
        k = "mirt" if kind == "mirt_nodrift" else kind
        mdl, res = run(k, drift, tr, va, len(ents), len(items), args.dim, args.epochs, args.lr,
                       args.wd, warm=warm if args.warm_start else None)
        results[label] = {**res, "n_params": sum(p.numel() for p in mdl.parameters())}
        models[label] = mdl
        print(f"{label:14s} rmse={res['rmse']:.5f} r2={res['r2']:.4f} {res['seconds']}s", flush=True)

    out = Path("data/processed")
    best = models.get("mirt", models[list(models)[0]])
    np.savez(out / f"{args.out_prefix}_{args.target}.npz",
             entity_ids=np.array(list(ents.keys())), chart_ids=np.array(list(items.keys())),
             P=best.P.weight.detach().numpy(), V=best.V.weight.detach().numpy() if hasattr(best, "V") else np.zeros_like(best.P.weight.detach().numpy()),
             C=best.C.weight.detach().numpy(),
             D=best.D.weight.detach().numpy() if hasattr(best, "D") else np.zeros_like(best.C.weight.detach().numpy()),
             b0=best.b0.detach().numpy(), bu=best.bu.weight.detach().numpy().ravel(),
             bm=best.bm.weight.detach().numpy().ravel())
    (out / f"{args.out_prefix}_summary_{args.target}.json").write_text(json.dumps(
        {"args": vars(args), "n_plays": int(len(plays)), "n_entities": len(ents), "n_charts": len(items),
         "results": results}, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
