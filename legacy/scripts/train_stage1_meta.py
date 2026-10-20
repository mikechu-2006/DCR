"""Does feeding the beatmap metadata into stage-1 help?

Variants (all on the same data / same optimiser):
  mirt           free C/D + biases                                  (current)
  mirt+meta_lin  + w . x_chart                                      (linear side branch)
  mirt+meta_mlp  + MLP(x_chart)                                     (non-linear side branch)
  mirt+meta_gen  C_c = C_free + W_C x_c   (metadata generates the level vector, free residual)

Two protocols:
  random cells : 10% of (player, chart) cells held out          -> interpolation
  cold charts  : 10% of charts held out entirely                -> a brand-new chart
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn

torch.manual_seed(0); np.random.seed(0)
META_COLS = ["max_combo", "countTotal", "playcount", "passcount"]


def load():
    cells = pd.read_parquet("data/processed/charts_v2_1k_v3.parquet",
                            columns=["user_id", "chart_id", "beatmap_id", "rate", "primary"])
    cells = cells[cells.primary]
    plays = pd.read_parquet("data/processed/plays_v2_1k_v3.parquet",
                            columns=["user_id", "entity_id", "chart_id", "beatmap_id", "rate", "tau", "loss"])
    plays = plays.merge(cells[["user_id", "chart_id"]], on=["user_id", "chart_id"], how="inner")
    return plays


def meta_matrix(beatmap_ids, rates):
    bm = pd.read_csv("data/interim/beatmaps.csv").set_index("beatmap_id")
    b = bm.reindex(beatmap_ids)
    x = np.column_stack([
        np.log1p(pd.to_numeric(b.max_combo, errors="coerce").values),
        np.log1p(pd.to_numeric(b.countTotal, errors="coerce").values),
        np.log1p(pd.to_numeric(b.playcount, errors="coerce").values),
        np.log1p(pd.to_numeric(b.passcount, errors="coerce").values),
        pd.to_numeric(b.passcount, errors="coerce").values / np.maximum(pd.to_numeric(b.playcount, errors="coerce").values, 1),
        (b.approved.values >= 1).astype(float),
        (rates == 1).astype(float), (rates == 2).astype(float),
    ])
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mu, sd = x.mean(0), x.std(0) + 1e-9
    return ((x - mu) / sd).astype("float32")


class M(nn.Module):
    def __init__(self, n_ent, n_chart, dim, kind, n_meta):
        super().__init__()
        self.kind = kind
        self.b0 = nn.Parameter(torch.zeros(1))
        self.bu = nn.Embedding(n_ent, 1); nn.init.zeros_(self.bu.weight)
        self.bm = nn.Embedding(n_chart, 1); nn.init.zeros_(self.bm.weight)
        self.P = nn.Embedding(n_ent, dim); nn.init.normal_(self.P.weight, std=0.05)
        self.V = nn.Embedding(n_ent, dim); nn.init.zeros_(self.V.weight)
        self.C = nn.Embedding(n_chart, dim); nn.init.normal_(self.C.weight, std=0.05)
        self.D = nn.Embedding(n_chart, dim); nn.init.normal_(self.D.weight, std=0.05)
        if kind == "meta_lin":
            self.w = nn.Linear(n_meta, 1); nn.init.zeros_(self.w.weight); nn.init.zeros_(self.w.bias)
        elif kind == "meta_mlp":
            self.g = nn.Sequential(nn.Linear(n_meta, 32), nn.ReLU(), nn.Linear(32, 1))
            nn.init.zeros_(self.g[-1].weight); nn.init.zeros_(self.g[-1].bias)
        elif kind == "meta_gen":
            self.W = nn.Linear(n_meta, dim); nn.init.zeros_(self.W.weight); nn.init.zeros_(self.W.bias)

    def forward(self, e, c, t, x):
        out = self.b0 + self.bu(e).squeeze(-1) + self.bm(c).squeeze(-1)
        if self.kind == "meta_lin":
            out = out + self.w(x).squeeze(-1)
        elif self.kind == "meta_mlp":
            out = out + self.g(x).squeeze(-1)
        p = self.P(e) + (t - 0.5).unsqueeze(-1) * self.V(e)
        cc = self.C(c)
        if self.kind == "meta_gen":
            cc = cc + self.W(x)
        return out + ((p - cc) * self.D(c)).sum(-1)


def run(kind, tr, va, n_ent, n_chart, dim, epochs=40, lr=0.02, wd=1e-4):
    m = M(n_ent, n_chart, dim, kind, tr[3].shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(tr[0]))
        for i in range(0, len(tr[0]), 8192):
            idx = perm[i:i + 8192]
            opt.zero_grad()
            nn.functional.mse_loss(m(tr[0][idx], tr[1][idx], tr[2][idx], tr[3][idx]), tr[4][idx]).backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        pv = m(va[0], va[1], va[2], va[3])
        e = pv - va[4]
        rmse = float(torch.sqrt((e ** 2).mean()))
        r2 = float(1 - (e ** 2).sum() / ((va[4] - va[4].mean()) ** 2).sum())
    return m, {"rmse": round(rmse, 5), "r2": round(r2, 4), "sec": round(time.time() - t0, 1)}


def main():
    plays = load()
    n = plays.groupby("chart_id").size()
    plays = plays[plays.chart_id.isin(n[n >= 10].index)].reset_index(drop=True)
    ents = {e: i for i, e in enumerate(plays.entity_id.unique())}
    charts = {c: i for i, c in enumerate(plays.chart_id.unique())}
    plays["ui"] = plays.entity_id.map(ents); plays["ci"] = plays.chart_id.map(charts)
    X = meta_matrix(plays.beatmap_id.values, plays.rate.values)
    print(f"plays={len(plays)} entities={len(ents)} charts={len(charts)} meta_dim={X.shape[1]}", flush=True)
    y = plays.loss.to_numpy("float32"); tau = plays.tau.to_numpy("float32")
    ui = plays.ui.to_numpy(); ci = plays.ci.to_numpy()
    key = plays.user_id.to_numpy("int64") * 1_000_000 + plays.chart_id.to_numpy("int64")
    rng = np.random.default_rng(0)

    def tensors(mask):
        return (torch.tensor(ui[mask]).long(), torch.tensor(ci[mask]).long(),
                torch.tensor(tau[mask]), torch.tensor(X[mask]), torch.tensor(y[mask]))

    results = {}
    # --- protocol 1: random cells ---
    uk = np.unique(key)
    hold = set(uk[rng.random(len(uk)) < 0.1].tolist())
    va_mask = np.fromiter((k in hold for k in key), dtype=bool, count=len(key))
    tr, va = tensors(~va_mask), tensors(va_mask)
    print("\n[random cells]", flush=True)
    for kind in ["mirt", "meta_lin", "meta_mlp", "meta_gen"]:
        m, res = run(kind, tr, va, len(ents), len(charts), 16)
        results[f"cells/{kind}"] = res
        print(f"  {kind:9s} rmse={res['rmse']:.5f} r2={res['r2']:.4f}", flush=True)
        if kind in ("mirt", "meta_mlp"):
            np.savez(f"data/processed/stage1_meta12_{kind}_loss.npz",
                     chart_ids=np.array(list(charts.keys())), C=m.C.weight.detach().numpy(),
                     D=m.D.weight.detach().numpy(), bm=m.bm.weight.detach().numpy().ravel(),
                     P=m.P.weight.detach().numpy())

    # --- protocol 2: cold charts ---
    chart_ids = np.array(list(charts.keys()))
    cold = set(chart_ids[rng.random(len(chart_ids)) < 0.1].tolist())
    cold_i = {charts[c] for c in cold}
    va2_mask = np.fromiter((c in cold_i for c in ci), dtype=bool, count=len(ci))
    tr2, va2 = tensors(~va2_mask), tensors(va2_mask)
    print(f"\n[cold charts] held-out charts={len(cold_i)} val plays={int(va2_mask.sum())}", flush=True)
    for kind in ["mirt", "meta_lin", "meta_mlp", "meta_gen"]:
        m, res = run(kind, tr2, va2, len(ents), len(charts), 16)
        results[f"cold/{kind}"] = res
        print(f"  {kind:9s} rmse={res['rmse']:.5f} r2={res['r2']:.4f}", flush=True)

    Path("data/processed/meta_stage1.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
