"""Why did free per-chart vectors lose to the shared linear-transform model?

Same data (charts_v2_1k, primary cells), item = chart (map x rate) in every variant; only the
regularisation / initialisation of the DT-HT embeddings changes.

  chart_mirt          free per-chart C/D                       (baseline)
  chart_mirt+wd{1e-4,1e-3}   same, stronger weight decay
  chart_mirt+warm     DT/HT rows initialised from the NM row of the same map, then left free
  map_linear          shared map vector + per-rate linear transform (reference)
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn

torch.manual_seed(0); np.random.seed(0)


class M(nn.Module):
    def __init__(self, n_user, n_item, dim, kind, n_rate=3):
        super().__init__()
        self.kind = kind
        self.b0 = nn.Parameter(torch.zeros(1))
        self.bu = nn.Embedding(n_user, 1); nn.init.zeros_(self.bu.weight)
        self.bm = nn.Embedding(n_item, 1); nn.init.zeros_(self.bm.weight)
        self.br = nn.Embedding(n_rate, 1); nn.init.zeros_(self.br.weight)
        self.P = nn.Embedding(n_user, dim); nn.init.normal_(self.P.weight, std=0.05)
        self.C = nn.Embedding(n_item, dim); nn.init.normal_(self.C.weight, std=0.05)
        self.D = nn.Embedding(n_item, dim); nn.init.normal_(self.D.weight, std=0.05)
        if kind == "linear":
            self.A = nn.Parameter(torch.zeros(n_rate, dim, dim))
            self.B = nn.Parameter(torch.zeros(n_rate, dim, dim))

    def forward(self, u, it, r):
        out = self.b0 + self.bu(u).squeeze(-1) + self.bm(it).squeeze(-1) + self.br(r).squeeze(-1)
        p, c, d = self.P(u), self.C(it), self.D(it)
        if self.kind == "linear":
            c = c + torch.einsum("bij,bj->bi", self.A[r], c)
            d = d + torch.einsum("bij,bj->bi", self.B[r], d)
        return out + ((p - c) * d).sum(-1)


def run(label, kind, item, tr, va, n_user, n_item, dim, epochs, lr, wd, warm=None, n_rate=3):
    m = M(n_user, n_item, dim, kind, n_rate)
    if warm is not None:
        with torch.no_grad():
            for dt_idx, nm_idx in warm:
                m.C.weight[dt_idx] = m.C.weight[nm_idx]
                m.D.weight[dt_idx] = m.D.weight[nm_idx]
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    t0 = time.time()
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(tr[0]))
        for i in range(0, len(tr[0]), 8192):
            idx = perm[i:i + 8192]
            opt.zero_grad()
            nn.functional.mse_loss(m(tr[0][idx], tr[1][idx], tr[2][idx]), tr[3][idx]).backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        def metrics(t):
            pv = m(t[0], t[1], t[2]); e = pv - t[3]
            return float(torch.sqrt((e ** 2).mean())), float(1 - (e ** 2).sum() / ((t[3] - t[3].mean()) ** 2).sum()), pv
        rtr, _, _ = metrics(tr)
        rva, r2va, pv = metrics(va)
        per = {}
        for rr in range(n_rate):
            mk = va[2] == rr
            if int(mk.sum()) > 30:
                e = pv[mk] - va[3][mk]
                per[int(rr)] = {"n": int(mk.sum()), "rmse": round(float(torch.sqrt((e ** 2).mean())), 5)}
    print(f"{label:22s} train={rtr:.5f} val={rva:.5f} r2={r2va:.4f} per_rate={per} {time.time()-t0:.0f}s", flush=True)
    return {"label": label, "train_rmse": round(rtr, 5), "val_rmse": round(rva, 5), "val_r2": round(r2va, 4),
            "per_rate": per}


def main():
    cells = pd.read_parquet("data/processed/charts_v2_1k.parquet")
    cells = cells[cells.primary].dropna(subset=["loss_med"])
    users = {u: i for i, u in enumerate(cells.user_id.unique())}
    charts = {c: i for i, c in enumerate(cells.chart_id.unique())}
    maps = {m: i for i, m in enumerate(cells.beatmap_id.unique())}
    cells["ui"] = cells.user_id.map(users); cells["ci"] = cells.chart_id.map(charts)
    cells["mi"] = cells.beatmap_id.map(maps)
    print(f"cells={len(cells)} users={len(users)} charts={len(charts)} maps={len(maps)}", flush=True)
    rng = np.random.default_rng(0); hold = rng.random(len(cells)) < 0.1
    tr_df, va_df = cells[~hold], cells[hold]

    def T(d, item):
        return (torch.tensor(d.ui.values).long(), torch.tensor(d[item].values).long(),
                torch.tensor(d.rate.values).long(), torch.tensor(d.loss_med.values, dtype=torch.float32))

    # warm start map: chart index of (map, rate 1/2) -> chart index of (map, rate 0)
    by_map = cells.groupby(["beatmap_id", "rate"]).ci.first().unstack()
    warm = [(int(r[1]), int(r[0])) for _, r in by_map.iterrows() if not pd.isna(r.get(1)) and not pd.isna(r.get(0))]
    warm += [(int(r[2]), int(r[0])) for _, r in by_map.iterrows() if not pd.isna(r.get(2)) and not pd.isna(r.get(0))]
    print("warm-start pairs:", len(warm), flush=True)

    res = []
    trc, vac = T(tr_df, "ci"), T(va_df, "ci")
    trm, vam = T(tr_df, "mi"), T(va_df, "mi")
    for wd in (1e-6, 1e-4, 1e-3):
        res.append(run(f"chart_mirt wd={wd:g}", "mirt", "ci", trc, vac, len(users), len(charts), 16, 60, 0.02, wd))
    res.append(run("chart_mirt +warm", "mirt", "ci", trc, vac, len(users), len(charts), 16, 60, 0.02, 1e-4, warm=warm))
    res.append(run("map_linear", "linear", "mi", trm, vam, len(users), len(maps), 16, 60, 0.02, 1e-6))
    Path("data/processed/compare_rate_models.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
