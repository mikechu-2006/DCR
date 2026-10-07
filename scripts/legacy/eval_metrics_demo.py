"""What do the stage-1 numbers actually mean?  Train the 1k v3 model and translate the
validation residuals back into ACC percentage points."""
import numpy as np, pandas as pd, torch, torch.nn as nn
torch.manual_seed(0); np.random.seed(0)

cells = pd.read_parquet("data/processed/charts_v2_1k_v3.parquet",
                        columns=["user_id", "chart_id", "primary", "loss_med"])
cells = cells[cells.primary]
plays = pd.read_parquet("data/processed/plays_v2_1k_v3.parquet",
                        columns=["user_id", "entity_id", "chart_id", "tau", "loss", "acc"])
plays = plays.merge(cells[["user_id", "chart_id"]], on=["user_id", "chart_id"], how="inner")
n = plays.groupby("chart_id").size()
plays = plays[plays.chart_id.isin(n[n >= 10].index)]

ents = {e: i for i, e in enumerate(plays.entity_id.unique())}
items = {c: i for i, c in enumerate(plays.chart_id.unique())}
ui = plays.entity_id.map(ents).to_numpy(); ii = plays.chart_id.map(items).to_numpy()
tau = plays.tau.to_numpy("float32"); y = plays.loss.to_numpy("float32"); acc = plays.acc.to_numpy("float32")
key = plays.user_id.to_numpy("int64") * 1_000_000 + plays.chart_id.to_numpy("int64")
uk = np.unique(key); rng = np.random.default_rng(0)
hold = set(uk[rng.random(len(uk)) < 0.1].tolist())
is_va = np.fromiter((k in hold for k in key), dtype=bool, count=len(key))
print(f"plays={len(plays)} train={int((~is_va).sum())} val={int(is_va.sum())} val cells={len(hold)}")

class M(nn.Module):
    def __init__(s, ne, ni, dim):
        super().__init__()
        s.b0 = nn.Parameter(torch.zeros(1))
        s.bu = nn.Embedding(ne, 1); nn.init.zeros_(s.bu.weight)
        s.bm = nn.Embedding(ni, 1); nn.init.zeros_(s.bm.weight)
        s.P = nn.Embedding(ne, dim); nn.init.normal_(s.P.weight, std=0.05)
        s.V = nn.Embedding(ne, dim); nn.init.zeros_(s.V.weight)
        s.C = nn.Embedding(ni, dim); nn.init.normal_(s.C.weight, std=0.05)
        s.D = nn.Embedding(ni, dim); nn.init.normal_(s.D.weight, std=0.05)
    def forward(s, e, i, t):
        p = s.P(e) + (t - 0.5).unsqueeze(-1) * s.V(e)
        return s.b0 + s.bu(e).squeeze(-1) + s.bm(i).squeeze(-1) + ((p - s.C(i)) * s.D(i)).sum(-1)

m = M(len(ents), len(items), 16)
opt = torch.optim.Adam(m.parameters(), lr=0.02, weight_decay=1e-4)
eu = torch.tensor(ui[~is_va]).long(); it = torch.tensor(ii[~is_va]).long()
tt = torch.tensor(tau[~is_va]); yy = torch.tensor(y[~is_va])
for ep in range(40):
    perm = torch.randperm(len(eu))
    for i in range(0, len(eu), 8192):
        idx = perm[i:i+8192]; opt.zero_grad()
        nn.functional.mse_loss(m(eu[idx], it[idx], tt[idx]), yy[idx]).backward(); opt.step()
m.eval()
with torch.no_grad():
    pred = m(torch.tensor(ui[is_va]).long(), torch.tensor(ii[is_va]).long(), torch.tensor(tau[is_va])).numpy()
true = y[is_va]
err = pred - true
rmse = float(np.sqrt((err**2).mean())); mae = float(np.abs(err).mean())
r2 = float(1 - (err**2).sum() / ((true - true.mean())**2).sum())
print(f"\nloss scale: true mean {true.mean():.3f} std {true.std():.3f}")
print(f"RMSE(ln) = {rmse:.4f}   MAE(ln) = {mae:.4f}   R2 = {r2:.4f}")
print(f"baseline (predict val mean): RMSE = {true.std():.4f}")
print(f"multiplicative error of (1-ACC): exp(+-{rmse:.3f}) = x{np.exp(rmse):.2f} / /{np.exp(rmse):.2f}")

acc_true = acc[is_va]; acc_pred = 1.0 - np.exp(pred)
d = np.abs(acc_pred - acc_true) * 100
print("\n|predicted ACC - true ACC| in percentage points:")
for q in [10, 25, 50, 75, 90, 99]:
    print(f"  p{q:>2}: {np.percentile(d, q):.4f} pp")
print(f"  mean {d.mean():.4f} pp   share > 1pp: {(d>1).mean():.4f}   > 5pp: {(d>5).mean():.4f}")
print("\ntrue ACC distribution of validation plays:")
for q in [1, 5, 25, 50, 75, 99]:
    print(f"  p{q:>2}: {np.percentile(acc_true, q)*100:.3f}%")
