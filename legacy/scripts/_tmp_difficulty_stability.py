"""Scratch 12: 无标签难度估计的稳定性 —— seed 稳定性 vs 原始统计量的 split-half 稳定性."""
import sys, numpy as np, pandas as pd, torch
from torch import nn
sys.path.insert(0, 'scripts')
import step2_predict as sp
torch.use_deterministic_algorithms(True)

df = pd.read_parquet('data/processed/step1_1k.parquet')
k = int(df.beatmap_id.max()) + 1
key = df.player_id.to_numpy(np.int64) * np.int64(k) + df.beatmap_id.to_numpy(np.int64)
uniq = np.unique(key); rng = np.random.default_rng(20260928); perm = rng.permutation(uniq.size)
n_tr = int(0.70 * uniq.size); is_te = np.isin(key, uniq[perm[n_tr:]])
tr = df[~is_te].reset_index(drop=True); te = df[is_te].reset_index(drop=True)
del df, key
ent_ids = np.unique(tr.player_id.to_numpy()); item_ids = np.unique(tr.beatmap_id.to_numpy())
em = {e: i for i, e in enumerate(ent_ids)}; im = {m: i for i, m in enumerate(item_ids)}
def enc(d):
    return (d.player_id.map(em).fillna(-1).to_numpy(np.int64),
            d.beatmap_id.map(im).fillna(-1).to_numpy(np.int64), d.loss.to_numpy(np.float32))
tr_e, tr_i, tr_y = enc(tr); te_e, te_i, te_y = enc(te)
ok = (tr_e >= 0) & (tr_i >= 0); teok = (te_e >= 0) & (te_i >= 0)
N_ENT, N_ITEM = len(ent_ids), len(item_ids)
u = torch.as_tensor(tr_e[ok]); i = torch.as_tensor(tr_i[ok]); y = torch.as_tensor(tr_y[ok])
tei = te_i[teok]; tey = te_y[teok]
print(f"train={len(tr):,} test={len(te):,}", flush=True)

class MLP(nn.Module):
    def __init__(self, seed=0, e_dim=32):
        super().__init__(); r = np.random.default_rng(seed)
        def draw(sh, std=0.05):
            return torch.as_tensor(np.ascontiguousarray(r.normal(0, std, sh).astype(np.float32)))
        g = torch.Generator().manual_seed(seed)
        self.b0 = nn.Parameter(torch.zeros(1)); self.bu = nn.Parameter(torch.zeros(N_ENT))
        self.bm = nn.Parameter(torch.zeros(N_ITEM))
        self.Eu = nn.Parameter(draw((N_ENT, e_dim))); self.Em = nn.Parameter(draw((N_ITEM, e_dim)))
        self.net = nn.Sequential(nn.Linear(2 * e_dim, 64), nn.ReLU(), nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 1))
        for lin in self.net:
            if isinstance(lin, nn.Linear):
                with torch.no_grad():
                    lin.weight.uniform_(-1 / lin.in_features ** 0.5, 1 / lin.in_features ** 0.5, generator=g)
                    lin.bias.uniform_(-1 / lin.in_features ** 0.5, 1 / lin.in_features ** 0.5, generator=g)
    def forward(self, a, b):
        return self.b0[0] + self.bu[a] + self.bm[b] + self.net(torch.cat([self.Eu[a], self.Em[b]], 1)).squeeze(1)

def sp_rho(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))

def difficulty(m, ref_idx):
    ref = torch.as_tensor(ref_idx); d = np.empty(N_ITEM)
    with torch.no_grad():
        for s0 in range(0, N_ITEM, 1500):
            s1 = min(s0 + 1500, N_ITEM); gg = torch.as_tensor(np.arange(s0, s1))
            U = ref.repeat_interleave(s1 - s0); M = gg.repeat(len(ref))
            base = m.b0[0] + m.bu[U] + m.bm[M]
            d[s0:s1] = (m.forward(U, M) - base).numpy().reshape(len(ref), s1 - s0).mean(0)
    return d + m.bm.detach().numpy()

def train_mlp(seed, epochs=60):
    torch.manual_seed(seed); m = MLP(seed=seed)
    opt = torch.optim.Adam([{"params": [m.b0, m.bu, m.bm, m.Eu, m.Em], "weight_decay": 1e-4},
                            {"params": list(m.net.parameters()), "weight_decay": 1e-4}], lr=0.02)
    rt = np.random.default_rng(seed); n = u.numel()
    for ep in range(epochs):
        pm = rt.permutation(n)
        for s in range(0, n, 8192):
            b = torch.as_tensor(pm[s:s + 8192])
            opt.zero_grad(set_to_none=True)
            ((m.forward(u[b], i[b]) - y[b]) ** 2).mean().backward()
            opt.step()
        if (ep + 1) % 30 == 0:
            with torch.no_grad():
                pv = m.forward(torch.as_tensor(te_e[teok]), torch.as_tensor(te_i[teok])).numpy()
            print(f"   [mlp s{seed}] ep{ep+1} test_rmse={np.sqrt(((pv-tey)**2).mean()):.4f}", flush=True)
    with torch.no_grad():
        pv = m.forward(torch.as_tensor(te_e[teok]), torch.as_tensor(te_i[teok])).numpy()
    print(f"[mlp seed={seed}] TEST rmse={np.sqrt(((pv-tey)**2).mean()):.4f}", flush=True)
    return m

REF = np.random.default_rng(7).choice(N_ENT, 600, replace=False)
REFA, REFB = REF[:300], REF[300:]
m0, m1 = train_mlp(0), train_mlp(1)
d0, d1 = difficulty(m0, REF), difficulty(m1, REF)
d0a, d0b = difficulty(m0, REFA), difficulty(m0, REFB)

# 原始统计量的 split-half (按 play 随机对半)
hal = np.random.default_rng(3).integers(0, 2, size=tei.size).astype(bool)
def rawstat(mask):
    c = np.bincount(tei[mask], minlength=N_ITEM).astype(float)
    return np.bincount(tei[mask], weights=tey[mask].astype(float), minlength=N_ITEM) / np.maximum(c, 1)
rawA, rawB = rawstat(~hal), rawstat(hal)
valid = (np.bincount(tei, minlength=N_ITEM) > 0) & (np.bincount(tei[~hal], minlength=N_ITEM) > 0) \
        & (np.bincount(tei[hal], minlength=N_ITEM) > 0)
print("\n=== 稳定性 (Spearman, 只算两半都有 play 的谱面) ===")
print(f"  MLP seed0 vs seed1            : {sp_rho(d0[valid], d1[valid]):.4f}")
print(f"  MLP 参考玩家 halfA vs halfB    : {sp_rho(d0a[valid], d0b[valid]):.4f}")
print(f"  raw 按 play 对半 A vs B        : {sp_rho(rawA[valid], rawB[valid]):.4f}")
print(f"  MLP seed0 vs seed1 (test RMSE 分别): {np.sqrt(((m0.forward(torch.as_tensor(te_e[teok]),torch.as_tensor(te_i[teok])).numpy()-tey)**2).mean()):.4f} / "
      f"{np.sqrt(((m1.forward(torch.as_tensor(te_e[teok]),torch.as_tensor(te_i[teok])).numpy()-tey)**2).mean()):.4f}")
print("ALL_DONE", flush=True)
