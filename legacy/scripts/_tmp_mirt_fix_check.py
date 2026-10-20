"""Scratch 2: (a) user's intent done right  (b) domain-free MLP baseline."""
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch import nn
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
print(f"rows train={len(tr):,} test={len(te):,} ents={len(ent_ids):,} items={len(item_ids):,}", flush=True)
N_ENT, N_ITEM, DIM = len(ent_ids), len(item_ids), 16

class Base(nn.Module):
    def __init__(self, seed=0, c_std=0.05):
        super().__init__(); r = np.random.default_rng(seed)
        def draw(sh, std=0.05):
            return torch.as_tensor(np.ascontiguousarray(r.normal(0, std, sh).astype(np.float32)))
        self.b0 = nn.Parameter(torch.zeros(1)); self.bu = nn.Parameter(torch.zeros(N_ENT))
        self.bm = nn.Parameter(torch.zeros(N_ITEM))
        self.P = nn.Parameter(draw((N_ENT, DIM))); self.C = nn.Parameter(draw((N_ITEM, DIM), c_std))
        self.D = nn.Parameter(draw((N_ITEM, DIM)))

class NewCorrect(Base):
    """用户意图的正确参数化: c=softmax(C) 当'成分占比', D=难度, 失分率在 ACC 空间相加."""
    def fwd(self, u, m):
        out = self.b0[0] + self.bu[u] + self.bm[m]
        return out + torch.logsumexp(F.log_softmax(self.C[m], dim=1) + self.D[m] - self.P[u], dim=1)

class MLP(Base):
    """零结构假设基线: 纯 ID embedding 拼起来过 MLP (NCF 式), 无 IRT/混合假设."""
    def __init__(self, seed=0, e_dim=32):
        super().__init__(seed=seed)
        self.Eu = nn.Parameter(self.P.detach().clone().repeat(1, 2).contiguous())
        self.Em = nn.Parameter(self.C.detach().clone().repeat(1, 2).contiguous())
        self.net = nn.Sequential(nn.Linear(2 * e_dim, 64), nn.ReLU(), nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 1))
    def fwd(self, u, m):
        out = self.b0[0] + self.bu[u] + self.bm[m]
        x = torch.cat([self.Eu[u], self.Em[m]], dim=1)
        return out + self.net(x).squeeze(1)

def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))

u = torch.as_tensor(tr_e[ok]); i = torch.as_tensor(tr_i[ok]); y = torch.as_tensor(tr_y[ok])
uu = torch.as_tensor(te_e[teok]); ii = torch.as_tensor(te_i[teok]); yy = te_y[teok]

def train(name, model, groups, epochs=60):
    opt = torch.optim.Adam(groups, lr=0.02)
    rt = np.random.default_rng(0); n = u.numel()
    for ep in range(epochs):
        pm = rt.permutation(n)
        for s in range(0, n, 8192):
            b = torch.as_tensor(pm[s:s + 8192])
            opt.zero_grad(set_to_none=True)
            ((model.fwd(u[b], i[b]) - y[b]) ** 2).mean().backward()
            opt.step()
        if (ep + 1) % 20 == 0:
            with torch.no_grad():
                ptr = model.fwd(u[:200000], i[:200000]).numpy()
            print(f"   [{name}] ep{ep+1} train_rmse={np.sqrt(((ptr-y[:200000].numpy())**2).mean()):.4f}", flush=True)
    with torch.no_grad():
        pred = model.fwd(uu, ii).numpy(); inter = (model.fwd(uu, ii) - (model.b0[0] + model.bu[uu] + model.bm[ii])).numpy()
        c = torch.softmax(model.C, dim=1).numpy()
        expo = F.log_softmax(model.C[ii], dim=1) + model.D[ii] - model.P[uu]
        w = torch.softmax(expo, dim=1).numpy()
    e = pred - yy
    print(f"[{name}] TEST rmse={np.sqrt((e**2).mean()):.4f} mae={np.abs(e).mean():.4f} spear={spearman(pred,yy):.4f} "
          f"| b0={float(model.b0[0]):+.3f} | inter mean={inter.mean():+.3f} sd={inter.std():.3f}", flush=True)
    with np.errstate(invalid='ignore'):
        ent = -((c * np.log(np.clip(c, 1e-12, 1))).sum(1) / np.log(c.shape[1])).mean()
        print(f"            C: 熵={ent:.4f} mean_max={c.max(1).mean():.4f} 跨图sd={c.std(0).mean():.5f} "
              f"| 实际占比w: 熵={(-(w*np.log(np.clip(w,1e-12,1))).sum(1)/np.log(w.shape[1])).mean():.4f} "
              f"corr(c,w)={np.corrcoef(c[ii.numpy()].ravel(), w.ravel())[0,1]:+.4f}", flush=True)

print("=== (a) 你的意图 + 正确参数化 (c=softmax(C) 当占比, D 自由当难度) ===", flush=True)
m1 = NewCorrect(c_std=0.5)
train('new_correct', m1, [{"params": [m1.b0, m1.bu, m1.bm, m1.P, m1.D], "weight_decay": 1e-4},
                          {"params": [m1.C], "weight_decay": 0.0}])
print("=== (b) 零 domain knowledge 基线: 纯 ID embedding + MLP ===", flush=True)
m2 = MLP()
train('mlp', m2, [{"params": [m2.b0, m2.bu, m2.bm], "weight_decay": 1e-4},
                  {"params": [m2.Eu, m2.Em], "weight_decay": 1e-4},
                  {"params": list(m2.net.parameters()), "weight_decay": 1e-4}])
print("ALL_DONE", flush=True)
