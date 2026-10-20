"""Scratch 10: can each step2 model's CHART-side output predict 官方定数 star? (step3 proxy)"""
import sys, numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch import nn
from argparse import Namespace
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
N_ENT, N_ITEM, DIM = len(ent_ids), len(item_ids), 16
print(f"rows train={len(tr):,} test={len(te):,} ents={N_ENT:,} items={N_ITEM:,}", flush=True)

u = torch.as_tensor(tr_e[ok]); i = torch.as_tensor(tr_i[ok]); y = torch.as_tensor(tr_y[ok])
uu = torch.as_tensor(te_e[teok]); ii = torch.as_tensor(te_i[teok]); yy = te_y[teok]

class MLP(nn.Module):
    """零结构假设: 纯 ID embedding + MLP (NCF)."""
    def __init__(self, seed=0, e_dim=32):
        super().__init__(); r = np.random.default_rng(seed)
        def draw(sh, std=0.05):
            return torch.as_tensor(np.ascontiguousarray(r.normal(0, std, sh).astype(np.float32)))
        self.b0 = nn.Parameter(torch.zeros(1)); self.bu = nn.Parameter(torch.zeros(N_ENT))
        self.bm = nn.Parameter(torch.zeros(N_ITEM))
        self.Eu = nn.Parameter(draw((N_ENT, e_dim))); self.Em = nn.Parameter(draw((N_ITEM, e_dim)))
        self.net = nn.Sequential(nn.Linear(2 * e_dim, 64), nn.ReLU(), nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, a, b):
        return self.b0[0] + self.bu[a] + self.bm[b] + self.net(torch.cat([self.Eu[a], self.Em[b]], 1)).squeeze(1)

def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))

def train(kind, epochs=60):
    torch.manual_seed(0)
    if kind == 'mlp':
        m = MLP()
        opt = torch.optim.Adam([{"params": [m.b0, m.bu, m.bm, m.Eu, m.Em], "weight_decay": 1e-4},
                                {"params": list(m.net.parameters()), "weight_decay": 1e-4}], lr=0.02)
        fix = lambda: None
    else:
        m = sp.EmbeddingModel(kind, N_ENT, N_ITEM, DIM, 'F0', seed=0, device=torch.device('cpu'),
                              p_drift='none', d_constraint='none', d_init_std=None, drop_bias='none',
                              content='none', chart_feats=None, c_init_std=None)
        ns = Namespace(lr=0.02, wd=1e-4, d_logit_wd=None, c_logit_wd=None)
        opt, wd_of = sp.build_optimizer(m, ns)
        print(f"  [{kind}] wd: " + ", ".join(f"{kk}={wd_of[kk]:g}" for kk in m.active), flush=True)
        fix = m.gauge_fix
    rt = np.random.default_rng(0); n = u.numel()
    for ep in range(epochs):
        pm = rt.permutation(n)
        for s in range(0, n, 8192):
            b = torch.as_tensor(pm[s:s + 8192])
            opt.zero_grad(set_to_none=True)
            ((m.forward(u[b], i[b]) - y[b]) ** 2).mean().backward()
            opt.step(); fix()
        if (ep + 1) % 20 == 0:
            with torch.no_grad():
                pv = m.forward(uu, ii).numpy()
            print(f"   [{kind}] ep{ep+1} test_rmse={np.sqrt(((pv-yy)**2).mean()):.4f}", flush=True)
    with torch.no_grad():
        pv = m.forward(uu, ii).numpy()
    print(f"[{kind}] TEST rmse={np.sqrt(((pv-yy)**2).mean()):.4f}", flush=True)
    with torch.no_grad():
        if kind == 'mlp':
            Cv = m.Em.detach().numpy(); Dv = np.zeros_like(Cv); bmv = m.bm.detach().numpy()
        else:
            Cv = m.export_C(); Dv = m.export_D(); bmv = m.np_array('bm')
        # 群体难度: 300 个采样实体的平均预测 loss
        sel = torch.as_tensor(np.random.default_rng(1).choice(N_ENT, 300, replace=False))
        pop = np.empty(N_ITEM, dtype=np.float64)
        for s0 in range(0, N_ITEM, 2000):
            s1 = min(s0 + 2000, N_ITEM)
            gg = torch.as_tensor(np.arange(s0, s1))
            U = sel.repeat_interleave(s1 - s0); M = gg.repeat(300)
            pop[s0:s1] = m.forward(U, M).numpy().reshape(300, s1 - s0).mean(0)
    return m, Cv, Dv, bmv, pop

def cv_r2(X, y, k=5, alpha=1.0, seed=0):
    X = np.asarray(X, dtype=np.float64); n = len(y)
    idx = np.random.default_rng(seed).permutation(n); pred = np.empty(n)
    for f in np.array_split(idx, k):
        tr_i2 = np.setdiff1d(idx, f)
        mu, sd = X[tr_i2].mean(0), X[tr_i2].std(0) + 1e-9
        Xa = (X[tr_i2] - mu) / sd; Xb = (X[f] - mu) / sd
        ym = y[tr_i2].mean()
        w = np.linalg.solve(Xa.T @ Xa + alpha * np.eye(X.shape[1]), Xa.T @ (y[tr_i2] - ym))
        pred[f] = Xb @ w + ym
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum(), spearman(pred, y)

meta = pd.read_csv('data/processed/beatmap_meta_1k.csv', usecols=['beatmap_id', 'star'])
star = pd.merge(pd.DataFrame({'beatmap_id': item_ids}), meta, on='beatmap_id', how='left')
star_v = star.star.to_numpy(np.float64)
print(f"star 覆盖率 {np.isfinite(star_v).mean():.2%}", flush=True)

fits = {}
for kind in ['mirt', 'mirt_exp', 'mlp']:
    print(f"===== {kind} =====", flush=True)
    fits[kind] = train(kind)

print("\n=== step3 代理: 用各模型的谱面侧产物回归官方定数 star (5-fold CV, 19,644 谱面) ===")
print(f"{'模型':<10} {'特征':<26} {'维度':>4} {'R2':>8} {'Spearman':>9}")
for kind in ['mirt', 'mirt_exp', 'mlp']:
    _, Cv, Dv, bmv, pop = fits[kind]
    sets = [('b_m 单独(谱面截距)', bmv[:, None]),
            ('群体平均 loss d_m', pop[:, None]),
            ('C 单独', Cv),
            ('D 单独', Dv),
            ('C+D+b_m', np.hstack([Cv, Dv, bmv[:, None]]))]
    if kind == 'mlp':
        sets = [('b_m 单独(谱面截距)', bmv[:, None]),
                ('群体平均 loss d_m', pop[:, None]),
                ('E_m 单独', Cv),
                ('E_m+b_m', np.hstack([Cv, bmv[:, None]]))]
    for name, X in sets:
        r2, sp_ = cv_r2(X, star_v)
        print(f"{kind:<10} {name:<26} {X.shape[1]:>4} {r2:>8.4f} {sp_:>9.4f}", flush=True)
print("ALL_DONE", flush=True)
