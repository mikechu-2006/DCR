"""Scratch 11: label-free chart difficulty from behaviour only (no star anywhere).

d_m = b_m + E_{u~ref}[interaction(u,m)]  -- 固定参考玩家总体上的期望 loss = 去混杂难度
raw_m = 该谱面 test play 的平均 loss      -- 无模型的原始难度估计(被"谁在打"污染)
mix_m = 打该谱面的玩家的(留一)平均 loss   -- 玩家构成项
"""
import sys, numpy as np, pandas as pd, torch
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
u = torch.as_tensor(tr_e[ok]); i = torch.as_tensor(tr_i[ok]); y = torch.as_tensor(tr_y[ok])
uu = torch.as_tensor(te_e[teok]); ii = torch.as_tensor(te_i[teok]); yy = te_y[teok]
print(f"train={len(tr):,} test={len(te):,} ents={N_ENT:,} items={N_ITEM:,}", flush=True)

# ---------- 无模型的原始难度: 每谱面 test 平均 loss ----------
cnt = np.bincount(te_i[teok], minlength=N_ITEM).astype(np.float64)
raw = np.bincount(te_i[teok], weights=yy.astype(np.float64), minlength=N_ITEM) / np.maximum(cnt, 1)
# ---------- 玩家构成项: 每个 play 的玩家"留一"平均 loss ----------
s_u = np.bincount(te_e[teok], weights=yy.astype(np.float64), minlength=N_ENT)
n_u = np.bincount(te_e[teok], minlength=N_ENT).astype(np.float64)
loo = (s_u[te_e[teok]] - yy) / np.maximum(n_u[te_e[teok]] - 1.0, 1.0)
mix = np.bincount(te_i[teok], weights=loo, minlength=N_ITEM) / np.maximum(cnt, 1)
fe = raw - mix                                   # 最朴素的去混杂(减掉玩家水平)
print(f"有 test play 的谱面 {int((cnt>0).sum()):,} / {N_ITEM:,}", flush=True)

class MLP(nn.Module):
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

def train(kind, seed=0, epochs=60):
    torch.manual_seed(seed)
    if kind == 'mlp':
        m = MLP(seed=seed)
        opt = torch.optim.Adam([{"params": [m.b0, m.bu, m.bm, m.Eu, m.Em], "weight_decay": 1e-4},
                                {"params": list(m.net.parameters()), "weight_decay": 1e-4}], lr=0.02)
        fix = lambda: None
    else:
        m = sp.EmbeddingModel(kind, N_ENT, N_ITEM, DIM, 'F0', seed=seed, device=torch.device('cpu'),
                              p_drift='none', d_constraint='none', d_init_std=None, drop_bias='none',
                              content='none', chart_feats=None, c_init_std=None)
        opt, _ = sp.build_optimizer(m, Namespace(lr=0.02, wd=1e-4, d_logit_wd=None, c_logit_wd=None))
        fix = m.gauge_fix
    rt = np.random.default_rng(seed); n = u.numel()
    for ep in range(epochs):
        pm = rt.permutation(n)
        for s in range(0, n, 8192):
            b = torch.as_tensor(pm[s:s + 8192])
            opt.zero_grad(set_to_none=True)
            ((m.forward(u[b], i[b]) - y[b]) ** 2).mean().backward()
            opt.step(); fix()
    with torch.no_grad():
        pv = m.forward(uu, ii).numpy()
    print(f"[{kind} seed={seed}] TEST rmse={np.sqrt(((pv-yy)**2).mean()):.4f}", flush=True)
    return m

def P_(m, name):
    """repo 的 EmbeddingModel 把参数放在 m.p 字典里, 本文件的 MLP 用普通属性."""
    return m.p[name] if hasattr(m, 'p') else getattr(m, name)

REF = np.random.default_rng(7).choice(N_ENT, 600, replace=False)
def difficulty(m):
    """b_m + 固定参考玩家总体上的期望交互项 (与'谁打过这张图'完全无关)."""
    ref = torch.as_tensor(REF); d = np.empty(N_ITEM, dtype=np.float64)
    with torch.no_grad():
        full = m.forward
        for s0 in range(0, N_ITEM, 1500):
            s1 = min(s0 + 1500, N_ITEM)
            gg = torch.as_tensor(np.arange(s0, s1))
            U = ref.repeat_interleave(s1 - s0); M = gg.repeat(len(REF))
            out = full(U, M)
            base = P_(m, 'b0')[0] + P_(m, 'bu')[U] + P_(m, 'bm')[M]
            d[s0:s1] = (out - base).numpy().reshape(len(REF), s1 - s0).mean(0)
        d += P_(m, 'bm').detach().numpy()
    return d

def sp_rho(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))

m_models = {}
for kind in ['mirt', 'mirt_exp', 'mlp']:
    m_models[kind] = difficulty(train(kind))

valid = cnt > 0
print("\n=== A. 各难度定义的排序一致性 (Spearman, 只算有 test play 的谱面) ===")
names = ['raw(原始均值)', 'fe(减玩家水平)', 'mirt d_m', 'mirt_exp d_m', 'mlp d_m']
vecs = [raw[valid], fe[valid], m_models['mirt'][valid], m_models['mirt_exp'][valid], m_models['mlp'][valid]]
print(f"{'':<16}" + "".join(f"{n:>16}" for n in names))
for a, va in zip(names, vecs):
    print(f"{a:<16}" + "".join(f"{sp_rho(va, vb):>16.4f}" for vb in vecs))

print("\n=== B. raw 的谱面间方差分解: raw ≈ 难度 + 玩家构成 ===")
vr, vd, vm = raw[valid].var(), m_models['mlp'][valid].var(), mix[valid].var()
cv_dm = np.cov(m_models['mlp'][valid], mix[valid])[0, 1]
print(f"  Var(raw)={vr:.4f}   Var(d_mlp)={vd:.4f} ({vd/vr:6.1%})   Var(mix)={vm:.4f} ({vm/vr:6.1%})   2Cov={2*cv_dm:+.4f} ({2*cv_dm/vr:6.1%})")
print(f"  corr(raw, mix) = {np.corrcoef(raw[valid], mix[valid])[0,1]:+.4f}   "
      f"corr(raw, d_mlp) = {np.corrcoef(raw[valid], m_models['mlp'][valid])[0,1]:+.4f}")

print("\n=== C. 与客观谱面属性 countTotal 的秩相关 (仅做 sanity check) ===")
meta = pd.read_csv('data/processed/beatmap_meta_1k.csv', usecols=['beatmap_id', 'count_total'])
mm = pd.merge(pd.DataFrame({'beatmap_id': item_ids}), meta, on='beatmap_id', how='left')
ct = mm.count_total.to_numpy(np.float64)
for n, v in zip(names, vecs):
    print(f"  Spearman({n:<16}, countTotal) = {sp_rho(v, ct[valid]):+.4f}")
print("\n=== D. 难度分布 ===")
for n, v in zip(names, vecs):
    print(f"  {n:<16} mean={v.mean():+.4f} sd={v.std():.4f} min={v.min():+.3f} max={v.max():+.3f}")
print("ALL_DONE", flush=True)
