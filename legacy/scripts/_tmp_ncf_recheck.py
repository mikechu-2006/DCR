"""Scratch 14: 用仓库当前协议(step1 自带 split 列)重验 NCF-MLP, 并查'自由谱面表到底有没有在干活'."""
import sys, numpy as np, pandas as pd, torch
from torch import nn
from argparse import Namespace
sys.path.insert(0, 'scripts')
import step2_predict as sp
torch.use_deterministic_algorithms(True)

df = pd.read_parquet('data/processed/step1_1k.parquet',
                     columns=['player_id', 'beatmap_id', 'loss', 'split'])
lab = df.split.astype(str).to_numpy()
tr = df[lab == 'train'].reset_index(drop=True)
te = df[lab == 'test'].reset_index(drop=True)
del df
print(f"train={len(tr):,} test={len(te):,} dropped={int((lab=='dropped').sum()):,}", flush=True)
ent_ids = np.unique(tr.player_id.to_numpy()); item_ids = np.unique(tr.beatmap_id.to_numpy())
em = {e: i for i, e in enumerate(ent_ids)}; im = {m: i for i, m in enumerate(item_ids)}
def enc(d):
    return (d.player_id.map(em).fillna(-1).to_numpy(np.int64),
            d.beatmap_id.map(im).fillna(-1).to_numpy(np.int64), d.loss.to_numpy(np.float32))
tr_e, tr_i, tr_y = enc(tr); te_e, te_i, te_y = enc(te)
ok = (tr_e >= 0) & (tr_i >= 0); teok = (te_e >= 0) & (te_i >= 0)
N_ENT, N_ITEM = len(ent_ids), len(item_ids)
u = torch.as_tensor(tr_e[ok]); i = torch.as_tensor(tr_i[ok]); y = torch.as_tensor(tr_y[ok])
uu = torch.as_tensor(te_e[teok]); ii = torch.as_tensor(te_i[teok]); yy = te_y[teok]
print(f"vocab ents={N_ENT:,} items={N_ITEM:,} | train in vocab {ok.mean():.2%} test covered {teok.mean():.2%}", flush=True)
print(f"每个谱面 train play 数中位 = {np.median(np.bincount(tr_i[ok], minlength=N_ITEM)[np.bincount(tr_i[ok], minlength=N_ITEM)>0]):.0f}", flush=True)

class MLP(nn.Module):
    def __init__(self, seed=0, e_dim=32):
        super().__init__(); self.e_dim = e_dim; r = np.random.default_rng(seed)
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
                    b = 1.0 / lin.in_features ** 0.5
                    lin.weight.uniform_(-b, b, generator=g); lin.bias.uniform_(-b, b, generator=g)
    def forward(self, a, b):
        return (self.b0[0] + self.bu[a] + self.bm[b]
                + self.net(torch.cat([self.Eu[a], self.Em[b]], 1)).squeeze(1))
    def inter(self, a, b):
        return self.net(torch.cat([self.Eu[a], self.Em[b]], 1)).squeeze(1)

def run(kind, e_dim=32, epochs=60):
    torch.manual_seed(0)
    if kind == 'mlp':
        m = MLP(e_dim=e_dim)
        opt = torch.optim.Adam(m.parameters(), lr=0.02, weight_decay=1e-4)
        nprm = sum(p.numel() for p in m.parameters())
        fix = lambda: None; inter = lambda a, b: m.inter(a, b); emb = (m.Eu, m.Em)
    else:
        m = sp.EmbeddingModel('mirt', N_ENT, N_ITEM, 16, 'F0', seed=0, device=torch.device('cpu'),
                              p_drift='none', d_constraint='none', d_init_std=None, drop_bias='none',
                              content='none', chart_feats=None, c_init_std=None)
        opt, _ = sp.build_optimizer(m, Namespace(lr=0.02, wd=1e-4, d_logit_wd=None, c_logit_wd=None))
        nprm = sum(m.p[k].numel() for k in m.active)
        fix = m.gauge_fix
        inter = lambda a, b: ((m.p['P'][a] - m.p['C'][b]) * m.p['D'][b]).sum(1)
        emb = (m.p['P'], m.p['C'], m.p['D'])
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
            print(f"   [{kind}{e_dim if kind=='mlp' else ''}] ep{ep+1} test_rmse={np.sqrt(((pv-yy)**2).mean()):.4f}", flush=True)
    with torch.no_grad():
        pr_tr = m.forward(u[:300000], i[:300000]).numpy(); pv = m.forward(uu, ii).numpy()
        base = (m.p['b0'][0] if kind != 'mlp' else m.b0[0]) + \
               (m.p['bu'][uu] if kind != 'mlp' else m.bu[uu]) + \
               (m.p['bm'][ii] if kind != 'mlp' else m.bm[ii])
        b_u = (m.p['bu'][uu] if kind != 'mlp' else m.bu[uu]).numpy()
        b_m = (m.p['bm'][ii] if kind != 'mlp' else m.bm[ii]).numpy()
        it = inter(uu, ii).numpy()
        norms = [float(e.detach().norm(dim=1).mean()) for e in emb]
    trr = np.sqrt(((pr_tr - y[:300000].numpy()) ** 2).mean()); ter = np.sqrt(((pv - yy) ** 2).mean())
    vt = pv.var()
    print(f"[{kind}{e_dim if kind=='mlp' else ''}] params={nprm:,} train_rmse={trr:.4f} TEST_rmse={ter:.4f} gap={ter-trr:+.4f}", flush=True)
    print(f"    Var 分解: b_u={b_u.var()/vt:5.1%} b_m={b_m.var()/vt:5.1%} inter={it.var()/vt:5.1%} | sd(inter)={it.std():.3f}", flush=True)
    print(f"    向量范数均值(init 期望 0.05*sqrt(d)={0.05*np.sqrt(8 if e_dim==8 else (16 if kind!='mlp' else 32)):.3f}): "
          + ", ".join(f"{x:.3f}" for x in norms), flush=True)
    return ter

print("\n=== 仓库当前协议 (train=split:train, test=split:test 未清洗) ===", flush=True)
r = {}
r['mirt'] = run('mirt')
r['mlp32'] = run('mlp', 32)
r['mlp8'] = run('mlp', 8)
print("\n=== 汇总 ===", flush=True)
for kk, v in r.items():
    print(f"  {kk:<6} TEST_rmse={v:.4f}", flush=True)
print("ALL_DONE", flush=True)
