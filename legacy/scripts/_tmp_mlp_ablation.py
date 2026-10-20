"""Scratch 13: MLP 的改进方向消融 (同一划分/seed/batch, 30 epoch 对齐比较).

四个变体, embedding 参数量完全相同 (987+19644)x32:
  base       : wd=1e-4 全部, lr=0.02 全部, 纯 concat->MLP
  wd0        : embedding 不吃 wd (wd=0), 其余同上
  wd0+lr     : 再加 dense 层 lr=1e-3 (embedding 仍 0.02)
  wd0+lr+gmf : 再加显式双线性支路 <Eu,Em> (NeuMF 式, 共享 embedding)
"""
import numpy as np, pandas as pd, torch
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
N_ENT, N_ITEM = len(ent_ids), len(item_ids)
u = torch.as_tensor(tr_e[ok]); i = torch.as_tensor(tr_i[ok]); y = torch.as_tensor(tr_y[ok])
uu = torch.as_tensor(te_e[teok]); ii = torch.as_tensor(te_i[teok]); yy = te_y[teok]
print(f"train={len(tr):,} test={len(te):,} ents={N_ENT:,} items={N_ITEM:,}", flush=True)

class MLP(nn.Module):
    def __init__(self, seed=0, e_dim=32, gmf=False):
        super().__init__(); self.gmf = gmf; r = np.random.default_rng(seed)
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
        eu, emb = self.Eu[a], self.Em[b]
        o = self.b0[0] + self.bu[a] + self.bm[b] + self.net(torch.cat([eu, emb], 1)).squeeze(1)
        if self.gmf:
            o = o + (eu * emb).sum(1)
        return o

def run(name, wd_emb, lr_emb, lr_dense, gmf, epochs=30, seed=0):
    torch.manual_seed(seed); m = MLP(seed=seed, gmf=gmf)
    emb_ids = {id(m.Eu), id(m.Em)}
    emb_p = [p for p in m.parameters() if id(p) in emb_ids]
    dense_p = [p for p in m.parameters() if id(p) not in emb_ids]
    opt = torch.optim.Adam([{"params": emb_p, "weight_decay": wd_emb, "lr": lr_emb},
                            {"params": dense_p, "weight_decay": 1e-4, "lr": lr_dense}])
    n_emb = sum(p.numel() for p in emb_p); n_dense = sum(p.numel() for p in dense_p)
    rt = np.random.default_rng(seed); n = u.numel()
    for ep in range(epochs):
        pm = rt.permutation(n)
        for s in range(0, n, 8192):
            bl = torch.as_tensor(pm[s:s + 8192])
            opt.zero_grad(set_to_none=True)
            ((m.forward(u[bl], i[bl]) - y[bl]) ** 2).mean().backward()
            opt.step()
    with torch.no_grad():
        ptr = m.forward(u[:150000], i[:150000]).numpy()
        pv = m.forward(uu, ii).numpy()
        base = (m.b0[0] + m.bu[uu] + m.bm[ii]).numpy()
        inter = pv - base
        eu = m.Eu.detach(); em_ = m.Em.detach()
        wnorm = float(eu.norm(dim=1).mean()); mnorm = float(em_.norm(dim=1).mean())
    trr = np.sqrt(((ptr - y[:150000].numpy()) ** 2).mean()); ter = np.sqrt(((pv - yy) ** 2).mean())
    print(f"[{name:<11}] emb={n_emb:,} dense={n_dense:,} | train_rmse={trr:.4f} TEST_rmse={ter:.4f} "
          f"| gap={ter-trr:+.4f} | |Eu|={wnorm:.3f} |Em|={mnorm:.3f} | sd(inter)={inter.std():.4f}", flush=True)
    return ter

print("\n=== MLP 变体消融 (30 epoch, batch 8192, seed 0) ===", flush=True)
res = {}
res['base'] = run('base', 1e-4, 0.02, 0.02, False)
res['wd0'] = run('wd0', 0.0, 0.02, 0.02, False)
res['wd0+lr'] = run('wd0+lr', 0.0, 0.02, 1e-3, False)
res['wd0+lr+gmf'] = run('wd0+lr+gmf', 0.0, 0.02, 1e-3, True)
print("\n=== 汇总 ===", flush=True)
for kk, v in res.items():
    print(f"  {kk:<11} TEST_rmse={v:.4f}   vs base {(v/res['base']-1)*100:+.2f}%", flush=True)
print("ALL_DONE", flush=True)
