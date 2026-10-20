"""Scratch 4: after freeing D -- isolating the remaining problems.
A) code as it now stands (P*softmax(C)+D, D free, repo default wd=1e-4 on everything)
B) user's stated intent (log_softmax(C)+D-P, D free with wd=0, C wd=0 + wide init)
"""
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
N_ENT, N_ITEM, DIM = len(ent_ids), len(item_ids), 16
print(f"rows train={len(tr):,} test={len(te):,} ents={N_ENT:,} items={N_ITEM:,}", flush=True)

class M(nn.Module):
    def __init__(self, kind, seed=0, c_std=0.05):
        super().__init__(); self.kind = kind; r = np.random.default_rng(seed)
        def draw(sh, std=0.05):
            return torch.as_tensor(np.ascontiguousarray(r.normal(0, std, sh).astype(np.float32)))
        self.b0 = nn.Parameter(torch.zeros(1)); self.bu = nn.Parameter(torch.zeros(N_ENT))
        self.bm = nn.Parameter(torch.zeros(N_ITEM))
        self.P = nn.Parameter(draw((N_ENT, DIM))); self.C = nn.Parameter(draw((N_ITEM, DIM), c_std))
        self.D = nn.Parameter(draw((N_ITEM, DIM)))          # D 自由 (无 softplus)
    def expo(self, u, m):
        P = self.P[u]; C = self.C[m]; D = self.D[m]
        if self.kind == 'A':      # 现行代码
            return P * torch.softmax(C, dim=1) + D
        return F.log_softmax(C, dim=1) + D - P              # B: 用户说的语义
    def fwd(self, u, m):
        return self.b0[0] + self.bu[u] + self.bm[m] + torch.logsumexp(self.expo(u, m), dim=1)

def spear(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))

u = torch.as_tensor(tr_e[ok]); i = torch.as_tensor(tr_i[ok]); y = torch.as_tensor(tr_y[ok])
uu = torch.as_tensor(te_e[teok]); ii = torch.as_tensor(te_i[teok]); yy = te_y[teok]

def run(name, kind, c_std, c_wd, d_wd):
    m = M(kind, c_std=c_std)
    if d_wd is None and c_wd is None:           # 完全按仓库默认: 单组 wd=1e-4
        opt = torch.optim.Adam([m.b0, m.bu, m.bm, m.P, m.C, m.D], lr=0.02, weight_decay=1e-4)
    else:
        opt = torch.optim.Adam([{"params": [m.b0, m.bu, m.bm, m.P], "weight_decay": 1e-4},
                                {"params": [m.C], "weight_decay": c_wd},
                                {"params": [m.D], "weight_decay": d_wd}], lr=0.02)
    rt = np.random.default_rng(0); n = u.numel(); first = {}
    for ep in range(60):
        pm = rt.permutation(n)
        for s in range(0, n, 8192):
            b = torch.as_tensor(pm[s:s + 8192])
            opt.zero_grad(set_to_none=True)
            ((m.fwd(u[b], i[b]) - y[b]) ** 2).mean().backward()
            if ep == 0 and s == 0:
                first = {kk: float(m._parameters[kk].grad.norm()) / m._parameters[kk].numel() ** 0.5
                         for kk in ['P', 'C', 'D']}
            opt.step()
        if (ep + 1) % 20 == 0:
            with torch.no_grad():
                ptr = m.fwd(u[:200000], i[:200000]).numpy()
            print(f"   [{name}] ep{ep+1} train_rmse={np.sqrt(((ptr-y[:200000].numpy())**2).mean()):.4f}", flush=True)
    with torch.no_grad():
        pred = m.fwd(uu, ii).numpy(); ex = m.expo(uu, ii)
        inter = torch.logsumexp(ex, dim=1).numpy()
        w = torch.softmax(ex, dim=1).numpy()
        c = torch.softmax(m.C, dim=1).numpy()
        D = m.D.detach().numpy()
    e = pred - yy
    ent = -((c * np.log(np.clip(c, 1e-12, 1))).sum(1) / np.log(c.shape[1])).mean()
    ent_w = -((w * np.log(np.clip(w, 1e-12, 1))).sum(1) / np.log(w.shape[1])).mean()
    print(f"[{name}] TEST rmse={np.sqrt((e**2).mean()):.4f} mae={np.abs(e).mean():.4f} spear={spear(pred,yy):.4f}", flush=True)
    print(f"     b0={float(m.b0[0]):+.3f}  inter: mean={inter.mean():+.3f} sd={inter.std():.3f}", flush=True)
    print(f"     D: mean={D.mean():+.3f} sd_overall={D.std():.3f} sd_across_charts(mean over dims)="
          f"{D.std(0).mean():.3f}  |D|范围=[{D.min():+.2f},{D.max():+.2f}]", flush=True)
    print(f"     C: 归一化熵={ent:.4f} 有效成分数={np.exp(ent*np.log(c.shape[1])):.2f} "
          f"mean_max={c.max(1).mean():.4f} | 实际占比w: 熵={ent_w:.4f} corr(c,w)="
          f"{np.corrcoef(c[ii.numpy()].ravel(), w.ravel())[0,1]:+.4f}", flush=True)
    print(f"     首步梯度RMS: P={first['P']:.3e} C={first['C']:.3e} D={first['D']:.3e} "
          f"(C/P={first['C']/first['P']:.5f})", flush=True)

print("=== A) 现行代码 + D 已自由 + 仓库默认 wd=1e-4(单组, 含 D) ===", flush=True)
run('A_code_default', 'A', 0.05, None, None)
print("=== B) 你说的语义(log_softmax) + D 自由且 wd=0 + C wd=0/宽初始化 ===", flush=True)
run('B_intent_best', 'B', 0.5, 0.0, 0.0)
print("ALL_DONE", flush=True)
