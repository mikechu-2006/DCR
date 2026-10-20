"""Scratch 3: plain mirt + mf_dot on the same synthetic split, as the fair anchor for the MLP."""
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
    def __init__(self, kind, seed=0):
        super().__init__(); self.kind = kind; r = np.random.default_rng(seed)
        def draw(sh, std=0.05):
            return torch.as_tensor(np.ascontiguousarray(r.normal(0, std, sh).astype(np.float32)))
        self.b0 = nn.Parameter(torch.zeros(1)); self.bu = nn.Parameter(torch.zeros(N_ENT))
        self.bm = nn.Parameter(torch.zeros(N_ITEM))
        self.P = nn.Parameter(draw((N_ENT, DIM))); self.C = nn.Parameter(draw((N_ITEM, DIM)))
        if kind != 'mf_dot':
            self.D = nn.Parameter(draw((N_ITEM, DIM)))
    def fwd(self, u, m):
        out = self.b0[0] + self.bu[u] + self.bm[m]; P = self.P[u]; C = self.C[m]
        if self.kind == 'mf_dot':
            return out + (P * C).sum(1)
        return out + ((P - C) * self.D[m]).sum(1)

def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))

u = torch.as_tensor(tr_e[ok]); i = torch.as_tensor(tr_i[ok]); y = torch.as_tensor(tr_y[ok])
uu = torch.as_tensor(te_e[teok]); ii = torch.as_tensor(te_i[teok]); yy = te_y[teok]
for kind in ['mirt', 'mf_dot']:
    m = M(kind)
    params = [m.b0, m.bu, m.bm, m.P, m.C] + ([m.D] if kind != 'mf_dot' else [])
    opt = torch.optim.Adam(params, lr=0.02, weight_decay=1e-4)
    rt = np.random.default_rng(0); n = u.numel()
    for ep in range(60):
        pm = rt.permutation(n)
        for s in range(0, n, 8192):
            b = torch.as_tensor(pm[s:s + 8192])
            opt.zero_grad(set_to_none=True)
            ((m.fwd(u[b], i[b]) - y[b]) ** 2).mean().backward()
            opt.step()
        if (ep + 1) % 20 == 0:
            with torch.no_grad():
                ptr = m.fwd(u[:200000], i[:200000]).numpy()
            print(f"   [{kind}] ep{ep+1} train_rmse={np.sqrt(((ptr-y[:200000].numpy())**2).mean()):.4f}", flush=True)
    with torch.no_grad():
        pred = m.fwd(uu, ii).numpy()
    e = pred - yy
    print(f"[{kind}] TEST rmse={np.sqrt((e**2).mean()):.4f} mae={np.abs(e).mean():.4f} "
          f"spear={spearman(pred,yy):.4f} b0={float(m.b0[0]):+.3f}", flush=True)
print("ALL_DONE", flush=True)
