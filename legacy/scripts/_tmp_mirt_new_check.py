"""Scratch: A/B compare original mirt_exp vs the modified line-304 formula."""
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch import nn
torch.use_deterministic_algorithms(True)

df = pd.read_parquet('data/processed/step1_1k.parquet')
k = int(df.beatmap_id.max()) + 1
key = df.player_id.to_numpy(np.int64) * np.int64(k) + df.beatmap_id.to_numpy(np.int64)
uniq = np.unique(key)
rng = np.random.default_rng(20260928); perm = rng.permutation(uniq.size)
n_tr = int(0.70 * uniq.size); te_set = uniq[perm[n_tr:]]
is_te = np.isin(key, te_set)
tr = df[~is_te].reset_index(drop=True); te = df[is_te].reset_index(drop=True)
del df, key
ent_ids = np.unique(tr.player_id.to_numpy()); item_ids = np.unique(tr.beatmap_id.to_numpy())
em = {e: i for i, e in enumerate(ent_ids)}; im = {m: i for i, m in enumerate(item_ids)}
def enc(d):
    return (d.player_id.map(em).fillna(-1).to_numpy(np.int64),
            d.beatmap_id.map(im).fillna(-1).to_numpy(np.int64),
            d.loss.to_numpy(np.float32))
tr_e, tr_i, tr_y = enc(tr); te_e, te_i, te_y = enc(te)
ok = (tr_e >= 0) & (tr_i >= 0); teok = (te_e >= 0) & (te_i >= 0)
print(f"rows train={len(tr):,} test={len(te):,} | ents={len(ent_ids):,} items={len(item_ids):,} "
      f"| test covered={teok.mean():.4%}", flush=True)

class M(nn.Module):
    def __init__(self, kind, n_ent, n_item, dim, seed=0, c_std=0.05):
        super().__init__(); self.kind = kind; r = np.random.default_rng(seed)
        def draw(sh, std=0.05):
            return torch.as_tensor(np.ascontiguousarray(r.normal(0, std, sh).astype(np.float32)))
        self.b0 = nn.Parameter(torch.zeros(1)); self.bu = nn.Parameter(torch.zeros(n_ent))
        self.bm = nn.Parameter(torch.zeros(n_item))
        self.P = nn.Parameter(draw((n_ent, dim))); self.C = nn.Parameter(draw((n_item, dim), c_std))
        self.D = nn.Parameter(draw((n_item, dim)))
        if kind != 'new_freeD':
            with torch.no_grad():
                self.D.add_(float(np.log(np.expm1(0.05))))
        self.active = ['b0', 'bu', 'bm', 'P', 'C', 'D']
    def softD(self):
        return F.softplus(self.D) if self.kind != 'new_freeD' else self.D
    def inter(self, u, m):
        P = self.P[u]; C = self.C[m]; D = self.softD()[m]
        if self.kind == 'orig':
            return torch.logsumexp(torch.log(D) - (P - C), dim=1)
        return torch.logsumexp(P * torch.softmax(C, dim=1) + D, dim=1)
    def fwd(self, u, m):
        return self.b0[0] + self.bu[u] + self.bm[m] + self.inter(u, m)

def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))

def train(kind, epochs=60, lr=0.02, wd=1e-4, c_wd=None, c_std=0.05):
    m = M(kind, len(ent_ids), len(item_ids), 16, seed=0, c_std=c_std)
    c_wd = wd if c_wd is None else c_wd
    opt = torch.optim.Adam([{"params": [m.b0, m.bu, m.bm, m.P], "weight_decay": wd},
                            {"params": [m.C], "weight_decay": c_wd},
                            {"params": [m.D], "weight_decay": 0.0}], lr=lr)
    u = torch.as_tensor(tr_e[ok]); i = torch.as_tensor(tr_i[ok]); y = torch.as_tensor(tr_y[ok])
    rt = np.random.default_rng(0); n = u.numel(); first = {}
    for ep in range(epochs):
        pm = rt.permutation(n)
        for s in range(0, n, 8192):
            b = torch.as_tensor(pm[s:s + 8192])
            opt.zero_grad(set_to_none=True)
            ((m.fwd(u[b], i[b]) - y[b]) ** 2).mean().backward()
            if ep == 0 and s == 0:
                first = {kk: float(m._parameters[kk].grad.norm()) / m._parameters[kk].numel() ** 0.5
                         for kk in m.active}
            opt.step()
        if (ep + 1) % 20 == 0:
            with torch.no_grad():
                pv = m.fwd(torch.as_tensor(te_e[teok]), torch.as_tensor(te_i[teok])).numpy()
            print(f"   [{kind}] ep{ep+1} rmse={np.sqrt(((pv - te_y[teok])**2).mean()):.4f}", flush=True)
    with torch.no_grad():
        uu = torch.as_tensor(te_e[teok]); ii = torch.as_tensor(te_i[teok])
        pred = m.fwd(uu, ii).numpy(); inter = m.inter(uu, ii).numpy()
        c = torch.softmax(m.C, dim=1).numpy(); D = m.softD().numpy()
        # 每个样本里"各维占比"的真实权重 w_i = exp(exponent_i)/sum
        expo = m.P[uu] * torch.softmax(m.C, dim=1)[ii] + m.softD()[ii] if kind != 'orig' else None
        if expo is not None:
            w = torch.softmax(expo, dim=1).numpy()
    yy = te_y[teok]; e = pred - yy
    ents = -((c * np.log(np.clip(c, 1e-12, 1))).sum(1)) / np.log(c.shape[1])
    out = dict(kind=kind, rmse=float(np.sqrt((e ** 2).mean())), mae=float(np.abs(e).mean()),
               spear=spearman(pred, yy), b0=float(m.b0[0]), sd_inter=float(inter.std()),
               inter_mean=float(inter.mean()), n_pos_inter=float((inter > 0).mean()),
               yhat_gt0=float((pred > 0).mean()), c_ent=float(ents.mean()),
               c_maxshare=float(c.max(1).mean()), c_sd=float(c.std(0).mean()),
               D_mean=float(D.mean()), P_norm=float(m.P.detach().norm().item()),
               g_C=first['C'], g_D=first['D'], g_P=first['P'])
    if expo is not None:
        out['w_maxshare'] = float(w.max(1).mean()); out['w_ent'] = float(
            (-(w * np.log(np.clip(w, 1e-12, 1))).sum(1)).mean() / np.log(w.shape[1]))
        out['corr_c_w'] = float(np.corrcoef(c[ii.numpy()].ravel(), w.ravel())[0, 1])
    print(f"[{kind:10s}] rmse={out['rmse']:.4f} mae={out['mae']:.4f} spear={out['spear']:.4f} "
          f"b0={out['b0']:+.3f} | inter mean={out['inter_mean']:+.3f} sd={out['sd_inter']:.3f} "
          f"| yhat>0={out['yhat_gt0']:.1%}", flush=True)
    print(f"             C: 归一化熵={out['c_ent']:.4f} mean_max_share={out['c_maxshare']:.4f} "
          f"跨图sd={out['c_sd']:.5f} | D_mean={out['D_mean']:+.3f} |P|={out['P_norm']:.2f}", flush=True)
    print(f"             首步梯度RMS: C={out['g_C']:.3e} D={out['g_D']:.3e} P={out['g_P']:.3e} "
          f"(C/D={out['g_C']/out['g_D']:.5f})", flush=True)
    if 'w_maxshare' in out:
        print(f"             实际占比 w: 熵={out['w_ent']:.4f} mean_max={out['w_maxshare']:.4f} "
              f"corr(c,w)={out['corr_c_w']:+.4f}", flush=True)
    return out

for kind, cw, cs in [('new', None, 0.05), ('new_freeD', None, 0.05), ('new_fixed', 0.0, 0.5)]:
    train(kind, c_wd=cw, c_std=cs)
print("ALL_DONE", flush=True)
