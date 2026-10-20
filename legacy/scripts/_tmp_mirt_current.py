"""Scratch 6: 用真实类跑用户当前公式 (D-P)*softmax(C), 优化器已修复."""
import sys, numpy as np, pandas as pd, torch
sys.path.insert(0, 'scripts')
from argparse import Namespace
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
print(f"rows train={len(tr):,} test={len(te):,}", flush=True)
u = torch.as_tensor(tr_e[ok]); i = torch.as_tensor(tr_i[ok]); y = torch.as_tensor(tr_y[ok])
uu = torch.as_tensor(te_e[teok]); ii = torch.as_tensor(te_i[teok]); yy = te_y[teok]

torch.manual_seed(0)
m = sp.EmbeddingModel('mirt_exp', len(ent_ids), len(item_ids), 16, 'F0', seed=0,
                      device=torch.device('cpu'), p_drift='none', d_constraint='none',
                      d_init_std=None, drop_bias='none', content='none',
                      chart_feats=None, c_init_std=None)
a = Namespace(d_logit_wd=None, c_logit_wd=None, wd=1e-4, lr=0.02)
opt, wd_of = sp.build_optimizer(m, a)
print(f"no_decay={sorted(m.no_decay)} wd={ {kk: wd_of[kk] for kk in sorted(wd_of)} }", flush=True)
rt = np.random.default_rng(0); n = u.numel()
for ep in range(60):
    pm = rt.permutation(n)
    for s in range(0, n, 8192):
        b = torch.as_tensor(pm[s:s + 8192])
        opt.zero_grad(set_to_none=True)
        ((m.forward(u[b], i[b]) - y[b]) ** 2).mean().backward()
        opt.step()
    if (ep + 1) % 20 == 0:
        with torch.no_grad():
            ptr = m.forward(u[:200000], i[:200000]).numpy()
        print(f"   ep{ep+1} train_rmse={np.sqrt(((ptr-y[:200000].numpy())**2).mean()):.4f}", flush=True)

with torch.no_grad():
    pred = m.forward(uu, ii).numpy()
    c = torch.softmax(m.p['C'], dim=1)[ii]
    ex = (m.p['D'][ii] - m.p['P'][uu]) * c
    inter = torch.logsumexp(ex, dim=1).numpy()
    w = torch.softmax(ex, dim=1).numpy()
    ent = -((c * torch.log(c.clamp_min(1e-12))).sum(1) / np.log(c.shape[1])).mean().item()
    Dn = m.p['D'].detach().numpy(); Pn = m.p['P'].detach().numpy()
e = pred - yy
ra = np.argsort(np.argsort(pred)).astype(float); rb = np.argsort(np.argsort(yy)).astype(float)
sp_ = float(((ra-ra.mean())*(rb-rb.mean())).sum()/np.sqrt(((ra-ra.mean())**2).sum()*((rb-rb.mean())**2).sum()))
print(f"\n[当前公式 (D-P)*softmax(C)] TEST rmse={np.sqrt((e**2).mean()):.4f} mae={np.abs(e).mean():.4f} "
      f"spear={sp_:.4f} b0={float(m.p['b0'][0]):+.3f}", flush=True)
print(f"   inter: mean={inter.mean():+.3f} sd={inter.std():.3f}", flush=True)
print(f"   C: 归一化熵={ent:.4f} 有效成分数={np.exp(ent*np.log(c.shape[1])):.2f} "
      f"corr(c,w)={float(np.corrcoef(c.numpy().ravel(), w.ravel())[0,1]):+.4f}", flush=True)
print(f"   D: mean={Dn.mean():+.3f} sd={Dn.std():.3f} range=[{Dn.min():+.2f},{Dn.max():+.2f}] | "
      f"P: sd={Pn.std():.3f} | 有效载荷 c 均值={c.mean():.4f} (1/16={1/16:.4f})", flush=True)
print("ALL_DONE", flush=True)
