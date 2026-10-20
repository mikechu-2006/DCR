"""Scratch 5: 用真实 EmbeddingModel + build_optimizer 验证修复前后 (公式 = 用户当前的 304 行)."""
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
N_ENT, N_ITEM = len(ent_ids), len(item_ids)
print(f"rows train={len(tr):,} test={len(te):,} ents={N_ENT:,} items={N_ITEM:,}", flush=True)

u = torch.as_tensor(tr_e[ok]); i = torch.as_tensor(tr_i[ok]); y = torch.as_tensor(tr_y[ok])
uu = torch.as_tensor(te_e[teok]); ii = torch.as_tensor(te_i[teok]); yy = te_y[teok]

def spear(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))

DEV = torch.device('cpu')
def run(tag, d_logit_wd, c_logit_wd, c_init_std):
    torch.manual_seed(0)
    m = sp.EmbeddingModel('mirt_exp', N_ENT, N_ITEM, 16, 'F0', seed=0, device=DEV,
                          p_drift='none', d_constraint='none', d_init_std=None,
                          drop_bias='none', content='none', chart_feats=None,
                          c_init_std=c_init_std)
    a = Namespace(d_logit_wd=d_logit_wd, c_logit_wd=c_logit_wd, wd=1e-4, lr=0.02)
    opt, wd_of = sp.build_optimizer(m, a)
    print(f"[{tag}] no_decay={sorted(m.no_decay)} wd={ {kk: wd_of[kk] for kk in sorted(wd_of)} } "
          f"C_init_std={float(m.p['C'].std()):.3f}", flush=True)
    rt = np.random.default_rng(0); n = u.numel()
    for ep in range(60):
        pm = rt.permutation(n)
        for s in range(0, n, 8192):
            b = torch.as_tensor(pm[s:s + 8192])
            opt.zero_grad(set_to_none=True)
            ((m.forward(u[b], i[b]) - y[b]) ** 2).mean().backward()
            opt.step()
    with torch.no_grad():
        pred = m.forward(uu, ii).numpy()
        P = m.p['P'][uu]; C = m.p['C'][ii]; D = m.p['D'][ii]
        ex = -P * torch.softmax(C, dim=1) + D
        inter = torch.logsumexp(ex, dim=1).numpy()
        w = torch.softmax(ex, dim=1).numpy()
        c = torch.softmax(m.p['C'], dim=1).numpy()
        Dn = m.p['D'].detach().numpy()
    e = pred - yy
    ent = -((c * np.log(np.clip(c, 1e-12, 1))).sum(1) / np.log(c.shape[1])).mean()
    print(f"[{tag}] TEST rmse={np.sqrt((e**2).mean()):.4f} mae={np.abs(e).mean():.4f} "
          f"spear={spear(pred, yy):.4f} b0={float(m.p['b0'][0]):+.3f}", flush=True)
    print(f"      inter: mean={inter.mean():+.3f} sd={inter.std():.3f} | "
          f"D: mean={Dn.mean():+.3f} sd={Dn.std():.3f} range=[{Dn.min():+.2f},{Dn.max():+.2f}]", flush=True)
    print(f"      C: 归一化熵={ent:.4f} 有效成分数={np.exp(ent*np.log(c.shape[1])):.2f} "
          f"corr(c,w)={np.corrcoef(c[ii.numpy()].ravel(), w.ravel())[0,1]:+.4f}", flush=True)
    return float(np.sqrt((e ** 2).mean()))

r_old = run('修复前(旧默认: C,D 都吃 wd=1e-4, C init 0.05)', 1e-4, 1e-4, 0.05)
r_new = run('修复后(新默认: C,D 都 wd=0, C init 0.5)', None, None, None)
print(f"\n=> {r_old:.4f} -> {r_new:.4f}  ({(r_new/r_old-1)*100:+.1f}%)", flush=True)
print("ALL_DONE", flush=True)
