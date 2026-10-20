"""Scratch 9: 对称 wd=0 + gauge 配置 是欠拟合还是过拟合? (train vs test)"""
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
em = {x: j for j, x in enumerate(ent_ids)}; im = {x: j for j, x in enumerate(item_ids)}
def enc(d):
    return (d.player_id.map(em).fillna(-1).to_numpy(np.int64),
            d.beatmap_id.map(im).fillna(-1).to_numpy(np.int64), d.loss.to_numpy(np.float32))
tr_e, tr_i, tr_y = enc(tr); te_e, te_i, te_y = enc(te)
ok = (tr_e >= 0) & (tr_i >= 0); teok = (te_e >= 0) & (te_i >= 0)
u = torch.as_tensor(tr_e[ok]); i = torch.as_tensor(tr_i[ok]); y = torch.as_tensor(tr_y[ok])
uu = torch.as_tensor(te_e[teok]); ii = torch.as_tensor(te_i[teok]); yy = te_y[teok]

def run(tag, symmetric_zero, epochs=60, wd_scale=1.0):
    torch.manual_seed(0)
    m = sp.EmbeddingModel('mirt_exp', len(ent_ids), len(item_ids), 16, 'F0', seed=0,
                          device=torch.device('cpu'), p_drift='none', d_constraint='none',
                          d_init_std=None, drop_bias='none', content='none',
                          chart_feats=None, c_init_std=None)
    if not symmetric_zero:
        m.no_decay.discard("D"); m.no_decay.discard("P")
    a = Namespace(d_logit_wd=None, c_logit_wd=None, wd=1e-4 * wd_scale, lr=0.02)
    opt, _ = sp.build_optimizer(m, a)
    rt = np.random.default_rng(0); n = u.numel()
    for ep in range(epochs):
        pm = rt.permutation(n)
        for s in range(0, n, 8192):
            b = torch.as_tensor(pm[s:s + 8192])
            opt.zero_grad(set_to_none=True)
            ((m.forward(u[b], i[b]) - y[b]) ** 2).mean().backward()
            opt.step(); m.gauge_fix()
    m.gauge_fix()
    with torch.no_grad():
        ptr = m.forward(u, i).numpy(); pte = m.forward(uu, ii).numpy()
        C = m.p['C'].detach(); sc = torch.softmax(C, dim=1)
        sci = sc[ii]
        ent = -((sc * torch.log(sc.clamp_min(1e-12))).sum(1) / np.log(16)).mean().item()
        inter = torch.logsumexp(sci*(m.p['D'][ii]-m.p['P'][uu]), dim=1)
    rtr = float(np.sqrt(((ptr - tr_y[ok])**2).mean())); rte = float(np.sqrt(((pte - yy)**2).mean()))
    print(f"[{tag}] TRAIN rmse={rtr:.4f}  TEST rmse={rte:.4f}  gap={rte-rtr:+.4f}  "
          f"sd(inter)={float(inter.std()):.3f}  交互|均值|={abs(float(inter.mean())):.3f}", flush=True)
    print(f"      D sd={float(m.p['D'].std()):.3f}  P sd={float(m.p['P'].std()):.3f}  C熵={ent:.4f}", flush=True)
    return rtr, rte

run('对称wd=0 + gauge (变体A)', True)
run('不对称 wd (P=1e-4, D=0) + 无gauge (变体作对照)', False)
print("ALL_DONE", flush=True)
