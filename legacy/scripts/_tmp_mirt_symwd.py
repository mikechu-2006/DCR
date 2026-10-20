"""Scratch 8: D/P 对称 wd 的两种选择 + gauge 修复."""
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
print(f"rows train={len(tr):,} test={len(te):,}", flush=True)

def run(tag, symmetric_zero):
    torch.manual_seed(0)
    m = sp.EmbeddingModel('mirt_exp', len(ent_ids), len(item_ids), 16, 'F0', seed=0,
                          device=torch.device('cpu'), p_drift='none', d_constraint='none',
                          d_init_std=None, drop_bias='none', content='none',
                          chart_feats=None, c_init_std=None)
    if not symmetric_zero:                      # 对照: D,P 都吃 wd=1e-4
        m.no_decay.discard("D"); m.no_decay.discard("P")
    a = Namespace(d_logit_wd=None, c_logit_wd=None, wd=1e-4, lr=0.02)
    opt, wd_of = sp.build_optimizer(m, a)
    print(f"[{tag}] wd: " + ", ".join(f"{kk}={wd_of[kk]:g}" for kk in sorted(wd_of)), flush=True)
    rt = np.random.default_rng(0); n = u.numel()
    for ep in range(60):
        pm = rt.permutation(n)
        for s in range(0, n, 8192):
            b = torch.as_tensor(pm[s:s + 8192])
            opt.zero_grad(set_to_none=True)
            ((m.forward(u[b], i[b]) - y[b]) ** 2).mean().backward()
            opt.step(); m.gauge_fix()
    m.gauge_fix()
    with torch.no_grad():
        pred = m.forward(uu, ii).numpy()
        C = m.p['C'].detach(); D = m.p['D'].detach(); P = m.p['P'].detach()
        sc = torch.softmax(C, dim=1); sci = sc[ii]
        ent = -((sc * torch.log(sc.clamp_min(1e-12))).sum(1) / np.log(sc.shape[1])).mean().item()
    e = pred - yy
    ra = np.argsort(np.argsort(pred)).astype(float); rb = np.argsort(np.argsort(yy)).astype(float)
    spp = float(((ra-ra.mean())*(rb-rb.mean())).sum()/np.sqrt(((ra-ra.mean())**2).sum()*((rb-rb.mean())**2).sum()))
    print(f"[{tag}] TEST rmse={np.sqrt((e**2).mean()):.4f} mae={np.abs(e).mean():.4f} spear={spp:.4f}", flush=True)
    print(f"     D: mean={float(D.mean()):+.3f} sd={float(D.std()):.3f} | P: mean={float(P.mean()):+.3f} sd={float(P.std()):.3f}"
          f" | sd(D)/sd(P)={float(D.std()/P.std()):.2f}", flush=True)
    print(f"     交互两边量级: sd(sc*D)={float((sci*D[ii]).std()):.4f}  sd(sc*P)={float((sci*P[uu]).std()):.4f}"
          f" | b_u sd={float(m.p['bu'].std()):.3f} b_m sd={float(m.p['bm'].std()):.3f}", flush=True)
    print(f"     C 归一化熵={ent:.4f} 有效成分数={np.exp(ent*np.log(16)):.2f}", flush=True)
    return float(np.sqrt((e ** 2).mean()))

print("=== A) D,P 都不过 wd + gauge 投影 ===", flush=True)
ra_ = run('A_sym0', True)
print("=== B) D,P 都吃 wd=1e-4（对称但被 L2 拉平）+ gauge 投影 ===", flush=True)
rb_ = run('B_sym1e4', False)
print(f"\n=> A {ra_:.4f} | B {rb_:.4f}", flush=True)
print("ALL_DONE", flush=True)
