"""Scratch 7: gauge 修复后的训练 + 定位 D 为什么被撑大."""
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
opt, _ = sp.build_optimizer(m, a)
rt = np.random.default_rng(0); n = u.numel()
for ep in range(60):
    pm = rt.permutation(n)
    for s in range(0, n, 8192):
        b = torch.as_tensor(pm[s:s + 8192])
        opt.zero_grad(set_to_none=True)
        ((m.forward(u[b], i[b]) - y[b]) ** 2).mean().backward()
        opt.step(); m.gauge_fix()
    if (ep + 1) % 20 == 0:
        with torch.no_grad():
            ptr = m.forward(u[:200000], i[:200000]).numpy()
        print(f"   ep{ep+1} train_rmse={np.sqrt(((ptr-y[:200000].numpy())**2).mean()):.4f}", flush=True)
m.gauge_fix()

with torch.no_grad():
    pred = m.forward(uu, ii).numpy()
    C = m.p['C'].detach(); D = m.p['D'].detach(); P = m.p['P'].detach()
    sc = torch.softmax(C, dim=1)
    sci = sc[ii]; Di = D[ii]; Pi = P[uu]
    ex = sci * (Di - Pi)
    eff_chart = sci * Di; eff_player = sci * Pi
    ent = -((sc * torch.log(sc.clamp_min(1e-12))).sum(1) / np.log(sc.shape[1])).mean().item()
    bU = m.p['bu'].detach(); bM = m.p['bm'].detach()
e = pred - yy
ra = np.argsort(np.argsort(pred)).astype(float); rb = np.argsort(np.argsort(yy)).astype(float)
sp_ = float(((ra-ra.mean())*(rb-rb.mean())).sum()/np.sqrt(((ra-ra.mean())**2).sum()*((rb-rb.mean())**2).sum()))
print(f"\n[gauge 修复后] TEST rmse={np.sqrt((e**2).mean()):.4f} mae={np.abs(e).mean():.4f} spear={sp_:.4f} b0={float(m.p['b0'][0]):+.3f}", flush=True)
print(f"  D: mean={float(D.mean()):+.3f} sd={float(D.std()):.3f} range=[{float(D.min()):+.2f},{float(D.max()):+.2f}]", flush=True)
print(f"  P: mean={float(P.mean()):+.3f} sd={float(P.std()):.3f}", flush=True)
print(f"  主效应 sd: b_u={float(bU.std()):.3f}  b_m={float(bM.std()):.3f}", flush=True)
print(f"  C: 归一化熵={ent:.4f} 有效成分数={np.exp(ent*np.log(16)):.2f} sc均值={float(sc.mean()):.4f}", flush=True)
print(f"  交互里两边的实际量级: sd(sc*D)={float(eff_chart.std()):.4f}  sd(sc*P)={float(eff_player.std()):.4f}", flush=True)
scf = sc.ravel(); Df = D.ravel().abs()
print(f"\n  [检验 D 是否被 1/sc 撑大]", flush=True)
print(f"    corr(|D|, 1/sc) = {np.corrcoef(Df.numpy(), (1/sc).ravel().numpy())[0,1]:+.3f}", flush=True)
dead = sc < 0.01
print(f"    sc<0.01 的格子占比 = {float(dead.float().mean()):.1%};  "
      f"这些格子上的 |D| 均值 = {float(D.abs()[dead].mean()) if bool(dead.any()) else float('nan'):.3f} "
      f"vs 其余 = {float(D.abs()[~dead].mean()):.3f}", flush=True)
print(f"    sd(sc*D) 的来源: 用 |D| 直接算 = {float((sc*D).std()):.4f};  "
      f"只看 sc>=0.01 的格子 = {float((sc*D)[~dead].std()):.4f}", flush=True)
print("ALL_DONE", flush=True)
