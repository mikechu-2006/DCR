"""Scratch 10: 原式 mirt_exp 的 train/test —— 用来判断"更优解"到底是找不到还是不存在."""
import sys, numpy as np, pandas as pd, torch, torch.nn.functional as F
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

class Orig(sp.EmbeddingModel):
    """原式: ln sum_i D_mi * exp(-(P_ui - C_mi)), D = softplus."""
    def forward(self, u, m, x1=None, tau=None, t=None):
        p = self.p
        out = p["b0"][0] + torch.zeros(u.shape[0])
        if "bu" in p: out = out + p["bu"][u]
        if "bm" in p: out = out + p["bm"][m]
        P = p["P"][u]; C = p["C"][m]; D = F.softplus(p["D"][m])
        return out + torch.logsumexp(torch.log(D) - (P - C), dim=1)

def run(tag, cls, no_decay, d_con, epochs=60):
    torch.manual_seed(0)
    m = cls('mirt_exp', len(ent_ids), len(item_ids), 16, 'F0', seed=0,
            device=torch.device('cpu'), p_drift='none', d_constraint=d_con,
            d_init_std=None, drop_bias='none', content='none',
            chart_feats=None, c_init_std=0.05)
    m.no_decay = set(no_decay)
    a = Namespace(d_logit_wd=None, c_logit_wd=None, wd=1e-4, lr=0.02)
    opt, wd_of = sp.build_optimizer(m, a)
    print(f"[{tag}] wd: " + ", ".join(f"{kk}={wd_of[kk]:g}" for kk in sorted(wd_of)), flush=True)
    rt = np.random.default_rng(0); n = u.numel()
    for ep in range(epochs):
        pm = rt.permutation(n)
        for s in range(0, n, 8192):
            b = torch.as_tensor(pm[s:s + 8192])
            opt.zero_grad(set_to_none=True)
            ((m.forward(u[b], i[b]) - y[b]) ** 2).mean().backward()
            opt.step()
    with torch.no_grad():
        ptr = m.forward(u, i).numpy(); pte = m.forward(uu, ii).numpy()
    rtr = float(np.sqrt(((ptr - tr_y[ok])**2).mean())); rte = float(np.sqrt(((pte - yy)**2).mean()))
    print(f"[{tag}] TRAIN rmse={rtr:.4f}  TEST rmse={rte:.4f}  gap={rte-rtr:+.4f}", flush=True)
    return rtr, rte

run('原式 mirt_exp (D=softplus, wd 在 C/P 上, D 上 0)', Orig, {"D"}, "nonneg")
print("ALL_DONE", flush=True)
