"""Scratch 11: MLP 在官方 split 上的表现 (协议与 main() 一致, 用 mirt 校准)."""
import sys, time, numpy as np, pandas as pd, torch
sys.path.insert(0, 'scripts')
from argparse import Namespace
import step2_predict as sp
from torch import nn
torch.use_deterministic_algorithms(True)

args = sp.parse_args([])                      # 全默认: dim16 lr0.02 wd1e-4 batch8192 epochs60 seed0
PROC = sp.PROC
cols = ["player_id", "beatmap_id", args.target]
tr_all = sp.load_play_table(args.tag, "step1", cols, proc_dir=PROC, optional_cols=(args.split_col,))
lab = tr_all[args.split_col].astype(str).to_numpy()
train = tr_all[lab == "train"].reset_index(drop=True)
test = tr_all[lab == "test"].reset_index(drop=True)
print(f"[split] train {len(train):,} / test {len(test):,} / dropped {int((lab=='dropped').sum()):,}", flush=True)
ent_ids = np.unique(train.player_id.to_numpy()); item_ids = np.unique(train.beatmap_id.to_numpy())
ents = {e: i for i, e in enumerate(ent_ids)}; items = {m: i for i, m in enumerate(item_ids)}
def encode(df):
    return (sp.map_ids(df.player_id.to_numpy(), ent_ids),
            sp.map_ids(df.beatmap_id.to_numpy(), item_ids),
            df[args.target].to_numpy(np.float32))
tr_e, tr_i, tr_y = encode(train); te_e, te_i, te_y = encode(test)
tr_ok = (tr_e >= 0) & (tr_i >= 0); te_ok = (te_e >= 0) & (te_i >= 0)
idx_tr = np.flatnonzero(tr_ok).astype(np.int64)
print(f"vocab {len(ents):,} ent x {len(items):,} item | test covered {te_ok.mean():.4%}", flush=True)
dev = torch.device('cpu')
tr_e_t = torch.as_tensor(tr_e); tr_i_t = torch.as_tensor(tr_i); tr_y_t = torch.as_tensor(tr_y)
te_e_t = torch.as_tensor(te_e[te_ok]); te_i_t = torch.as_tensor(te_i[te_ok]); te_y_np = te_y[te_ok]

class MLP(nn.Module):
    """零结构假设: 纯 ID embedding 拼接 -> 3 层 MLP。无 IRT / 混合 / logsumexp 假设。"""
    def __init__(self, n_ent, n_item, d_emb, h, seed=0, scale=1.0):
        super().__init__(); r = np.random.default_rng(seed)
        def draw(sh, std): return torch.as_tensor(r.normal(0, std, sh).astype(np.float32))
        self.p = {"b0": nn.Parameter(torch.zeros(1)),
                  "bu": nn.Parameter(torch.zeros(n_ent)),
                  "bm": nn.Parameter(torch.zeros(n_item)),
                  "P": nn.Parameter(draw((n_ent, d_emb), 0.05)),
                  "C": nn.Parameter(draw((n_item, d_emb), 0.05)),
                  "W1": nn.Parameter(draw((2 * d_emb, h), scale / np.sqrt(2 * d_emb))),
                  "b1": nn.Parameter(torch.zeros(h)),
                  "W2": nn.Parameter(draw((h, d_emb), scale / np.sqrt(h))),
                  "b2": nn.Parameter(torch.zeros(d_emb)),
                  "W3": nn.Parameter(draw((d_emb, 1), scale / np.sqrt(d_emb))),
                  "b3": nn.Parameter(torch.zeros(1))}
        self.active = list(self.p); self.no_decay = set()
    def forward(self, u, m):
        x = torch.cat([self.p["P"][u], self.p["C"][m]], dim=1)
        h = torch.relu(x @ self.p["W1"] + self.p["b1"])
        h = torch.relu(h @ self.p["W2"] + self.p["b2"])
        return (self.p["b0"][0] + self.p["bu"][u] + self.p["bm"][m]
                + (h @ self.p["W3"] + self.p["b3"][0]).squeeze(1))
    @torch.no_grad()
    def predict(self, u, m): return self.forward(u, m).numpy()

def run(tag, build):
    t0 = time.time()
    model = build()
    opt, wd_of = sp.build_optimizer(model, args) if hasattr(model, "no_decay") else (None, None)
    if opt is None:
        opt = torch.optim.Adam([model.p[k] for k in model.active], lr=args.lr, weight_decay=args.wd)
    n_par = sum(v.numel() for v in model.p.values())
    rng_tr = np.random.default_rng(args.seed)
    for ep in range(args.epochs):
        perm = rng_tr.permutation(idx_tr.size)
        for s in range(0, perm.size, args.batch):
            b = torch.as_tensor(idx_tr[perm[s:s + args.batch]])
            opt.zero_grad(set_to_none=True)
            ((model.forward(tr_e_t[b], tr_i_t[b]) - tr_y_t[b]) ** 2).mean().backward()
            opt.step()
            if hasattr(model, "gauge_fix"): model.gauge_fix()
    if hasattr(model, "gauge_fix"): model.gauge_fix()
    pv = model.predict(te_e_t, te_i_t)
    rep = sp.report(pv, te_y_np, len(test))
    n_pos = int((pv > 0).sum())
    print(f"[{tag:9s}] rmse={rep['rmse']:.4f} mae={rep['mae']:.4f} spear={rep['spearman']:.4f} "
          f"pp_p50={rep['pp_p50']:.3f} pp_p90={rep['pp_p90']:.3f} pp_mean={rep['pp_mean']:.2f} "
          f"cov={rep['coverage']:.4f} | 参数={n_par:,} | pred>0 = {n_pos} | {time.time()-t0:.0f}s", flush=True)
    return rep['rmse']

print("--- 校准: mirt 应复现 main() 的 1.4598 ---", flush=True)
run('mirt', lambda: sp.EmbeddingModel('mirt', len(ents), len(items), args.dim, 'F0', seed=args.seed,
    device=dev, p_drift='none', d_constraint='none', d_init_std=None, drop_bias='none',
    content='none', chart_feats=None, c_init_std=None))
for d_emb, h in [(16, 64), (32, 64), (32, 128)]:
    run(f'mlp_{d_emb}x{h}', lambda de=d_emb, hh=h: MLP(len(ents), len(items), de, hh, seed=args.seed))
print("ALL_DONE", flush=True)
