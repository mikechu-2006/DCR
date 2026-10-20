#!/usr/bin/env python
"""step2_CM3P_MLP -- content-anchored MLP on the frozen CM3P chart vectors.

Design note: legacy/docs/cm3p_step2_mlp.md (architecture) built on legacy/docs/cm3p_step2_integration_plan.md
(the linear model this one extends) and legacy/docs/step2_capacity_dim_sweep.md (why capacity is not
the bottleneck).  Everything the linear pipeline does is kept; exactly one thing is replaced.

    linear (scripts/step2_predict.py --chart-content cm3p --content-reduce raw):
        L_hat = b0 + bu[u] + bm[m] + <P_u, C_m>
        C_m   = the frozen CM3P vector, d = 512
        P     : (n_ent, 512) free          <- d = 512  =>  504,832 free player parameters

    this script (anchored MLP):
        L_hat = b0 + bu[u] + bm[m] + w * <P_u, D_m>
        D_m   = g_theta(LayerNorm(C_m))    g = Linear(hid -> d) . GELU . Linear(512 -> hid)
        P_u   = LayerNorm( p_tower(Eu[u]) )        p_tower = GELU . Linear(d -> d), optional
        w     : scalar, init 1;  b_h : scalar, init 0
        C_m is FROZEN (buffer) and the MLP weights are SHARED across every chart
        => per-chart free parameters = bm only, exactly as in the linear content model

Why this shape (see the design note for the full argument):

  * **parameter budget**.  In 1k-covered (983 players / 11,214 charts / 378,750 train plays) the
    baseline NCF shape -- one free embedding per chart plus an MLP head -- is 685,777 parameters,
    of which 628,352 are the chart table, i.e. 1.0 plays/parameter.  Here the chart side has ZERO
    free parameters and the whole thing is ~63k (6.0 plays/param).  The MLP *layers* are only
    ~34k of that; the rest is bm and the player table.  Layer count was never the budget problem.
  * **depth = 2**.  A 1-layer g is a linear map, i.e. exactly the PCA/linear path already measured;
    2 layers is the smallest depth that can produce something new.  More depth buys nothing here:
    the target is a 2-D interaction surface (player x chart), not a feature hierarchy, and the repo
    has three independent measurements that extra capacity only adds variance (free rank K 2->64:
    test RMSE 1.5527 -> 1.6671; CM3P d 32->512: 1.57595 -> 1.59759; FM dim 4 == dim 16).
  * **the interaction stays multiplicative**.  Concatenating the raw feature blocks into one MLP
    would throw away the low-rank structure that is the only thing this dataset supports
    (legacy/docs/step2_capacity_dim_sweep.md section 6.1: an additive model explains 0% of the interaction
    variance, the best low-rank model 11%).
  * **main effects never enter the MLP** (--drop-bias u costs +3.3% in the linear pipeline).
  * **input scale**.  CM3P rows are unit-norm (component sd 0.028).  Measured on the real 512-d
    vectors: a Kaiming Linear(512 -> 64) starts with pre-activations of sd 0.0597 and gradients
    ||dL/dW1|| = 4.2e2, versus 1.345 and 7.2e3 with LayerNorm in front -- a 23x / 17x conditioning
    gap, which matters over a fixed 60-epoch budget.  It is NOT a survival issue (an earlier draft
    of this note claimed near-silent activations of order 1e-4; the measurement says otherwise).
    LayerNorm also makes the dot product scale-free, so --ln is on by default and --ln off is an
    ablation rather than a usable configuration.

Reproducing the linear baseline inside this script:

    --nonlinearity off    <=>  w*<P_u, D_m> with D_m linear and P_u linear
                           <=>  the linear content model up to the LayerNorm on both branches.

Training protocol is step2_predict.py's, deliberately: same vocab rules (from train rows only),
same seeded permutation (np.random.default_rng(seed).permutation(idx_tr.size)), same batching,
same fixed 60 epochs with no early stopping, same nine-step-style checklist, same npz/summary
layout, so numbers are directly comparable with legacy/docs/cm3p_step2_results.md section 4.2.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

# 仓库里其它脚本都是 `python scripts/xxx.py` 跑法（scripts/ 不在 sys.path 上），
# 这里显式加上，顺便让 `cd scripts && python step2_cm3p_mlp.py` 也能用。
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import step2_predict as sp          # noqa: E402

PROC = Path("data/processed")
BASE_COLS = sp.BASE_COLS


# --------------------------------------------------------------------------- model
class AnchoredMLP(nn.Module):
    """L_hat = b0 + bu[u] + bm[m] + b_h + w * <P_u, D_m>,  D_m = g_theta(LayerNorm(C_m)).

    Parameter groups (all reported in the [params] block at construction):

      content tower   g_theta        shared by every chart; the only place nonlinearity lives
      player tower    p_tower        optional (--player-tower off drops it)
      player table    Eu             the one free per-entity table
      main effects    b0 / bu / bm   b_m is the only remaining per-chart free parameter
      interaction     w / b_h        init 1 / 0 -> step 0 reproduces the linear model exactly

    Init order mirrors step2_predict.EmbeddingModel: b0/bu/bm at zero, then the drawn tables,
    using numpy default_rng(seed) so a run is reproducible bit for bit.

    NB (inherited from EmbeddingModel, not a regression introduced here): b0/bu/bm live in a plain
    dict, so nn.Module does not register them.  Autograd, Adam and count_params() see them (they are
    ordinary leaf Parameters held in an attribute), but named_parameters() and state_dict() do NOT.
    Everything here is built on the target device with float32, so nothing calls .to()/.half() on
    the module and the difference stays invisible; anything that reintroduces state_dict() or a
    dtype cast must first move these three into an nn.ParameterDict.
    """

    def __init__(self, n_ent, n_item, d_raw, d_out, features="F0", seed=0,
                 device=torch.device("cpu"), hid=64, depth=2, dropout=0.0,
                 ln=True, player_tower=True, nonlinearity=True, head="dot",
                 field_weight=False, hidden_layers=None, hidden_act="gelu",
                 hidden_norm=False, out_init_std=0.01, gate_init=1.0,
                 gate_learnable=True, gate_bound="none", drop_bias="none",
                 p_drift="none", mlp_wd=1e-4, eu_wd=1e-4, main_wd=1e-4):
        super().__init__()
        if not nonlinearity and depth != 2:
            # --nonlinearity off means "no hidden layer at all"; --depth is then meaningless, so it
            # is forced rather than rejected (rejecting would make "--depth 3 --nonlinearity off"
            # fail for no user-visible reason).
            depth = 1
        self.d_raw, self.d_out = int(d_raw), int(d_out)
        self.features, self.head = features, head
        self.ln_on, self.player_tower_on = bool(ln), bool(player_tower)
        self.nonlinearity = bool(nonlinearity)
        self.field_weight_on = bool(field_weight)
        self.hidden_act = hidden_act
        self.hidden_norm = bool(hidden_norm)
        self.gate_bound, self.gate_learnable = gate_bound, bool(gate_learnable)
        self.n_ent, self.n_item = int(n_ent), int(n_item)
        self.mlp_wd, self.eu_wd, self.main_wd = float(mlp_wd), float(eu_wd), float(main_wd)
        rng = np.random.default_rng(seed)

        def draw(shape, std=0.05, scale=1.0):
            a = np.ascontiguousarray(rng.normal(0, std, shape).astype(np.float32) * scale)
            return torch.as_tensor(a, dtype=torch.float32, device=device)

        # --- main effects.  Only the intercepts this run actually uses become parameters, so
        #     --drop-bias changes the parameter count rather than adding dead weight.
        self.p = {"b0": nn.Parameter(torch.zeros(1, dtype=torch.float32, device=device))}
        if drop_bias not in ("u", "both"):
            self.p["bu"] = nn.Parameter(torch.zeros(n_ent, dtype=torch.float32, device=device))
        if drop_bias not in ("m", "both"):
            self.p["bm"] = nn.Parameter(torch.zeros(n_item, dtype=torch.float32, device=device))

        # --- content tower: the shared function chart -> interaction vector
        self.raw_ln = nn.LayerNorm(d_raw) if ln else nn.Identity()
        # --depth = the number of Linear layers in the content tower, counted INCLUDING the output
        # layer.  depth 1 is therefore a bare projection (the already-measured linear/PCA path) and
        # depth 2 -- the default -- is "one hidden layer", as the design note says.  An earlier
        # version used [hid] * max(1, depth - 1), which silently made depth 1 and depth 2 identical.
        # --nonlinearity off removes every hidden layer regardless of --depth.
        if hidden_layers is not None:
            hidden_layers = [int(h) for h in hidden_layers]
        elif not nonlinearity or depth <= 1:
            hidden_layers = []
        else:
            hidden_layers = [hid] * (depth - 1)
        c_layers, prev = [], d_raw
        for i, h in enumerate(hidden_layers):
            c_layers.append(nn.Linear(prev, h))
            if i < len(hidden_layers) - 1 and hidden_norm:
                c_layers.append(nn.LayerNorm(h))
            c_layers.append(self._act())
            if dropout:
                c_layers.append(nn.Dropout(dropout))
            prev = h
        self.c_out = nn.Linear(prev, d_out)
        # Small output layer: the MLP enters as a correction to the linear term instead of
        # dominating it at step 0.  Drawn directly at the requested std rather than by rescaling
        # nn.Linear's own init, because whatever feeds this layer has already been LayerNorm'ed
        # (or is the raw content when --ln off), so the scale is ~1 either way and does not depend
        # on fan_in.  An earlier version multiplied by out_init_std/0.05, which only produced the
        # requested std when the preceding layer was a LayerNorm.
        with torch.no_grad():
            nn.init.normal_(self.c_out.weight, 0.0, float(out_init_std))
            self.c_out.bias.zero_()
        # nn.Sequential() with no children IS the identity map, and unlike nn.Identity() it stays
        # iterable/indexable -- code that walks the tower's Linear layers must not have to special-case
        # the depth-1 ("bare projection") case.
        self.c_tower = nn.Sequential(*c_layers)

        # --- player branch: free per-entity table (+ optional one hidden layer)
        self.Eu = nn.Parameter(draw((n_ent, d_out), 0.05))
        # --player-tower off means "no learnable layers on the player branch": Eu is used as-is.
        # The LayerNorm belongs to the tower bundle -- keeping it while dropping the Linear would
        # apply it to the RAW Eu (sd 0.05) and blow the dot product's initial scale up by ~1/0.05,
        # which silently makes the on/off ablation incomparable (it changes the interaction's
        # starting magnitude, not just its capacity).
        if player_tower:
            self.p_tower = nn.Sequential(nn.Linear(d_out, d_out), self._act())
            self.p_ln = nn.LayerNorm(d_out) if ln else nn.Identity()
        else:
            self.p_tower, self.p_ln = None, nn.Identity()

        # --- interaction head.  Multiplicative by construction: <P_u, D_m> factors first, and the
        #     head only reweights (field weight) or squares (factored bilinear) the per-dimension
        #     products.  The 1025-parameter full bilinear is a different, higher-capacity choice.
        self.field_weight_m = nn.Parameter(torch.zeros(d_out, dtype=torch.float32, device=device)) \
            if field_weight else None
        self.q = nn.Parameter(torch.zeros(d_out, dtype=torch.float32, device=device)) \
            if head == "factored" else None

        # --- explicit gate.  init 1 (learnable) or a constant; --gate-bound sigmoid keeps w >= 0
        g0 = float(gate_init)
        if gate_bound == "sigmoid":
            g0 = min(max(g0, 1e-6), 1 - 1e-6)
            raw = torch.log(torch.tensor(g0 / (1.0 - g0)))
            self.gate_raw = nn.Parameter(raw.to(dtype=torch.float32, device=device),
                                         requires_grad=bool(gate_learnable))
        else:
            self.gate_raw = nn.Parameter(torch.tensor(g0, dtype=torch.float32, device=device),
                                         requires_grad=bool(gate_learnable))
        self.b_h = nn.Parameter(torch.zeros(1, dtype=torch.float32, device=device))

        # --- frozen content, aligned to the training item order by the caller
        self.register_buffer("C_feat", torch.zeros(max(n_item, 1), d_raw,
                                                   dtype=torch.float32, device=device))
        self.C_feat = None          # replaced by attach_content(); None is a hard error in forward

        # --- drift, same parameterisation as the linear pipeline
        if p_drift in ("logpc", "time"):
            self.p["Pc"] = nn.Parameter(torch.zeros((n_ent, d_out), dtype=torch.float32,
                                                    device=device))
        if features == "F2":
            self.p["V"] = nn.Parameter(torch.zeros((n_ent, d_out), dtype=torch.float32, device=device))
        if features in ("F1", "F2"):
            self.p["w1"] = nn.Parameter(torch.zeros(1, dtype=torch.float32, device=device))
        self.p_drift = p_drift
        self.drop_bias = drop_bias

        # --- what the optimiser sees; wd groups are resolved by build_optimizer below
        self.active = [k for k in ("b0", "bu", "bm") if k in self.p]
        self.active += ["Eu"]
        if p_drift in ("logpc", "time"):
            self.active += ["Pc"]
        if features == "F2":
            self.active += ["V"]
        if features in ("F1", "F2"):
            self.active += ["w1"]
        self.mlp_params = []          # named separately so --mlp-wd can target them
        self.active += ["gate_raw", "b_h"]
        self.mlp_params += ["gate_raw", "b_h"]

    # ------------------------------------------------------------------ helpers
    def _act(self):
        return {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU,
                "tanh": nn.Tanh}[self.hidden_act]()

    def attach_content(self, Z: torch.Tensor):
        """Align the frozen chart matrix.  C[i] must belong to item_ids[i] -- asserted upstream."""
        self.C_feat = Z.to(dtype=torch.float32)

    def gate(self):
        w = self.gate_raw
        return torch.sigmoid(w) if self.gate_bound == "sigmoid" else w

    def player_vectors(self, u, Pc_term_u=None):
        """P_u for the whole batch: table lookup, optional drift, optional tower + LayerNorm."""
        P = self.Eu[u]
        if Pc_term_u is not None:
            P = P + Pc_term_u
        if self.p_tower is not None:
            return self.p_ln(self.p_tower(P))
        return P

    def chart_vectors(self, m):
        """D_m = g_theta(LayerNorm(C_m)) for the batch."""
        C = self.raw_ln(self.C_feat[m])
        return self.c_out(self.c_tower(C))

    def forward(self, u, m, x1=None, tau=None, t=None):
        p = self.p
        drift = None
        if self.p_drift == "logpc":
            drift = p["Pc"][u] * x1.unsqueeze(1)
        elif self.p_drift == "time":
            drift = p["Pc"][u] * t.unsqueeze(1)
        P = self.player_vectors(u, drift)
        if self.features == "F2":
            P = P + (tau - 0.5).unsqueeze(1) * p["V"][u]
        D = self.chart_vectors(m)

        out = p["b0"][0] + torch.zeros(u.shape[0], dtype=torch.float32, device=u.device)
        if "bu" in p:
            out = out + p["bu"][u]
        if "bm" in p:
            out = out + p["bm"][m]
        if self.features in ("F1", "F2"):
            out = out + p["w1"][0] * x1

        if self.head == "factored":
            # sum_i (P_i * D_i)^2 * exp(q_i): a multiplicative, diagonal upgrade of the dot
            # product.  q=0 init -> plain squared dot product.
            prod = P * D
            inter = (prod * prod * torch.exp(self.q).unsqueeze(0)).sum(1)
        elif self.field_weight_m is not None:
            # <P, s * D> with s = 1 + m: a learned per-dimension importance.  s=1 init -> dot.
            inter = (P * D * (1.0 + self.field_weight_m).unsqueeze(0)).sum(1)
        else:
            inter = (P * D).sum(1)
        return out + self.b_h[0] + self.gate() * inter

    # ------------------------------------------------------------------ reporting
    @torch.no_grad()
    def predict(self, u, m, x1=None, tau=None, t=None) -> np.ndarray:
        return self.forward(u, m, x1, tau, t).detach().to("cpu").numpy()

    def np_array(self, name, fallback=None):
        """Parameter as a numpy array; fallback when the model has none.

        名字/签名与 step2_predict.EmbeddingModel.np_array 一致 —— 导出那段代码是共用的写法，
        少了这个方法只会在"跑完训练、准备落盘"时炸，前 60 个 epoch 白跑。
        """
        if name in self.p:
            return self.p[name].detach().to("cpu").numpy()
        return fallback

    @torch.no_grad()
    def export_C(self) -> np.ndarray:
        """Effective chart-side matrix = g_theta's output (identical to export_D here).

        线性管线里 C 是谱面因子、D 是区分度向量；内容锚定 MLP 把两者合成成了一个交互方向，
        所以两者导出相同。见 legacy/docs/cm3p_step2_mlp.md section 6。
        """
        return self.export_D()

    @torch.no_grad()
    def export_D(self) -> np.ndarray:
        """Effective chart-side interaction vector for the whole chart vocabulary."""
        m = torch.arange(self.n_item, device=self.C_feat.device)
        return self.chart_vectors(m).detach().to("cpu").numpy()

    @torch.no_grad()
    def export_P(self) -> np.ndarray:
        """Effective player vectors (post tower/LayerNorm), in entity id order."""
        m0 = torch.zeros(1, dtype=torch.long, device=self.C_feat.device)
        out = []
        for s in range(0, self.n_ent, 20000):
            uu = torch.arange(s, min(s + 20000, self.n_ent), device=m0.device)
            out.append(self.player_vectors(uu).detach().to("cpu").numpy())
        return np.concatenate(out, 0)

    def param_groups(self):
        """[(name, module_or_parameter, tensor, wd)] for the optimiser + the [params] table."""
        # 主效应跟 --wd 走（与 step2_predict.build_optimizer 的默认行为一致：那里除了 no_decay
        # 里的参数，其它全部按 args.wd 处理）。早先这里写 None（= wd 0）只是参数表印错了 —— 优化器
        # 本来就给了它们 --wd，是本表的其它行与之对不上。
        rows = [("main effects b0/bu/bm", None,
                 [self.p[k] for k in ("b0", "bu", "bm") if k in self.p], self.main_wd)]
        rows.append(("player table Eu", None, [self.Eu], self.eu_wd))
        if self.p_tower is not None:
            rows.append(("player tower + LayerNorm",
                         None, list(self.p_tower.parameters()) + list(self.p_ln.parameters()),
                         self.mlp_wd))
        rows.append(("content tower g_theta", None,
                     list(self.c_tower.parameters()) + list(self.c_out.parameters())
                     + (list(self.raw_ln.parameters()) if self.ln_on else []), self.mlp_wd))
        rows.append(("interaction w / b_h",
                     None, [self.gate_raw, self.b_h], 0.0))
        if self.field_weight_m is not None:
            rows.append(("field weight s", None, [self.field_weight_m], self.mlp_wd))
        if self.q is not None:
            rows.append(("factored head q", None, [self.q], self.mlp_wd))
        if "Pc" in self.p:
            rows.append(("drift Pc", None, [self.p["Pc"]], self.eu_wd))
        if "V" in self.p:
            rows.append(("drift V", None, [self.p["V"]], self.eu_wd))
        if "w1" in self.p:
            rows.append(("sequence w1", None, [self.p["w1"]], self.mlp_wd))
        return rows


# ---------------------------------------------------------------------- checklist
def determinism_selftest(device) -> bool:
    """Same seed and same batches -> bitwise identical parameters, new model included."""

    def run_once():
        torch.manual_seed(0)
        m = AnchoredMLP(24, 17, 8, 4, "F2", seed=0, device=device, hid=6, depth=3,
                        player_tower=True, field_weight=True, p_drift="logpc")
        m.attach_content(torch.as_tensor(
            np.random.default_rng(11).normal(0, 1, (17, 8)).astype(np.float32), device=device))
        opt = build_optimizer(m, _Args(lr=0.02, mlp_wd=1e-4, eu_wd=1e-4, wd=1e-4))
        r = np.random.default_rng(7)
        u = torch.as_tensor(r.integers(0, 24, 96), dtype=torch.long, device=device)
        i = torch.as_tensor(r.integers(0, 17, 96), dtype=torch.long, device=device)
        y = torch.as_tensor(r.normal(-3, 1, 96).astype(np.float32), device=device)
        x1 = torch.as_tensor(r.normal(1, 0.5, 96).astype(np.float32), device=device)
        tau = torch.as_tensor(r.random(96).astype(np.float32), device=device)
        rng = np.random.default_rng(3)
        for _ in range(3):
            for s in range(0, 96, 32):
                b = torch.as_tensor(rng.permutation(96)[s:s + 32], dtype=torch.long, device=device)
                opt.zero_grad(set_to_none=True)
                ((m.forward(u[b], i[b], x1[b], tau[b]) - y[b]) ** 2).mean().backward()
                opt.step()
        return {k: v.detach().clone() for k, v in list(m.named_parameters())}

    a, b = run_once(), run_once()
    return all(torch.equal(a[k], b[k]) for k in a)


def build_optimizer(model, args):
    """Adam with three wd groups: main effects (--wd), MLP function (--mlp-wd), tables (--eu-wd).

    The distinction matters: Adam's wd is a constant pull towards zero, which is a fine prior for
    a free per-entity table (that is where step2_predict puts it) but is the wrong prior for a
    shared function whose zero point is not special.  --mlp-wd 0 reproduces that "no prior on the
    scale" stance taken for mirt_exp's D.
    """
    named, seen = [], set()
    for name, _, params, wd in model.param_groups():
        for q in params:
            if q is None or id(q) in seen or not q.requires_grad:
                continue
            seen.add(id(q))
            named.append((name, q, wd))
    # anything the group list missed still gets optimised -- never silently dropped
    for nm, q in model.named_parameters():
        if id(q) not in seen and q.requires_grad:
            named.append((f"other:{nm}", q, args.mlp_wd))
            seen.add(id(q))
    groups = {}
    for _, q, wd in named:
        groups.setdefault(0.0 if wd is None else float(wd), []).append(q)
    opt = torch.optim.Adam([{"params": ps, "weight_decay": w} for w, ps in groups.items()],
                           lr=args.lr)
    return opt


class _Args:
    """Tiny stand-in for argparse.Namespace inside the determinism self-test."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def count_params(model):
    named, seen = [], set()
    for _, _, params, _ in model.param_groups():
        for q in params:
            if q is None or id(q) in seen or not q.requires_grad:
                continue
            seen.add(id(q))
            named.append(q)
    for _, q in model.named_parameters():
        if id(q) not in seen and q.requires_grad:
            seen.add(id(q))
            named.append(q)
    return sum(int(q.numel()) for q in named), named


def param_table(model) -> list:
    """[params] block: what was actually allocated, per group -- the design note's claim, measured."""
    rows, seen = [], set()
    for name, _, params, wd in model.param_groups():
        n = 0
        for q in params:
            if q is None or id(q) in seen or not q.requires_grad:
                continue
            seen.add(id(q))
            n += int(q.numel())
        rows.append({"group": name, "params": n,
                     "wd": None if wd is None else float(wd)})
    extra = 0
    for nm, q in model.named_parameters():
        if id(q) not in seen and q.requires_grad:
            extra += int(q.numel())
            rows.append({"group": f"other:{nm}", "params": int(q.numel()), "wd": model.mlp_wd})
    total = sum(r["params"] for r in rows)
    return {"groups": rows, "total": total, "unaccounted": extra}


# --------------------------------------------------------------------------- args
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="step2_CM3P_MLP -- content-anchored MLP head")
    # data / split (identical semantics to scripts/step2_predict.py)
    ap.add_argument("--tag", default="1k")
    ap.add_argument("--target", default="loss")
    ap.add_argument("--proc-dir", type=Path, default=PROC)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--train-path", type=Path, default=None)
    ap.add_argument("--split-col", default="split")
    ap.add_argument("--features", default="F0", choices=["F0", "F1", "F2"])
    ap.add_argument("--drop-bias", default="none", choices=["none", "u", "m", "both"])
    ap.add_argument("--p-drift", default="none", choices=["none", "logpc", "time"])
    ap.add_argument("--out-prefix", default="step2")
    # content: this script is the content-anchored model, so content is mandatory
    ap.add_argument("--content-path", default=str(PROC / "chart_content_cm3p.parquet"),
                    help="pipeline A's parquet; column 'element' holds the vector")
    ap.add_argument("--content-dim", type=int, default=512,
                    help="input dimension fed to the MLP (512 = raw CM3P; <512 = PCA to that "
                         "width first, fitted on train charts only)")
    ap.add_argument("--content-reduce", default="raw", choices=["raw", "pca"],
                    help="raw (default): feed the table's own 512-d vectors, let the MLP learn the "
                         "projection.  pca: whiten to --content-dim first, as the linear pipeline does.")
    # architecture
    ap.add_argument("--hid", type=int, default=64,
                    help="width of the content tower's hidden layer(s)")
    ap.add_argument("--depth", type=int, default=2,
                    help="number of Linear layers in the content tower, INCLUDING the output layer. "
                         "2 (default) = one hidden layer of --hid; 1 = a bare projection, i.e. the "
                         "already measured linear/PCA path.  --nonlinearity off forces 1")
    ap.add_argument("--hidden-layers", default=None,
                    help="explicit comma-separated widths, e.g. 128,32; overrides --hid/--depth")
    ap.add_argument("--hidden-act", default="gelu", choices=["gelu", "relu", "silu", "tanh"])
    ap.add_argument("--hidden-norm", default="off", choices=["off", "on"],
                    help="LayerNorm after each hidden Linear (unlike the ablated input scale, this "
                         "changes optimisation; off by default so the residual is clean)")
    ap.add_argument("--dim", type=int, default=16,
                    help="d: output width of g_theta, i.e. the interaction space (and P's width)")
    ap.add_argument("--out-init-std", type=float, default=0.01,
                    help="std of the content tower's output layer; small so the MLP starts as a "
                         "correction to the linear term rather than dominating it")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--player-tower", default="on", choices=["on", "off"],
                    help="on: P_u = LayerNorm(Linear(Eu[u])); off: P_u = Eu[u] (saves 272 params)")
    ap.add_argument("--nonlinearity", default="on", choices=["on", "off"],
                    help="content tower only: off makes g_theta a single Linear + LayerNorm, i.e. "
                         "a capacity control (NOT the linear baseline -- LayerNorm makes it "
                         "nonlinear in C_m, see legacy/docs/cm3p_step2_mlp.md section 5.2; use "
                         "--player-tower off as well to strip the player tower)")
    ap.add_argument("--ln", default="on", choices=["on", "off"],
                    help="LayerNorm on both branches.  CM3P rows are unit-norm (component sd "
                         "0.028), so off makes the first layer learn from ~0 -- ablation only")
    ap.add_argument("--head", default="dot", choices=["dot", "factored"],
                    help="dot: <P,D>; factored: sum_i (P_i*D_i)^2 * exp(q_i) (diagonal NFM-style)")
    ap.add_argument("--field-weight", default="off", choices=["off", "on"],
                    help="on: a learned per-dimension importance s = 1 + m, i.e. <P, s*D>. "
                         "1025-parameter full bilinear heads are deliberately NOT offered")
    # gate
    ap.add_argument("--gate-init", type=float, default=1.0)
    ap.add_argument("--gate-learnable", default="on", choices=["on", "off"])
    ap.add_argument("--gate-bound", default="none", choices=["none", "sigmoid"],
                    help="sigmoid keeps w >= 0; 'none' lets the MLP learn a negative correction "
                         "(an unrestricted but strictly nested extension of the linear model)")
    # optimisation
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--wd", type=float, default=1e-4,
                    help="weight decay on the main-effect intercepts")
    ap.add_argument("--mlp-wd", type=float, default=1e-4,
                    help="weight decay on the shared content/player towers.  0 = no prior on the "
                         "function's scale (cf. mirt_exp's D)")
    ap.add_argument("--eu-wd", type=float, default=1e-4,
                    help="weight decay on the free per-entity table Eu")
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--save-preds", action="store_true",
                    help="write {out_dir}/{prefix}_{tag}_preds_{target}.npz (test ids + labels + "
                         "predictions) for scripts/compare_preds.py.  NOTE the name carries only "
                         "--out-prefix and --target, so two runs that share a prefix overwrite each "
                         "other -- give each configuration its own --out-prefix")
    return ap.parse_args(argv)


# --------------------------------------------------------------------------- main
def main():
    args = parse_args()
    t_start = time.time()
    out_dir = Path(args.out_dir) if args.out_dir else args.proc_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = sp.pick_device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    print(f"engine: torch {torch.__version__} on {device}, threads={torch.get_num_threads()}, "
          f"deterministic_algorithms={torch.are_deterministic_algorithms_enabled()}", flush=True)

    if args.ln == "off":
        print("[warn] --ln off: CM3P rows are unit-norm with component sd ~0.028, so the first "
              "layer starts near-silent.  This is an ablation, not a configuration.", flush=True)
    if args.content_reduce == "raw" and args.content_dim != 512:
        raise SystemExit("--content-reduce raw needs --content-dim 512")

    # ------------------------------------------------------------------ load + filter
    need_pc = args.features in ("F1", "F2") or args.p_drift == "logpc"
    need_ts = args.features == "F2" or args.p_drift == "time"
    cols = BASE_COLS + [args.target] \
        + (["playcount_cur"] if need_pc else []) + (["timestamp"] if need_ts else [])

    tr_all = sp.load_play_table(args.tag, "step1", cols, path=args.train_path,
                                proc_dir=args.proc_dir, optional_cols=(args.split_col,))

    content_ids, content_E = sp.load_content_table(args.content_path)
    have = set(content_ids.tolist())
    n_all = len(tr_all)
    tr_all = tr_all[tr_all.beatmap_id.isin(have)].reset_index(drop=True)
    content_drop = {"rows_dropped": n_all - len(tr_all), "rows_total": n_all}
    print(f"[content] charts with a vector: {len(have):,}; kept {len(tr_all):,}/{n_all:,} rows "
          f"(train and test filtered identically, BEFORE the split)", flush=True)
    if len(tr_all) == 0:
        raise SystemExit("no plays left after the content filter")

    # ------------------------------------------------------------------ split (from step1)
    if args.split_col not in tr_all.columns:
        raise SystemExit(f"step1 table has no '{args.split_col}' column -- rerun step1")
    lab = tr_all[args.split_col].astype(str).to_numpy()
    unknown = sorted(set(lab.tolist()) - {"train", "test", "dropped"})
    if unknown:
        raise SystemExit(f"'{args.split_col}' has unknown labels {unknown}")
    tr_mask, te_mask = lab == "train", lab == "test"
    train = tr_all[tr_mask].reset_index(drop=True)
    test = tr_all[te_mask].reset_index(drop=True)
    n_dropped = int((lab == "dropped").sum())
    if len(train) == 0 or len(test) == 0:
        raise SystemExit(f"train={len(train):,} / test={len(test):,} rows")

    split_spec = sp.read_split_spec(args.train_path, args.proc_dir, args.tag)
    k = int(max(int(train.beatmap_id.max()), int(test.beatmap_id.max()))) + 1
    row_keys = sp.pair_keys(tr_all.player_id.to_numpy(), tr_all.beatmap_id.to_numpy(), k)
    unit_kind = (split_spec or {}).get("unit", "pair")
    if unit_kind == "beatmap":
        ids_ = tr_all.beatmap_id.to_numpy()
        unit_tr, unit_te = np.unique(ids_[lab != "test"]), np.unique(ids_[te_mask])
    else:
        unit_tr = np.unique(row_keys[lab != "test"])
        unit_te = np.unique(row_keys[te_mask])
    units_all = np.union1d(unit_tr, unit_te)
    is_test = np.isin(units_all, unit_te)
    n_pairs, n_test_pairs = int(units_all.size), int(unit_te.size)
    cross_units = np.intersect1d(unit_tr, unit_te, assume_unique=True)
    test_pairs = np.unique(row_keys[te_mask])
    tr_pairs = np.unique(row_keys[tr_mask])
    expected_test_frac = float((split_spec or {}).get("actual_test_frac")
                               or (n_test_pairs / max(n_pairs, 1)))
    frac_tol = 0.02
    print(f"[split] train {len(train):,} / test {len(test):,} / dropped {n_dropped:,} rows | "
          f"unit {unit_kind}: train {unit_tr.size:,} / test {unit_te.size:,}", flush=True)
    del row_keys, lab, tr_all
    gc.collect()

    chk = sp.Checklist()
    chk.check("pair_never_crosses_split", cross_units.size == 0,
              f"{cross_units.size} units on both sides")
    tr_keys = sp.pair_keys(train.player_id.to_numpy(), train.beatmap_id.to_numpy(), k)
    te_keys = sp.pair_keys(test.player_id.to_numpy(), test.beatmap_id.to_numpy(), k)
    tr_in_test = np.isin(tr_keys, test_pairs)
    te_in_test = np.isin(te_keys, test_pairs)
    chk.check("train_and_test_disjoint",
              (not tr_in_test.any()) and bool(te_in_test.all()),
              f"train rows on a held-out pair = {int(tr_in_test.sum()):,}; "
              f"test rows on a held-out pair = {int(te_in_test.sum()):,}/{len(test):,}")
    if split_spec:
        frac_ok = abs(float(is_test.mean()) - expected_test_frac) <= frac_tol
        frac_detail = (f"observed {is_test.mean():.4%} vs step1 report {expected_test_frac:.2%}")
    else:
        frac_ok, frac_detail = True, f"no step1 report; observed {is_test.mean():.4%}"
    chk.check("test_pair_frac", frac_ok, frac_detail)
    del tr_keys, te_keys, tr_in_test, te_in_test
    gc.collect()

    # ------------------------------------------------------------------ vocab
    ent_ids = np.unique(train.player_id.to_numpy())
    item_ids = np.unique(train.beatmap_id.to_numpy())
    ents = {e: i for i, e in enumerate(ent_ids)}
    items = {m: i for i, m in enumerate(item_ids)}
    print(f"train vocab: {len(ents):,} entities, {len(items):,} items", flush=True)

    d_raw = None
    Z, content_info = sp.align_content(content_ids, content_E, item_ids,
                                       args.content_dim, args.content_reduce)
    d_raw = int(Z.shape[1])
    chart_feats = torch.as_tensor(Z, dtype=torch.float32, device=device)
    # g_theta's OUTPUT width is the interaction space; its INPUT width is d_raw.  Unlike the
    # linear pipeline, --dim no longer forces --content-dim: the MLP is what does the projection.
    print(f"[content] input {d_raw}-d (mode={args.content_reduce}) -> interaction space d={args.dim}",
          flush=True)

    t_ref = 0.0
    if args.p_drift == "time":
        t_ref = float(pd.to_datetime(train.timestamp).to_numpy(dtype="datetime64[D]")
                      .astype(np.float64).mean() / 365.25)
        print(f"[time-drift] t_ref = {t_ref:.3f} yr CE", flush=True)

    def encode(df):
        e = sp.map_ids(df.player_id.to_numpy(), ent_ids)
        i = sp.map_ids(df.beatmap_id.to_numpy(), item_ids)
        x1 = np.log(df.playcount_cur.to_numpy(np.float64)).astype(np.float32) if need_pc else None
        tau = ((pd.to_datetime(df.timestamp).dt.dayofyear.to_numpy() - 1) / 365.0).astype(np.float32) \
            if args.features == "F2" else None
        tv = None
        if args.p_drift == "time":
            days = pd.to_datetime(df.timestamp).to_numpy(dtype="datetime64[D]").astype(np.float64)
            tv = (days / 365.25 - t_ref).astype(np.float32)
        return e, i, x1, tau, tv, df[args.target].to_numpy(np.float32)

    tr_e, tr_i, tr_x, tr_tau, tr_t, tr_y = encode(train)
    te_e, te_i, te_x, te_tau, te_t, te_y = encode(test)
    tr_ok = (tr_e >= 0) & (tr_i >= 0)
    te_ok = (te_e >= 0) & (te_i >= 0)
    print(f"train rows in vocab = {int(tr_ok.sum()):,}/{len(train):,}   "
          f"test rows covered = {int(te_ok.sum()):,}/{len(test):,} ({te_ok.mean():.2%})", flush=True)

    if args.tag in sp.LOSS_RANGE:
        lo, hi = sp.LOSS_RANGE[args.tag]
        loss_ok = bool(np.isfinite(te_y).all() and te_y.min() >= lo - 1e-3 and te_y.max() <= hi + 1e-3)
        loss_detail = f"test({args.tag}) loss in [{te_y.min():.4f}, {te_y.max():.4f}], expected [{lo}, {hi}]"
    else:
        loss_ok = bool(np.isfinite(te_y).all() and (te_y <= 0).all())
        loss_detail = f"test({args.tag}) loss in [{te_y.min():.4f}, {te_y.max():.4f}]"
    chk.check("loss_in_range", loss_ok, loss_detail)
    chk.check("vocab_from_train_only",
              np.array_equal(ent_ids, np.unique(train.player_id.to_numpy()))
              and np.array_equal(item_ids, np.unique(train.beatmap_id.to_numpy())),
              f"{ent_ids.size:,} entities / {item_ids.size:,} items == unique train ids")
    chk.check("determinism_selftest", determinism_selftest(device),
              "same seed -> bitwise identical parameters (AnchoredMLP + grouped Adam)")
    step = max(1, len(item_ids) // 64)
    probe = np.arange(len(item_ids))[::step][:64]
    pos = pd.Index(content_ids).get_indexer(item_ids[probe])
    norms = chart_feats.norm(dim=1)
    chk.check("content_aligned_with_item_ids",
              bool((pos >= 0).all()) and len(chart_feats) == len(item_ids),
              f"{len(item_ids):,} train charts; {len(probe)} probe ids resolve; "
              f"row norm in [{float(norms.min()):.4f}, {float(norms.max()):.4f}] "
              f"(unit-norm vectors are what g_theta's LayerNorm expects)")

    # ------------------------------------------------------------------ tensors
    def dev(a):
        return torch.as_tensor(a, device=device)

    tr_e_t, tr_i_t, tr_y_t = dev(tr_e), dev(tr_i), dev(tr_y)
    tr_x_t = dev(tr_x) if tr_x is not None else None
    tr_tau_t = dev(tr_tau) if tr_tau is not None else None
    tr_t_t = dev(tr_t) if tr_t is not None else None
    te_e_t, te_i_t = dev(te_e[te_ok]), dev(te_i[te_ok])
    te_y_np = te_y[te_ok]
    te_x_t = dev(te_x[te_ok]) if te_x is not None else None
    te_tau_t = dev(te_tau[te_ok]) if te_tau is not None else None
    te_t_t = dev(te_t[te_ok]) if te_t is not None else None
    idx_tr = np.flatnonzero(tr_ok).astype(np.int64)
    del tr_e, tr_i, tr_x, tr_tau, tr_t, tr_y, te_e, te_i, te_x, te_tau, te_t, te_y
    gc.collect()

    # ------------------------------------------------------------------ one model
    hidden_layers = [int(s) for s in args.hidden_layers.split(",")] if args.hidden_layers else None
    model = AnchoredMLP(len(ents), len(items), d_raw, args.dim, args.features, args.seed, device,
                        hid=args.hid, depth=args.depth, dropout=args.dropout,
                        ln=(args.ln == "on"), player_tower=(args.player_tower == "on"),
                        nonlinearity=(args.nonlinearity == "on"), head=args.head,
                        field_weight=(args.field_weight == "on"), hidden_layers=hidden_layers,
                        hidden_act=args.hidden_act, hidden_norm=(args.hidden_norm == "on"),
                        out_init_std=args.out_init_std, gate_init=args.gate_init,
                        gate_learnable=(args.gate_learnable == "on"), gate_bound=args.gate_bound,
                        drop_bias=args.drop_bias, p_drift=args.p_drift,
                        mlp_wd=args.mlp_wd, eu_wd=args.eu_wd, main_wd=args.wd)
    model.attach_content(chart_feats)

    pt = param_table(model)
    total_params = pt["total"]
    train_plays = int(tr_ok.sum())
    print(f"\n[params] {total_params:,} trainable  "
          f"({train_plays:,} train plays -> {train_plays / max(total_params, 1):.1f} plays/param)")
    for row in pt["groups"]:
        wd = "none" if row["wd"] is None else f"wd={row['wd']:g}"
        print(f"    {row['group']:<34}{row['params']:>9,}   {wd}")
    if pt["unaccounted"]:
        print(f"    [warn] {pt['unaccounted']:,} trainable parameter(s) outside the declared groups")
    print()

    opt = build_optimizer(model, args)
    wd_of = {}
    for g in opt.param_groups:
        for q in g["params"]:
            wd_of[id(q)] = g["weight_decay"]

    # ------------------------------------------------------------------ train
    rng_tr = np.random.default_rng(args.seed)
    t0 = time.time()
    curve = []
    for ep in range(args.epochs):
        perm = rng_tr.permutation(idx_tr.size)
        for s0 in range(0, perm.size, args.batch):
            b = torch.as_tensor(idx_tr[perm[s0:s0 + args.batch]], device=device)
            x1 = tr_x_t[b] if tr_x_t is not None else None
            tau = tr_tau_t[b] if tr_tau_t is not None else None
            tt = tr_t_t[b] if tr_t_t is not None else None
            opt.zero_grad(set_to_none=True)
            pred = model.forward(tr_e_t[b], tr_i_t[b], x1, tau, tt)
            ((pred - tr_y_t[b]) ** 2).mean().backward()
            opt.step()
        if args.eval_every and (ep + 1) % args.eval_every == 0:
            pv = model.predict(te_e_t, te_i_t, te_x_t, te_tau_t, te_t_t)
            rm = float(np.sqrt(((pv - te_y_np) ** 2).mean()))
            curve.append({"epoch": ep + 1, "test_rmse": round(rm, 6), "seconds": round(time.time() - t0, 1)})
            print(f"  [mlp] epoch {ep+1:3d}  test RMSE = {rm:.5f}  ({time.time()-t0:.1f}s)", flush=True)

    pv = model.predict(te_e_t, te_i_t, te_x_t, te_tau_t, te_t_t)
    res = sp.report(pv, te_y_np, len(test))
    tr_pred = model.predict(tr_e_t[idx_tr], tr_i_t[idx_tr],
                            None if tr_x_t is None else tr_x_t[idx_tr],
                            None if tr_tau_t is None else tr_tau_t[idx_tr],
                            None if tr_t_t is None else tr_t_t[idx_tr])
    res["train_rmse"] = round(float(np.sqrt(((tr_pred - tr_y_t[idx_tr].cpu().numpy()) ** 2).mean())), 6)
    res["test_minus_train"] = round(res["rmse"] - res["train_rmse"], 6)
    print(f"\nmlp      test rmse={res['rmse']:.5f}  train rmse={res['train_rmse']:.5f}  "
          f"gap={res['test_minus_train']:+.5f}  cov={res['coverage']:.3f}", flush=True)

    gate = float(model.gate().detach())
    degeneracy = {"gate_w": round(gate, 6),
                  "D_row_sd": round(float(np.std(model.export_D(), axis=0).mean()), 6),
                  "P_row_sd": round(float(np.std(model.export_P(), axis=0).mean()), 6),
                  "content_tower_weight_norm": round(
                      float(torch.cat([q.detach().flatten() for q in model.c_tower.parameters()]
                                      + [model.c_out.weight.detach().flatten()]).norm()), 6)}
    print(f"[gate] w = {gate:.5f}  (init {args.gate_init}; 1.0 means the MLP is carrying the "
          f"interaction at full strength, ~0 means the tower collapsed)", flush=True)
    if abs(gate) < 1e-3:
        print("  [warn] gate is ~0: the tower learned nothing and the model is behaving additively",
              flush=True)

    # ------------------------------------------------------------------ checks + save
    chk.check("coverage_reported",
              all(("coverage" in r) and ("n_scored" in r) and r["coverage"] is not None
                  for r in [res]),
              f"{len([res])} result block carries coverage + n_scored: "
              f"coverage={res['coverage']:.4f}, n_scored={res['n_scored']:,}")
    leak = np.isin(sp.pair_keys(train.player_id.to_numpy()[idx_tr],
                                train.beatmap_id.to_numpy()[idx_tr], k), test_pairs).sum()
    chk.check("no_test_leak", bool(leak == 0 and idx_tr.size == int(tr_ok.sum())),
              f"training rows used = {idx_tr.size:,}; on a held-out pair = {int(leak):,}")
    del test_pairs, tr_pairs
    gc.collect()

    inv_e = {v: kk for kk, v in ents.items()}
    inv_i = {v: kk for kk, v in items.items()}
    entity_ids = np.array(list(inv_e.values()), dtype=np.int64)
    beatmap_ids = np.array(list(inv_i.values()), dtype=np.int32)

    variant = f"mlp{args.dim}_h{args.hid}_L{args.depth}"
    if args.hidden_layers:
        variant += "x" + args.hidden_layers.replace(",", "-")
    if args.nonlinearity == "off":
        variant += "_lin"
    if args.player_tower == "off":
        variant += "_nopt"
    if args.ln == "off":
        variant += "_noln"
    if args.head != "dot":
        variant += f"_{args.head}"
    if args.field_weight == "on":
        variant += "_fw"
    if args.hidden_norm == "on":
        variant += "_hn"
    if args.hidden_act != "gelu":
        variant += f"_{args.hidden_act}"
    if args.content_reduce != "raw":
        variant += f"_in{args.content_dim}{args.content_reduce}"
    if args.p_drift != "none":
        variant += {"logpc": "_pd", "time": "_pdt"}[args.p_drift]
    if args.drop_bias != "none":
        variant += {"u": "_nobu", "m": "_nobm", "both": "_nobubm"}[args.drop_bias]
    if args.features != "F0":
        variant += f"_{args.features}"
    if args.gate_init != 1.0 or args.gate_learnable == "off" or args.gate_bound != "none":
        variant += f"_g{args.gate_init:g}{'' if args.gate_learnable == 'on' else 'f'}" \
                   f"{'' if args.gate_bound == 'none' else args.gate_bound[:3]}"

    D_eff = model.export_D()
    P_eff = model.export_P()
    extra = {}
    if "Pc" in model.p:
        extra["Pc"] = model.np_array("Pc")
    npz_path = out_dir / f"{args.out_prefix}_{args.tag}_{variant}_cm3p_{args.target}.npz"
    np.savez(npz_path,
             entity_ids=entity_ids, beatmap_ids=beatmap_ids,
             P=P_eff, V=model.np_array("V", np.zeros_like(P_eff)),
             C=D_eff, D=D_eff,
             b0=model.np_array("b0"),
             bu=model.np_array("bu", np.zeros(len(ents), np.float32)),
             bm=model.np_array("bm", np.zeros(len(items), np.float32)),
             **extra)
    with np.load(npz_path) as z:
        ok9 = (len(z["entity_ids"]) == len(ents) and len(z["beatmap_ids"]) == len(items)
               and z["P"].shape == (len(ents), args.dim)
               and z["C"].shape == (len(items), args.dim)
               and z["D"].shape == z["C"].shape and z["V"].shape == z["P"].shape
               and z["b0"].shape == (1,) and len(z["bu"]) == len(ents) and len(z["bm"]) == len(items))
    chk.check("npz_matches_summary", bool(ok9),
              f"P {P_eff.shape}, C/D {D_eff.shape} (C = D = g_theta(C_m), the matrix actually "
              f"used by the interaction), bm {len(items):,}")

    preds_path = None
    if args.save_preds:
        preds_path = out_dir / f"{args.out_prefix}_{args.tag}_preds_{args.target}.npz"
        np.savez(preds_path, player_id=test.player_id.to_numpy()[te_ok],
                 beatmap_id=test.beatmap_id.to_numpy()[te_ok],
                 y=te_y_np, pred_mlp=pv, variant=variant, **extra)
        print(f"wrote {preds_path.name} ({len(pv):,} test rows)")

    chk.require()

    summary = {
        "variant": variant,
        "model_class": "AnchoredMLP",
        "arch": {"kind": "anchored_mlp", "nonlinearity": args.nonlinearity,
                 "content_tower": {"in": d_raw, "hidden": hidden_layers or [args.hid] * max(1, args.depth - 1),
                                   "out": args.dim, "act": args.hidden_act,
                                   "hidden_norm": args.hidden_norm, "dropout": args.dropout,
                                   "out_init_std": args.out_init_std},
                 "player_tower": args.player_tower, "ln": args.ln, "head": args.head,
                 "field_weight": args.field_weight, "gate": {"init": args.gate_init,
                 "learnable": args.gate_learnable, "bound": args.gate_bound, "final": round(gate, 6)}},
        "params": {"total": total_params, "train_plays": train_plays,
                   "plays_per_param": round(train_plays / max(total_params, 1), 3),
                   "groups": pt["groups"]},
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "engine": {"name": "torch", "version": torch.__version__, "device": str(device),
                   "threads": torch.get_num_threads(),
                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
        "split": {"from": "step1", "col": args.split_col, "spec": split_spec, "unit": unit_kind,
                  "n_units": n_pairs, "n_test_units": n_test_pairs,
                  "observed_test_frac": float(is_test.mean()),
                  "rows": {"train": int(len(train)), "test": int(len(test)), "dropped": n_dropped}},
        "sizes": {"train_plays": int(len(train)), "test_plays": int(len(test)),
                  "dropped_rows": n_dropped, "n_entities": len(ents), "n_items": len(items),
                  "train_rows_in_vocab": int(tr_ok.sum()), "test_rows_covered": int(te_ok.sum())},
        "content": {"mode": "cm3p", "path": str(args.content_path), "dim": d_raw, "reduce": args.content_reduce,
                    "n_charts_with_vector": int(len(content_ids)), "filtered_rows": content_drop,
                    "note": "C is frozen, g_theta is shared across charts: per-chart free params = bm only"},
        "time_drift": None if args.p_drift != "time" else {"t_ref_year_ce": round(t_ref, 4),
                                                           "unit": "years"},
        "results": {f"mlp{args.dim}": res},
        "curve": curve,
        "degeneracy": degeneracy,
        "checks": chk.items,
        "seconds": round(time.time() - t_start, 1),
        "preds_path": None if preds_path is None else str(preds_path),
    }
    merged_path = out_dir / f"{args.out_prefix}_{args.tag}_summary_{args.target}.json"
    merged = json.loads(merged_path.read_text()) if merged_path.exists() else {"runs": {}}
    key = variant if variant not in merged["runs"] else f"{variant}__{int(time.time())}"
    merged["runs"][key] = summary
    merged_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    print(f"\nwrote {npz_path.name} + {merged_path.name} ({summary['seconds']}s total)")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
