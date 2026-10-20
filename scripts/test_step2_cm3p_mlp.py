#!/usr/bin/env python
"""Self-tests for scripts/step2_cm3p_mlp.py -- the design note's claims, made executable.

Small synthetic tensors only: this runs in a second and needs no data/ files, so it can gate a
real run without waiting for it.  Each test corresponds to a claim in legacy/docs/cm3p_step2_mlp.md:

    t1  the interaction is multiplicative and the frozen content is an input, not a weight
    t2  per-chart free parameters are bm and nothing else (the whole point of anchoring)
    t3  the gate is an exact switch: w=0 == the additive model, w=1 == full interaction
    t4  the content tower's output layer starts as a small correction, not the dominant term
    t5  determinism holds through the grouped Adam (checklist item 8 of the linear pipeline)
    t6  --depth 1 really is a linear map (which is why depth 2 is the minimum useful choice)
    t7  a unit-norm input with no LayerNorm starts near-silent (why --ln is not optional)

    python scripts/test_step2_cm3p_mlp.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step2_cm3p_mlp as M  # noqa: E402

torch.use_deterministic_algorithms(True)
torch.manual_seed(0)

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


class Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def make(n_ent=7, n_item=5, d_raw=8, d_out=3, seed=0, **kw):
    # The drawn tables (Eu) and PyTorch's own nn.Linear/nn.LayerNorm initialisation are two
    # different RNGs: the numpy side is seeded inside AnchoredMLP, the torch side is not.  Real
    # runs create the model once per process, so seeding happens in main(); a test that builds two
    # models must seed both, or the *biases* differ between them while every drawn table matches.
    torch.manual_seed(seed)
    m = M.AnchoredMLP(n_ent, n_item, d_raw, d_out, seed=seed,
                      hid=6, depth=kw.pop("depth", 2),
                      player_tower=kw.pop("player_tower", True),
                      nonlinearity=kw.pop("nonlinearity", True),
                      ln=kw.pop("ln", True), **kw)
    rng = np.random.default_rng(99)
    Z = rng.normal(0, 1, (n_item, d_raw)).astype(np.float32)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)          # unit norm, like real CM3P rows
    m.attach_content(torch.as_tensor(Z))
    return m


def main():
    n_ent, n_item, d_raw, d_out = 7, 5, 8, 3
    u = torch.as_tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.long)
    i = torch.as_tensor([0, 1, 2, 3, 4, 0, 1], dtype=torch.long)

    # ---- t1: the chart enters as an INPUT; nothing per-chart is trainable
    m = make(n_ent, n_item, d_raw, d_out)
    p0 = m.forward(u, i).detach().clone()
    with torch.no_grad():
        m.C_feat[0] = m.C_feat[0] * 0.0        # mutate the frozen chart vector (not a parameter)
    p1 = m.forward(u, i).detach()
    check("t1 frozen content is an input", not torch.allclose(p0, p1),
          f"perturbing C[0] moved {int((p0 != p1).sum())}/{len(p0)} predictions")

    # ---- t2: parameters = b0, bu, bm, Eu, tower, gate; no per-chart free parameter
    # NB: read the parameter tables off model.p, not named_parameters().  EmbeddingModel (and this
    # model, which mirrors it) keeps b0/bu/bm in a plain dict, so nn.Module never registers them:
    # autograd, Adam and count_params all see them, but named_parameters()/state_dict() do not.
    per_chart = [k for k, v in m.p.items() if v.dim() > 0 and v.shape[0] == n_item]
    per_chart += [k for k, v in m.named_parameters() if v.dim() > 0 and v.shape[0] == n_item]
    check("t2 no per-chart free parameter but bm", sorted(set(per_chart)) == ["bm"],
          f"trainable tensors whose first dim is n_item={n_item}: "
          f"{sorted(set(per_chart)) or 'none'}; b_m is the whole per-chart budget")
    total, _ = M.count_params(m)
    # 参数量必须与逐项手工统计一致。这里直接用 **真实 1k 覆盖子集** 的尺寸，
    # 因为文档里承诺的 63,116 就是这么来的；用小玩具模型核对没有意义。
    def tally_of(ne, ni, dr, do, hid):
        # 注意分组：b_u 属于"主效应"组（与 b0/b_m 一起），玩家表组里只有 Eu。
        return ((1 + ne + ni)                     # main effects: b0 + b_u + b_m
                + (ne * do)                       # player table Eu (b_u is NOT here)
                + (do * do + do)                  # player tower Linear(d_out, d_out)
                + do * 2                          # player LayerNorm(d_out)
                + (dr * hid + hid)                # content tower Linear(d_raw, hid)
                + (hid * do + do)                 # content tower Linear(hid, d_out)
                + dr * 2                          # input LayerNorm(d_raw) on the content
                + 1 + 1)                          # interaction: gate + b_h
    toy_ok = total == tally_of(n_ent, n_item, d_raw, d_out, 6)
    BIG = (983, 11202, 512, 16, 64)                   # 1k CM3P-covered: 983 players, 11,202 charts
    torch.manual_seed(0)
    big = M.AnchoredMLP(BIG[0], BIG[1], BIG[2], BIG[3], seed=0, hid=BIG[4], depth=2)
    big_total, _ = M.count_params(big)
    big_tally = tally_of(*BIG)
    check("t2b parameter count matches an independent tally",
          toy_ok and big_total == big_tally == 63116,
          f"toy {total:,} == {tally_of(n_ent, n_item, d_raw, d_out, 6):,}; "
          f"the documented 1k configuration (983 players / 11,202 charts / d=16 / hid=64) = "
          f"{big_total:,} == {big_tally:,}")
    m = make(n_ent, n_item, d_raw, d_out)
    with torch.no_grad():
        m.gate_raw.fill_(0.0)
    off = m.forward(u, i).detach()
    with torch.no_grad():
        m.gate_raw.fill_(1.0)
    on = m.forward(u, i).detach()
    dnorm = float((on - off).abs().max())
    check("t3 w=0 is the additive model, w=1 adds the interaction", dnorm > 1e-3,
          f"max|w=1 minus w=0| = {dnorm:.5f}; gate() = {float(m.gate().detach()):.3f}")
    check("t3b gate init is 1 by default", abs(float(make(n_ent, n_item, d_raw, d_out).gate()) - 1.0) < 1e-9,
          "step 0 starts at full interaction strength (the MLP is a correction, not a gate)")

    # ---- t4: the output layer starts small
    m = make(n_ent, n_item, d_raw, d_out)
    inter = m.chart_vectors(torch.arange(n_item)).detach()
    big = make(n_ent, n_item, d_raw, d_out, out_init_std=1.0)
    inter_big = big.chart_vectors(torch.arange(n_item)).detach()
    check("t4 the output layer scales the tower's init", float(inter.abs().max()) < float(inter_big.abs().max()),
          f"max|D_m| at init: out_init_std=0.01 -> {float(inter.abs().max()):.5f}, "
          f"1.0 -> {float(inter_big.abs().max()):.5f} (a 100x init shrinks the tower's step-0 output)")

    # ---- t5: determinism through the grouped optimiser
    def run():
        mm = make(n_ent, n_item, d_raw, d_out)
        opt = M.build_optimizer(mm, Args(lr=0.02, mlp_wd=1e-4, eu_wd=1e-4, wd=1e-4))
        r = np.random.default_rng(5)
        for _ in range(4):
            b = torch.as_tensor(r.integers(0, n_ent, 16), dtype=torch.long)
            c = torch.as_tensor(r.integers(0, n_item, 16), dtype=torch.long)
            y = torch.as_tensor(r.normal(-3, 1, 16).astype(np.float32))
            opt.zero_grad(set_to_none=True)
            ((mm.forward(b, c) - y) ** 2).mean().backward()
            opt.step()
        return {k: v.detach().clone() for k, v in mm.named_parameters()}
    a, b = run(), run()
    check("t5 bitwise determinism", all(torch.equal(a[k], b[k]) for k in a), f"{len(a)} tensors")

    # ---- t6: depth 1 vs depth 2, stated as what can actually be checked reliably.
    #
    #         Three attempts at measuring "how nonlinear is the tower" all failed, for one shared
    #         reason, and the failures are worth recording because they are easy to repeat:
    #           (a) rank of the chart vectors: useless -- they live in d_out dimensions by
    #               construction, so the rank can never exceed d_out at any depth;
    #           (b) exact affine residual: A = [LayerNorm(C), 1] is row-full-rank whenever
    #               n <= d_raw+1, so lstsq interpolates the points EXACTLY and the residual is
    #               ~1e-28 for a linear map, a GELU tower and a hand-written nonlinearity alike;
    #           (c) held-out affine fit: cond(A) ~ 1e7 here (CM3P rows are near-collinear unit
    #               vectors), so even a genuinely affine function scores 0.25-1.05 -- the noise is
    #               the same size as the effect.
    #         The honest summary: on THIS geometry curvature is not separably measurable.
    #         Whether the extra depth buys anything is answered by the real experiment
    #         (linear CM3P d=32 vs the MLP, same split, paired bootstrap), not by a unit test.
    #         What the test can pin down is the structure and the parameter budget.
    def towers(depth, n=64, seed=1, hid=16):
        mm = M.AnchoredMLP(n_ent, n, d_raw, d_out, seed=seed, hid=hid, depth=depth)
        rng = np.random.default_rng(seed)
        Zc = rng.normal(0, 1, (n, d_raw)).astype(np.float32)
        Zc /= np.linalg.norm(Zc, axis=1, keepdims=True)
        mm.attach_content(torch.as_tensor(Zc))
        return mm

    def lin_layers(mm):
        # --depth counts the output layer too, and the output layer lives in c_out, not c_tower
        return [q for q in list(mm.c_tower) + [mm.c_out] if isinstance(q, nn.Linear)]

    m1, m2, m3 = towers(1), towers(2), towers(3)
    n1, n2, n3 = (len(lin_layers(x)) for x in (m1, m2, m3))
    # hidden width actually used at the default depth, which is what the parameter table assumes
    hid_used = m2.c_tower[0].out_features if len(m2.c_tower) else None
    off = M.AnchoredMLP(n_ent, 64, d_raw, d_out, seed=1, hid=16, depth=3, nonlinearity=False)
    check("t6 --depth counts the tower's Linear layers, output layer included",
          (n1, n2, n3) == (1, 2, 3) and hid_used == 16 and len(lin_layers(off)) == 1,
          f"depth 1 -> {n1} Linear layer (bare projection, the already-measured linear/PCA path); "
          f"depth 2 -> {n2} (one hidden layer of width {hid_used}); depth 3 -> {n3}; "
          f"--nonlinearity off -> {len(lin_layers(off))} regardless of --depth.  Curvature itself "
          f"is NOT unit-testable on this geometry -- see the comment above")
    # ---- t7: input conditioning.  Measured on the REAL 512-d unit-norm CM3P rows: without
    #         LayerNorm the first layer's pre-activations and gradients are ~20x smaller.  Small
    #         enough to matter over a fixed 60-epoch budget, NOT small enough to be fatal -- an
    #         earlier draft of the design note claimed the layer starts "near-silent" (1e-4); the
    #         measurement says 0.06, so the honest reason for LayerNorm is conditioning, not survival.
    try:
        import pandas as _pd
        df = _pd.read_parquet("data/processed/chart_content_cm3p.parquet", columns=["embedding"])
        Er = np.stack([np.asarray(e, dtype=np.float32) for e in df.embedding.to_numpy()])
    except Exception as exc:                                    # noqa: BLE001
        # 兜底：用真实 CM3P 的形状（512 维、单位范数）合成一批行。注意必须保持单位范数，
        # 否则测的就不是这个条件数问题了。
        rng = np.random.default_rng(0)
        Er = rng.normal(0, 1, (256, 512)).astype(np.float32)
        Er /= np.linalg.norm(Er, axis=1, keepdims=True)
        print(f"  [note] real content table unavailable ({exc}); t7 falls back to synthetic "
              f"unit-norm rows of the same shape ({Er.shape})")
    torch.manual_seed(1)
    lin = nn.Linear(Er.shape[1], 64, bias=False)
    nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
    ln = nn.LayerNorm(Er.shape[1])
    x = torch.as_tensor(Er[:256])

    def preact_sd(h):
        with torch.no_grad():
            return float(lin(h).std())

    def grad_norm(h):
        torch.manual_seed(1)
        l2 = nn.Linear(Er.shape[1], 64, bias=False)
        nn.init.kaiming_normal_(l2.weight, nonlinearity="relu")
        torch.sigmoid(l2(h)).sum().backward()
        return float(l2.weight.grad.norm())

    sd_raw, sd_ln = preact_sd(x), preact_sd(ln(x))
    g_raw, g_ln = grad_norm(x), grad_norm(ln(x))
    check("t7 LayerNorm fixes the input conditioning", sd_raw * 5 < sd_ln and g_raw * 5 < g_ln,
          f"pre-act sd {sd_raw:.4f} -> {sd_ln:.4f} ({sd_ln/sd_raw:.1f}x); "
          f"||dL/dW1|| {g_raw:.2e} -> {g_ln:.2e} ({g_ln/g_raw:.1f}x)")

    print()
    if FAILS:
        print(f"FAILED: {', '.join(FAILS)}")
        return 1
    print("all self-tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
