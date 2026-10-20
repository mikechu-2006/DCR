"""P1: torch 前向（与 minacalc.core.score 数值一致），带 aggregate 隐函数反传。

结构（与 core.py 一一对应，见 legacy/docs/step2_fit_skills.md §4.1）：

    d  = diffs * stamina_mods(d, x, stam)        # (N,6,2,L)  逐区间疲劳
    t  = clip(1 - x/d, 0, 1)
    q  = c1 t + c2 t^2 + c3 t^3                  # 三次式，c = c_tech / c_rest
    v  = q * pts                                 # (N,2,6,L)  每手每区间每维失分
    v[:,:,3] = jack 事件模型（amp*erf(slope*(y'-x))）
    Lc = mult * aggregate(v; delta, beta)        # (N,2,L)    softplus + Newton
    T  = sqrt(min(0.1*Max, 0.85*sum(amp*erf(slope*(y'-0.75x_tech)))))
    score = 1 - (sum Lc + T) / Max

aggregate 的 12 轮 Newton **不进计算图**：backward 用隐函数定理，
dr/dv_j = -(dF/dv_j) / (dF/dr)，代价 O(1)、无额外显存。
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .data import Charts

DIMS = ("stream", "js", "hs", "jackspeed", "cj", "tech")
TECH = 5
JACK = 3

_FOUR_OVER_SQRT_PI = 4.0 / math.sqrt(math.pi)
_LOG2 = math.log(2.0)

# jack_stam 的冻结常数（ceil, mag, fscale, prop, super_ceil, 起始 floor）
JACK_STAM = (1.05234, 23.0, 750.0, 0.49424, 1.01, 0.95)


TH0 = (2.0, -1.0, 0.0,            # c_tech
       1.7, -0.595, -0.0595,        # c_rest
       0.25,                        # delta
       12.0, 0.04,                  # jack amp / slope
       1.075234, 243.0, 500.0, 0.69424, 1.09, 0.95)   # stamina_mods
TH_SCALE = (1.7, 0.6, 0.06, 1.7, 0.6, 0.06, 0.25, 12.0, 0.04,
            1.075234, 243.0, 500.0, 0.69424, 1.09, 0.95)


class Theta(nn.Module):
    """全部全局参数（docs §4.3）。th0 严格等于 MinaCalc 505。

    参数化：v = TH0 + TH_SCALE * du，du 是优化变量（初值 0）。
    分两块是为了阶段 B1（损失形状/聚合）与 B2（疲劳常数）能分别优化：
        du1[:6]  = c（tech 在前 3，其余五维共享后 3）
        du1[6:9] = delta, jack_amp, jack_slope
        du2      = stamina_mods 的 6 个常数
    """

    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.du1 = nn.Parameter(torch.zeros(9, dtype=dtype))
        self.du2 = nn.Parameter(torch.zeros(6, dtype=dtype))
        self.register_buffer("th0", torch.tensor(TH0, dtype=dtype))
        self.register_buffer("scale", torch.tensor(TH_SCALE, dtype=dtype))
        self.mult = 1.11          # 冻结
        self.beta = 5.0           # 冻结
        self.iters = 32

    def vec(self):
        return self.th0 + self.scale * torch.cat([self.du1, self.du2])

    @property
    def c_tech(self):
        return self.vec()[0:3]

    @property
    def c_rest(self):
        return self.vec()[3:6]

    @property
    def delta(self):
        return self.vec()[6].clamp_min(0.02)

    @property
    def jack_amp(self):
        return self.vec()[7].clamp_min(0.5)

    @property
    def jack_slope(self):
        return self.vec()[8].clamp_min(0.005)

    @property
    def stam(self):
        """区间疲劳 6 常数，夹到物理合理范围（防止 L-BFGS 跑飞）。"""
        v = self.vec()[9:15]
        lo = torch.tensor([1.00, 50.0, 100.0, 0.30, 1.00, 0.80],
                          dtype=v.dtype, device=v.device)
        hi = torch.tensor([1.20, 600.0, 2000.0, 1.00, 1.20, 1.00],
                          dtype=v.dtype, device=v.device)
        return torch.minimum(torch.maximum(v, lo), hi)

    def coef(self):
        """(6,3) 逐维系数：tech 用 c_tech，其余 c_rest。"""
        v = self.vec()
        c = v[3:6].unsqueeze(0).expand(6, 3).clone()
        c[TECH] = v[0:3]
        return c

    def prior_penalty(self):
        return (self.du1 ** 2).mean() + (self.du2 ** 2).mean()


# ---------------------------------------------------------------- stamina ----

def stamina_mods_t(d, x, stam):
    """core.stamina_mods 的向量化版：返回**倍率** mods，与 d 同形。

    d (N,6,2,L)，x (N,6)，stam (6,)。调用方自己做 d * mods（与 numpy 版一致）。
    """
    ceil, mag, fscale, prop, super_ceil, floor0 = stam
    ref = (prop * x.clamp_min(0.0)).clamp_min(1e-6)          # (N,6)
    ref = ref.unsqueeze(-1)                                   # (N,6,1)
    floor = torch.zeros_like(d[..., 0]) + floor0
    mod = torch.zeros_like(d[..., 0]) + floor0
    prev = torch.zeros_like(d[..., 0])
    out = torch.empty_like(d)
    cap = ceil * floor
    for i in range(d.shape[-1]):
        cur = d[..., i]
        mod = mod + (((prev + cur) * 0.5) / ref - 1.0) / mag
        floor = floor + torch.where(mod > 0.95, (mod - 0.95) / fscale,
                                    torch.zeros_like(mod))
        mod = torch.maximum(torch.minimum(mod, cap), floor)
        mod = torch.minimum(mod, super_ceil)
        out[..., i] = mod
        prev = cur
        cap = ceil * floor
    # x<=0 的维度 numpy 版直接返回 1.0（ref=0 分支）
    dead = (x <= 0.0)
    if dead.any():
        out = torch.where(dead[:, :, None, None],
                          torch.ones_like(out), out)
    return out


def jack_stam_t(y, x, stam_j):
    """core.jack_stam 的向量化版。y (N,2,E)，x (N,)，返回 (N,2,E)。"""
    ceil, mag, fscale, prop, super_ceil, floor0 = stam_j
    ref = (prop * x.clamp_min(0.0)).clamp_min(1e-6)           # (N,)
    floor = torch.zeros_like(y[..., 0]) + floor0             # (N,2)
    mod = torch.ones_like(floor)
    prev = torch.zeros_like(floor)
    out = torch.empty_like(y)
    cap = ceil * floor
    r = ref.unsqueeze(-1)
    for i in range(y.shape[-1]):
        cur = y[..., i]
        mod = mod + (((prev + cur) * 0.5) / r - 1.0) / mag
        floor = floor + torch.where(mod > 0.95, (mod - 0.95) / fscale,
                                    torch.zeros_like(mod))
        mod = torch.maximum(torch.minimum(mod, cap), floor)
        mod = torch.minimum(mod, super_ceil)
        out[..., i] = cur * mod
        prev = cur
        cap = ceil * floor
    dead = (x <= 0.0)
    if dead.any():
        out = torch.where(dead[:, None, None], y, out)
    return out


# -------------------------------------------------------------- aggregate ----

def _agg_fw(v, r, delta, beta):
    x = torch.clamp(delta * (v - r.unsqueeze(-2)), -27.0, 27.0)
    e = torch.erfc(x)
    # 2/erfc - 2 == 2*erf/erfc；后者在 x->0 时没有相消误差
    Y = torch.clamp(2.0 * torch.erf(x) / e, -1.0, 200.0)
    w = torch.nn.functional.softplus(beta * Y) / beta
    budget = torch.pow(2.0, 0.1 * r)
    return w.sum(-2) - budget, w


def _solve_cols(vf, delta, beta, iters):
    """vf (K, P) -> r (P,)，纯二分。"""
    K, P = vf.shape
    lo = torch.zeros(P, dtype=vf.dtype, device=vf.device)
    hi = vf.max(0).values + 10.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        x = torch.clamp(delta * (vf - mid.unsqueeze(0)), -27.0, 27.0)
        e = torch.erfc(x)
        z = torch.clamp(2.0 * torch.erf(x) / e, -1.0, 200.0)
        w = torch.nn.functional.softplus(beta * z) / beta
        f = w.sum(0) - torch.pow(2.0, 0.1 * mid)
        lo = torch.where(f > 0.0, mid, lo)
        hi = torch.where(f < 0.0, mid, hi)
    return 0.5 * (lo + hi)


def agg_newton(v, delta, beta, iters=32):
    """v (N,2,K,L) -> r (N,2,L)。纯二分 + **整列跳过**。

    F(r) = sum_j w_j(r) - 2^(0.1r) 对 r 严格单调递减，二分必然收敛。
    6 维全为 0 的列（玩家全面打得动这段）解必为 0（F(0) = 6*ln2/beta - 1 < 0），
    直接跳过；这是恒等变换，不是近似——v 在 clip 边界上，上游梯度本来也是 0。
    """
    N, H, K, L = v.shape
    vf = v.permute(2, 0, 1, 3).reshape(K, -1)          # (K, N*H*L)
    active = vf.abs().sum(0) > 0                        # 全零列
    r = torch.zeros(vf.shape[1], dtype=v.dtype, device=v.device)
    if bool(active.any()):
        r = r.index_copy(0, active.nonzero(as_tuple=True)[0],
                         _solve_cols(vf[:, active], delta, beta, iters))
    return r.reshape(N, H, L)


class AggregateImplicit(torch.autograd.Function):
    """forward: Newton；backward: 隐函数定理（不对迭代反传）。"""

    @staticmethod
    def forward(ctx, v, delta, beta, iters):
        N, H, K, L = v.shape
        vf = v.permute(2, 0, 1, 3).reshape(K, -1)
        active = vf.abs().sum(0) > 0
        r = torch.zeros(vf.shape[1], dtype=v.dtype, device=v.device)
        if bool(active.any()):
            idx = active.nonzero(as_tuple=True)[0]
            r = r.index_copy(0, idx, _solve_cols(vf[:, active], delta, beta, iters))
        ctx.save_for_backward(v, r.reshape(N, H, L), active)
        ctx.delta, ctx.beta, ctx.iters = delta, beta, iters
        return r.reshape(N, H, L)

    @staticmethod
    def backward(ctx, g):
        v, r, active = ctx.saved_tensors
        delta = ctx.delta
        x = torch.clamp(delta * (v - r.unsqueeze(-2)), -27.0, 27.0)
        e = torch.erfc(x)
        Y = torch.clamp(2.0 * torch.erf(x) / e, -1.0, 200.0)
        sig = 0.5 * (1.0 + torch.tanh(0.5 * ctx.beta * Y))
        A = sig * _FOUR_OVER_SQRT_PI * torch.exp(-x * x) / (e * e)   # dw/dx
        budget = torch.pow(2.0, 0.1 * r)
        G = -(delta * A).sum(-2) - 0.1 * _LOG2 * budget              # dF/dr
        gv = (-g.unsqueeze(-2) * delta * A) / G.unsqueeze(-2)
        gd = (-g * (A * (v - r.unsqueeze(-2))).sum(-2)) / G
        # 全零列是边界解（r=0 由约束决定，不是 F=0），其梯度必须置 0
        N, H, K, L = v.shape
        act4 = active.reshape(N, H, 1, L).to(gv.dtype)
        gv = gv * act4
        gd = gd * active.reshape(N, H, L)
        return gv, gd.sum(), None, None


# ------------------------------------------------------------------ model ----

class TorchScorer:
    def __init__(self, charts: Charts, device="cpu", dtype=torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype
        z = lambda a: torch.as_tensor(a, device=self.device)
        self.n = z(charts.n).long()
        self.off = z(charts.off).long()
        self.diffs = z(charts.diffs)                       # (6,2,T)
        self.pts = z(charts.pts)                           # (2,T)
        self.jack_off = z(charts.jack_off).long()          # (2,Nc+1)
        self.jack_itv = z(charts.jack_itv).long()
        self.jack_y = z(charts.jack_y)

    # ---- gather ----
    def gather(self, cidx: torch.Tensor, L: int):
        n = self.n[cidx]
        ar = torch.arange(L, device=self.device)
        flat = torch.clamp(self.off[cidx][:, None] + ar[None, :], max=int(self.off[-1]) - 1)
        mask = ar[None, :] < n[:, None]
        d = self.diffs[:, :, flat].permute(2, 0, 1, 3) * mask[:, None, None, :]
        p = self.pts[:, flat].permute(1, 0, 2) * mask[:, None, :]
        return d, p, mask

    def jack_events(self, cidx: torch.Tensor, L: int):
        """把每手 jack 事件展开到 (N,2,E)，E = 本 batch 最大事件数。"""
        N = cidx.shape[0]
        starts = self.jack_off[:, cidx]                       # (2,N)
        ends = self.jack_off[:, cidx + 1]
        cnt = (ends - starts).T                               # (N,2)
        E = int(cnt.max()) if N else 0
        if E == 0:
            z = torch.zeros((N, 2, 0), device=self.device)
            return z, z, z.long(), torch.ones((N, 2), dtype=torch.bool, device=self.device)
        pos = torch.arange(E, device=self.device)
        idx = starts.T[:, :, None] + pos[None, None, :].to(starts.dtype)  # (N,2,E)
        valid = pos[None, None, :] < cnt[:, :, None]
        idx = torch.clamp(idx, max=max(int(self.jack_off.max()) - 1, 0))
        y = self.jack_y[idx] * valid
        itv = self.jack_itv[idx] * valid
        return y, valid, itv, valid

    # ---- forward ----
    def forward(self, X, pidx, cidx, theta: Theta, L: int | None = None,
                detach_stam: bool = False, debug: bool = False):
        """X (P,6) 玩家技能；pidx (N,)、cidx (N,)。返回 dict。"""
        N = cidx.shape[0]
        L = int(self.n[cidx].max()) if L is None else int(L)
        d, pts, mask = self.gather(cidx, L)
        x = X[pidx]                                            # (N,6)
        x_st = x.detach() if detach_stam else x
        d = d * stamina_mods_t(d, x_st, theta.stam)

        u = torch.where(d > 0, x[:, :, None, None] / d.clamp_min(1e-9),
                        torch.full_like(d, 1e9))
        t = torch.clamp(1.0 - u, 0.0, 1.0)
        c = theta.coef()                                       # (6,3)
        q = (c[:, 0][None, :, None, None] * t
             + c[:, 1][None, :, None, None] * t * t
             + c[:, 2][None, :, None, None] * t * t * t)
        q = torch.clamp(q, 0.0, 1.0)
        v = q * pts[:, None, :, :]                             # (N,6,2,L)

        # ---- jack（事件模型）----
        T_extra = torch.zeros(N, device=self.device)
        jy, jvalid, jitv, _ = self.jack_events(cidx, L)
        if jy.shape[-1] > 0:
            # jack_stam 的 7 个常数冻结（§4.3-F），与区间版 stamina_mods 无关
            stam_j = torch.tensor(list(JACK_STAM), device=self.device,
                                  dtype=self.dtype)
            ys = jack_stam_t(jy, x_st[:, JACK], stam_j)
            pen = theta.jack_amp * torch.erf(theta.jack_slope * (ys - x_st[:, JACK, None, None]))
            pen = torch.where((ys > x_st[:, JACK, None, None]) & jvalid, pen,
                              torch.zeros_like(pen))
            vj = torch.zeros((N, 2, L), device=self.device, dtype=self.dtype)
            vj.scatter_add_(2, jitv.clamp(0, L - 1),
                            pen * jvalid.to(self.dtype))
            v = v.clone()
            v[:, JACK] = vj
            # tech 额外项：**另一次** jack_stam，参考值是 0.75*x_tech（不是 x_jackspeed）
            x_tech75 = 0.75 * x_st[:, TECH]
            ys_t = jack_stam_t(jy, x_tech75, stam_j)
            pen_t = theta.jack_amp * torch.erf(
                theta.jack_slope * (ys_t - x_tech75[:, None, None]))
            pen_t = torch.where((ys_t > x_tech75[:, None, None]) & jvalid, pen_t,
                                torch.zeros_like(pen_t))
            cap = 0.1 * pts.sum((1, 2))                       # (N,)
            per_hand = pen_t.sum(-1)                          # (N,2)
            T_extra = torch.sqrt(
                torch.minimum(cap[:, None], 0.85 * per_hand)).sum(1)

        v = v.permute(0, 2, 1, 3)                              # (N,2,6,L)
        r = AggregateImplicit.apply(v, theta.delta, theta.beta, theta.iters)
        S = theta.mult * (r * mask[:, None, :]).sum((1, 2))
        Max = pts.sum((1, 2))
        score = 1.0 - (S + T_extra) / Max.clamp_min(1e-9)
        out = {"score": score, "S": S, "T": T_extra, "Max": Max}
        if debug:
            out.update({"v": v, "r": r, "mask": mask, "jy": jy, "jvalid": jvalid,
                        "jitv": jitv, "vj": vj if jy.shape[-1] > 0 else None,
                        "pen_t": pen_t if jy.shape[-1] > 0 else None,
                        "ys_t": ys_t if jy.shape[-1] > 0 else None})
        return out
