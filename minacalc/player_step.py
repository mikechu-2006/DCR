"""逐个玩家精确求解技能向量（Levenberg-Marquardt，6 维）。

玩家之间互相独立：给定 θ，X 的最优解 = 891 个互不相关的 6 维最小二乘问题。

    min_x  Σ_i [ ln((S_i+T_i)/Max_i) - loss_i ]^2  +  lam * ||x - mu||^2
           s.t. x ∈ [0, 60]

Jacobian 用 6 次有限差分（参数只有 6 个，比反向的 n 次 vjp 便宜得多）。
未激活维（该列 J 全零）冻结在先验 mu。
"""

from __future__ import annotations

import numpy as np
import torch


def _resid(sc, x, pidx, cidx, loss, theta, L):
    out = sc.forward(x, pidx, cidx, theta, L=L)
    tot = (out["S"] + out["T"]).clamp_min(1e-6)
    return torch.log(tot) - torch.log(out["Max"].clamp_min(1e-9)) - loss


@torch.no_grad()
def solve_player(sc, plays_idx, x0, cidx, loss, theta, dev,
                 iters: int = 12, lam: float = 1e-3, h: float = 1e-3,
                 bounds=(0.0, 60.0), tol: float = 1e-7):
    """返回 (x (6,), info dict)。plays_idx 只是长度。"""
    n = len(loss)
    pidx = torch.zeros(n, dtype=torch.long, device=dev)
    cidx = cidx.to(dev); loss = loss.to(dev)
    L = int(sc.n[cidx].max())
    x = x0.clone().to(dev)
    mu = x0.clone().to(dev).detach()
    r = _resid(sc, x.unsqueeze(0), pidx, cidx, loss, theta, L)
    cost = float((r * r).sum()) + lam * float(((x - mu) ** 2).sum())
    tau = 1e-3
    for _ in range(iters):
        J = torch.zeros(n, 6, device=dev)
        for k in range(6):
            xp = x.clone(); xp[k] += h
            J[:, k] = (_resid(sc, xp.unsqueeze(0), pidx, cidx, loss, theta, L) - r) / h
        act = J.norm(dim=0) > 1e-4 * max(n, 1) ** 0.5
        JtJ = J.T @ J
        g = J.T @ r
        A = JtJ + lam * torch.eye(6, device=dev)
        A = A + torch.diag(torch.where(act, torch.full((6,), tau, device=dev),
                                       torch.full((6,), 1e9, device=dev)))
        b = -(g + lam * (x - mu))
        try:
            dx = torch.linalg.solve(A, b)
        except Exception:
            break
        x_new = torch.clamp(x + dx, bounds[0], bounds[1])
        r_new = _resid(sc, x_new.unsqueeze(0), pidx, cidx, loss, theta, L)
        cost_new = float((r_new * r_new).sum()) + lam * float(((x_new - mu) ** 2).sum())
        if cost_new < cost:
            x, r, cost = x_new, r_new, cost_new
            tau = max(tau / 3.0, 1e-6)
            if float((dx * dx).abs().max()) < tol:
                break
        else:
            tau = min(tau * 10.0, 1e8)
    rmse = float(torch.sqrt((r * r).mean())) if n else float("nan")
    return x, {"rmse": rmse, "n": int(n), "active": act.cpu().numpy()}


@torch.no_grad()
def init_player(sc, cidx, loss, theta, dev, mu, grid=np.arange(6.0, 34.5, 1.5)):
    """一维扫描：找使预测失分最接近观测中位数的那条等技能线。"""
    med = float(loss.median())
    best, bx = 1e9, float(mu.mean())
    for g in grid:
        x = torch.full((1, 6), float(g), device=dev)
        r = _resid(sc, x, torch.zeros(len(loss), dtype=torch.long, device=dev),
                   cidx.to(dev), torch.full_like(loss.to(dev), med),
                   theta, int(sc.n[cidx].max()))
        v = abs(float(r.mean()))
        if v < best:
            best, bx = v, float(g)
    return torch.full((6,), bx, device=dev)
